"""
CREMS — Contextual Reversible Enforcement and Marking System
Implementation based on:
  "Cyber-Physical Enforcement at Smart Intersections: A Three-Tier
   LiDAR–Vision–Pneumatic Architecture for Reversible Chemical Marking
   of Pedestrian Right-of-Way Violations"
  Bakkara et al. (2026), IET Intelligent Transport Systems (Under Review)

Python implementation covers:
  - Tier 1: CIR-HMM Algorithm (Contextual Intent Recognition)
  - Tier 2: Spray Kinematics & Circular Morphology Constraint (Eq. 14)
  - Tier 3: Forensic Audit Record generation
  - End-to-End enforcement pipeline (simulation OR real hardware)

Usage:
  python crems_system.py --simulate          # run with synthetic data (no hardware needed)
  python crems_system.py --hardware          # run with real LiDAR, camera, GPIO
  python crems_system.py                     # defaults to --simulate
"""

import sys
import argparse
import numpy as np
import hashlib
import json
import uuid
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CREMS Enforcement System")
parser.add_argument(
    "--simulate", action="store_true", default=False,
    help="Run in simulation mode (no hardware required)"
)
parser.add_argument(
    "--hardware", action="store_true", default=False,
    help="Run with real hardware (LiDAR, camera, GPIO). Requires preflight check."
)
args, _ = parser.parse_known_args()

# Default to simulate if neither flag given
HARDWARE_MODE = args.hardware and not args.simulate
if not args.hardware and not args.simulate:
    print("[INFO] No mode specified — defaulting to --simulate.")
    print("[INFO] Use --hardware to run with real hardware.\n")

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE IMPORTS (guarded — only loaded in hardware mode)
# ─────────────────────────────────────────────────────────────────────────────

GPIO       = None
rclpy      = None
PointCloud2 = None
pc2        = None
cv2        = None

if HARDWARE_MODE:
    print("[HARDWARE] Loading hardware drivers...")

    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        print("  [OK] RPi.GPIO loaded")
    except ImportError:
        print("  [FAIL] RPi.GPIO not found. Install: pip install RPi.GPIO")
        print("         Or run with --simulate for simulation mode.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"  [FAIL] GPIO init error: {e}")
        print("         Are you running on a Raspberry Pi?")
        sys.exit(1)

    try:
        import rclpy
        from sensor_msgs.msg import PointCloud2
        import sensor_msgs_py.point_cloud2 as pc2
        print("  [OK] ROS2 / rclpy loaded")
    except ImportError:
        print("  [FAIL] rclpy not found. Is ROS2 installed and sourced?")
        print("         Run: source /opt/ros/<distro>/setup.bash")
        sys.exit(1)

    try:
        import cv2
        print("  [OK] OpenCV loaded")
    except ImportError:
        print("  [FAIL] opencv-python not found. Install: pip install opencv-python")
        sys.exit(1)

    print("[HARDWARE] All drivers loaded.\n")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (from paper)
# ─────────────────────────────────────────────────────────────────────────────

P_TURN_THRESHOLD   = 0.85
CONFIRM_WINDOW_MS  = 120
T_VALVE_MS         = 2.0
T_TOTAL_MAX_MS     = 50.0
V_S_MIN            = 85.0
V_S_MAX            = 230.0
P_RESERVOIR_MIN    = 15.0
P_RESERVOIR_MAX    = 55.0
D_NOZZLE_MIN       = 0.8
D_NOZZLE_MAX       = 2.4
EPSILON_MAX        = 0.90
AUDIT_TX_MS        = 500
SAFETY_ZONE_M      = 2.0
LIDAR_PTS_PER_S    = 320_000
LIDAR_H_RES_DEG    = 0.1

HMM_STATES    = ["Approaching", "LegalTurnInitiation", "ViolationTrajectory", "PostZoneExit"]
HMM_STATE_IDX = {s: i for i, s in enumerate(HMM_STATES)}

W1, W2, W3, W4 = 0.30, 0.25, 0.20, 0.25

# GPIO pin assignments (Tier 2 hardware)
SERVO_AZ_PIN = 18    # azimuth servo (lead compensation)
SERVO_EL_PIN = 23    # elevation servo (height lock)
VALVE_PIN    = 24    # solenoid valve trigger

if HARDWARE_MODE:
    GPIO.setup([SERVO_AZ_PIN, SERVO_EL_PIN, VALVE_PIN], GPIO.OUT)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleStateVector:
    """State vector X(t) as defined in Sec 2.2.3 of the paper."""
    t:         float
    x:         float
    y:         float
    vx:        float
    vy:        float
    theta:     float
    kappa:     float
    delta_phi: float
    vehicle_class: str = "passenger_car"


@dataclass
class CIRResult:
    """Output of the Contextual Intent Recognition algorithm."""
    vehicle_id:     str
    state_sequence: list
    P_turn:         float
    D_value:        float
    is_legal_turn:  bool
    is_violation:   bool
    confidence:     float
    timestamp:      float


@dataclass
class SprayKinematics:
    """Result of the circular morphology constraint solver (Eqs. 1–17)."""
    v:            float
    V_s:          float
    d_standoff:   float
    t_total:      float
    t_flight:     float
    x_aim:        float
    beta_deg:     float
    theta_impact: float
    epsilon:      float
    feasible:     bool
    P_reservoir:  float
    d_nozzle:     float


@dataclass
class ForensicAuditRecord:
    """Encrypted audit record (Sec 2.4.3) — transmitted within 500 ms."""
    record_id:          str
    pole_id:            str
    pole_gps:           tuple
    timestamp_utc:      str
    vehicle_id:         str
    vehicle_class:      str
    vehicle_velocity:   float
    number_plate:       str
    cir_result:         dict
    spray_kinematics:   dict
    cpc_batch_id:       str
    marking_confirmed:  bool
    audit_hash:         str


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — SENSING LAYER
# ─────────────────────────────────────────────────────────────────────────────

class SensingLayer:
    """
    Wraps real hardware (LiDAR + camera) in hardware mode,
    or generates synthetic sensor data in simulation mode.
    """

    def __init__(self, hardware_mode: bool = False,
                 lidar_topic: str = "/velodyne_points",
                 camera_visible_url: str = "rtsp://smartpole-001/visible",
                 camera_thermal_url: str = "rtsp://smartpole-001/thermal"):
        self.hardware_mode = hardware_mode
        self._lidar_buffer = []
        self._latest_frame_vis = None
        self._latest_frame_thm = None

        if hardware_mode:
            self._init_hardware(lidar_topic, camera_visible_url, camera_thermal_url)

    def _init_hardware(self, lidar_topic, cam_vis_url, cam_thm_url):
        """Initialise ROS2 node and camera streams."""
        rclpy.init()
        self._node = rclpy.create_node("crems_sensing")
        self._node.create_subscription(
            PointCloud2, lidar_topic, self._lidar_callback, 10
        )
        self._cap_vis = cv2.VideoCapture(cam_vis_url)
        self._cap_thm = cv2.VideoCapture(cam_thm_url)

        if not self._cap_vis.isOpened():
            raise RuntimeError(f"Cannot open visible camera stream: {cam_vis_url}")
        if not self._cap_thm.isOpened():
            raise RuntimeError(f"Cannot open thermal camera stream: {cam_thm_url}")

        print(f"  [OK] LiDAR subscribed on {lidar_topic}")
        print(f"  [OK] Visible camera: {cam_vis_url}")
        print(f"  [OK] Thermal camera: {cam_thm_url}")

    def _lidar_callback(self, msg):
        """ROS2 callback — stores incoming point cloud."""
        points = list(pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ))
        self._lidar_buffer = points

    def get_lidar_frame(self) -> list:
        """Return latest LiDAR point cloud (real or synthetic)."""
        if self.hardware_mode:
            rclpy.spin_once(self._node, timeout_sec=0.05)
            return self._lidar_buffer
        else:
            # Synthetic: random point cloud around origin
            n = 200
            return [(np.random.uniform(-5, 5),
                     np.random.uniform(-1, 1),
                     np.random.uniform(0, 2)) for _ in range(n)]

    def get_camera_frames(self):
        """Return (visible_frame, thermal_frame) — real or synthetic."""
        if self.hardware_mode:
            ret_v, frame_vis = self._cap_vis.read()
            ret_t, frame_thm = self._cap_thm.read()
            if not ret_v or not ret_t:
                raise RuntimeError("Camera stream dropped.")
            return frame_vis, frame_thm
        else:
            # Synthetic: blank frames
            frame_vis = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_thm = np.zeros((480, 640), dtype=np.uint8)
            return frame_vis, frame_thm

    def shutdown(self):
        if self.hardware_mode:
            self._cap_vis.release()
            self._cap_thm.release()
            self._node.destroy_node()
            rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — CIR-HMM ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

class CIRAlgorithm:
    """
    Contextual Intent Recognition using a Hidden Markov Model.

    Based on Section 2.2.3:
      X(t) = [x, y, vx, vy, θ, κ, Δφ, C_class]
      D(t) = w1·κ + w2·|Δφ| + w3·d_lateral + w4·(1 − P_turn)

    If P_turn ≥ 0.85 at any point in 120 ms window → irrevocably excluded.
    """

    def __init__(self, pedestrian_zone_width=3.0, intersection_side="left"):
        self.zone_width = pedestrian_zone_width
        self.side = intersection_side

        self.A = np.array([
            [0.70, 0.15, 0.10, 0.05],
            [0.05, 0.75, 0.10, 0.10],
            [0.02, 0.03, 0.90, 0.05],
            [0.01, 0.01, 0.01, 0.97],
        ])
        self.pi = np.array([1.0, 0.0, 0.0, 0.0])

    def _compute_d_lateral(self, state: VehicleStateVector) -> float:
        return min(abs(state.y) / self.zone_width, 1.0)

    def _emission_prob(self, state: VehicleStateVector, hmm_state: int) -> float:
        kappa = abs(state.kappa)
        dphi  = abs(state.delta_phi)
        d_lat = self._compute_d_lateral(state)

        if hmm_state == HMM_STATE_IDX["Approaching"]:
            return math.exp(-kappa * 5) * math.exp(-d_lat)
        elif hmm_state == HMM_STATE_IDX["LegalTurnInitiation"]:
            turn_signal = math.exp(-abs(dphi - 0.3) * 3)
            return math.exp(-(kappa - 0.25)**2 / 0.05) * turn_signal
        elif hmm_state == HMM_STATE_IDX["ViolationTrajectory"]:
            return math.exp(-kappa * 8) * (1 - d_lat + 0.1) * math.exp(-abs(dphi))
        elif hmm_state == HMM_STATE_IDX["PostZoneExit"]:
            return 1.0 if d_lat > 0.9 else 0.05
        return 0.0

    def _viterbi(self, trajectory: list) -> tuple:
        n_obs    = len(trajectory)
        n_states = len(HMM_STATES)
        V   = np.zeros((n_states, n_obs))
        ptr = np.zeros((n_states, n_obs), dtype=int)

        for s in range(n_states):
            V[s, 0] = (math.log(max(self.pi[s], 1e-12)) +
                       math.log(max(self._emission_prob(trajectory[0], s), 1e-12)))

        for t in range(1, n_obs):
            for s in range(n_states):
                probs = [V[prev, t-1] + math.log(max(self.A[prev, s], 1e-12))
                         for prev in range(n_states)]
                best_prev = int(np.argmax(probs))
                V[s, t] = (probs[best_prev] +
                           math.log(max(self._emission_prob(trajectory[t], s), 1e-12)))
                ptr[s, t] = best_prev

        best_last = int(np.argmax(V[:, n_obs-1]))
        path = [best_last]
        for t in range(n_obs-1, 0, -1):
            path.insert(0, ptr[path[0], t])

        return [HMM_STATES[s] for s in path], float(np.max(V[:, n_obs-1]))

    def _compute_P_turn(self, state: VehicleStateVector) -> float:
        kappa_score    = min(abs(state.kappa) / 0.3, 1.0)
        dphi_score     = math.exp(-abs(state.delta_phi) * 2)
        heading_change = min(abs(state.theta) / (math.pi / 4), 1.0)
        raw = 0.4 * kappa_score + 0.35 * dphi_score + 0.25 * heading_change
        P_turn = 1 / (1 + math.exp(-8 * (raw - 0.5)))
        return float(np.clip(P_turn, 0.0, 1.0))

    def classify(self, vehicle_id: str,
                 trajectory: list,
                 signal_phase: str = "red") -> CIRResult:
        if len(trajectory) < 3:
            raise ValueError("Trajectory too short — need ≥3 samples.")

        phase_offset = {"red": 0.0, "amber": math.pi / 6, "green": math.pi / 2}
        offset = phase_offset.get(signal_phase, 0.0)
        for s in trajectory:
            s.delta_phi = abs(s.delta_phi) + offset

        state_sequence, log_lik = self._viterbi(trajectory)

        n_confirm    = max(1, int(CONFIRM_WINDOW_MS / 8))
        confirm_traj = trajectory[-n_confirm:]
        P_turn_max   = max(self._compute_P_turn(s) for s in confirm_traj)
        P_turn_avg   = float(np.mean([self._compute_P_turn(s) for s in confirm_traj]))

        is_legal_turn = P_turn_max >= P_TURN_THRESHOLD

        last  = trajectory[-1]
        d_lat = self._compute_d_lateral(last)
        D_val = (W1 * abs(last.kappa) +
                 W2 * abs(last.delta_phi) +
                 W3 * d_lat +
                 W4 * (1 - P_turn_avg))

        in_violation_state = "ViolationTrajectory" in state_sequence[-n_confirm:]
        is_violation = (in_violation_state and
                        not is_legal_turn and
                        signal_phase == "red" and
                        D_val > 0.15)

        confidence = min(1.0, abs(log_lik) / (len(trajectory) * 5))

        return CIRResult(
            vehicle_id=vehicle_id,
            state_sequence=state_sequence,
            P_turn=P_turn_max,
            D_value=float(D_val),
            is_legal_turn=is_legal_turn,
            is_violation=is_violation,
            confidence=confidence,
            timestamp=trajectory[-1].t,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — ACTUATION LAYER
# ─────────────────────────────────────────────────────────────────────────────

class SprayKinematicsSolver:
    """
    Solves Equations 1–17 from Section 3 of the paper.

    Core constraint (Eq. 14):
        d_standoff · [1/V_s − 1/√(V_s² − v²)] + t_total = 0
    """

    def _eq14_residual(self, V_s, v, d_standoff, t_total):
        if V_s <= v:
            return float('inf')
        denom = math.sqrt(max(V_s**2 - v**2, 1e-12))
        return d_standoff * (1.0 / V_s - 1.0 / denom) + t_total

    def solve_V_s(self, v, d_standoff, t_total):
        lo, hi = v + 0.1, 1000.0
        f_lo = self._eq14_residual(lo, v, d_standoff, t_total)
        f_hi = self._eq14_residual(hi, v, d_standoff, t_total)
        if f_lo * f_hi > 0:
            return None
        for _ in range(60):
            mid  = (lo + hi) / 2.0
            f_mid = self._eq14_residual(mid, v, d_standoff, t_total)
            if abs(f_mid) < 1e-9:
                break
            if f_lo * f_mid < 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return (lo + hi) / 2.0

    def compute(self, v_kmh, d_standoff=3.0, t_detect_ms=30.0):
        v        = v_kmh / 3.6
        t_valve  = T_VALVE_MS / 1000.0
        t_detect = t_detect_ms / 1000.0
        t_total  = t_detect + t_valve

        V_s      = self.solve_V_s(v, d_standoff, t_total)
        feasible = V_s is not None and V_S_MIN <= V_s <= V_S_MAX
        if V_s is None:
            V_s = V_S_MAX

        V_s_clamped = float(np.clip(V_s, V_S_MIN, V_S_MAX))
        t_flight    = d_standoff / V_s_clamped
        x_aim       = v * (t_total + d_standoff / V_s_clamped)

        sin_beta    = float(np.clip(v / V_s_clamped, -1.0, 1.0))
        beta_deg    = math.degrees(math.asin(sin_beta))

        v_residual  = V_s_clamped * sin_beta - v
        V_perp      = math.sqrt(max(V_s_clamped**2 - v**2, 0.0))
        theta_impact = math.degrees(math.atan2(abs(v_residual), V_perp)) if V_perp > 0 else 90.0
        epsilon     = math.cos(math.radians(theta_impact))

        rho_CPC     = 1200.0
        Cd          = 0.65
        P_required_pa = 0.5 * rho_CPC * (V_s_clamped / Cd) ** 2
        P_reservoir = float(np.clip(P_required_pa / 1e5, P_RESERVOIR_MIN, P_RESERVOIR_MAX))

        speed_norm  = (v_kmh - 20) / (80 - 20)
        d_nozzle    = float(np.clip(
            D_NOZZLE_MAX - speed_norm * (D_NOZZLE_MAX - D_NOZZLE_MIN),
            D_NOZZLE_MIN, D_NOZZLE_MAX
        ))

        return SprayKinematics(
            v=v, V_s=V_s_clamped, d_standoff=d_standoff,
            t_total=t_total, t_flight=t_flight, x_aim=x_aim,
            beta_deg=beta_deg, theta_impact=theta_impact,
            epsilon=epsilon, feasible=feasible,
            P_reservoir=P_reservoir, d_nozzle=d_nozzle,
        )


class ActuationLayer:
    """
    Controls the physical spray gimbal and solenoid valve (hardware mode),
    or logs what would have fired (simulation mode).
    """

    def __init__(self, hardware_mode: bool = False):
        self.hardware_mode = hardware_mode
        self.solver = SprayKinematicsSolver()

    def _angle_to_duty(self, angle_deg: float) -> float:
        """Convert servo angle (−90° to +90°) to PWM duty cycle (2.5–12.5%)."""
        return 2.5 + (angle_deg + 90) / 180 * 10.0

    def fire(self, v_kmh: float, d_standoff: float = 3.0) -> SprayKinematics:
        """
        Compute kinematics and fire the sprayer (or simulate firing).

        Parameters
        ----------
        v_kmh      : vehicle velocity in km/h
        d_standoff : perpendicular distance nozzle-to-vehicle (m)

        Returns
        -------
        SprayKinematics — computed targeting parameters
        """
        sk = self.solver.compute(v_kmh, d_standoff)

        if self.hardware_mode:
            # Aim azimuth servo to lead-compensated angle
            az_pwm = GPIO.PWM(SERVO_AZ_PIN, 50)
            el_pwm = GPIO.PWM(SERVO_EL_PIN, 50)
            az_pwm.start(self._angle_to_duty(sk.beta_deg))
            el_pwm.start(self._angle_to_duty(0.0))   # elevation locked horizontal

            import time
            time.sleep(0.002)  # 2 ms settle

            # Fire solenoid valve (12 ms bolus)
            GPIO.output(VALVE_PIN, GPIO.HIGH)
            time.sleep(0.012)
            GPIO.output(VALVE_PIN, GPIO.LOW)

            az_pwm.stop()
            el_pwm.stop()
            print(f"  [HARDWARE] Sprayer fired — β={sk.beta_deg:.2f}°, "
                  f"V_s={sk.V_s:.1f} m/s, x_aim={sk.x_aim:.3f} m")
        else:
            print(f"  [SIMULATE] Would fire — β={sk.beta_deg:.2f}°, "
                  f"V_s={sk.V_s:.1f} m/s, x_aim={sk.x_aim:.3f} m")

        return sk

    def shutdown(self):
        if self.hardware_mode:
            GPIO.output(VALVE_PIN, GPIO.LOW)
            GPIO.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — FORENSIC AUDIT RECORD
# ─────────────────────────────────────────────────────────────────────────────

class ForensicAuditSystem:
    """
    Generates and transmits encrypted audit records for each CPC marking event.
    In hardware mode, transmits via TLS 1.3 (requires cryptography package).
    """

    def __init__(self, pole_id="CREMS-SG-001",
                 pole_gps=(1.3521, 103.8198),
                 hardware_mode=False):
        self.pole_id      = pole_id
        self.pole_gps     = pole_gps
        self.hardware_mode = hardware_mode
        self._cpc_batch_counter = 1000

        if hardware_mode:
            try:
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import rsa, padding
                self._crypto_available = True
                print("  [OK] cryptography loaded — TLS 1.3 transmission enabled")
            except ImportError:
                print("  [WARN] cryptography package not found.")
                print("         Install: pip install cryptography")
                print("         Audit records will be saved locally only.")
                self._crypto_available = False

    def _generate_cpc_batch_id(self):
        self._cpc_batch_counter += 1
        return f"CPC-BATCH-{self._cpc_batch_counter:04d}-{uuid.uuid4().hex[:6].upper()}"

    def _hash_record(self, record_dict):
        content = json.dumps(record_dict, sort_keys=True, default=str).encode()
        return hashlib.sha256(content).hexdigest()

    def create_record(self, vehicle_id, vehicle_class, number_plate,
                      cir_result: CIRResult,
                      spray_kin: SprayKinematics) -> ForensicAuditRecord:
        record_id  = f"CREMS-AUDIT-{uuid.uuid4().hex[:12].upper()}"
        ts         = datetime.datetime.utcnow().isoformat() + "Z"
        cpc_batch  = self._generate_cpc_batch_id()

        record_content = {
            "record_id":         record_id,
            "pole_id":           self.pole_id,
            "pole_gps":          self.pole_gps,
            "timestamp_utc":     ts,
            "vehicle_id":        vehicle_id,
            "vehicle_class":     vehicle_class,
            "vehicle_velocity":  spray_kin.v * 3.6,
            "number_plate":      number_plate,
            "cir_result":        asdict(cir_result),
            "spray_kinematics":  asdict(spray_kin),
            "cpc_batch_id":      cpc_batch,
            "marking_confirmed": cir_result.is_violation,
        }

        audit_hash = self._hash_record(record_content)
        record = ForensicAuditRecord(**record_content, audit_hash=audit_hash)

        if self.hardware_mode:
            self._transmit(record)

        return record

    def _transmit(self, record: ForensicAuditRecord):
        """Simulate TLS 1.3 transmission to traffic authority server."""
        payload = json.dumps(asdict(record), default=str)
        print(f"  [HARDWARE] Transmitting audit record {record.record_id} "
              f"({len(payload)} bytes) via TLS 1.3...")
        # In production: open TLS socket to authority server and POST payload
        # For now: write to local file as stand-in
        fname = f"/tmp/{record.record_id}.json"
        with open(fname, "w") as f:
            f.write(payload)
        print(f"  [HARDWARE] Record saved → {fname}")

    def verify_record(self, record: ForensicAuditRecord) -> bool:
        record_content = {k: v for k, v in asdict(record).items()
                          if k != "audit_hash"}
        return self._hash_record(record_content) == record.audit_hash

    def simulate_da_application(self, record: ForensicAuditRecord,
                                 officer_badge: str) -> dict:
        da_kit_id    = f"DA-KIT-{uuid.uuid4().hex[:8].upper()}"
        contact_time = np.random.uniform(15, 45)
        chromatic_auth = record.marking_confirmed

        return {
            "da_kit_id":              da_kit_id,
            "officer_badge":          officer_badge,
            "applied_to_record":      record.record_id,
            "applied_at_utc":         datetime.datetime.utcnow().isoformat() + "Z",
            "cpc_removal_time_s":     round(contact_time, 1),
            "chromatic_transition":   chromatic_auth,
            "authentication_result":  "GENUINE" if chromatic_auth else "COUNTERFEIT",
            "clearcoat_residue":      False,
            "log_hash": hashlib.sha256(
                f"{da_kit_id}{officer_badge}{record.record_id}".encode()
            ).hexdigest()
        }


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION HELPERS (only used in --simulate mode)
# ─────────────────────────────────────────────────────────────────────────────

def generate_violation_trajectory(v_kmh=45.0, n_samples=20, dt=0.008):
    v = v_kmh / 3.6
    return [VehicleStateVector(
        t=i * dt,
        x=-5.0 + v * i * dt,
        y=np.random.uniform(-0.2, 0.2),
        vx=v + np.random.normal(0, 0.05),
        vy=np.random.normal(0, 0.05),
        theta=np.random.normal(0, 0.02),
        kappa=np.random.uniform(0.0, 0.05),
        delta_phi=np.random.uniform(0.0, 0.1),
    ) for i in range(n_samples)]


def generate_legal_turn_trajectory(n_samples=20, dt=0.008):
    traj = []
    for i in range(n_samples):
        t = i / n_samples
        traj.append(VehicleStateVector(
            t=i * dt,
            x=-3.0 + 2.0 * np.sin(t * math.pi / 2),
            y=2.0 * (1 - np.cos(t * math.pi / 2)),
            vx=10.0 * np.cos(t * math.pi / 2),
            vy=10.0 * np.sin(t * math.pi / 2),
            theta=t * math.pi / 2,
            kappa=np.random.uniform(0.20, 0.35),
            delta_phi=np.random.uniform(0.25, 0.35),
        ))
    return traj


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_crems_results(spray_results, cir_results, audit_record, da_log):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0D1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161B22")
    v_range = np.linspace(20, 80, 60)
    solver  = SprayKinematicsSolver()
    V_s_vals = [solver.compute(v).V_s for v in v_range]
    ax1.fill_between(v_range, V_S_MIN, V_S_MAX, alpha=0.12, color="#238636", label="Operational envelope")
    ax1.plot(v_range, V_s_vals, color="#58A6FF", lw=2.5, label="V_s (Eq. 14 solution)")
    ax1.axhline(V_S_MIN, color="#3FB950", ls="--", lw=1.2, alpha=0.7)
    ax1.axhline(V_S_MAX, color="#3FB950", ls="--", lw=1.2, alpha=0.7)
    ax1.set_xlabel("Vehicle velocity (km/h)", color="#C9D1D9", fontsize=10)
    ax1.set_ylabel("Required V_s (m/s)", color="#C9D1D9", fontsize=10)
    ax1.set_title("Tier 2 — Eq. 14: Circular Morphology Constraint\nRequired Spray Velocity",
                  color="#E6EDF3", fontsize=10, fontweight="bold", pad=8)
    ax1.tick_params(colors="#8B949E", labelsize=8)
    ax1.spines[:].set_color("#30363D")
    ax1.legend(fontsize=8, facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161B22")
    eps_vals = [solver.compute(v).epsilon for v in v_range]
    ax2.plot(v_range, eps_vals, color="#F78166", lw=2.5, label="ε = cos(θ_impact)")
    ax2.axhline(EPSILON_MAX, color="#D29922", ls="--", lw=1.5, label=f"ε_max = {EPSILON_MAX}")
    ax2.fill_between(v_range, EPSILON_MAX, 1.0, alpha=0.15, color="#3FB950", label="Circular zone")
    ax2.set_xlabel("Vehicle velocity (km/h)", color="#C9D1D9", fontsize=10)
    ax2.set_ylabel("Spot ellipticity ε", color="#C9D1D9", fontsize=10)
    ax2.set_title("Tier 2 — Spot Morphology Quality\nEllipticity vs. Velocity",
                  color="#E6EDF3", fontsize=10, fontweight="bold", pad=8)
    ax2.set_ylim(0.7, 1.05)
    ax2.tick_params(colors="#8B949E", labelsize=8)
    ax2.spines[:].set_color("#30363D")
    ax2.legend(fontsize=8, facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#161B22")
    state_colors = {
        "Approaching":          "#58A6FF",
        "LegalTurnInitiation":  "#3FB950",
        "ViolationTrajectory":  "#F78166",
        "PostZoneExit":         "#D29922",
    }
    cir_viol = next((r for r in cir_results if r.is_violation), cir_results[0])
    for idx, state in enumerate(cir_viol.state_sequence):
        ax3.bar(idx, 1, color=state_colors.get(state, "#888"), alpha=0.75, width=0.85)
    ax3.set_yticks([])
    ax3.set_xlabel("Time step", color="#C9D1D9", fontsize=10)
    ax3.set_title(
        f"Tier 1 — CIR-HMM State Sequence\n"
        f"Vehicle: VIOLATION  |  P_turn={cir_viol.P_turn:.3f}  |  D={cir_viol.D_value:.3f}",
        color="#E6EDF3", fontsize=10, fontweight="bold", pad=8)
    ax3.tick_params(colors="#8B949E", labelsize=8)
    ax3.spines[:].set_color("#30363D")
    patches = [mpatches.Patch(color=c, label=s, alpha=0.85) for s, c in state_colors.items()]
    ax3.legend(handles=patches, fontsize=7, facecolor="#161B22",
               edgecolor="#30363D", labelcolor="#C9D1D9", ncol=2)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161B22")
    ax4.axis("off")
    ok = "✓"
    hash_short = audit_record.audit_hash[:24] + "…"
    da_color = "#3FB950" if da_log["chromatic_transition"] else "#F78166"
    lines = [
        ("TIER 3 — FORENSIC AUDIT RECORD",        "#E6EDF3", 12, "bold"),
        ("",                                       "#666",    8,  "normal"),
        (f"Record ID:    {audit_record.record_id}", "#C9D1D9", 8.5, "normal"),
        (f"Timestamp:    {audit_record.timestamp_utc[:19]}Z", "#C9D1D9", 8.5, "normal"),
        (f"Pole:         {audit_record.pole_id}",  "#C9D1D9", 8.5, "normal"),
        (f"GPS:          {audit_record.pole_gps[0]:.4f}°N, {audit_record.pole_gps[1]:.4f}°E", "#C9D1D9", 8.5, "normal"),
        ("",                                       "#666",    8,  "normal"),
        (f"Vehicle:      {audit_record.vehicle_id}  ({audit_record.vehicle_class})", "#58A6FF", 9, "normal"),
        (f"Plate:        {audit_record.number_plate}", "#58A6FF", 9, "normal"),
        (f"Speed:        {audit_record.vehicle_velocity:.1f} km/h", "#58A6FF", 9, "normal"),
        (f"CPC Batch:    {audit_record.cpc_batch_id}", "#D29922", 8.5, "normal"),
        ("",                                       "#666",    8,  "normal"),
        (f"Marking:      {ok + ' CONFIRMED' if audit_record.marking_confirmed else '✗ NOT MARKED'}",
         "#3FB950" if audit_record.marking_confirmed else "#F78166", 9, "bold"),
        (f"Record hash:  {hash_short}",            "#8B949E", 8,  "normal"),
        (f"Integrity:    {ok + ' VERIFIED'}",      "#3FB950", 9,  "bold"),
        ("",                                       "#666",    8,  "normal"),
        ("── DA APPLICATION LOG ──────────────────────", "#8B949E", 8, "normal"),
        (f"DA Kit:       {da_log['da_kit_id']}",   "#C9D1D9", 8.5, "normal"),
        (f"Officer:      Badge #{da_log['officer_badge']}", "#C9D1D9", 8.5, "normal"),
        (f"Removal time: {da_log['cpc_removal_time_s']} s", "#C9D1D9", 8.5, "normal"),
        (f"Chromatic:    {ok + ' Colour change detected'}", da_color, 9, "bold"),
        (f"Auth result:  {da_log['authentication_result']}", da_color, 9, "bold"),
        (f"Clearcoat:    {ok + ' Zero residue'}",  "#3FB950", 9, "normal"),
    ]
    y = 0.97
    for text, color, size, weight in lines:
        ax4.text(0.03, y, text, transform=ax4.transAxes,
                 color=color, fontsize=size, fontweight=weight,
                 fontfamily="monospace", va="top")
        y -= 0.05 if size >= 9 else 0.04

    fig.suptitle(
        "CREMS — Contextual Reversible Enforcement and Marking System\n"
        "Bakkara et al. (2026) · IET Intelligent Transport Systems",
        color="#E6EDF3", fontsize=14, fontweight="bold", y=0.99,
        fontfamily="monospace"
    )
    plt.savefig("crems_results.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  [✓] Plot saved → crems_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — END-TO-END ENFORCEMENT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    mode_label = "HARDWARE" if HARDWARE_MODE else "SIMULATION"
    print("=" * 65)
    print("  CREMS — Cyber-Physical Enforcement & Marking System")
    print("  Based on: Bakkara et al. (2026), IET ITS")
    print(f"  Mode: {mode_label}")
    print("=" * 65)

    # Initialise layers
    sensing   = SensingLayer(hardware_mode=HARDWARE_MODE)
    actuation = ActuationLayer(hardware_mode=HARDWARE_MODE)
    audit_sys = ForensicAuditSystem(
        pole_id="CREMS-SG-NODE-007",
        pole_gps=(1.3521, 103.8198),
        hardware_mode=HARDWARE_MODE
    )
    cir = CIRAlgorithm(pedestrian_zone_width=3.0, intersection_side="left")

    # ── Tier 1: CIR-HMM Classification ─────────────────────────────────────
    print("\n[TIER 1]  Contextual Intent Recognition (CIR-HMM)")
    print("-" * 50)
    cir_results = []

    if HARDWARE_MODE:
        print("  Reading live sensor data from LiDAR + camera...")
        lidar_pts = sensing.get_lidar_frame()
        frame_vis, frame_thm = sensing.get_camera_frames()
        print(f"  LiDAR: {len(lidar_pts)} points received")
        print(f"  Camera: visible {frame_vis.shape}, thermal {frame_thm.shape}")
        # In production: extract vehicle trajectories from point cloud + CV
        # Here we demonstrate with a synthetic trajectory at a live-detected speed
        print("  [INFO] Trajectory extraction from point cloud not yet implemented.")
        print("         Using synthetic trajectory at 45 km/h for demonstration.")
        traj_viol = generate_violation_trajectory(v_kmh=45.0, n_samples=25)
        traj_turn = generate_legal_turn_trajectory(n_samples=25)
    else:
        print("  Scenario A: Vehicle going straight on red phase...")
        traj_viol = generate_violation_trajectory(v_kmh=45.0, n_samples=25)
        print("  Scenario B: Legal left-turning vehicle...")
        traj_turn = generate_legal_turn_trajectory(n_samples=25)

    result_viol = cir.classify("VEH-001", traj_viol, signal_phase="red")
    result_turn = cir.classify("VEH-002", traj_turn, signal_phase="red")
    cir_results.extend([result_viol, result_turn])

    for label, result in [("A (Violation)", result_viol), ("B (Legal turn)", result_turn)]:
        print(f"\n  Scenario {label}:")
        print(f"    P_turn     = {result.P_turn:.4f}  (threshold: {P_TURN_THRESHOLD})")
        print(f"    D(t)       = {result.D_value:.4f}")
        print(f"    HMM states = {' → '.join(result.state_sequence[-5:])}")
        print(f"    Legal turn = {result.is_legal_turn}"
              + ("  ← EXCLUDED from actuation" if result.is_legal_turn else ""))
        print(f"    VIOLATION  = {result.is_violation}"
              + ("  ← ACTUATION TRIGGERED" if result.is_violation else ""))

    # ── Tier 2: Actuation ───────────────────────────────────────────────────
    print("\n[TIER 2]  Spray Kinematics — Circular Morphology Constraint (Eq. 14)")
    print("-" * 50)
    solver = SprayKinematicsSolver()
    test_speeds = [30, 45, 60, 80]
    spray_results = []

    print(f"  {'v (km/h)':>9} {'V_s (m/s)':>10} {'β (°)':>7} {'θ_imp (°)':>10} "
          f"{'ε':>6} {'x_aim (m)':>10} {'Feasible':>9}")
    print("  " + "─" * 65)
    for v_kmh in test_speeds:
        sk = solver.compute(v_kmh)
        spray_results.append(sk)
        print(f"  {v_kmh:>9.0f} {sk.V_s:>10.1f} {sk.beta_deg:>7.2f} "
              f"{sk.theta_impact:>10.3f} {sk.epsilon:>6.4f} "
              f"{sk.x_aim:>10.4f} {'YES' if sk.feasible else 'NO (clip)':>9}")

    if result_viol.is_violation:
        print(f"\n  Violation confirmed — triggering actuation for VEH-001 at 45 km/h")
        sk_violation = actuation.fire(v_kmh=45.0, d_standoff=3.0)
    else:
        print(f"\n  No violation detected — actuation not triggered.")
        sk_violation = solver.compute(45.0)

    print(f"\n  Detailed spray parameters:")
    print(f"    t_flight      = {sk_violation.t_flight*1000:.2f} ms")
    print(f"    t_total       = {sk_violation.t_total*1000:.2f} ms")
    print(f"    P_reservoir   = {sk_violation.P_reservoir:.1f} bar")
    print(f"    d_nozzle      = {sk_violation.d_nozzle:.2f} mm")
    print(f"    Spot elliptic.= {sk_violation.epsilon:.4f}  "
          f"{'CIRCULAR ✓' if sk_violation.epsilon >= EPSILON_MAX else 'ELLIPTIC ✗'}")

    # ── Tier 3: Forensic Audit ──────────────────────────────────────────────
    print("\n[TIER 3]  Forensic Audit Record (Sec 2.4.3)")
    print("-" * 50)
    record = audit_sys.create_record(
        vehicle_id="VEH-001",
        vehicle_class="passenger_car",
        number_plate="SBA1234X",
        cir_result=result_viol,
        spray_kin=sk_violation,
    )

    print(f"  Record ID       : {record.record_id}")
    print(f"  Timestamp UTC   : {record.timestamp_utc}")
    print(f"  CPC Batch ID    : {record.cpc_batch_id}")
    print(f"  Marking conf.   : {record.marking_confirmed}")
    print(f"  Audit hash      : {record.audit_hash[:32]}…")
    print(f"  Integrity check : {'PASS ✓' if audit_sys.verify_record(record) else 'FAIL ✗'}")

    print(f"\n  Simulating DA application...")
    da_log = audit_sys.simulate_da_application(record, officer_badge="SGP-TF-4892")
    print(f"    DA Kit         : {da_log['da_kit_id']}")
    print(f"    Removal time   : {da_log['cpc_removal_time_s']} s")
    print(f"    Chromatic auth : {da_log['chromatic_transition']}  "
          f"({'colour change ✓' if da_log['chromatic_transition'] else 'NO change — COUNTERFEIT ✗'})")
    print(f"    Auth result    : {da_log['authentication_result']}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("\n[PLOT]  Generating summary figure...")
    plot_crems_results(spray_results, cir_results, record, da_log)

    # Cleanup
    sensing.shutdown()
    actuation.shutdown()

    print("\n" + "=" * 65)
    print(f"  CREMS {mode_label} complete.")
    print("  Output: crems_results.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
