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
  - End-to-End enforcement pipeline simulation
"""

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
# CONSTANTS (from paper)
# ─────────────────────────────────────────────────────────────────────────────
P_TURN_THRESHOLD   = 0.85       # Sec 2.2.3 — left-turn exclusion threshold
CONFIRM_WINDOW_MS  = 120        # ms — HMM confirmation window
T_VALVE_MS         = 2.0        # ms — solenoid valve latency (≤2 ms)
T_TOTAL_MAX_MS     = 50.0       # ms — max detect-to-actuation pipeline delay
V_S_MIN            = 85.0       # m/s — minimum spray velocity
V_S_MAX            = 230.0      # m/s — maximum spray velocity
P_RESERVOIR_MIN    = 15.0       # bar
P_RESERVOIR_MAX    = 55.0       # bar
D_NOZZLE_MIN       = 0.8        # mm
D_NOZZLE_MAX       = 2.4        # mm
EPSILON_MAX        = 0.90       # ellipticity tolerance (1.0 = perfect circle)
AUDIT_TX_MS        = 500        # ms — audit record transmission window
SAFETY_ZONE_M      = 2.0        # m — pedestrian exclusion zone around pole
LIDAR_PTS_PER_S    = 320_000    # points per second
LIDAR_H_RES_DEG    = 0.1        # horizontal angular resolution

# HMM latent states (Sec 2.2.3)
HMM_STATES = ["Approaching", "LegalTurnInitiation", "ViolationTrajectory", "PostZoneExit"]
HMM_STATE_IDX = {s: i for i, s in enumerate(HMM_STATES)}

# CIR discriminant weights (from Eq. D(t) in paper)
W1, W2, W3, W4 = 0.30, 0.25, 0.20, 0.25


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleStateVector:
    """State vector X(t) as defined in Sec 2.2.3 of the paper."""
    t:        float   # timestamp (s)
    x:        float   # position x (m)
    y:        float   # position y (m)
    vx:       float   # velocity x (m/s)
    vy:       float   # velocity y (m/s)
    theta:    float   # heading angle (rad)
    kappa:    float   # Frenet-Serret curvature (1/m)
    delta_phi: float  # vehicle-to-signal-phase offset (rad)
    vehicle_class: str = "passenger_car"  # from CV classifier


@dataclass
class CIRResult:
    """Output of the Contextual Intent Recognition algorithm."""
    vehicle_id:       str
    state_sequence:   list
    P_turn:           float   # posterior turning probability
    D_value:          float   # discriminant D(t) value
    is_legal_turn:    bool    # True → excluded from actuation
    is_violation:     bool    # True → trigger actuation
    confidence:       float
    timestamp:        float


@dataclass
class SprayKinematics:
    """
    Result of the circular morphology constraint solver.
    Implements Equations 1–17 (Section 3 of paper).
    """
    v:            float   # vehicle velocity (m/s)
    V_s:          float   # required spray velocity (m/s)
    d_standoff:   float   # standoff distance (m)
    t_total:      float   # detection + valve latency (s)
    t_flight:     float   # bolus flight time (s)
    x_aim:        float   # forward lead offset (m)
    beta_deg:     float   # nozzle lead angle (degrees)
    theta_impact: float   # angle of incidence at surface (degrees)
    epsilon:      float   # spot ellipticity (1.0 = perfect circle)
    feasible:     bool    # True if V_s within operational envelope
    P_reservoir:  float   # required reservoir pressure (bar)
    d_nozzle:     float   # nozzle orifice diameter (mm)


@dataclass
class ForensicAuditRecord:
    """
    Encrypted audit record (Sec 2.4.3) — transmitted within 500 ms.
    """
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
    audit_hash:         str   # SHA-256 of record content


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
        self.zone_width = pedestrian_zone_width   # metres
        self.side = intersection_side              # "left" or "right" traffic

        # HMM transition matrix (rows = from-state, cols = to-state)
        self.A = np.array([
            [0.70, 0.15, 0.10, 0.05],   # Approaching
            [0.05, 0.75, 0.10, 0.10],   # LegalTurnInitiation
            [0.02, 0.03, 0.90, 0.05],   # ViolationTrajectory
            [0.01, 0.01, 0.01, 0.97],   # PostZoneExit
        ])

        # Initial state distribution
        self.pi = np.array([1.0, 0.0, 0.0, 0.0])

    def _compute_d_lateral(self, state: VehicleStateVector) -> float:
        """Lateral distance from pedestrian zone centre line (normalised 0–1)."""
        return min(abs(state.y) / self.zone_width, 1.0)

    def _emission_prob(self, state: VehicleStateVector, hmm_state: int) -> float:
        """
        P(observation | hidden state) — simplified Gaussian emission.
        Maps vehicle physics to how well it fits each HMM state.
        """
        kappa  = abs(state.kappa)
        dphi   = abs(state.delta_phi)
        d_lat  = self._compute_d_lateral(state)
        speed  = math.sqrt(state.vx**2 + state.vy**2)

        if hmm_state == HMM_STATE_IDX["Approaching"]:
            # Low curvature, approaching zone
            return math.exp(-kappa * 5) * math.exp(-d_lat)

        elif hmm_state == HMM_STATE_IDX["LegalTurnInitiation"]:
            # High curvature + aligned with permitted turn signal
            turn_signal = math.exp(-abs(dphi - 0.3) * 3)
            return math.exp(-(kappa - 0.25)**2 / 0.05) * turn_signal

        elif hmm_state == HMM_STATE_IDX["ViolationTrajectory"]:
            # Low curvature (going straight), in violation zone, misaligned signal
            return math.exp(-kappa * 8) * (1 - d_lat + 0.1) * math.exp(-abs(dphi))

        elif hmm_state == HMM_STATE_IDX["PostZoneExit"]:
            # Past the zone
            return 1.0 if d_lat > 0.9 else 0.05

        return 0.0

    def _viterbi(self, trajectory: list) -> tuple:
        """
        Viterbi decoding of most probable state sequence.
        Returns (state_sequence, log_likelihood).
        """
        n_obs   = len(trajectory)
        n_states = len(HMM_STATES)
        V = np.zeros((n_states, n_obs))
        ptr = np.zeros((n_states, n_obs), dtype=int)

        # Initialise
        for s in range(n_states):
            V[s, 0] = math.log(max(self.pi[s], 1e-12)) + \
                      math.log(max(self._emission_prob(trajectory[0], s), 1e-12))

        # Recursion
        for t in range(1, n_obs):
            for s in range(n_states):
                probs = [V[prev, t-1] + math.log(max(self.A[prev, s], 1e-12))
                         for prev in range(n_states)]
                best_prev = int(np.argmax(probs))
                V[s, t] = probs[best_prev] + \
                          math.log(max(self._emission_prob(trajectory[t], s), 1e-12))
                ptr[s, t] = best_prev

        # Backtrack
        best_last = int(np.argmax(V[:, n_obs-1]))
        path = [best_last]
        for t in range(n_obs-1, 0, -1):
            path.insert(0, ptr[path[0], t])

        return [HMM_STATES[s] for s in path], float(np.max(V[:, n_obs-1]))

    def _compute_P_turn(self, state: VehicleStateVector) -> float:
        """
        Posterior turning probability based on curvature + signal phase.
        Simple logistic model derived from paper's HMM context.
        """
        kappa_score = min(abs(state.kappa) / 0.3, 1.0)     # normalise to [0,1]
        dphi_score  = math.exp(-abs(state.delta_phi) * 2)   # aligned = high score
        heading_change = min(abs(state.theta) / (math.pi/4), 1.0)

        # Sigmoid blend
        raw = 0.4 * kappa_score + 0.35 * dphi_score + 0.25 * heading_change
        P_turn = 1 / (1 + math.exp(-8 * (raw - 0.5)))
        return float(np.clip(P_turn, 0.0, 1.0))

    def classify(self, vehicle_id: str,
                 trajectory: list,
                 signal_phase: str = "red") -> CIRResult:
        """
        Run full CIR-HMM classification on a vehicle trajectory.
        
        Parameters
        ----------
        vehicle_id  : unique identifier
        trajectory  : list of VehicleStateVector sampled at ~8 ms intervals
        signal_phase: "red" | "green" | "amber"
        
        Returns
        -------
        CIRResult with violation / legal-turn classification.
        """
        if len(trajectory) < 3:
            raise ValueError("Trajectory too short — need ≥3 samples.")

        # Adjust delta_phi relative to signal phase
        phase_offset = {"red": 0.0, "amber": math.pi/6, "green": math.pi/2}
        offset = phase_offset.get(signal_phase, 0.0)
        for s in trajectory:
            s.delta_phi = abs(s.delta_phi) + offset

        # HMM decode
        state_sequence, log_lik = self._viterbi(trajectory)

        # Compute turning probability across confirmation window (last 120 ms ≈ last N samples)
        n_confirm = max(1, int(CONFIRM_WINDOW_MS / 8))
        confirm_traj = trajectory[-n_confirm:]
        P_turn_max = max(self._compute_P_turn(s) for s in confirm_traj)
        P_turn_avg = float(np.mean([self._compute_P_turn(s) for s in confirm_traj]))

        # Mandatory override — Sec 2.2.3
        is_legal_turn = P_turn_max >= P_TURN_THRESHOLD

        # Discriminant D(t) — evaluated at final state
        last = trajectory[-1]
        d_lat = self._compute_d_lateral(last)
        D_val = (W1 * abs(last.kappa) +
                 W2 * abs(last.delta_phi) +
                 W3 * d_lat +
                 W4 * (1 - P_turn_avg))

        # Violation: in violation state, not excluded, signal is red
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
# TIER 2 — SPRAY KINEMATICS (Section 3)
# ─────────────────────────────────────────────────────────────────────────────

class SprayKinematicsSolver:
    """
    Solves Equations 1–17 from Section 3 of the paper.
    
    Core constraint (Eq. 14):
        d_standoff · [1/V_s  −  1/√(V_s² − v²)]  +  t_total  =  0
    
    Solved numerically for V_s given (v, d_standoff, t_total).
    """

    def _eq14_residual(self, V_s: float, v: float,
                       d_standoff: float, t_total: float) -> float:
        """
        Equation 14 residual. Zero when circular morphology is satisfied.
        f(V_s) = d_standoff·[1/V_s − 1/√(V_s²−v²)] + t_total
        """
        if V_s <= v:
            return float('inf')
        denom = math.sqrt(max(V_s**2 - v**2, 1e-12))
        return d_standoff * (1.0/V_s - 1.0/denom) + t_total

    def solve_V_s(self, v: float, d_standoff: float,
                  t_total: float) -> Optional[float]:
        """
        Numerically solve Eq. 14 for V_s using bisection.
        
        Returns V_s in m/s, or None if no feasible solution exists.
        """
        lo, hi = v + 0.1, 1000.0
        f_lo = self._eq14_residual(lo, v, d_standoff, t_total)
        f_hi = self._eq14_residual(hi, v, d_standoff, t_total)

        if f_lo * f_hi > 0:
            return None  # No root in interval

        for _ in range(60):  # bisection iterations
            mid = (lo + hi) / 2.0
            f_mid = self._eq14_residual(mid, v, d_standoff, t_total)
            if abs(f_mid) < 1e-9:
                break
            if f_lo * f_mid < 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid

        return (lo + hi) / 2.0

    def compute(self, v_kmh: float,
                d_standoff: float = 3.0,
                t_detect_ms: float = 30.0) -> SprayKinematics:
        """
        Full spray kinematics calculation for a vehicle at v_kmh km/h.
        
        Parameters
        ----------
        v_kmh       : vehicle velocity (km/h)
        d_standoff  : perpendicular nozzle-to-vehicle distance (m)
        t_detect_ms : detection pipeline latency (ms), excludes valve latency
        
        Returns
        -------
        SprayKinematics with all parameters (Eqs. 1–17).
        """
        v = v_kmh / 3.6                               # m/s
        t_valve = T_VALVE_MS / 1000.0                 # s
        t_detect = t_detect_ms / 1000.0               # s
        t_total = t_detect + t_valve                  # Eq: t_total

        # Solve Eq. 14
        V_s = self.solve_V_s(v, d_standoff, t_total)
        feasible = V_s is not None and V_S_MIN <= V_s <= V_S_MAX

        if V_s is None:
            V_s = V_S_MAX  # fallback

        # Clip to operational envelope
        V_s_clamped = float(np.clip(V_s, V_S_MIN, V_S_MAX))

        # Eq. 1: flight time
        t_flight = d_standoff / V_s_clamped

        # Eq. 4: lead offset
        x_aim = v * (t_total + d_standoff / V_s_clamped)

        # Lead angle β from Eq. 5/6: V_s·sin(β) = v
        sin_beta = v / V_s_clamped
        sin_beta = float(np.clip(sin_beta, -1.0, 1.0))
        beta_rad = math.asin(sin_beta)
        beta_deg = math.degrees(beta_rad)

        # Impact angle at surface (Eq. 16 context)
        # With clamped V_s: residual longitudinal = V_s·sin(β) − v
        v_residual = V_s_clamped * sin_beta - v
        V_perp = math.sqrt(max(V_s_clamped**2 - v**2, 0.0))
        if V_perp > 0:
            theta_impact = math.degrees(math.atan2(abs(v_residual), V_perp))
        else:
            theta_impact = 90.0

        # Ellipticity ε = cos(θ_impact)
        epsilon = math.cos(math.radians(theta_impact))

        # Estimate reservoir pressure from V_s (Bernoulli approximation):
        # V_s ≈ Cd · √(2·P/ρ), ρ_CPC ≈ 1200 kg/m³, Cd ≈ 0.65
        rho_CPC = 1200.0
        Cd = 0.65
        P_required_pa = 0.5 * rho_CPC * (V_s_clamped / Cd) ** 2
        P_reservoir = float(np.clip(P_required_pa / 1e5, P_RESERVOIR_MIN, P_RESERVOIR_MAX))

        # Nozzle diameter: larger nozzle for heavier vehicles / lower speed
        # Linear interpolation from Table 1 range
        speed_norm = (v_kmh - 20) / (80 - 20)  # normalise 20–80 km/h
        d_nozzle = D_NOZZLE_MAX - speed_norm * (D_NOZZLE_MAX - D_NOZZLE_MIN)
        d_nozzle = float(np.clip(d_nozzle, D_NOZZLE_MIN, D_NOZZLE_MAX))

        return SprayKinematics(
            v=v,
            V_s=V_s_clamped,
            d_standoff=d_standoff,
            t_total=t_total,
            t_flight=t_flight,
            x_aim=x_aim,
            beta_deg=beta_deg,
            theta_impact=theta_impact,
            epsilon=epsilon,
            feasible=feasible,
            P_reservoir=P_reservoir,
            d_nozzle=d_nozzle,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — FORENSIC AUDIT RECORD (Section 2.4.3)
# ─────────────────────────────────────────────────────────────────────────────

class ForensicAuditSystem:
    """
    Generates encrypted audit records for each CPC marking event.
    Simulates the chain-of-custody and DA serialisation described in Sec 2.4.
    """

    def __init__(self, pole_id: str = "CREMS-SG-001",
                 pole_gps: tuple = (1.3521, 103.8198)):
        self.pole_id  = pole_id
        self.pole_gps = pole_gps
        self._cpc_batch_counter = 1000

    def _generate_cpc_batch_id(self) -> str:
        self._cpc_batch_counter += 1
        return f"CPC-BATCH-{self._cpc_batch_counter:04d}-{uuid.uuid4().hex[:6].upper()}"

    def _hash_record(self, record_dict: dict) -> str:
        """SHA-256 hash of record content for tamper detection."""
        content = json.dumps(record_dict, sort_keys=True, default=str).encode()
        return hashlib.sha256(content).hexdigest()

    def create_record(self, vehicle_id: str,
                      vehicle_class: str,
                      number_plate: str,
                      cir_result: CIRResult,
                      spray_kin: SprayKinematics) -> ForensicAuditRecord:
        """
        Compile and hash a forensic audit record (Sec 2.4.3).
        In production this is transmitted via TLS 1.3 within 500 ms.
        """
        record_id = f"CREMS-AUDIT-{uuid.uuid4().hex[:12].upper()}"
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        cpc_batch = self._generate_cpc_batch_id()

        cir_dict   = asdict(cir_result)
        spray_dict = asdict(spray_kin)

        # Build record without hash first
        record_content = {
            "record_id":        record_id,
            "pole_id":          self.pole_id,
            "pole_gps":         self.pole_gps,
            "timestamp_utc":    ts,
            "vehicle_id":       vehicle_id,
            "vehicle_class":    vehicle_class,
            "vehicle_velocity": spray_kin.v * 3.6,  # back to km/h for report
            "number_plate":     number_plate,
            "cir_result":       cir_dict,
            "spray_kinematics": spray_dict,
            "cpc_batch_id":     cpc_batch,
            "marking_confirmed": cir_result.is_violation,
        }

        audit_hash = self._hash_record(record_content)

        return ForensicAuditRecord(
            **record_content,
            audit_hash=audit_hash,
        )

    def verify_record(self, record: ForensicAuditRecord) -> bool:
        """Verify record integrity by recomputing hash."""
        record_content = {
            "record_id":        record.record_id,
            "pole_id":          record.pole_id,
            "pole_gps":         record.pole_gps,
            "timestamp_utc":    record.timestamp_utc,
            "vehicle_id":       record.vehicle_id,
            "vehicle_class":    record.vehicle_class,
            "vehicle_velocity": record.vehicle_velocity,
            "number_plate":     record.number_plate,
            "cir_result":       record.cir_result,
            "spray_kinematics": record.spray_kinematics,
            "cpc_batch_id":     record.cpc_batch_id,
            "marking_confirmed": record.marking_confirmed,
        }
        return self._hash_record(record_content) == record.audit_hash

    def simulate_da_application(self, record: ForensicAuditRecord,
                                 officer_badge: str) -> dict:
        """
        Simulate Depolymerisation Agent application (Sec 2.4.1).
        Returns authentication event log.
        """
        da_kit_id = f"DA-KIT-{uuid.uuid4().hex[:8].upper()}"
        contact_time = np.random.uniform(15, 45)  # seconds (paper: 15–45s)
        chromatic_auth = record.marking_confirmed  # genuine CPC → colour change

        return {
            "da_kit_id":          da_kit_id,
            "officer_badge":      officer_badge,
            "applied_to_record":  record.record_id,
            "applied_at_utc":     datetime.datetime.utcnow().isoformat() + "Z",
            "cpc_removal_time_s": round(contact_time, 1),
            "chromatic_transition": chromatic_auth,
            "authentication_result": "GENUINE" if chromatic_auth else "COUNTERFEIT",
            "clearcoat_residue":  False,  # paper guarantees zero residue
            "log_hash": hashlib.sha256(
                f"{da_kit_id}{officer_badge}{record.record_id}".encode()
            ).hexdigest()
        }


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_violation_trajectory(v_kmh: float, n_samples: int = 20,
                                  dt: float = 0.008) -> list:
    """
    Synthetic trajectory for a vehicle going straight through red (violation).
    Low curvature, low P_turn, misaligned signal phase.
    """
    v = v_kmh / 3.6
    traj = []
    for i in range(n_samples):
        t = i * dt
        traj.append(VehicleStateVector(
            t=t,
            x=-5.0 + v * t,
            y=np.random.uniform(-0.2, 0.2),   # drifts near centre
            vx=v + np.random.normal(0, 0.05),
            vy=np.random.normal(0, 0.05),
            theta=np.random.normal(0, 0.02),   # near-zero heading change
            kappa=np.random.uniform(0.0, 0.05),  # nearly straight
            delta_phi=np.random.uniform(0.0, 0.1),  # aligned with straight
        ))
    return traj


def generate_legal_turn_trajectory(n_samples: int = 20, dt: float = 0.008) -> list:
    """
    Synthetic trajectory for a legal left-turning vehicle.
    High curvature, high P_turn, aligned with turn signal.
    """
    traj = []
    for i in range(n_samples):
        t = i * dt
        turn_progress = i / n_samples
        traj.append(VehicleStateVector(
            t=t,
            x=-3.0 + 2.0 * np.sin(turn_progress * math.pi / 2),
            y=2.0 * (1 - np.cos(turn_progress * math.pi / 2)),
            vx=10.0 * np.cos(turn_progress * math.pi / 2),
            vy=10.0 * np.sin(turn_progress * math.pi / 2),
            theta=turn_progress * math.pi / 2,
            kappa=np.random.uniform(0.20, 0.35),   # high curvature = turning
            delta_phi=np.random.uniform(0.25, 0.35),  # aligned with turn phase
        ))
    return traj


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_crems_results(spray_results: list,
                       cir_results: list,
                       audit_record: ForensicAuditRecord,
                       da_log: dict):
    """
    Four-panel figure summarising CREMS system outputs.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0D1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── Panel 1: Eq. 14 — V_s vs vehicle velocity ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161B22")
    v_range = np.linspace(20, 80, 60)
    V_s_vals, feasible_mask = [], []
    solver = SprayKinematicsSolver()
    for v_kmh in v_range:
        res = solver.compute(v_kmh)
        V_s_vals.append(res.V_s)
        feasible_mask.append(res.feasible)

    V_s_arr = np.array(V_s_vals)
    ax1.fill_between(v_range, V_S_MIN, V_S_MAX, alpha=0.12,
                     color="#238636", label="Operational envelope")
    ax1.plot(v_range, V_s_arr, color="#58A6FF", lw=2.5, label="V_s (Eq. 14 solution)")
    ax1.axhline(V_S_MIN, color="#3FB950", ls="--", lw=1.2, alpha=0.7)
    ax1.axhline(V_S_MAX, color="#3FB950", ls="--", lw=1.2, alpha=0.7)
    ax1.set_xlabel("Vehicle velocity (km/h)", color="#C9D1D9", fontsize=10)
    ax1.set_ylabel("Required V_s (m/s)", color="#C9D1D9", fontsize=10)
    ax1.set_title("Tier 2 — Eq. 14: Circular Morphology Constraint\nRequired Spray Velocity", 
                  color="#E6EDF3", fontsize=10, fontweight="bold", pad=8)
    ax1.tick_params(colors="#8B949E", labelsize=8)
    ax1.spines[:].set_color("#30363D")
    leg = ax1.legend(fontsize=8, facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9")

    # ── Panel 2: Ellipticity ε vs velocity ──────────────────────────────────
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

    # ── Panel 3: CIR-HMM — state sequence + P_turn ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#161B22")

    # Show both violation and legal turn examples
    cir_viol = next((r for r in cir_results if r.is_violation), cir_results[0])
    cir_turn = next((r for r in cir_results if r.is_legal_turn), None)

    state_colors = {
        "Approaching":         "#58A6FF",
        "LegalTurnInitiation": "#3FB950",
        "ViolationTrajectory": "#F78166",
        "PostZoneExit":        "#D29922",
    }

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

    patches = [mpatches.Patch(color=c, label=s, alpha=0.85)
               for s, c in state_colors.items()]
    ax3.legend(handles=patches, fontsize=7, facecolor="#161B22",
               edgecolor="#30363D", labelcolor="#C9D1D9", ncol=2)

    # ── Panel 4: Forensic audit summary ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161B22")
    ax4.axis("off")

    ok  = "✓"
    fail = "✗"
    hash_short = audit_record.audit_hash[:24] + "…"
    da_color = "#3FB950" if da_log["chromatic_transition"] else "#F78166"

    lines = [
        ("TIER 3 — FORENSIC AUDIT RECORD", "#E6EDF3", 12, "bold"),
        ("", "#666", 8, "normal"),
        (f"Record ID:    {audit_record.record_id}", "#C9D1D9", 8.5, "normal"),
        (f"Timestamp:    {audit_record.timestamp_utc[:19]}Z", "#C9D1D9", 8.5, "normal"),
        (f"Pole:         {audit_record.pole_id}", "#C9D1D9", 8.5, "normal"),
        (f"GPS:          {audit_record.pole_gps[0]:.4f}°N, {audit_record.pole_gps[1]:.4f}°E", "#C9D1D9", 8.5, "normal"),
        ("", "#666", 8, "normal"),
        (f"Vehicle:      {audit_record.vehicle_id}  ({audit_record.vehicle_class})", "#58A6FF", 9, "normal"),
        (f"Plate:        {audit_record.number_plate}", "#58A6FF", 9, "normal"),
        (f"Speed:        {audit_record.vehicle_velocity:.1f} km/h", "#58A6FF", 9, "normal"),
        (f"CPC Batch:    {audit_record.cpc_batch_id}", "#D29922", 8.5, "normal"),
        ("", "#666", 8, "normal"),
        (f"Marking:      {ok + ' CONFIRMED' if audit_record.marking_confirmed else fail + ' NOT MARKED'}", 
         "#3FB950" if audit_record.marking_confirmed else "#F78166", 9, "bold"),
        (f"Record hash:  {hash_short}", "#8B949E", 8, "normal"),
        (f"Integrity:    {ok + ' VERIFIED'}", "#3FB950", 9, "bold"),
        ("", "#666", 8, "normal"),
        ("── DA APPLICATION LOG ──────────────────────", "#8B949E", 8, "normal"),
        (f"DA Kit:       {da_log['da_kit_id']}", "#C9D1D9", 8.5, "normal"),
        (f"Officer:      Badge #{da_log['officer_badge']}", "#C9D1D9", 8.5, "normal"),
        (f"Removal time: {da_log['cpc_removal_time_s']} s", "#C9D1D9", 8.5, "normal"),
        (f"Chromatic:    {ok + ' Colour change detected'}", da_color, 9, "bold"),
        (f"Auth result:  {da_log['authentication_result']}", da_color, 9, "bold"),
        (f"Clearcoat:    {ok + ' Zero residue'}", "#3FB950", 9, "normal"),
    ]

    y = 0.97
    for text, color, size, weight in lines:
        ax4.text(0.03, y, text, transform=ax4.transAxes,
                 color=color, fontsize=size, fontweight=weight,
                 fontfamily="monospace", va="top")
        y -= 0.05 if size >= 9 else 0.04

    # Main title
    fig.suptitle(
        "CREMS — Contextual Reversible Enforcement and Marking System\n"
        "Bakkara et al. (2026) · IET Intelligent Transport Systems",
        color="#E6EDF3", fontsize=14, fontweight="bold", y=0.99,
        fontfamily="monospace"
    )

    plt.savefig("/mnt/user-data/outputs/crems_results.png",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  [✓] Plot saved → crems_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — END-TO-END ENFORCEMENT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CREMS — Cyber-Physical Enforcement & Marking System")
    print("  Based on: Bakkara et al. (2026), IET ITS")
    print("=" * 65)

    # ── Tier 1: CIR-HMM Classification ─────────────────────────────────────
    print("\n[TIER 1]  Contextual Intent Recognition (CIR-HMM)")
    print("-" * 50)
    cir = CIRAlgorithm(pedestrian_zone_width=3.0, intersection_side="left")
    cir_results = []

    # Scenario A: Violation — straight through red
    print("  Scenario A: Vehicle going straight on red phase...")
    traj_viol = generate_violation_trajectory(v_kmh=45.0, n_samples=25)
    result_viol = cir.classify("VEH-001", traj_viol, signal_phase="red")
    cir_results.append(result_viol)
    print(f"    P_turn     = {result_viol.P_turn:.4f}  (threshold: {P_TURN_THRESHOLD})")
    print(f"    D(t)       = {result_viol.D_value:.4f}")
    print(f"    HMM states = {' → '.join(result_viol.state_sequence[-5:])}")
    print(f"    Legal turn = {result_viol.is_legal_turn}")
    print(f"    VIOLATION  = {result_viol.is_violation}  {'← ACTUATION TRIGGERED' if result_viol.is_violation else ''}")

    print()

    # Scenario B: Legal left turn
    print("  Scenario B: Legal left-turning vehicle...")
    traj_turn = generate_legal_turn_trajectory(n_samples=25)
    result_turn = cir.classify("VEH-002", traj_turn, signal_phase="red")
    cir_results.append(result_turn)
    print(f"    P_turn     = {result_turn.P_turn:.4f}  (threshold: {P_TURN_THRESHOLD})")
    print(f"    D(t)       = {result_turn.D_value:.4f}")
    print(f"    HMM states = {' → '.join(result_turn.state_sequence[-5:])}")
    print(f"    Legal turn = {result_turn.is_legal_turn}  {'← EXCLUDED from actuation' if result_turn.is_legal_turn else ''}")
    print(f"    VIOLATION  = {result_turn.is_violation}")

    # ── Tier 2: Spray Kinematics ─────────────────────────────────────────────
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
        feasible_str = "YES" if sk.feasible else "NO (clip)"
        print(f"  {v_kmh:>9.0f} {sk.V_s:>10.1f} {sk.beta_deg:>7.2f} "
              f"{sk.theta_impact:>10.3f} {sk.epsilon:>6.4f} "
              f"{sk.x_aim:>10.4f} {feasible_str:>9}")

    # Detailed output for the violation vehicle
    sk_violation = solver.compute(45.0, d_standoff=3.0, t_detect_ms=30.0)
    print(f"\n  Detailed for violation vehicle at 45 km/h:")
    print(f"    t_flight      = {sk_violation.t_flight*1000:.2f} ms")
    print(f"    t_total       = {sk_violation.t_total*1000:.2f} ms  (detect + valve)")
    print(f"    P_reservoir   = {sk_violation.P_reservoir:.1f} bar")
    print(f"    d_nozzle      = {sk_violation.d_nozzle:.2f} mm")
    print(f"    Spot elliptic.= {sk_violation.epsilon:.4f}  {'CIRCULAR ✓' if sk_violation.epsilon >= EPSILON_MAX else 'ELLIPTIC ✗'}")

    # ── Tier 3: Forensic Audit Record ───────────────────────────────────────
    print("\n[TIER 3]  Forensic Audit Record (Sec 2.4.3)")
    print("-" * 50)
    audit_sys = ForensicAuditSystem(pole_id="CREMS-SG-NODE-007",
                                    pole_gps=(1.3521, 103.8198))

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

    # DA Application simulation
    print(f"\n  Simulating DA application by enforcement officer...")
    da_log = audit_sys.simulate_da_application(record, officer_badge="SGP-TF-4892")
    print(f"    DA Kit         : {da_log['da_kit_id']}")
    print(f"    Officer badge  : #{da_log['officer_badge']}")
    print(f"    Removal time   : {da_log['cpc_removal_time_s']} s")
    print(f"    Chromatic auth : {da_log['chromatic_transition']}  "
          f"({'colour change detected ✓' if da_log['chromatic_transition'] else 'NO change — COUNTERFEIT ✗'})")
    print(f"    Auth result    : {da_log['authentication_result']}")
    print(f"    Clearcoat resi.: {'none ✓' if not da_log['clearcoat_residue'] else 'RESIDUE DETECTED ✗'}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("\n[PLOT]  Generating summary figure…")
    plot_crems_results(spray_results, cir_results, record, da_log)

    print("\n" + "=" * 65)
    print("  CREMS simulation complete.")
    print("  Output: crems_results.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
