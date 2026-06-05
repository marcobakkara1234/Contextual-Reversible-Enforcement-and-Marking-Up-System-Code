# CREMS — Contextual Reversible Enforcement and Marking System

> **Cyber-Physical Enforcement at Smart Intersections: A Three-Tier LiDAR–Vision–Pneumatic Architecture for Reversible Chemical Marking of Pedestrian Right-of-Way Violations**
>
> Bakkara et al. (2026) · *IET Intelligent Transport Systems* (Under Review)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review%20%7C%20SN%20IJITSR-orange)](https://link.springer.com/journal/13177)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0004--0959--6450-brightgreen?logo=orcid)](https://orcid.org/0009-0004-0959-6450)

---

## Overview

CREMS is a **three-tier Cyber-Physical System (CPS)** for smart intersection enforcement. Instead of mailing a citation days later, CREMS responds to a pedestrian zone violation **within 50 ms** — firing a precision pneumatic jet of reversible Cross-linked Polymer Coating (CPC) onto the violating vehicle's lateral door panel.

```
VIOLATION DETECTED
       │
  ┌────▼────┐     ┌─────────────┐     ┌──────────────────┐
  │ TIER 1  │────▶│   TIER 2    │────▶│     TIER 3       │
  │  LiDAR  │     │  Pneumatic  │     │  Forensic Audit  │
  │  + CIR  │     │   Sprayer   │     │  + Chain Custody │
  └─────────┘     └─────────────┘     └──────────────────┘
  Sensing Layer   Actuation Layer      Forensic Layer
```

| Problem with current systems | CREMS solution |
|---|---|
| E-TLE / ANPR: citation arrives weeks later | Physical mark applied within **50 ms** of violation |
| Zebra crossings invisible at night / rain | Active LiDAR virtual boundary — **24/7, weather-proof** |
| Physical barriers cause secondary crashes | Lateral panel spray — **no traffic disruption** |
| ANPR cites registered owner, not driver | Mark stays on vehicle — **any officer can verify** |
| Corrupt officers can suppress citations | CPC chemistry makes suppression **physically impossible** |

---

## ⚠️ Hardware Prerequisites

**CREMS requires all physical hardware to be connected and verified before running `crems_system.py`.** The system will check for each component at startup and will refuse to proceed if any required device is missing or unreachable.

### Required Hardware Checklist

| # | Component | Interface | How to Verify |
|---|---|---|---|
| 1 | **Velodyne VLP-32C LiDAR** | ROS2 topic `/velodyne_points` | `ros2 topic echo /velodyne_points --once` |
| 2 | **Visible + NIR + Thermal Camera Array** | RTSP stream | `ffprobe rtsp://smartpole-001/visible` |
| 3 | **Pneumatic Sprayer + Solenoid Valve** | GPIO pin 24 (Raspberry Pi) | `gpio read 24` |
| 4 | **2-Axis Servo Gimbal** | GPIO pins 18 & 23 (Raspberry Pi) | `gpio read 18 && gpio read 23` |
| 5 | **Edge CPS Node** | GigE LAN (≤ 2 ms/hop) | `ping smartpole-001 -c 4` |
| 6 | **CPC Reservoir** | Pressure sensor reading 15–55 bar | Verify via pneumatic controller panel |

### Pre-flight Check Script

Run this before starting the main system to confirm all hardware is online:

```bash
python crems_preflight.py
```

Expected output when all hardware is ready:

```
[OK] LiDAR      — VLP-32C responding on /velodyne_points (320,000 pts/s)
[OK] Camera     — Visible stream online @ rtsp://smartpole-001/visible
[OK] Camera     — Thermal stream online @ rtsp://smartpole-001/thermal
[OK] GPIO 18    — Servo azimuth pin ready
[OK] GPIO 23    — Servo elevation pin ready
[OK] GPIO 24    — Solenoid valve pin ready
[OK] Pressure   — CPC reservoir at 42 bar (within 15–55 bar range)
[OK] Network    — Edge node latency 1.4 ms (within ≤ 2 ms requirement)

✅ All systems nominal. Safe to run: python crems_system.py
```

If any check fails, the output will show:

```
[FAIL] LiDAR   — No data on /velodyne_points. Is the VLP-32C powered and connected?
[FAIL] GPIO 24 — Pin unreadable. Is the Raspberry Pi GPIO interface enabled?

❌ Hardware check failed. Do NOT run crems_system.py until all components are connected.
```

> **The system will not arm the actuation pipeline (Tier 2) unless all Tier 1 sensors report nominal status.** This is a hard safety interlock — not a software warning.

---

## System Architecture

### Tier 1 — Sensing Layer

**Hardware (per smart pole node):**

| Component | Specification |
|---|---|
| 3D LiDAR Scanner | 320,000 pts/s · 0.1° H-res · 0.2° V-res · 200 m range |
| Multi-spectral Camera Array | 4K / 60 fps · Visible + NIR + Thermal |
| Mounting Height | 5.5–6.0 m on smart pole |
| Edge CPS Node | Real-time compute; deterministic LAN ≤ 2 ms/hop |
| LED Road Indicator | Luminous pedestrian zone boundary (replaces painted line) |

**Real sensor integration (Python example):**

```python
# LiDAR — Velodyne VLP-32C via ROS2
import rclpy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

def lidar_callback(msg):
    points = list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True))
    # Feed into CIR algorithm
    crems.tier1.ingest_lidar_frame(points, timestamp=msg.header.stamp.sec)

node.create_subscription(PointCloud2, '/velodyne_points', lidar_callback, 10)
```

```python
# Camera — OpenCV + RTSP stream (visible + thermal)
import cv2

cap_visible = cv2.VideoCapture("rtsp://smartpole-001/visible")
cap_thermal  = cv2.VideoCapture("rtsp://smartpole-001/thermal")

while True:
    ret_v, frame_vis = cap_visible.read()
    ret_t, frame_thm = cap_thermal.read()
    if ret_v and ret_t:
        crems.tier1.ingest_camera_frame(frame_vis, frame_thm)
```

**CIR-HMM Algorithm:**

The Contextual Intent Recognition algorithm classifies each tracked vehicle using a Hidden Markov Model over the state vector:

```
X(t) = [x, y, vx, vy, θ, κ, Δφ, C_class]
```

Where `κ` = Frenet-Serret curvature, `Δφ` = vehicle-to-signal-phase offset.

Discriminant function:
```
D(t) = w₁·κ(t) + w₂·|Δφ(t)| + w₃·d_lateral(t) + w₄·(1 − P_turn(t))
```

**Mandatory left-turn exclusion rule:** if `P_turn ≥ 0.85` at any point in the 120 ms confirmation window → vehicle is **irrevocably excluded** from the actuation pipeline.

```
HMM States:
  Approaching ──▶ LegalTurnInitiation ──▶ [EXCLUDED — no spray]
       │
       └────────▶ ViolationTrajectory ──▶ [TRIGGER Tier 2]
```

---

### Tier 2 — Actuation Layer

**Hardware:**

| Component | Specification |
|---|---|
| Pneumatic Sprayer | 15–55 bar reservoir pressure |
| Solenoid Valve | ≤ 2 ms actuation latency |
| Nozzle Orifice | 0.8–2.4 mm (vehicle-class adaptive) |
| 2-Axis Servo Gimbal | 500 Hz update rate; hard windshield exclusion mask |
| Spray Velocity | 85–230 m/s |

**Circular Morphology Constraint — Equation 14:**

The core kinematic requirement: the CPC bolus must arrive at the vehicle's lateral panel at **normal incidence** (θ_impact = 0°), guaranteeing a circular spot.

```
d_standoff · [ 1/V_s  −  1/√(V_s² − v²) ]  +  t_total  =  0
```

This is solved numerically for `V_s` given vehicle velocity `v`, standoff distance `d_standoff`, and pipeline latency `t_total`.

```python
from crems_system import SprayKinematicsSolver

solver = SprayKinematicsSolver()
result = solver.compute(v_kmh=60.0, d_standoff=3.0, t_detect_ms=30.0)

print(f"Required V_s  : {result.V_s:.1f} m/s")
print(f"Lead angle β  : {result.beta_deg:.2f}°")
print(f"Spot elliptic : {result.epsilon:.4f}")   # 1.0 = perfect circle
print(f"x_aim offset  : {result.x_aim:.4f} m")
```

**Real servo-gimbal integration (Raspberry Pi / ROS2 example):**

```python
import RPi.GPIO as GPIO
from crems_system import SprayKinematicsSolver

SERVO_AZ_PIN = 18   # azimuth (lead compensation)
SERVO_EL_PIN = 23   # elevation (height lock)
VALVE_PIN    = 24   # solenoid trigger

GPIO.setmode(GPIO.BCM)
GPIO.setup([SERVO_AZ_PIN, SERVO_EL_PIN, VALVE_PIN], GPIO.OUT)

solver = SprayKinematicsSolver()

def fire_crems(v_kmh: float, d_standoff: float = 3.0):
    sk = solver.compute(v_kmh, d_standoff)

    # Aim servo to lead-compensated position
    az_pwm = GPIO.PWM(SERVO_AZ_PIN, 50)
    az_pwm.start(angle_to_duty(sk.beta_deg))

    # Fire solenoid (valve latency ≤ 2 ms)
    GPIO.output(VALVE_PIN, GPIO.HIGH)
    time.sleep(0.012)   # 12 ms bolus duration
    GPIO.output(VALVE_PIN, GPIO.LOW)
```

**Targeting safety:** the servo gimbal enforces a **hard exclusion mask** covering the windshield and all glazed surfaces. The spray envelope is limited strictly to the lateral door panel between the front wheel arch and the B-pillar (0.8–1.2 m above road level).

---

### Tier 3 — Forensic Layer

**CPC (Cross-linked Polymer Coating) resistance profile:**

| Challenge Agent | Result |
|---|---|
| Rainwater / aqueous wash | ✗ Resistant |
| Car shampoo / surfactants | ✗ Resistant |
| Gasoline / diesel | ✗ Resistant |
| Acetone / isopropanol | ✗ Resistant |
| Toluene / aromatic solvents | ✗ Resistant |
| Mechanical abrasion | ✗ Resistant (UV-detectable smear) |
| **Depolymerisation Agent (DA)** | **✓ Complete removal in 15–45 s** |

**Depolymerisation Agent chain-of-custody:**
- DA supplied only to authorised law enforcement
- Each kit individually serialised: `DA-KIT-XXXXXXXX`
- Every application logged: officer badge + timestamp + audit record ID
- Chromatic colour change on application = authentication of genuine CPC

**Forensic audit record (transmitted within 500 ms via TLS 1.3):**

```python
from crems_system import ForensicAuditSystem

audit = ForensicAuditSystem(pole_id="CREMS-SG-NODE-007",
                             pole_gps=(1.3521, 103.8198))

record = audit.create_record(
    vehicle_id="VEH-001",
    vehicle_class="passenger_car",
    number_plate="SBA1234X",
    cir_result=cir_result,
    spray_kin=spray_result,
)

# Verify tamper-proof hash
assert audit.verify_record(record)  # SHA-256 integrity check
```

---

## Full Hardware Integration Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     SMART POLE NODE                         │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  VLP-32C     │   │  Camera Array│   │ Signal Control │  │
│  │  LiDAR       │   │  Vis+NIR+Thm │   │ Interface      │  │
│  └──────┬───────┘   └──────┬───────┘   └───────┬────────┘  │
│         │                  │                   │           │
│         └──────────────────┴───────────────────┘           │
│                            │                               │
│                   ┌────────▼────────┐                      │
│                   │  Edge CPS Node  │  GigE LAN ≤2ms/hop   │
│                   │  CIR-HMM Engine │                      │
│                   └────────┬────────┘                      │
│                            │                               │
│              ┌─────────────┴──────────────┐                │
│              │                            │                │
│     ┌────────▼────────┐         ┌─────────▼──────────┐    │
│     │ Pneumatic Ctrl  │         │  Audit Transmitter  │    │
│     │ Servo Gimbal    │         │  4G/5G · TLS 1.3    │    │
│     │ Solenoid Valve  │         │  → Traffic Authority│    │
│     └────────┬────────┘         └────────────────────┘    │
│              │                                             │
└──────────────┼─────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   CPC RESERVOIR     │   15–55 bar · 0.8–2.4 mm nozzle
    │   + SPRAY GIMBAL    │   V_s = 85–230 m/s
    └─────────────────────┘
              │
              ▼  (fires at lateral door panel, never windshield)
    ══════════════════════
         VEHICLE
    ══════════════════════
```

---

## Installation

```bash
git clone https://github.com/marcobakkara1234/Contextual-Reversible-Enforcement-and-Marking-Up-System-Code.git
cd Contextual-Reversible-Enforcement-and-Marking-Up-System-Code

pip install -r requirements.txt
```

> **Do not run `crems_system.py` directly.** Always run the pre-flight check first:

```bash
# Step 1 — verify all hardware is connected
python crems_preflight.py

# Step 2 — only if preflight passes, start the main system
python crems_system.py
```

**requirements.txt:**
```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
hmmlearn>=0.3
```

**For real hardware deployment, additionally:**
```
rclpy          # ROS2 Python client — LiDAR/camera integration
opencv-python  # Camera frame processing
RPi.GPIO       # Servo gimbal + solenoid control (Raspberry Pi)
cryptography   # TLS 1.3 audit record transmission
```

---

## Repository Structure

```
.
├── crems_system.py       # Full CREMS implementation (Tier 1–3)
├── crems_preflight.py    # Hardware pre-flight check (run before crems_system.py)
├── crems_results.png     # Simulation output plots
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── docs/
    └── index.html        # GitHub Pages site
```

---

## Authors

| Name | Affiliation | Role |
|---|---|---|
| **Marco Julius Andreas Bakkara** | Independent Researcher | Conceptualisation, system architecture, CIR-HMM, kinematics, materials, legal analysis |
| Akmal Hasan Hasibuan | Universitas Padjajaran, Dept. Informatics Engineering | CIR algorithm & HMM software implementation |
| Fretyna Afesa Simarmata | Politeknik Negeri Medan, Informatics Management | Pneumatic controller, servo routines, audit transmission |
| Queena Itsuka Umri | Politeknik Negeri Medan, Multimedia Graphic Engineering | Multi-spectral camera system & CV integration |

**Corresponding author:** Marco Julius Andreas Bakkara
📧 tupabakara@gmail.com · ORCID: [0009-0004-0959-6450](https://orcid.org/0009-0004-0959-6450)

---

## Citation

```bibtex
@article{bakkara2026crems,
  title   = {Cyber-Physical Enforcement at Smart Intersections: A Three-Tier
             LiDAR–Vision–Pneumatic Architecture for Reversible Chemical
             Marking of Pedestrian Right-of-Way Violations},
  author  = {Bakkara, Marco Julius Andreas and Hasibuan, Akmal Hasan and
             Simarmata, Fretyna Afesa and Umri, Queena Itsuka},
  journal = {IET Intelligent Transport Systems},
  year    = {2026},
  note    = {Under Review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

*This repository accompanies the above paper submitted to IET Intelligent Transport Systems.*
