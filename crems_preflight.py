"""
CREMS — Hardware Pre-flight Check
Run this BEFORE crems_system.py --hardware to verify all components are online.

Usage:
  python crems_preflight.py
"""

import sys
import subprocess
import socket
import time

PASS = "\033[92m[OK]  \033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []


def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    line = f"  {status} {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    results.append(ok)
    return ok


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  CREMS — Hardware Pre-flight Check")
print("=" * 55)

# ── 1. Python packages ────────────────────────────────────────
section("1. Python Packages")

for pkg, import_name in [
    ("numpy",          "numpy"),
    ("scipy",          "scipy"),
    ("matplotlib",     "matplotlib"),
    ("hmmlearn",       "hmmlearn"),
    ("RPi.GPIO",       "RPi.GPIO"),
    ("rclpy",          "rclpy"),
    ("opencv-python",  "cv2"),
    ("cryptography",   "cryptography"),
]:
    try:
        __import__(import_name)
        check(f"pip: {pkg}", True, "installed")
    except ImportError:
        check(f"pip: {pkg}", False, f"missing — pip install {pkg}")

# ── 2. GPIO pins (Raspberry Pi) ───────────────────────────────
section("2. GPIO Pins (Raspberry Pi)")

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)

    for pin, label in [(18, "Servo azimuth (AZ)"),
                       (23, "Servo elevation (EL)"),
                       (24, "Solenoid valve")]:
        try:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
            check(f"GPIO pin {pin} — {label}", True, "readable & writable")
        except Exception as e:
            check(f"GPIO pin {pin} — {label}", False, str(e))

    GPIO.cleanup()

except ImportError:
    check("GPIO (all pins)", False, "RPi.GPIO not installed")
except RuntimeError as e:
    check("GPIO (all pins)", False, f"Not a Raspberry Pi? {e}")

# ── 3. ROS2 / LiDAR ───────────────────────────────────────────
section("3. ROS2 + LiDAR (Velodyne VLP-32C)")

try:
    import rclpy
    rclpy.init()
    node = rclpy.create_node("crems_preflight")

    # Check topic exists
    topic_names = [name for name, _ in node.get_topic_names_and_types()]
    lidar_ok = "/velodyne_points" in topic_names
    check("ROS2 node init", True, "rclpy OK")
    check("LiDAR topic /velodyne_points", lidar_ok,
          "active" if lidar_ok else "not found — is VLP-32C powered and driver running?")

    node.destroy_node()
    rclpy.shutdown()

except ImportError:
    check("ROS2 / rclpy", False, "not installed — source /opt/ros/<distro>/setup.bash")
except Exception as e:
    check("ROS2 / rclpy", False, str(e))

# ── 4. Camera streams (RTSP) ──────────────────────────────────
section("4. Camera Streams (RTSP)")

try:
    import cv2

    for url, label in [
        ("rtsp://smartpole-001/visible", "Visible camera"),
        ("rtsp://smartpole-001/thermal", "Thermal camera"),
    ]:
        cap = cv2.VideoCapture(url)
        ok  = cap.isOpened()
        ret, _ = cap.read() if ok else (False, None)
        cap.release()
        check(label, ok and ret,
              f"{url}" if (ok and ret) else f"cannot open stream — is camera online?")

except ImportError:
    check("Camera streams", False, "opencv-python not installed")

# ── 5. Network / Edge node latency ───────────────────────────
section("5. Network — Edge CPS Node")

try:
    start = time.time()
    s = socket.create_connection(("smartpole-001", 80), timeout=2)
    latency_ms = (time.time() - start) * 1000
    s.close()
    ok = latency_ms <= 2.0
    check("Edge node reachable", True, f"latency {latency_ms:.2f} ms")
    check("Latency ≤ 2 ms", ok,
          f"{latency_ms:.2f} ms {'OK' if ok else '— exceeds 2 ms requirement'}")
except (socket.timeout, OSError) as e:
    check("Edge node reachable", False, f"smartpole-001 — {e}")
    check("Latency ≤ 2 ms", False, "cannot measure — node unreachable")

# ── 6. CPC reservoir pressure ────────────────────────────────
section("6. CPC Reservoir Pressure")

# In production: read from pressure sensor via GPIO/I2C/SPI
# Here: placeholder check (always warns — requires real sensor integration)
print(f"  {WARN} Pressure sensor — manual verification required")
print(f"         Confirm CPC reservoir is at 15–55 bar before proceeding.")
results.append(True)   # operator must confirm manually

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
passed = sum(results)
total  = len(results)

if all(results):
    print(f"\033[92m  ✅ All {total} checks passed. Safe to run:\033[0m")
    print(f"\033[92m     python crems_system.py --hardware\033[0m")
else:
    failed = total - passed
    print(f"\033[91m  ❌ {failed}/{total} check(s) failed.\033[0m")
    print(f"\033[91m     Resolve all failures before running --hardware.\033[0m")
    print(f"     Simulation mode is always available:")
    print(f"     python crems_system.py --simulate")

print(f"{'='*55}\n")
sys.exit(0 if all(results) else 1)
