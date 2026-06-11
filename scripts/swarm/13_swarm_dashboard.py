"""Live OpenCV dashboard for Tello swarm bench tests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

try:
    import cv2
except ModuleNotFoundError as error:
    print("Missing Python package: cv2")
    print("Run this dashboard with the project Windows venv:")
    print(r"  .\venv-win\Scripts\python.exe scripts\swarm\13_swarm_dashboard.py")
    raise SystemExit(1) from error


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swarm_dashboard_app import (  # noqa: E402
    DEFAULT_THERMAL_COOLING_START_C,
    DEFAULT_THERMAL_COOLING_STOP_C,
    DEFAULT_LOCAL_VIDEO_PORT,
    DEFAULT_STATUS_EVERY_SECONDS,
    DEFAULT_THERMAL_VIDEO_STOP_C,
    DEFAULT_THERMAL_WARNING_C,
    DashboardApp,
    SwarmDashboardConfig,
)
from swarm_utils import DEFAULT_DRONE_IPS_FILE, load_registered_ips, unique_ips  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual dashboard and motor-spin bench test runner for a Tello swarm.")
    parser.add_argument(
        "ips",
        nargs="*",
        help="Drone IP address(es). If omitted, uses scripts/swarm/drone_ips.txt.",
    )
    parser.add_argument(
        "--status-retries",
        type=int,
        default=2,
        help="Retries per drone for background status refresh. Default: 2.",
    )
    parser.add_argument(
        "--motor-spin-seconds",
        type=float,
        default=2.0,
        help="Seconds to spin props after the last M/button press. Default: 2.0.",
    )
    parser.add_argument(
        "--no-camera-wall",
        action="store_true",
        help="Disable live video streams if you only want status/control.",
    )
    parser.add_argument(
        "--video-port-start",
        type=int,
        default=DEFAULT_LOCAL_VIDEO_PORT,
        help=f"First local UDP video port after demux. Each drone uses +1. Default: {DEFAULT_LOCAL_VIDEO_PORT}.",
    )
    parser.add_argument(
        "--status-every",
        type=float,
        default=DEFAULT_STATUS_EVERY_SECONDS,
        help=f"Background status refresh interval. Default: {DEFAULT_STATUS_EVERY_SECONDS}.",
    )
    parser.add_argument("--width", type=int, default=1480, help="Dashboard window width.")
    parser.add_argument("--height", type=int, default=860, help="Dashboard window height.")
    parser.add_argument(
        "--thermal-warning-c",
        type=int,
        default=DEFAULT_THERMAL_WARNING_C,
        help=f"Warn visually at this reported main-board temperature. Default: {DEFAULT_THERMAL_WARNING_C}.",
    )
    parser.add_argument(
        "--thermal-video-stop-c",
        type=int,
        default=DEFAULT_THERMAL_VIDEO_STOP_C,
        help=f"Auto-stop camera streams at this reported temperature. Default: {DEFAULT_THERMAL_VIDEO_STOP_C}.",
    )
    parser.add_argument(
        "--thermal-cooling-start-c",
        type=int,
        default=DEFAULT_THERMAL_COOLING_START_C,
        help=f"Auto-start motor-on cooling at this temperature. Default: {DEFAULT_THERMAL_COOLING_START_C}.",
    )
    parser.add_argument(
        "--thermal-cooling-stop-c",
        type=int,
        default=DEFAULT_THERMAL_COOLING_STOP_C,
        help=f"Stop motor-on cooling once temperature falls to this value. Default: {DEFAULT_THERMAL_COOLING_STOP_C}.",
    )
    parser.add_argument(
        "--no-thermal-video-stop",
        action="store_true",
        help="Do not auto-stop video when drones report high temperature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.status_every <= 0:
        raise ValueError("--status-every must be greater than 0")
    if args.status_retries <= 0:
        raise ValueError("--status-retries must be greater than 0")
    if args.motor_spin_seconds <= 0:
        raise ValueError("--motor-spin-seconds must be greater than 0")
    if args.video_port_start < 1025:
        raise ValueError("--video-port-start must be >= 1025")
    if args.thermal_warning_c <= 0:
        raise ValueError("--thermal-warning-c must be greater than 0")
    if args.thermal_video_stop_c <= 0:
        raise ValueError("--thermal-video-stop-c must be greater than 0")
    if args.thermal_video_stop_c < args.thermal_warning_c:
        raise ValueError("--thermal-video-stop-c must be >= --thermal-warning-c")
    if args.thermal_cooling_start_c <= 0:
        raise ValueError("--thermal-cooling-start-c must be greater than 0")
    if args.thermal_cooling_stop_c <= 0:
        raise ValueError("--thermal-cooling-stop-c must be greater than 0")
    if args.thermal_cooling_start_c < args.thermal_warning_c:
        raise ValueError("--thermal-cooling-start-c must be >= --thermal-warning-c")
    if args.thermal_cooling_stop_c >= args.thermal_cooling_start_c:
        raise ValueError("--thermal-cooling-stop-c must be < --thermal-cooling-start-c")
    if args.thermal_video_stop_c < args.thermal_cooling_start_c:
        raise ValueError("--thermal-video-stop-c must be >= --thermal-cooling-start-c")

    ips = unique_ips(args.ips or load_registered_ips())
    if not ips:
        raise RuntimeError(f"No IPs provided and no registered drones found in {DEFAULT_DRONE_IPS_FILE}.")

    config = SwarmDashboardConfig(
        ips=ips,
        status_retries=args.status_retries,
        motor_spin_seconds=args.motor_spin_seconds,
        camera_wall_enabled=not args.no_camera_wall,
        video_port_start=args.video_port_start,
        status_every=args.status_every,
        width=args.width,
        height=args.height,
        thermal_warning_c=args.thermal_warning_c,
        thermal_cooling_start_c=args.thermal_cooling_start_c,
        thermal_cooling_stop_c=args.thermal_cooling_stop_c,
        thermal_video_stop_c=args.thermal_video_stop_c,
        auto_stop_video_on_heat=not args.no_thermal_video_stop,
    )
    app = DashboardApp(config)
    print(f"Loaded {len(ips)} drone(s): {', '.join(ips)}")
    app.run()


if __name__ == "__main__":
    main()
