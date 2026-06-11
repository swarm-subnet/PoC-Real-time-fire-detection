"""Preview-first one-drone stationary person-search validation run.

The script connects to the first reachable drone, starts video + person
detection, and waits for the operator. Press G to take off, yaw-search in place
until a person is repeatedly detected, then land. It never moves forward or
sideways.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swarm_detector import (  # noqa: E402
    DEFAULT_PERSON_CONFIDENCE,
    DEFAULT_PERSON_IMGSZ,
    DEFAULT_PERSON_MODEL,
    PersonDetectorConfig,
)
from swarm_single_search import (  # noqa: E402
    SingleDroneSearchConfig,
    SingleDroneSearchPreviewApp,
    SingleDroneSearchRunner,
)
from tello_direct_single import DirectTelloCamera, DirectTelloController, DirectTelloLink  # noqa: E402
from swarm_utils import (  # noqa: E402
    DEFAULT_DRONE_IPS_FILE,
    TelloUdpClient,
    load_registered_ips,
    query_battery,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open one Tello live preview, then press G to yaw-search for a person and land."
    )
    parser.add_argument(
        "ip",
        nargs="?",
        help="Drone IP. If omitted, probes scripts/swarm/drone_ips.txt and uses the first reachable drone.",
    )
    parser.add_argument("--probe-timeout", type=float, default=2.0, help="Per-drone probe timeout when no IP is supplied. Default: 2.")
    parser.add_argument("--probe-retries", type=int, default=1, help="Per-drone probe retries when no IP is supplied. Default: 1.")
    parser.add_argument("--min-battery", type=int, default=20, help="Minimum battery for takeoff. Default: 20.")
    parser.add_argument("--status-retries", type=int, default=2, help="Preflight retries per command. Default: 2.")
    parser.add_argument("--max-search-seconds", type=float, default=90.0, help="Land after this many seconds if no person is confirmed.")
    parser.add_argument("--yaw-step-degrees", type=int, default=20, help="Yaw step during search. Default: 20.")
    parser.add_argument("--yaw-interval", type=float, default=1.2, help="Seconds between yaw steps. Default: 1.2.")
    parser.add_argument("--takeoff-settle", type=float, default=3.0, help="Seconds to settle after takeoff. Default: 3.")
    parser.add_argument("--confirm-detections", type=int, default=5, help="Detections required before landing. Default: 5.")
    parser.add_argument("--confirm-window", type=float, default=3.0, help="Confirmation window in seconds. Default: 3.")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview and run the old immediate takeoff/search/land flow.")
    parser.add_argument("--detection-hold", type=float, default=1.5, help="Seconds to keep the detection box visible before landing. Default: 1.5.")
    parser.add_argument("--person-model", default=DEFAULT_PERSON_MODEL, help=f"YOLO model. Default: {DEFAULT_PERSON_MODEL}.")
    parser.add_argument("--person-imgsz", type=int, default=DEFAULT_PERSON_IMGSZ, help=f"Inference image size. Default: {DEFAULT_PERSON_IMGSZ}.")
    parser.add_argument("--person-conf", type=float, default=DEFAULT_PERSON_CONFIDENCE, help=f"Minimum person confidence. Default: {DEFAULT_PERSON_CONFIDENCE}.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    ip = args.ip or _first_reachable_registered_ip(args.probe_timeout, args.probe_retries)

    config = SingleDroneSearchConfig(
        min_battery=args.min_battery,
        status_retries=args.status_retries,
        max_search_seconds=args.max_search_seconds,
        yaw_step_degrees=args.yaw_step_degrees,
        yaw_interval_seconds=args.yaw_interval,
        takeoff_settle_seconds=args.takeoff_settle,
        confirmation_detections=args.confirm_detections,
        confirmation_window_seconds=args.confirm_window,
        min_confidence=args.person_conf,
        preview_enabled=not args.no_preview,
        detection_hold_seconds=args.detection_hold,
    )
    detector_config = PersonDetectorConfig(
        model_name=args.person_model,
        imgsz=args.person_imgsz,
        confidence=args.person_conf,
    )

    print(f"[{timestamp()}] Single-drone person search: {ip}", flush=True)
    print("Expected behavior: live preview first; press G for takeoff -> yaw in place -> person detected -> land.", flush=True)
    print("No forward/sideways movement is sent by this script.", flush=True)
    if not args.no_preview:
        print("Live preview enabled: P=preflight, G=start search, L=land, Q/Esc=quit-land.", flush=True)
    else:
        print("Preview disabled: running immediate takeoff/search/land flow.", flush=True)

    link = DirectTelloLink(ip)
    runner_cls = SingleDroneSearchRunner if args.no_preview else SingleDroneSearchPreviewApp
    runner = runner_cls(
        ip,
        detector_config,
        config,
        controller=DirectTelloController(link),  # type: ignore[arg-type]
        camera=DirectTelloCamera(link),  # type: ignore[arg-type]
    )
    result = runner.run(status=lambda message: print(f"[{timestamp()}] {message}", flush=True))
    outcome = "DETECTED" if result.detected else "NOT DETECTED"
    print(f"[{timestamp()}] Result: {outcome}; landed={result.landed}; reason={result.reason}", flush=True)


def _first_reachable_registered_ip(timeout: float, retries: int) -> str:
    ips = load_registered_ips()
    if not ips:
        raise RuntimeError(f"No IP provided and no registered drones found in {DEFAULT_DRONE_IPS_FILE}.")

    print(f"[{timestamp()}] No IP supplied. Probing registered drones: {', '.join(ips)}", flush=True)
    with TelloUdpClient() as client:
        for ip in ips:
            try:
                print(f"[{timestamp()}] Probing {ip}...", flush=True)
                client.send_one(ip, "command", timeout=timeout, retries=retries)
                battery, _latency_ms = query_battery(client, ip, timeout=timeout, retries=retries)
                print(f"[{timestamp()}] Selected reachable drone {ip}; battery={battery}%", flush=True)
                return ip
            except Exception as error:
                print(f"[{timestamp()}] {ip} not reachable: {error}", flush=True)
    raise RuntimeError("No registered drone responded to command/battery probe.")


def _validate_args(args: argparse.Namespace) -> None:
    if args.min_battery < 0:
        raise ValueError("--min-battery must be 0 or greater")
    if args.status_retries <= 0:
        raise ValueError("--status-retries must be greater than 0")
    if args.probe_timeout <= 0:
        raise ValueError("--probe-timeout must be greater than 0")
    if args.probe_retries <= 0:
        raise ValueError("--probe-retries must be greater than 0")
    if args.max_search_seconds <= 0:
        raise ValueError("--max-search-seconds must be greater than 0")
    if not (1 <= args.yaw_step_degrees <= 360):
        raise ValueError("--yaw-step-degrees must be in [1, 360]")
    if args.yaw_interval <= 0:
        raise ValueError("--yaw-interval must be greater than 0")
    if args.takeoff_settle < 0:
        raise ValueError("--takeoff-settle must be 0 or greater")
    if args.confirm_detections <= 0:
        raise ValueError("--confirm-detections must be greater than 0")
    if args.confirm_window <= 0:
        raise ValueError("--confirm-window must be greater than 0")
    if args.detection_hold < 0:
        raise ValueError("--detection-hold must be 0 or greater")
    if args.person_imgsz <= 0:
        raise ValueError("--person-imgsz must be greater than 0")
    if not (0.0 < args.person_conf <= 1.0):
        raise ValueError("--person-conf must be in (0, 1]")


if __name__ == "__main__":
    main()
