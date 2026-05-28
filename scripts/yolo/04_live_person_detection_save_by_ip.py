"""Live YOLO person detection from the first reachable Tello camera.

The script tries registered station-mode drone IPs first, then falls back to
the direct Tello Wi-Fi IP. It displays a live preview with person boxes and
saves one annotated frame per second while it runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_IP_FILE = ROOT_DIR / "scripts" / "swarm" / "drone_ips.txt"
DEFAULT_SAVE_DIR = ROOT_DIR / "captures" / "yolo_live"
DEFAULT_CONFIDENCE_THRESHOLD = 0.80

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from media_utils import save_frame_once_per_second
from tello_stream import (
    build_candidate_ips as build_stream_candidate_ips,
    connect_first_streaming_drone,
    resolve_path as resolve_root_path,
    stop_streaming_tello,
)
from yolo_utils import DEFAULT_MODEL_NAME, detect_people_in_image, draw_detections, load_yolo_model


WINDOW_NAME = "Tello YOLO Live Person Detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show live Tello video, detect people with YOLO, and save one annotated frame per second."
    )
    parser.add_argument(
        "--ip",
        action="append",
        help="Drone IP to try. Can be passed multiple times. If omitted, uses scripts/swarm/drone_ips.txt.",
    )
    parser.add_argument(
        "--ip-file",
        default=str(DEFAULT_IP_FILE),
        help="Text file with one station-mode drone IP per line.",
    )
    parser.add_argument(
        "--no-direct-fallback",
        action="store_true",
        help="Do not try 192.168.10.1 after registered IPs fail.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="YOLO model file/name.")
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Person detection confidence threshold. Default: 0.80.",
    )
    parser.add_argument("--every", type=int, default=3, help="Run YOLO every N frames to reduce CPU load.")
    parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_SAVE_DIR),
        help="Directory where annotated frames are saved once per second.",
    )
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for video frames before trying the next IP.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    return resolve_root_path(ROOT_DIR, path_text)


def build_candidate_ips(args: argparse.Namespace) -> list[str]:
    return build_stream_candidate_ips(
        ROOT_DIR,
        args.ip,
        args.ip_file,
        include_direct_fallback=not args.no_direct_fallback,
    )


def draw_status_overlay(
    frame_bgr,
    ip: str,
    battery: int | None,
    detections_count: int,
    saved_count: int,
    confidence_threshold: float,
    inference_ms: float | None,
) -> None:
    has_person = detections_count > 0
    threshold_percent = int(confidence_threshold * 100)
    status = f"PERSON >= {threshold_percent}%: {detections_count}" if has_person else f"NO PERSON >= {threshold_percent}%"
    color = (0, 255, 0) if has_person else (0, 0, 255)
    inference_text = f"{inference_ms:.0f} ms" if inference_ms is not None else "waiting"

    overlay_height = 116
    y0 = max(0, frame_bgr.shape[0] - overlay_height)
    cv2.rectangle(frame_bgr, (0, y0), (frame_bgr.shape[1], frame_bgr.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame_bgr, status, (12, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame_bgr,
        f"Drone: {ip}  Battery: {battery if battery is not None else '?'}%  YOLO: {inference_text}",
        (12, y0 + 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"Saved: {saved_count}  Press q to quit",
        (12, y0 + 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    candidate_ips = build_candidate_ips(args)
    save_dir = resolve_path(args.save_dir)

    tello = None
    frame_read = None
    latest_detections = []
    latest_inference_ms: float | None = None
    frame_count = 0
    saved_count = 0
    saved_second: int | None = None
    last_reported_count: int | None = None

    try:
        print(f"Loading YOLO model: {args.model}")
        model = load_yolo_model(args.model)

        tello, frame_read, drone_ip, battery = connect_first_streaming_drone(
            candidate_ips,
            frame_timeout=args.frame_timeout,
            usage_label="for live detection",
        )

        print(f"Saving one annotated frame per second to: {save_dir}")
        print("Video window open. Press 'q' to quit.")

        while True:
            frame_rgb = frame_read.frame
            if frame_rgb is None:
                time.sleep(0.05)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_count += 1

            if frame_count % max(1, args.every) == 0:
                inference_start = time.perf_counter()
                latest_detections = detect_people_in_image(
                    model,
                    frame_bgr,
                    confidence_threshold=args.conf,
                )
                latest_inference_ms = (time.perf_counter() - inference_start) * 1000
                if len(latest_detections) != last_reported_count:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] People >= {int(args.conf * 100)}%: "
                        f"{len(latest_detections)}  YOLO: {latest_inference_ms:.0f} ms"
                    )
                    last_reported_count = len(latest_detections)

            display_frame = draw_detections(frame_bgr, latest_detections)
            draw_status_overlay(
                display_frame,
                drone_ip,
                battery,
                len(latest_detections),
                saved_count,
                args.conf,
                latest_inference_ms,
            )

            saved_second, did_save = save_frame_once_per_second(
                display_frame,
                save_dir,
                drone_ip,
                saved_second,
            )
            saved_count += did_save

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Live saved person detection failed: {error}")
    finally:
        stop_streaming_tello(tello, frame_read)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
