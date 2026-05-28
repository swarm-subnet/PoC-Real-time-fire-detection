"""Live fire detection from the first reachable Tello camera.

This script does not fly. It only opens the camera stream, runs the
SuperBitDev/fire1 ONNX model, draws hazard boxes, and saves annotated frames.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_IP_FILE = ROOT_DIR / "scripts" / "swarm" / "drone_ips.txt"
DEFAULT_SAVE_DIR = ROOT_DIR / "captures" / "fire_live"
DEFAULT_VIDEO_DIR = ROOT_DIR / "captures" / "fire_videos"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_RECORD_FPS = 20.0
DEFAULT_DETECTION_PROFILE = "candle"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_utils import FIRE_MODEL_REPO_ID, FireOnnxDetector, get_fire_model_path, keep_fire_detections, predict_center_zoom, predict_tiled
from media_utils import (
    VideoRecording,
    close_video_recording,
    create_video_recording,
    save_frame_once_per_second,
    write_video_frame,
)
from tello_stream import (
    build_candidate_ips as build_stream_candidate_ips,
    connect_first_streaming_drone,
    resolve_path as resolve_root_path,
    stop_streaming_tello,
)
from yolo_utils import detect_labels_in_image, draw_detections, load_yolo_model


WINDOW_NAME = "Tello YOLO Live Fire Detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show live Tello video, detect fire, and save annotated frames/video."
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
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Local YOLO .pt/.onnx path. If omitted, downloads weights.onnx from SuperBitDev/fire1.",
    )
    parser.add_argument("--repo-id", default=FIRE_MODEL_REPO_ID, help="Hugging Face model repo ID.")
    parser.add_argument(
        "--profile",
        choices=("candle", "miner"),
        default=DEFAULT_DETECTION_PROFILE,
        help="ONNX post-processing profile. 'miner' matches the published miner thresholds; 'candle' is more sensitive.",
    )
    parser.add_argument("--include-smoke", action="store_true", help="Also include smoke detections.")
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for .pt fallback models. ONNX uses --profile thresholds.",
    )
    parser.add_argument("--every", type=int, default=3, help="Run YOLO every N frames to reduce CPU load.")
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Run overlapping 2x2 tiled detection to improve small fire detection.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        help="Fractional overlap for 2x2 tiled detection. Default: 0.25.",
    )
    parser.add_argument(
        "--tile-only",
        action="store_true",
        help="When --tiled is enabled, skip the full-frame pass and run only tiles.",
    )
    parser.add_argument(
        "--center-zoom",
        action="store_true",
        help="Run inference on a center crop to improve small-object detection in a controlled scene.",
    )
    parser.add_argument(
        "--center-crop",
        type=float,
        default=0.70,
        help="Fraction of frame width/height used by --center-zoom. Default: 0.70.",
    )
    parser.add_argument(
        "--center-include-full-frame",
        action="store_true",
        help="When --center-zoom is enabled, also run one full-frame pass.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="Keep showing the last positive fire detection for this long to reduce flicker. Default: 1.0.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_SAVE_DIR),
        help="Directory where annotated frames are saved once per second.",
    )
    parser.add_argument(
        "--video-dir",
        default=str(DEFAULT_VIDEO_DIR),
        help="Directory where the annotated MP4 recording is saved.",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=DEFAULT_RECORD_FPS,
        help="FPS to write into the annotated video file. Default: 20.",
    )
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Disable annotated video recording.",
    )
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for video frames before trying the next IP.",
    )
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    return resolve_root_path(ROOT_DIR, path_text)


def build_candidate_ips(args: argparse.Namespace) -> list[str]:
    return build_stream_candidate_ips(
        ROOT_DIR,
        args.ip,
        args.ip_file,
        include_direct_fallback=not args.no_direct_fallback,
    )


def detect_fire_with_fallback(
    model,
    frame_bgr,
    confidence_threshold: float,
    include_smoke: bool,
    tiled: bool,
    tile_overlap: float,
    tile_only: bool,
    center_zoom: bool,
    center_crop: float,
    center_include_full_frame: bool,
):
    if isinstance(model, FireOnnxDetector):
        if center_zoom:
            detections = predict_center_zoom(
                model,
                frame_bgr,
                crop_fraction=center_crop,
                include_full_frame=center_include_full_frame,
            )
        elif tiled:
            detections = predict_tiled(
                model,
                frame_bgr,
                include_full_frame=not tile_only,
                overlap=tile_overlap,
            )
        else:
            detections = model.predict(frame_bgr)
        return keep_fire_detections(
            detections,
            include_smoke=include_smoke,
        )

    target_substrings = {"fire", "flame", "smoke"} if include_smoke else {"fire", "flame"}
    detections = detect_labels_in_image(
        model,
        frame_bgr,
        target_substrings=target_substrings,
        confidence_threshold=confidence_threshold,
    )
    if not detections and len(model.names) == 1:
        detections = detect_labels_in_image(model, frame_bgr, confidence_threshold=confidence_threshold)
    return detections


def draw_status_overlay(
    frame_bgr,
    ip: str,
    battery: int | None,
    detections,
    saved_count: int,
    profile: str,
    tiled: bool,
    center_zoom: bool,
    inference_ms: float | None,
    include_smoke: bool,
) -> None:
    fire_count = sum(1 for detection in detections if detection.label.lower() == "fire")
    smoke_count = sum(1 for detection in detections if detection.label.lower() == "smoke")
    has_hazard = fire_count > 0 or smoke_count > 0
    if include_smoke:
        status = f"FIRE: {fire_count}  SMOKE: {smoke_count}" if has_hazard else "NO FIRE/SMOKE"
    else:
        status = f"FIRE: {fire_count}" if fire_count else "NO FIRE"
    color = (0, 0, 255) if fire_count else ((180, 180, 180) if smoke_count else (0, 255, 0))
    inference_text = f"{inference_ms:.0f} ms" if inference_ms is not None else "waiting"

    overlay_height = 116
    y0 = max(0, frame_bgr.shape[0] - overlay_height)
    cv2.rectangle(frame_bgr, (0, y0), (frame_bgr.shape[1], frame_bgr.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame_bgr, status, (12, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    mode = " center" if center_zoom else (" tiled" if tiled else "")
    cv2.putText(
        frame_bgr,
        f"Drone: {ip}  Battery: {battery if battery is not None else '?'}%  ONNX: {inference_text}  Profile: {profile}{mode}",
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


def run_fire_detection(
    model,
    frame_bgr,
    confidence_threshold: float,
    include_smoke: bool,
    tiled: bool,
    tile_overlap: float,
    tile_only: bool,
    center_zoom: bool,
    center_crop: float,
    center_include_full_frame: bool,
):
    inference_start = time.perf_counter()
    detections = detect_fire_with_fallback(
        model,
        frame_bgr,
        confidence_threshold=confidence_threshold,
        include_smoke=include_smoke,
        tiled=tiled,
        tile_overlap=tile_overlap,
        tile_only=tile_only,
        center_zoom=center_zoom,
        center_crop=center_crop,
        center_include_full_frame=center_include_full_frame,
    )
    inference_ms = (time.perf_counter() - inference_start) * 1000
    return detections, inference_ms


def format_hazard_counts(detections, include_smoke: bool) -> str:
    fire_count = sum(1 for detection in detections if detection.label.lower() == "fire")
    smoke_count = sum(1 for detection in detections if detection.label.lower() == "smoke")
    if include_smoke:
        return f"Fire: {fire_count}  Smoke: {smoke_count}"
    return f"Fire: {fire_count}"


def main() -> None:
    args = parse_args()
    candidate_ips = build_candidate_ips(args)
    save_dir = resolve_path(args.save_dir)
    video_dir = resolve_path(args.video_dir)
    model_path = resolve_path(args.model) if args.model is not None else get_fire_model_path(args.repo_id)

    tello = None
    frame_read = None
    latest_detections = []
    last_positive_detections = []
    last_positive_time = 0.0
    latest_inference_ms: float | None = None
    frame_count = 0
    saved_count = 0
    saved_second: int | None = None
    last_reported_count: int | None = None
    pending_future = None
    video_recording: VideoRecording | None = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        print(f"Loading fire model: {model_path}")
        if model_path.suffix.lower() == ".onnx":
            model = FireOnnxDetector(model_path, profile=args.profile)
            print(f"Model labels: {model.class_names}")
            print(f"Detection profile: {model.profile}")
        else:
            model = load_yolo_model(str(model_path))
            print(f"Model labels: {model.names}")

        tello, frame_read, drone_ip, battery = connect_first_streaming_drone(
            candidate_ips,
            frame_timeout=args.frame_timeout,
            usage_label="for live fire detection",
        )

        print("Safety note: this script only observes. Do not fly near real fire, heat, smoke, or people.")
        if args.center_zoom:
            print(
                "Center-zoom detection enabled: "
                f"center crop={args.center_crop:.2f}, full_frame={args.center_include_full_frame}."
            )
        elif args.tiled:
            print(
                "Tiled detection enabled: 2x2 overlapping tiles "
                f"(overlap={args.tile_overlap:.2f}, full_frame={not args.tile_only})."
            )
        print(f"Saving one annotated frame per second to: {save_dir}")
        if args.no_record_video:
            print("Annotated video recording is disabled.")
        else:
            print(f"Saving annotated video to: {video_dir}")
        print("Video window open. Press 'q' to quit.")

        while True:
            if pending_future is not None and pending_future.done():
                try:
                    latest_detections, latest_inference_ms = pending_future.result()
                    if latest_detections:
                        last_positive_detections = latest_detections
                        last_positive_time = time.monotonic()

                    if len(latest_detections) != last_reported_count:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] {format_hazard_counts(latest_detections, args.include_smoke)}  "
                            f"ONNX: {latest_inference_ms:.0f} ms"
                        )
                        last_reported_count = len(latest_detections)
                except Exception as error:
                    print(f"[{time.strftime('%H:%M:%S')}] Fire inference failed: {error}")
                finally:
                    pending_future = None

            frame_rgb = frame_read.frame
            if frame_rgb is None:
                time.sleep(0.05)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_count += 1

            if pending_future is None and frame_count % max(1, args.every) == 0:
                pending_future = executor.submit(
                    run_fire_detection,
                    model,
                    frame_bgr.copy(),
                    args.conf,
                    args.include_smoke,
                    args.tiled,
                    args.tile_overlap,
                    args.tile_only,
                    args.center_zoom,
                    args.center_crop,
                    args.center_include_full_frame,
                )

            visible_detections = latest_detections
            if not visible_detections and time.monotonic() - last_positive_time <= args.hold_seconds:
                visible_detections = last_positive_detections

            display_frame = draw_detections(frame_bgr, visible_detections)
            draw_status_overlay(
                display_frame,
                drone_ip,
                battery,
                visible_detections,
                saved_count,
                args.profile,
                args.tiled,
                args.center_zoom,
                latest_inference_ms,
                args.include_smoke,
            )

            if not args.no_record_video:
                if video_recording is None:
                    video_recording = create_video_recording(
                        video_dir,
                        drone_ip,
                        display_frame,
                        args.record_fps,
                        temp_subdir="fire_videos",
                    )
                    print(f"Recording annotated video: {video_recording.final_path}")
                write_video_frame(video_recording, display_frame)

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
        print(f"Live saved fire detection failed: {error}")
    finally:
        if pending_future is not None:
            pending_future.cancel()
        stop_streaming_tello(tello, frame_read)
        if video_recording is not None:
            close_video_recording(video_recording)
        executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
