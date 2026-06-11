"""Capture metric depth from images, folders, or a live Tello camera.

Examples:
  python scripts/depth/01_depth_capture.py image path/to/image.jpg
  python scripts/depth/01_depth_capture.py folder captures/depth_metric_live/<run-id> --limit 10
  python scripts/depth/01_depth_capture.py live --ip 192.168.1.132
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_IP_FILE = ROOT_DIR / "scripts" / "swarm" / "config" / "drone_ips.txt"
DEFAULT_IMAGE_OUTPUT_DIR = ROOT_DIR / "captures" / "depth_images"
DEFAULT_FOLDER_OUTPUT_DIR = ROOT_DIR / "captures" / "depth_folder"
DEFAULT_LIVE_SAVE_DIR = ROOT_DIR / "captures" / "depth_live"
DEFAULT_VIDEO_DIR = ROOT_DIR / "captures" / "depth_videos"
DEFAULT_RECORD_FPS = 20.0

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from depth_utils import (  # noqa: E402
    DEFAULT_CPU_THREADS,
    DEFAULT_METRIC_DEPTH_MODEL,
    DepthPrediction,
    MetricDepthEstimator,
    draw_depth_status,
    make_depth_display,
    make_depth_side_by_side,
    save_depth_sample,
)
from media_utils import VideoRecording, close_video_recording, create_video_recording, write_video_frame  # noqa: E402
from tello_stream import (  # noqa: E402
    build_candidate_ips as build_stream_candidate_ips,
    connect_first_streaming_drone,
    resolve_path as resolve_root_path,
    stop_streaming_tello,
)


WINDOW_NAME = "Tello Metric Depth Capture"


@dataclass
class DepthJobResult:
    frame_bgr: Any
    prediction: DepthPrediction
    finished_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metric indoor depth on Tello camera data.")
    parser.add_argument(
        "--model",
        default=DEFAULT_METRIC_DEPTH_MODEL,
        help=f"Hugging Face metric depth model. Default: {DEFAULT_METRIC_DEPTH_MODEL}",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Inference device. Default: cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Model dtype. Default: auto.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help=f"PyTorch CPU threads to use. Default: {DEFAULT_CPU_THREADS}. Use 0 to leave PyTorch default.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use optional dynamic int8 quantization on CPU. Faster in tests, but depth values can shift slightly.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="Run depth on one local image.")
    image_parser.add_argument("image", type=Path, help="Path to an RGB image.")
    image_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_IMAGE_OUTPUT_DIR,
        help="Directory for RGB, depth, and CSV outputs.",
    )

    folder_parser = subparsers.add_parser("folder", help="Run depth on captured '*_rgb.jpg' images.")
    folder_parser.add_argument("folder", type=Path, help="Folder containing captured '*_rgb.jpg' images.")
    folder_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FOLDER_OUTPUT_DIR,
        help="Directory for depth outputs.",
    )
    folder_parser.add_argument("--limit", type=int, default=0, help="Maximum images to process. Use 0 for all.")
    folder_parser.add_argument("--stride", type=int, default=1, help="Process every Nth image. Default: 1.")

    live_parser = subparsers.add_parser("live", help="Run live Tello video depth capture.")
    live_parser.add_argument(
        "--ip",
        action="append",
        help="Drone IP to try. Can be passed multiple times. If omitted, uses scripts/swarm/config/drone_ips.txt.",
    )
    live_parser.add_argument(
        "--ip-file",
        default=str(DEFAULT_IP_FILE),
        help="Text file with one station-mode drone IP per line.",
    )
    live_parser.add_argument(
        "--no-direct-fallback",
        action="store_true",
        help="Do not try 192.168.10.1 after registered IPs fail.",
    )
    live_parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Submit depth inference every N video frames when the model is idle. Default: 1.",
    )
    live_parser.add_argument(
        "--view",
        choices=("side-by-side", "depth"),
        default="depth",
        help="Live window/recording layout. Default: depth.",
    )
    live_parser.add_argument(
        "--smooth",
        action="store_true",
        help="Compatibility flag for the current defaults: infer every frame, save every 5s, and show depth only.",
    )
    live_parser.add_argument(
        "--hide-overlay",
        action="store_true",
        help="Hide status text on the live preview and recording.",
    )
    live_parser.add_argument(
        "--save-every-seconds",
        type=float,
        default=5.0,
        help="Save at most one processed sample every N seconds. Default: 5.",
    )
    live_parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_LIVE_SAVE_DIR),
        help="Root directory for depth artifacts. A timestamped run folder is created inside it.",
    )
    live_parser.add_argument(
        "--video-dir",
        default=str(DEFAULT_VIDEO_DIR),
        help="Directory where the side-by-side MP4 recording is saved.",
    )
    live_parser.add_argument(
        "--record-fps",
        type=float,
        default=DEFAULT_RECORD_FPS,
        help="FPS to write into the depth video file. Default: 20.",
    )
    live_parser.add_argument("--no-record-video", action="store_true", help="Disable side-by-side video recording.")
    live_parser.add_argument(
        "--frame-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for video frames before trying the next IP.",
    )

    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    return resolve_root_path(ROOT_DIR, path_text)


def load_estimator(args: argparse.Namespace) -> MetricDepthEstimator:
    print(f"Loading metric depth model: {args.model}")
    estimator = MetricDepthEstimator(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        cpu_threads=args.cpu_threads,
        quantize=args.quantize,
    )
    if estimator.cpu_threads:
        print(f"Using PyTorch CPU threads: {estimator.cpu_threads}")
    if estimator.quantized:
        print("Using dynamic int8 quantization for CPU inference.")
    return estimator


def run_image(args: argparse.Namespace) -> None:
    image_path = resolve_path(args.image)
    output_dir = resolve_path(args.output_dir)
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    estimator = load_estimator(args)
    print(f"Running metric depth estimation on: {image_path}")
    prediction = estimator.predict(frame_bgr)
    sample_id = time.strftime("%Y%m%d_%H%M%S")
    row = save_depth_sample(output_dir, sample_id, "local-image", frame_bgr, prediction)

    print(f"Metric depth inference: {prediction.inference_ms:.0f} ms")
    print("Raw depth is in meters. Normalized depth uses subnet 0.5m..20m convention.")
    print(f"Saved side-by-side image: {row['side_by_side_path']}")
    print(f"Saved stats CSV: {output_dir / 'depth_stats.csv'}")


def run_folder(args: argparse.Namespace) -> None:
    input_dir = resolve_path(args.folder)
    output_dir = resolve_path(args.output_dir)
    image_paths = sorted(input_dir.glob("*_rgb.jpg"))[:: max(1, args.stride)]
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No '*_rgb.jpg' images found in: {input_dir}")

    estimator = load_estimator(args)
    print(f"Processing {len(image_paths)} image(s) from: {input_dir}")
    print(f"Saving metric depth run to: {output_dir}")

    for index, image_path in enumerate(image_paths, start=1):
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            print(f"[{index}/{len(image_paths)}] Skipping unreadable image: {image_path}")
            continue

        prediction = estimator.predict(frame_bgr)
        sample_id = image_path.stem.removesuffix("_rgb")
        row = save_depth_sample(output_dir, sample_id, "folder", frame_bgr, prediction)
        print(
            f"[{index}/{len(image_paths)}] {image_path.name}: "
            f"{prediction.inference_ms:.0f} ms -> {row['side_by_side_path']}"
        )

    print(f"Saved stats CSV: {output_dir / 'depth_stats.csv'}")


def build_candidate_ips(args: argparse.Namespace) -> list[str]:
    return build_stream_candidate_ips(
        ROOT_DIR,
        args.ip,
        args.ip_file,
        include_direct_fallback=not args.no_direct_fallback,
    )


def run_depth_job(estimator: MetricDepthEstimator, frame_bgr) -> DepthJobResult:
    prediction = estimator.predict(frame_bgr)
    return DepthJobResult(frame_bgr=frame_bgr, prediction=prediction, finished_at=time.time())


def make_display_frame(
    latest_frame_bgr,
    latest_prediction: DepthPrediction | None,
    latest_depth_frame_bgr,
    drone_ip: str,
    battery: int | None,
    saved_count: int,
    view: str,
    with_overlay: bool,
    depth_frame_count: int,
):
    if latest_prediction is None or latest_depth_frame_bgr is None:
        blank_depth = latest_frame_bgr.copy()
        cv2.putText(
            blank_depth,
            "Waiting for metric depth model...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        display = blank_depth if view == "depth" else cv2.hconcat([latest_frame_bgr, blank_depth])
        if with_overlay:
            draw_depth_status(display, drone_ip, battery, None, saved_count)
        return display

    if view == "depth":
        frame_size = (latest_depth_frame_bgr.shape[1], latest_depth_frame_bgr.shape[0])
        return make_depth_display(
            latest_prediction,
            frame_size,
            with_overlay=with_overlay,
            drone_ip=drone_ip,
            battery=battery,
            saved_count=saved_count,
            depth_frame_count=depth_frame_count,
        )

    display = make_depth_side_by_side(latest_depth_frame_bgr, latest_prediction)
    if with_overlay:
        draw_depth_status(display, drone_ip, battery, latest_prediction, saved_count)
    return display


def run_live(args: argparse.Namespace) -> None:
    if args.smooth:
        args.every = 1
        args.save_every_seconds = max(args.save_every_seconds, 5.0)
        args.view = "depth"

    candidate_ips = build_candidate_ips(args)
    save_root = resolve_path(args.save_dir)
    video_dir = resolve_path(args.video_dir)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = save_root / run_id

    tello = None
    frame_read = None
    pending_future = None
    video_recording: VideoRecording | None = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    frame_count = 0
    saved_count = 0
    last_save_time = 0.0
    latest_prediction: DepthPrediction | None = None
    latest_depth_frame_bgr = None
    latest_reported_ms: int | None = None
    depth_frame_count = 0

    try:
        estimator = load_estimator(args)
        tello, frame_read, drone_ip, battery = connect_first_streaming_drone(
            candidate_ips,
            frame_timeout=args.frame_timeout,
            usage_label="for live metric depth capture",
        )

        print("This script only observes. It does not take off or move the drone.")
        print("Raw metric output is in meters; saved normalized depth uses subnet 0.5m..20m convention.")
        print(f"Live view: {args.view}. Submit depth inference every {max(1, args.every)} frame(s).")
        print(f"Saving metric depth run to: {run_dir}")
        if args.no_record_video:
            print("Depth video recording is disabled.")
        else:
            print(f"Saving depth video to: {video_dir}")
        print("Video window open. Press 'q' to quit.")

        while True:
            if pending_future is not None and pending_future.done():
                try:
                    result = pending_future.result()
                    latest_prediction = result.prediction
                    latest_depth_frame_bgr = result.frame_bgr
                    depth_frame_count += 1

                    rounded_ms = int(round(latest_prediction.inference_ms))
                    if rounded_ms != latest_reported_ms or args.view == "depth":
                        print(
                            f"[{time.strftime('%H:%M:%S')}] Metric depth inference: "
                            f"{latest_prediction.inference_ms:.0f} ms "
                            f"(depth frame #{depth_frame_count})"
                        )
                        latest_reported_ms = rounded_ms

                    if result.finished_at - last_save_time >= max(0.1, args.save_every_seconds):
                        sample_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(result.finished_at))
                        row = save_depth_sample(run_dir, sample_id, drone_ip, result.frame_bgr, latest_prediction)
                        saved_count += 1
                        last_save_time = result.finished_at
                        print(f"[{time.strftime('%H:%M:%S')}] Saved metric depth sample: {row['side_by_side_path']}")
                except Exception as error:
                    print(f"[{time.strftime('%H:%M:%S')}] Metric depth inference failed: {error}")
                finally:
                    pending_future = None

            frame_rgb = frame_read.frame
            if frame_rgb is None:
                time.sleep(0.05)
                continue

            latest_frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_count += 1

            if pending_future is None and frame_count % max(1, args.every) == 0:
                pending_future = executor.submit(run_depth_job, estimator, latest_frame_bgr.copy())

            display_frame = make_display_frame(
                latest_frame_bgr,
                latest_prediction,
                latest_depth_frame_bgr,
                drone_ip,
                battery,
                saved_count,
                args.view,
                not args.hide_overlay,
                depth_frame_count,
            )

            if not args.no_record_video:
                if video_recording is None:
                    video_recording = create_video_recording(
                        video_dir,
                        drone_ip,
                        display_frame,
                        args.record_fps,
                        temp_subdir="depth_videos",
                    )
                    print(f"Recording side-by-side video: {video_recording.final_path}")
                write_video_frame(video_recording, display_frame)

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Live metric depth capture failed: {error}")
    finally:
        if pending_future is not None:
            pending_future.cancel()
        stop_streaming_tello(tello, frame_read)
        if video_recording is not None:
            close_video_recording(video_recording)
        executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.command == "image":
        run_image(args)
    elif args.command == "folder":
        run_folder(args)
    elif args.command == "live":
        run_live(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
