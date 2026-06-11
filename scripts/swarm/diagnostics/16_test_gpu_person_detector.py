"""Verify CUDA YOLO person-detector inference for the swarm dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swarm_detector import (  # noqa: E402
    DEFAULT_PERSON_CONFIDENCE,
    DEFAULT_PERSON_IMGSZ,
    DEFAULT_PERSON_MODEL,
    PersonDetectorConfig,
    SwarmPersonDetector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail fast unless YOLO person detection works on CUDA GPU, then benchmark it.")
    parser.add_argument(
        "--model",
        default=DEFAULT_PERSON_MODEL,
        help=f"Ultralytics YOLO model to test. Default: {DEFAULT_PERSON_MODEL}.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_PERSON_IMGSZ,
        help=f"Inference image size. Default: {DEFAULT_PERSON_IMGSZ}.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_PERSON_CONFIDENCE,
        help=f"Minimum person confidence. Default: {DEFAULT_PERSON_CONFIDENCE}.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Measured benchmark iterations after warmup. Default: 100.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before measuring. Default: 10.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Frames per inference batch. Use 5 to mimic five drones. Default: 5.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0")
    if not (0.0 < args.conf <= 1.0):
        raise ValueError("--conf must be in (0, 1]")
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be 0 or greater")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    try:
        import torch

        print(f"torch={torch.__version__}", flush=True)
        print(f"cuda_available={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)
            print(f"cuda_device_0={torch.cuda.get_device_name(0)}", flush=True)

        benchmark_ips = [f"gpu_test_{index}" for index in range(args.batch_size)]
        detector = SwarmPersonDetector(
            benchmark_ips,
            PersonDetectorConfig(
                model_name=args.model,
                imgsz=args.imgsz,
                confidence=args.conf,
            ),
        )

        startup_started = time.perf_counter()
        detector.ensure_ready()
        startup_ms = (time.perf_counter() - startup_started) * 1000.0

        frames = _make_test_frames(args.batch_size, args.imgsz)
        for _ in range(args.warmup):
            _run_batch(detector, benchmark_ips, frames)

        batch_times_ms: list[float] = []
        wall_started = time.perf_counter()
        for _ in range(args.iterations):
            batch_times_ms.append(_run_batch(detector, benchmark_ips, frames))
        wall_ms = (time.perf_counter() - wall_started) * 1000.0

        stats = detector.snapshot_stats()

        if stats.status != "running:0":
            raise RuntimeError(f"detector ended in unexpected status: {stats.status}; {stats.last_error}")

        summary = _summarize(batch_times_ms)
        measured_frames = args.iterations * args.batch_size
        fps = measured_frames / max(wall_ms / 1000.0, 1e-9)
        per_frame_avg = summary["avg"] / args.batch_size

        print(
            "SUCCESS: GPU person detector inference works\n"
            f"(model={args.model}, imgsz={args.imgsz}, device=cuda:0, fp16={stats.half}, "
            f"batch_size={args.batch_size})\n"
            f"startup_ready_ms={startup_ms:.0f}  "
            f"warmup_iterations={args.warmup}  benchmark_iterations={args.iterations}\n"
            f"batch_ms: min={summary['min']:.1f} median={summary['median']:.1f} "
            f"p95={summary['p95']:.1f} avg={summary['avg']:.1f} max={summary['max']:.1f}\n"
            f"per_frame_avg_ms={per_frame_avg:.1f}  throughput_fps={fps:.1f}",
            flush=True,
        )
        return 0
    except Exception as error:
        print(f"FAIL: GPU person detector inference is not working: {error}", file=sys.stderr, flush=True)
        return 1


def _make_test_frames(batch_size: int, imgsz: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for index in range(batch_size):
        frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        # Add deterministic structure so preprocessing is closer to real frames than a pure black image.
        frame[:, :, 1] = (index * 37) % 255
        frame[imgsz // 4 : imgsz // 2, imgsz // 4 : imgsz // 2, 0] = 180
        frame[imgsz // 2 : (3 * imgsz) // 4, imgsz // 2 : (3 * imgsz) // 4, 2] = 220
        frames.append(frame)
    return frames


def _run_batch(detector: SwarmPersonDetector, ips: list[str], frames: list[np.ndarray]) -> float:
    versions = [index for index in range(len(frames))]
    detector._process_batch(ips, frames, versions)
    stats = detector.snapshot_stats()
    if stats.status != "running:0" or stats.last_inference_ms is None:
        raise RuntimeError(f"benchmark inference failed: {stats.status}; {stats.last_error}")
    return stats.last_inference_ms


def _summarize(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, int(round(0.95 * (len(sorted_values) - 1))))
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": sorted_values[p95_index],
        "avg": statistics.fmean(values),
        "max": max(values),
    }


if __name__ == "__main__":
    raise SystemExit(main())
