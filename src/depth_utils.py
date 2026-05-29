"""Utilities for Tello metric-depth experiments.

The repository uses Depth Anything V2 Metric Indoor Small to estimate depth in
meters from the Tello RGB camera. Saved normalized depth arrays follow the
subnet contract: 0.0 = 0.5m or closer, 1.0 = 20m or farther.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_METRIC_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEFAULT_CPU_THREADS = min(6, os.cpu_count() or 1)
DEPTH_128_SIZE = (128, 128)
SUBNET_DEPTH_MIN_M = 0.5
SUBNET_DEPTH_MAX_M = 20.0

SECTOR_NAMES = (
    "top_left",
    "top_center",
    "top_right",
    "mid_left",
    "mid_center",
    "mid_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)


@dataclass
class DepthPrediction:
    """Depth model output plus normalized forms used for display and datasets."""

    raw: np.ndarray
    normalized: np.ndarray
    depth_128: np.ndarray
    color_bgr: np.ndarray
    inference_ms: float
    model_name: str
    metadata: dict[str, str | float] = field(default_factory=dict)


class MetricDepthEstimator:
    """Depth Anything V2 Metric Indoor wrapper."""

    def __init__(
        self,
        model_name: str = DEFAULT_METRIC_DEPTH_MODEL,
        device: str = "cpu",
        dtype: str = "auto",
        cpu_threads: int = DEFAULT_CPU_THREADS,
        quantize: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:
            raise RuntimeError(
                "Metric depth estimation requires torch, transformers, and pillow. "
                "Install them with: pip install -r requirements.txt"
            ) from error

        self.torch = torch
        self.device = resolve_torch_device(torch, device)
        self.dtype = resolve_torch_dtype(torch, dtype, self.device)
        self.model_name = model_name
        self.cpu_threads = configure_torch_cpu_threads(torch, cpu_threads, self.device)
        self.quantized = False

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        load_kwargs: dict[str, Any] = {}
        if self.dtype is not None:
            load_kwargs["torch_dtype"] = self.dtype

        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, **load_kwargs)
        if quantize:
            if self.device.type != "cpu":
                raise ValueError("--quantize is only supported for CPU inference.")
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.quantized = True

        self.model.to(self.device)
        self.model.eval()

    def predict(self, frame_bgr: np.ndarray) -> DepthPrediction:
        """Estimate metric depth for one BGR OpenCV frame."""
        from PIL import Image

        started = time.perf_counter()
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(frame_rgb), return_tensors="pt").to(self.device)
        if self.dtype is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        with self.torch.inference_mode():
            outputs = self.model(**inputs)

        prediction = self.torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0

        depth_meters = tensor_to_numpy(prediction.squeeze()).astype(np.float32)
        normalized = normalize_metric_depth_to_subnet(depth_meters)

        return DepthPrediction(
            raw=depth_meters,
            normalized=normalized,
            depth_128=cv2.resize(normalized, DEPTH_128_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32),
            color_bgr=colorize_depth(normalized),
            inference_ms=inference_ms,
            model_name=self.model_name,
            metadata={
                "depth_type": "metric_depth_anything_v2_indoor",
                "depth_unit": "meters",
                "normalization": "subnet_metric_0.5m_20.0m",
                "display_mode": "metric indoor depth 0.5m..20m",
                "metric_min_m": SUBNET_DEPTH_MIN_M,
                "metric_max_m": SUBNET_DEPTH_MAX_M,
                "cpu_threads": float(self.cpu_threads),
                "quantized": float(self.quantized),
            },
        )


def resolve_torch_device(torch_module, device: str):
    """Resolve a user-facing device string to a torch.device."""
    if device == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    return torch_module.device(device)


def resolve_torch_dtype(torch_module, dtype: str, device) -> Any | None:
    """Pick a safe inference dtype.

    On the tested GTX 1650, CUDA float32 is substantially faster than float16
    for this model, so auto keeps the model in float32 unless explicitly
    overridden.
    """
    if dtype == "auto":
        return None
    if dtype == "float16":
        return torch_module.float16
    if dtype == "bfloat16":
        return torch_module.bfloat16
    if dtype == "float32":
        return None
    raise ValueError(f"Unsupported dtype: {dtype}")


def configure_torch_cpu_threads(torch_module, cpu_threads: int, device) -> int:
    """Set a conservative CPU thread count before inference starts."""
    if device.type != "cpu":
        return 0

    if cpu_threads <= 0:
        return torch_module.get_num_threads()

    threads = max(1, int(cpu_threads))
    torch_module.set_num_threads(threads)
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return threads


def tensor_to_numpy(value) -> np.ndarray:
    """Convert torch tensors or array-like values to numpy arrays."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_metric_depth_to_subnet(
    depth_meters: np.ndarray,
    min_m: float = SUBNET_DEPTH_MIN_M,
    max_m: float = SUBNET_DEPTH_MAX_M,
) -> np.ndarray:
    """Normalize metric meters to the subnet convention."""
    depth = np.asarray(depth_meters, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    safe = depth.copy()
    safe[~valid] = max_m
    clipped = np.clip(safe, min_m, max_m)
    return ((clipped - min_m) / (max_m - min_m)).astype(np.float32)


def colorize_depth(normalized: np.ndarray) -> np.ndarray:
    """Create a readable color depth visualization."""
    depth_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def compute_depth_stats(raw: np.ndarray, normalized: np.ndarray) -> dict[str, float]:
    """Compute stable per-frame statistics for real-vs-sim comparison."""
    stats: dict[str, float] = {}
    stats.update(_array_stats("raw", np.asarray(raw, dtype=np.float32)))
    stats.update(_array_stats("norm", np.asarray(normalized, dtype=np.float32)))
    stats["high_fraction_080"] = float(np.mean(normalized >= 0.80))
    stats["high_fraction_090"] = float(np.mean(normalized >= 0.90))

    height, width = normalized.shape[:2]
    row_edges = [0, height // 3, (2 * height) // 3, height]
    col_edges = [0, width // 3, (2 * width) // 3, width]
    sector_index = 0
    for row in range(3):
        for col in range(3):
            sector = normalized[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]]
            stats[f"sector_{SECTOR_NAMES[sector_index]}"] = float(np.mean(sector))
            sector_index += 1

    return stats


def _array_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
        }

    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p05": float(np.percentile(finite, 5)),
        f"{prefix}_p50": float(np.percentile(finite, 50)),
        f"{prefix}_p95": float(np.percentile(finite, 95)),
    }


def make_depth_side_by_side(frame_bgr: np.ndarray, prediction: DepthPrediction) -> np.ndarray:
    """Return RGB frame and colorized depth image beside each other."""
    depth_bgr = prediction.color_bgr
    if depth_bgr.shape[:2] != frame_bgr.shape[:2]:
        depth_bgr = cv2.resize(depth_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return np.hstack([frame_bgr, depth_bgr])


def draw_depth_status(
    frame_bgr: np.ndarray,
    drone_ip: str,
    battery: int | None,
    prediction: DepthPrediction | None,
    saved_count: int,
) -> None:
    """Draw compact status text at the bottom of a depth preview frame."""
    overlay_height = 64
    y0 = max(0, frame_bgr.shape[0] - overlay_height)
    cv2.rectangle(frame_bgr, (0, y0), (frame_bgr.shape[1], frame_bgr.shape[0]), (0, 0, 0), -1)

    inference_text = "waiting" if prediction is None else f"{prediction.inference_ms:.0f} ms"

    cv2.putText(
        frame_bgr,
        "MODE: DEPTH_CAPTURE",
        (12, y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"Drone: {drone_ip}  Battery: {battery if battery is not None else '?'}%  Depth: {inference_text}",
        (12, y0 + 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def make_depth_display(
    prediction: DepthPrediction,
    frame_size: tuple[int, int],
    with_overlay: bool,
    drone_ip: str,
    battery: int | None,
    saved_count: int,
    depth_frame_count: int = 0,
) -> np.ndarray:
    """Return a display frame containing only the colorized depth map."""
    width, height = frame_size
    display = prediction.color_bgr
    if display.shape[:2] != (height, width):
        display = cv2.resize(display, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        display = display.copy()

    if with_overlay:
        draw_depth_status(display, drone_ip, battery, prediction, saved_count)
    return display


def save_depth_sample(
    run_dir: Path,
    sample_id: str,
    source_id: str,
    frame_bgr: np.ndarray,
    prediction: DepthPrediction,
) -> dict[str, str | float]:
    """Save RGB, visual depth, 128x128 depth, raw depth, side-by-side, and stats."""
    run_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_depth_stats(prediction.raw, prediction.normalized)

    safe_source = source_id.replace(".", "-").replace("/", "-").replace("\\", "-")
    base_name = f"{sample_id}_{safe_source}"
    rgb_path = run_dir / f"{base_name}_rgb.jpg"
    color_path = run_dir / f"{base_name}_depth_color.jpg"
    norm_path = run_dir / f"{base_name}_depth_norm.png"
    depth_128_png_path = run_dir / f"{base_name}_depth_128.png"
    depth_128_npy_path = run_dir / f"{base_name}_depth_128.npy"
    raw_npy_path = run_dir / f"{base_name}_depth_raw.npy"
    side_by_side_path = run_dir / f"{base_name}_side_by_side.jpg"

    cv2.imwrite(str(rgb_path), frame_bgr)
    cv2.imwrite(str(color_path), prediction.color_bgr)
    cv2.imwrite(str(norm_path), np.clip(prediction.normalized * 65535.0, 0, 65535).astype(np.uint16))
    cv2.imwrite(str(depth_128_png_path), np.clip(prediction.depth_128 * 65535.0, 0, 65535).astype(np.uint16))
    cv2.imwrite(str(side_by_side_path), make_depth_side_by_side(frame_bgr, prediction))
    np.save(depth_128_npy_path, prediction.depth_128.astype(np.float32))
    np.save(raw_npy_path, prediction.raw.astype(np.float32))

    row: dict[str, str | float] = {
        "sample_id": sample_id,
        "source_id": source_id,
        "model": prediction.model_name,
        "frame_width": frame_bgr.shape[1],
        "frame_height": frame_bgr.shape[0],
        "inference_ms": prediction.inference_ms,
        "rgb_path": str(rgb_path),
        "depth_color_path": str(color_path),
        "depth_norm_path": str(norm_path),
        "depth_128_png_path": str(depth_128_png_path),
        "depth_128_npy_path": str(depth_128_npy_path),
        "depth_raw_npy_path": str(raw_npy_path),
        "side_by_side_path": str(side_by_side_path),
    }
    row.update(prediction.metadata)
    row.update(stats)
    append_depth_stats(run_dir / "depth_stats.csv", row)
    return row


def append_depth_stats(csv_path: Path, row: dict[str, str | float]) -> None:
    """Append one depth sample row to a CSV, creating the header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
