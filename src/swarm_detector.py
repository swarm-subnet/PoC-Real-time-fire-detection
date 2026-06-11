"""Person-only YOLO detector for swarm dashboard video frames."""

from __future__ import annotations

import builtins
from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading
import time
from typing import Any

import numpy as np


PERSON_CLASS_ID = 0
DEFAULT_PERSON_MODEL = "yolo11s.pt"
DEFAULT_PERSON_IMGSZ = 640
DEFAULT_PERSON_CONFIDENCE = 0.20
DEFAULT_MAX_BATCHES_PER_SECOND = 5.0
DETECTION_RETENTION_SECONDS = 1.25


@dataclass(frozen=True)
class PersonDetectorConfig:
    model_name: str = DEFAULT_PERSON_MODEL
    imgsz: int = DEFAULT_PERSON_IMGSZ
    confidence: float = DEFAULT_PERSON_CONFIDENCE
    max_batches_per_second: float = DEFAULT_MAX_BATCHES_PER_SECOND


@dataclass(frozen=True)
class PersonDetection:
    ip: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    frame_shape: tuple[int, int]
    detected_at_monotonic: float


@dataclass(frozen=True)
class PersonDetectorStats:
    loaded: bool
    status: str
    model_name: str
    imgsz: int
    confidence: float
    device: str
    half: bool
    total_inferences: int
    total_batches: int
    last_batch_size: int
    total_detections: int
    last_inference_ms: float | None
    average_inference_ms: float | None
    overall_fps: float
    last_error: str


@contextmanager
def _ultralytics_windows_os_release_guard():
    """Avoid an Ultralytics Windows/UNC-path probe of /etc/os-release."""
    if os.name != "nt":
        yield
        return

    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if str(file).replace("\\", "/") == "/etc/os-release":
            raise FileNotFoundError("/etc/os-release")
        return original_open(file, *args, **kwargs)

    builtins.open = guarded_open
    try:
        yield
    finally:
        builtins.open = original_open


class SwarmPersonDetector:
    """Shared CUDA YOLO detector over latest available frames from each drone."""

    def __init__(self, ips: list[str], config: PersonDetectorConfig) -> None:
        self.ips = list(ips)
        self.config = config
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames: dict[str, np.ndarray] = {}
        self._frame_versions: dict[str, int] = {}
        self._processed_versions: dict[str, int] = {}
        self._detections: dict[str, list[PersonDetection]] = {ip: [] for ip in ips}
        self._model: Any | None = None
        self._device = "cuda:0"
        self._half = True
        self._loaded = False
        self._status = "stopped"
        self._last_error = ""
        self._started_at = 0.0
        self._total_inferences = 0
        self._total_batches = 0
        self._last_batch_size = 0
        self._total_detections = 0
        self._last_inference_ms: float | None = None
        self._average_inference_ms: float | None = None
        self._last_batch_started_at = 0.0

    def ensure_ready(self) -> None:
        try:
            self._ensure_model_loaded()
        except Exception as error:
            self._set_error(f"detector unavailable: {error}")
            raise RuntimeError(f"GPU person detector unavailable: {error}") from error

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def update_frames(self, frames: dict[str, np.ndarray], frame_versions: dict[str, int] | None = None) -> None:
        with self._lock:
            for ip, frame in frames.items():
                if ip not in self.ips or frame is None:
                    continue
                if frame_versions is None:
                    next_version = self._frame_versions.get(ip, 0) + 1
                else:
                    next_version = frame_versions.get(ip, 0)
                    if next_version <= self._frame_versions.get(ip, 0):
                        continue
                self._frames[ip] = frame
                self._frame_versions[ip] = next_version

    def get_detections(self) -> dict[str, list[PersonDetection]]:
        with self._lock:
            return {ip: list(detections) for ip, detections in self._detections.items()}

    def snapshot_stats(self) -> PersonDetectorStats:
        with self._lock:
            elapsed = max(1e-6, time.monotonic() - self._started_at) if self._started_at else 0.0
            overall_fps = self._total_inferences / elapsed if elapsed else 0.0
            return PersonDetectorStats(
                loaded=self._loaded,
                status=self._status,
                model_name=self.config.model_name,
                imgsz=self.config.imgsz,
                confidence=self.config.confidence,
                device=self._device,
                half=self._half,
                total_inferences=self._total_inferences,
                total_batches=self._total_batches,
                last_batch_size=self._last_batch_size,
                total_detections=self._total_detections,
                last_inference_ms=self._last_inference_ms,
                average_inference_ms=self._average_inference_ms,
                overall_fps=overall_fps,
                last_error=self._last_error,
            )

    def _run_loop(self) -> None:
        try:
            self._ensure_model_loaded()
            self._set_status(f"running:{self._device}")
        except Exception as error:
            self._set_error(f"detector unavailable: {error}")
            return

        while not self._stop_event.is_set():
            if not self._wait_for_inference_slot():
                continue
            batch = self._next_latest_batch()
            if batch is None:
                self._stop_event.wait(0.005)
                continue
            ips, frames, versions = batch
            self._process_batch(ips, frames, versions)
        self._set_status("stopped")

    def _ensure_model_loaded(self) -> None:
        with self._lock:
            if self._loaded and self._model is not None:
                return
            self._status = "loading model"

        self._verify_cuda()
        model = self._load_model()
        self._verify_model_inference(model)
        with self._lock:
            self._model = model
            self._loaded = True
            self._status = f"ready:{self._device}"
            self._last_error = ""

    def _load_model(self):
        with _ultralytics_windows_os_release_guard():
            try:
                from ultralytics import YOLO
            except ModuleNotFoundError as error:
                raise RuntimeError("install ultralytics to enable person detection") from error
            return YOLO(self.config.model_name)

    def _verify_cuda(self) -> None:
        try:
            import torch
        except Exception as error:
            raise RuntimeError("PyTorch with CUDA is required for person detection") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required; torch.cuda.is_available() is False")
        if torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA GPU is required; torch.cuda.device_count() is 0")

    def _verify_model_inference(self, model) -> None:
        dummy = np.zeros((max(32, self.config.imgsz), max(32, self.config.imgsz), 3), dtype=np.uint8)
        kwargs = self._predict_kwargs()
        with _ultralytics_windows_os_release_guard():
            model.predict([dummy], **kwargs)
        self._synchronize_cuda()

    def _predict_kwargs(self) -> dict[str, object]:
        return {
            "imgsz": self.config.imgsz,
            "classes": [PERSON_CLASS_ID],
            "conf": self.config.confidence,
            "verbose": False,
            "device": self._device,
            "half": self._half,
        }

    def _next_latest_batch(self) -> tuple[list[str], list[np.ndarray], list[int]] | None:
        ips: list[str] = []
        frames: list[np.ndarray] = []
        versions: list[int] = []
        with self._lock:
            for ip in self.ips:
                version = self._frame_versions.get(ip, 0)
                if version <= 0 or self._processed_versions.get(ip) == version:
                    continue
                frame = self._frames.get(ip)
                if frame is None:
                    continue
                self._processed_versions[ip] = version
                ips.append(ip)
                frames.append(frame)
                versions.append(version)
        if not frames:
            return None
        return ips, frames, versions

    def _process_batch(self, ips: list[str], frames: list[np.ndarray], versions: list[int]) -> None:
        if self._model is None:
            return
        try:
            started = time.perf_counter()
            self._last_batch_started_at = started
            with _ultralytics_windows_os_release_guard():
                results = self._model.predict(frames, **self._predict_kwargs())
            self._synchronize_cuda()
            inference_ms = (time.perf_counter() - started) * 1000.0
            detections_by_ip = {
                ip: self._parse_result(ip, frame, result)
                for ip, frame, result in zip(ips, frames, results)
            }
            self._record_batch(ips, versions, detections_by_ip, inference_ms)
        except Exception as error:
            self._set_error(f"detector error: {error}")
            time.sleep(0.25)

    def _parse_result(self, ip: str, frame: np.ndarray, result) -> list[PersonDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        height, width = frame.shape[:2]
        now = time.monotonic()
        xyxy_values = boxes.xyxy.cpu().numpy()
        conf_values = boxes.conf.cpu().numpy()
        detections: list[PersonDetection] = []
        for xyxy, confidence in zip(xyxy_values, conf_values):
            coords = np.asarray(xyxy, dtype=np.float32).reshape(-1)
            confidence_value = float(confidence)
            if coords.shape[0] != 4 or not np.all(np.isfinite(coords)) or not np.isfinite(confidence_value):
                continue
            x1, y1, x2, y2 = coords.tolist()
            x1 = float(np.clip(x1, 0, width - 1))
            x2 = float(np.clip(x2, 0, width - 1))
            y1 = float(np.clip(y1, 0, height - 1))
            y2 = float(np.clip(y2, 0, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                PersonDetection(
                    ip=ip,
                    xyxy=(x1, y1, x2, y2),
                    confidence=confidence_value,
                    frame_shape=(height, width),
                    detected_at_monotonic=now,
                )
            )
        return detections

    def _record_batch(
        self,
        ips: list[str],
        versions: list[int],
        detections_by_ip: dict[str, list[PersonDetection]],
        inference_ms: float,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            for ip, version in zip(ips, versions):
                self._processed_versions[ip] = max(version, self._processed_versions.get(ip, 0))
                detections = detections_by_ip.get(ip, [])
                if detections:
                    self._detections[ip] = detections
                else:
                    self._detections[ip] = [
                        detection
                        for detection in self._detections.get(ip, [])
                        if now - detection.detected_at_monotonic <= DETECTION_RETENTION_SECONDS
                    ]
            self._total_inferences += len(ips)
            self._total_batches += 1
            self._last_batch_size = len(ips)
            self._total_detections += sum(len(value) for value in detections_by_ip.values())
            self._last_inference_ms = inference_ms
            self._average_inference_ms = (
                inference_ms
                if self._average_inference_ms is None
                else 0.9 * self._average_inference_ms + 0.1 * inference_ms
            )
            self._status = f"running:{self._device}"
            self._last_error = ""

    def _wait_for_inference_slot(self) -> bool:
        max_batches = self.config.max_batches_per_second
        if max_batches <= 0:
            return True
        min_interval = 1.0 / max_batches
        elapsed = time.perf_counter() - self._last_batch_started_at
        remaining = min_interval - elapsed
        if remaining <= 0:
            return True
        self._stop_event.wait(min(remaining, 0.05))
        return False

    @staticmethod
    def _synchronize_cuda() -> None:
        try:
            import torch
            torch.cuda.synchronize(0)
        except Exception:
            pass

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._status = "error"
            self._last_error = message
