"""Small YOLO helpers for image and Tello camera tests."""

from __future__ import annotations

import builtins
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from ultralytics import YOLO


DEFAULT_MODEL_NAME = "yolo11n.pt"
PERSON_CLASS_NAME = "person"


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


@contextmanager
def _ultralytics_windows_unc_import_compat():
    """Let Ultralytics' Linux probe fail cleanly on Windows UNC paths."""
    original_open = builtins.open

    def patched_open(file, *args, **kwargs):
        path = os.fspath(file) if isinstance(file, (str, os.PathLike)) else file
        if isinstance(path, str) and path.replace("\\", "/") == "/etc/os-release":
            try:
                return original_open(file, *args, **kwargs)
            except OSError as error:
                raise FileNotFoundError(path) from error

        return original_open(file, *args, **kwargs)

    builtins.open = patched_open
    try:
        yield
    finally:
        builtins.open = original_open


def load_yolo_model(model_name: str = DEFAULT_MODEL_NAME) -> "YOLO":
    """Load a YOLO model. The .pt file is downloaded by Ultralytics if missing."""
    with _ultralytics_windows_unc_import_compat():
        from ultralytics import YOLO

    return YOLO(model_name)


def detect_labels_in_image(
    model: YOLO,
    image_bgr,
    target_labels: set[str] | None = None,
    target_substrings: set[str] | None = None,
    confidence_threshold: float = 0.35,
) -> list[Detection]:
    """Run YOLO on one BGR image and return matching detections."""
    results = model.predict(image_bgr, conf=confidence_threshold, verbose=False)
    detections: list[Detection] = []

    if not results:
        return detections

    result = results[0]
    names = result.names
    normalized_labels = {label.lower() for label in target_labels or set()}
    normalized_substrings = {substring.lower() for substring in target_substrings or set()}

    for box in result.boxes:
        class_id = int(box.cls[0])
        label = names[class_id]
        confidence = float(box.conf[0])
        normalized_label = label.lower()

        if normalized_labels and normalized_label not in normalized_labels:
            continue
        if normalized_substrings and not any(substring in normalized_label for substring in normalized_substrings):
            continue

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                xyxy=(x1, y1, x2, y2),
            )
        )

    return detections


def detect_people_in_image(
    model: YOLO,
    image_bgr,
    confidence_threshold: float = 0.35,
) -> list[Detection]:
    """Run YOLO on one BGR image and return person detections."""
    return detect_labels_in_image(
        model,
        image_bgr,
        target_labels={PERSON_CLASS_NAME},
        confidence_threshold=confidence_threshold,
    )


def detect_fire_in_image(
    model: YOLO,
    image_bgr,
    confidence_threshold: float = 0.35,
) -> list[Detection]:
    """Run YOLO on one BGR image and return fire-like detections."""
    return detect_labels_in_image(
        model,
        image_bgr,
        target_substrings={"fire", "flame"},
        confidence_threshold=confidence_threshold,
    )


def draw_detections(image_bgr, detections: list[Detection]):
    """Draw detections onto a copy of the image."""
    output = image_bgr.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        label = f"{detection.label} {detection.confidence:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def save_annotated_image(input_path: Path, output_path: Path, detections: list[Detection], image_bgr) -> None:
    """Save an annotated image and print a compact detection summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_detections(image_bgr, detections)
    cv2.imwrite(str(output_path), annotated)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Detections: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        print(f"  {index}. confidence={detection.confidence:.2f}, box={detection.xyxy}")
