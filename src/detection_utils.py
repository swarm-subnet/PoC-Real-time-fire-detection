"""Generic detection drawing helpers used by the fire PoC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


def draw_detections(image_bgr, detections: list[Detection]):
    """Draw detections onto a copy of an OpenCV BGR image."""
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
        print(f"  {index}. label={detection.label}, confidence={detection.confidence:.2f}, box={detection.xyxy}")
