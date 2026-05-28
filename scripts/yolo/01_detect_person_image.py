"""Run YOLO person detection on one local image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo_utils import DEFAULT_MODEL_NAME, detect_people_in_image, load_yolo_model, save_annotated_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people in one image using YOLO.")
    parser.add_argument("image", type=Path, help="Path to an input image.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="YOLO model file/name.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated output path. Defaults to captures/yolo_<input-name>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    model = load_yolo_model(args.model)
    detections = detect_people_in_image(model, image_bgr, confidence_threshold=args.conf)

    output_path = args.output or ROOT_DIR / "captures" / f"yolo_{args.image.name}"
    save_annotated_image(args.image, output_path, detections, image_bgr)


if __name__ == "__main__":
    main()
