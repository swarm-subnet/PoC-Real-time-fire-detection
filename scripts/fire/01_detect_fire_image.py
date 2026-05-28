"""Download a sample fire image and run the SuperBitDev/fire1 ONNX model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_utils import FIRE_MODEL_REPO_ID, FireOnnxDetector, download_fire_sample, get_fire_model_path, keep_fire_detections
from yolo_utils import detect_labels_in_image, load_yolo_model, save_annotated_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect fire in one image using the SuperBitDev/fire1 model.")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help="Path to an input image. If omitted, downloads a sample fire image.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Local YOLO .pt/.onnx path. If omitted, downloads weights.onnx from SuperBitDev/fire1.",
    )
    parser.add_argument("--repo-id", default=FIRE_MODEL_REPO_ID, help="Hugging Face model repo ID.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for .pt fallback models. ONNX uses --profile thresholds.",
    )
    parser.add_argument(
        "--profile",
        choices=("candle", "miner"),
        default="candle",
        help="ONNX post-processing profile. 'miner' matches the published miner thresholds; 'candle' is more sensitive.",
    )
    parser.add_argument("--include-smoke", action="store_true", help="Also include smoke detections in the output.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated output path. Defaults to captures/fire/fire_<input-name>.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def main() -> None:
    args = parse_args()
    image_path = resolve_path(args.image) if args.image is not None else download_fire_sample()
    model_path = resolve_path(args.model) if args.model is not None else get_fire_model_path(args.repo_id)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    print(f"Loading fire model: {model_path}")
    if model_path.suffix.lower() == ".onnx":
        model = FireOnnxDetector(model_path, profile=args.profile)
        print(f"Model labels: {model.class_names}")
        detections = keep_fire_detections(
            model.predict(image_bgr),
            include_smoke=args.include_smoke,
        )
    else:
        model = load_yolo_model(str(model_path))
        print(f"Model labels: {model.names}")
        target_substrings = {"fire", "flame", "smoke"} if args.include_smoke else {"fire", "flame"}
        detections = detect_labels_in_image(
            model,
            image_bgr,
            target_substrings=target_substrings,
            confidence_threshold=args.conf,
        )
        if not detections and len(model.names) == 1:
            print("No 'fire'/'flame' label matched, but this model has one class. Treating that class as fire.")
            detections = detect_labels_in_image(model, image_bgr, confidence_threshold=args.conf)

    output_path = args.output or ROOT_DIR / "captures" / "fire" / f"fire_{image_path.name}"
    save_annotated_image(image_path, output_path, detections, image_bgr)

    if detections:
        labels = ", ".join(sorted({detection.label for detection in detections}))
        print(f"Detected: {labels}.")
    else:
        print("No fire detected at this threshold/profile.")


if __name__ == "__main__":
    main()
