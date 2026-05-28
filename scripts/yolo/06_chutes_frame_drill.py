"""Ask Chutes what command it would output for one saved human-detection frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_FRAME_PATH = ROOT_DIR / "captures" / "yolo_live" / "20260527_135407_192-168-1-132.jpg"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chutes_agent import (
    DEFAULT_CHUTES_MODEL,
    add_control_state,
    get_chutes_api_key,
    load_dotenv_if_present,
    request_tello_command_from_chutes,
)
from yolo_utils import Detection, DEFAULT_MODEL_NAME, detect_people_in_image, load_yolo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Chutes command-generation drill from a saved frame.")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_FRAME_PATH),
        help="Saved image to detect a human in.",
    )
    parser.add_argument("--yolo-model", default=DEFAULT_MODEL_NAME, help="YOLO model file/name.")
    parser.add_argument("--conf", type=float, default=0.80, help="YOLO person confidence threshold.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of consecutive Chutes calls to make.")
    parser.add_argument(
        "--stop-coverage",
        type=float,
        default=40.0,
        help="Coverage threshold for near_enough. Default: 40.",
    )
    parser.add_argument("--chutes-model", default=DEFAULT_CHUTES_MODEL, help="Chutes LLM model ID.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def select_target_person(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None

    return max(
        detections,
        key=lambda detection: (
            (detection.xyxy[2] - detection.xyxy[0]) * (detection.xyxy[3] - detection.xyxy[1]),
            detection.confidence,
        ),
    )


def build_detection_context(
    image_bgr,
    target: Detection | None,
    stop_coverage_percent: float,
    airborne: bool,
    last_command: str,
) -> dict:
    frame_height, frame_width = image_bgr.shape[:2]
    context = {
        "target": "person",
        "frame_width": frame_width,
        "frame_height": frame_height,
        "airborne": airborne,
        "last_command": last_command,
        "stop_coverage_threshold_percent": stop_coverage_percent,
        "person_found": target is not None,
    }

    if target is None:
        return add_control_state(context)

    x1, y1, x2, y2 = target.xyxy
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    box_area = box_width * box_height
    frame_area = frame_width * frame_height
    center_x = x1 + box_width / 2
    center_y = y1 + box_height / 2
    coverage_percent = (box_area / frame_area) * 100 if frame_area else 0

    context.update(
        {
            "person_confidence": round(target.confidence, 3),
            "person_coordinate": {
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
            },
            "person_center": {
                "x": round(center_x, 1),
                "y": round(center_y, 1),
            },
            "center_offset_x_percent": round(((center_x - frame_width / 2) / (frame_width / 2)) * 100, 1),
            "center_offset_y_percent": round(((center_y - frame_height / 2) / (frame_height / 2)) * 100, 1),
            "object_coverage_percentage": round(coverage_percent, 2),
            "near_enough": coverage_percent >= stop_coverage_percent,
        }
    )
    return add_control_state(context)


def main() -> None:
    args = parse_args()
    image_path = resolve_path(args.image)

    load_dotenv_if_present(ROOT_DIR / ".env")
    api_key = get_chutes_api_key()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    yolo_model = load_yolo_model(args.yolo_model)
    detections = detect_people_in_image(yolo_model, image_bgr, confidence_threshold=args.conf)
    target = select_target_person(detections)
    context = build_detection_context(
        image_bgr,
        target,
        stop_coverage_percent=args.stop_coverage,
        airborne=True,
        last_command="",
    )

    print(f"Image: {image_path}")
    print(f"People detected above {args.conf:.2f}: {len(detections)}")
    print("Detection context sent to Chutes:")
    print(json.dumps(context, indent=2))

    repeat = max(1, args.repeat)
    latencies = []

    for index in range(1, repeat + 1):
        decision = request_tello_command_from_chutes(
            context,
            api_key=api_key,
            model=args.chutes_model,
        )
        latencies.append(decision.latency_ms)

        print("")
        print(f"Call {index}/{repeat}")
        print(f"Chutes model: {decision.model}")
        print(f"Chutes latency: {decision.latency_ms:.0f} ms")
        print(f"Chutes raw output: {decision.raw_response!r}")
        print(f"Validated command: {decision.command!r}")

    if len(latencies) > 1:
        print("")
        print("Latency summary:")
        print(f"  min: {min(latencies):.0f} ms")
        print(f"  max: {max(latencies):.0f} ms")
        print(f"  avg: {sum(latencies) / len(latencies):.0f} ms")


if __name__ == "__main__":
    main()
