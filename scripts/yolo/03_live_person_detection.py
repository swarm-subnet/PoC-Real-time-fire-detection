"""Live YOLO person detection on the Tello camera stream."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tello_utils import connect_tello
from yolo_utils import DEFAULT_MODEL_NAME, detect_people_in_image, draw_detections, load_yolo_model


WINDOW_NAME = "Tello YOLO Person Detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live YOLO person detection on Tello video.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="YOLO model file/name.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--every", type=int, default=3, help="Run YOLO every N frames to reduce CPU load.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tello = None
    frame_read = None
    latest_detections = []
    frame_count = 0

    try:
        model = load_yolo_model(args.model)

        tello = connect_tello()
        print(f"Battery: {tello.get_battery()}%")

        try:
            tello.streamoff()
        except Exception:
            pass

        print("Starting video stream...")
        tello.streamon()
        frame_read = tello.get_frame_read()
        print("Press 'q' to quit.")

        while True:
            frame_rgb = frame_read.frame
            if frame_rgb is None:
                time.sleep(0.05)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_count += 1

            if frame_count % max(1, args.every) == 0:
                latest_detections = detect_people_in_image(
                    model,
                    frame_bgr,
                    confidence_threshold=args.conf,
                )
                print(f"People detected: {len(latest_detections)}")

            display_frame = draw_detections(frame_bgr, latest_detections)
            cv2.putText(
                display_frame,
                "Press q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Live detection failed: {error}")
    finally:
        if frame_read is not None:
            try:
                frame_read.stop()
            except Exception:
                pass

        if tello is not None:
            try:
                tello.streamoff()
            except Exception:
                pass
            try:
                tello.end()
            except Exception:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
