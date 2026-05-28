"""Capture still images from the Tello camera for YOLO testing."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture images from the Tello video stream.")
    parser.add_argument("--count", type=int, default=5, help="Number of images to save.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between captures.")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "captures", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tello = None
    frame_read = None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tello = connect_tello()
        print(f"Battery: {tello.get_battery()}%")

        try:
            tello.streamoff()
        except Exception:
            pass

        print("Starting video stream...")
        tello.streamon()
        frame_read = tello.get_frame_read()
        time.sleep(2)

        for index in range(1, args.count + 1):
            frame_rgb = frame_read.frame
            if frame_rgb is None:
                print("No frame yet, waiting...")
                time.sleep(args.interval)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            output_path = args.output_dir / f"tello_capture_{int(time.time())}_{index:02d}.jpg"
            cv2.imwrite(str(output_path), frame_bgr)
            print(f"Saved {output_path}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Image capture failed: {error}")
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


if __name__ == "__main__":
    main()
