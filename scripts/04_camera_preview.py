"""Live camera preview window for RoboMaster TT / Tello Talent."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tello_utils import connect_tello


WINDOW_NAME = "Tello Camera Preview"


def main() -> None:
    tello = None
    frame_read = None

    try:
        tello = connect_tello()
        battery = tello.get_battery()
        print(f"Battery: {battery}%")

        print("Starting video stream...")
        try:
            tello.streamoff()
        except Exception:
            pass

        tello.streamon()
        frame_read = tello.get_frame_read()

        print("Video window open. Press 'q' to quit.")

        while True:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.05)
                continue

            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Camera preview failed: {error}")
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
