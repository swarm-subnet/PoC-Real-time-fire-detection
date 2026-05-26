"""Cycle the RoboMaster TT / Tello Talent top LED without taking off."""

from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tello_utils import connect_tello


STEP_SECONDS = 1.0
LED_SEQUENCE = [
    ("red", "led 255 0 0"),
    ("green", "led 0 255 0"),
    ("blue", "led 0 0 255"),
    ("off", "led 0 0 0"),
]


def main() -> None:
    tello = None

    try:
        tello = connect_tello(wait_for_state=False)

        print("Cycling the top LED. This script does not take off.")
        for label, expansion_command in LED_SEQUENCE:
            print(f"Setting LED to {label}...")
            tello.send_expansion_command(expansion_command)
            time.sleep(STEP_SECONDS)

        print("LED test completed.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print("LED test failed.")
        print("This command usually requires a RoboMaster TT / Tello Talent expansion board.")
        print(f"Error: {error}")
    finally:
        if tello is not None:
            try:
                tello.send_expansion_command("led 0 0 0")
            except Exception:
                pass

            try:
                tello.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
