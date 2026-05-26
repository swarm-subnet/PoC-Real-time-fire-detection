"""Mission pad detection test for RoboMaster TT / Tello Talent."""

from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tello_utils import connect_tello, print_basic_status, safe_land


MIN_BATTERY_PERCENT = 30
TAKEOFF_FOR_TEST = False
TEST_DURATION_SECONDS = 10
POLL_INTERVAL_SECONDS = 0.5


def main() -> None:
    tello = None
    has_taken_off = False

    try:
        tello = connect_tello()
        print_basic_status(tello)

        if TAKEOFF_FOR_TEST:
            battery = tello.get_battery()
            if battery < MIN_BATTERY_PERCENT:
                print(f"Battery too low for flight. Charge to at least {MIN_BATTERY_PERCENT}%.")
                return

        print("Enabling mission pad detection...")
        tello.enable_mission_pads()

        print("Setting mission pad detection direction to downward...")
        tello.set_mission_pad_detection_direction(0)

        if TAKEOFF_FOR_TEST:
            print("Taking off for mission pad test...")
            tello.takeoff()
            has_taken_off = True
            print("Hovering over the mission pad area...")
            time.sleep(2)
        else:
            print("TAKEOFF_FOR_TEST is False, so the drone will stay on the ground.")
            print("If you want an in-air test, change TAKEOFF_FOR_TEST to True.")

        print(f"Reading mission pad detections for {TEST_DURATION_SECONDS} seconds...")

        end_time = time.time() + TEST_DURATION_SECONDS
        while time.time() < end_time:
            pad_id = tello.get_mission_pad_id()

            if pad_id == -1:
                print("No mission pad detected.")
            else:
                print(f"Detected mission pad ID: {pad_id}")

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Mission pad test failed: {error}")
    finally:
        if tello is not None:
            if has_taken_off:
                safe_land(tello)

            try:
                print("Disabling mission pad detection...")
                tello.disable_mission_pads()
            except Exception as error:
                print(f"Could not disable mission pads: {error}")

            try:
                tello.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
