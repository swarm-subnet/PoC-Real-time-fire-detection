"""Bench-only propeller spin test without takeoff."""

from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tello_utils import connect_tello, print_query_status


SPIN_SECONDS = 3
MIN_BATTERY_PERCENT = 20


def main() -> None:
    tello = None
    motors_on = False

    try:
        tello = connect_tello(wait_for_state=False)
        print_query_status(tello)

        battery = tello.query_battery()
        if battery < MIN_BATTERY_PERCENT:
            print(f"Battery too low for motor test. Charge to at least {MIN_BATTERY_PERCENT}%.")
            return

        print("WARNING: The propellers will spin without taking off.")
        print("Keep hands, hair, clothing, cables, and loose objects away from the drone.")
        print("Place the drone on a flat surface with clear space around it.")
        print(f"Spinning motors for {SPIN_SECONDS} seconds...")

        tello.turn_motor_on()
        motors_on = True
        time.sleep(SPIN_SECONDS)

        print("Stopping motors...")
        tello.turn_motor_off()
        motors_on = False

        print("Motor spin test completed.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Motor spin test failed: {error}")
    finally:
        if tello is not None:
            if motors_on:
                try:
                    tello.turn_motor_off()
                except Exception:
                    pass

            try:
                tello.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
