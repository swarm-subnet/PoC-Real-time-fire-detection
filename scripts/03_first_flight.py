"""Minimal first-flight script with conservative safety behavior."""

import time

from djitellopy import Tello


MIN_BATTERY_PERCENT = 30
# Set this to False if the room is too small for the 20 cm climb test.
ALLOW_MOVE_UP_TEST = True


def main() -> None:
    tello = Tello()
    has_taken_off = False

    try:
        print("Connecting to drone...")
        tello.connect()

        battery = tello.get_battery()
        print(f"Battery: {battery}%")

        if battery < MIN_BATTERY_PERCENT:
            print(f"Battery too low for flight. Charge to at least {MIN_BATTERY_PERCENT}%.")
            return

        print("Taking off...")
        tello.takeoff()
        has_taken_off = True

        print("Hovering...")
        time.sleep(2)

        if ALLOW_MOVE_UP_TEST:
            print("Moving up 20 cm...")
            tello.move_up(20)
        else:
            print("Skipping upward movement test because ALLOW_MOVE_UP_TEST is False.")

        print("Rotating clockwise 30 degrees...")
        tello.rotate_clockwise(30)

        print("Landing...")
        tello.land()
        has_taken_off = False

        print("Done.")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Error: {error}")
    finally:
        if has_taken_off:
            print("Attempting emergency safe landing...")
            try:
                tello.land()
            except Exception as landing_error:
                print(f"Landing failed: {landing_error}")

        try:
            tello.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
