"""Bench-only motor spin test for two Tello drones on the same Wi-Fi network."""

from __future__ import annotations

import time

from djitellopy import TelloSwarm


# Replace these with the IP addresses assigned by your router/hotspot.
DRONE_IPS = [
    "192.168.1.101",
    "192.168.1.102",
]

SPIN_SECONDS = 3
MIN_BATTERY_PERCENT = 20


def main() -> None:
    swarm = None
    motors_on = False

    try:
        print("Creating swarm...")
        swarm = TelloSwarm.fromIps(DRONE_IPS)

        print("Connecting to drones...")
        swarm.parallel(lambda i, tello: tello.connect(wait_for_state=False))

        print("Checking batteries...")
        batteries: list[int] = []
        for i, tello in enumerate(swarm):
            battery = tello.query_battery()
            batteries.append(battery)
            print(f"Drone {i} ({DRONE_IPS[i]}): {battery}%")

        low_batteries = [
            f"drone {i} ({battery}%)"
            for i, battery in enumerate(batteries)
            if battery < MIN_BATTERY_PERCENT
        ]
        if low_batteries:
            print(f"Battery too low for motor test: {', '.join(low_batteries)}")
            print(f"Charge every drone to at least {MIN_BATTERY_PERCENT}%.")
            return

        print("WARNING: The propellers on both drones will spin without taking off.")
        print("Keep hands, hair, clothing, cables, and loose objects away from both drones.")
        print("Place both drones on flat surfaces with clear space around them.")
        print(f"Spinning motors for {SPIN_SECONDS} seconds...")

        swarm.parallel(lambda i, tello: tello.turn_motor_on())
        motors_on = True
        time.sleep(SPIN_SECONDS)

        print("Stopping motors...")
        swarm.parallel(lambda i, tello: tello.turn_motor_off())
        motors_on = False

        print("Swarm motor spin test completed.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Swarm motor spin test failed: {error}")
    finally:
        if swarm is not None:
            if motors_on:
                print("Attempting to stop all motors...")
                try:
                    swarm.parallel(lambda i, tello: tello.turn_motor_off())
                except Exception as stop_error:
                    print(f"Could not stop every motor cleanly: {stop_error}")

            for tello in swarm:
                try:
                    tello.end()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
