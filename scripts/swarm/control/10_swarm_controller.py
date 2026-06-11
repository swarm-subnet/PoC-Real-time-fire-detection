"""Interactive controller for a station-mode RoboMaster TT / Tello swarm."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
import time


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
MIN_FLIGHT_BATTERY_PERCENT = 30
DEFAULT_SPIN_SECONDS = 3

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swarm_utils import (  # noqa: E402
    DEFAULT_DRONE_IPS_FILE,
    TelloUdpClient,
    load_registered_ips,
    query_battery,
)


@dataclass
class DroneStatus:
    ip: str
    ok: bool
    battery: int | None = None
    response: str | None = None
    error: str | None = None


class SwarmFlightController:
    def __init__(self, ips: list[str]) -> None:
        self.ips = ips
        self.client = TelloUdpClient()
        self.has_taken_off = False
        self.motors_on = False

    def close(self) -> None:
        self.client.close()

    def initialize(self) -> None:
        print(f"Loaded {len(self.ips)} drone(s): {', '.join(self.ips)}")
        print("Entering SDK mode on all drones...")
        self.client.send_all(self.ips, "command")
        self.status()

    def status(self) -> list[DroneStatus]:
        print("Reading swarm status...")
        statuses: list[DroneStatus] = []
        for ip in self.ips:
            try:
                self.client.send_one(ip, "command", timeout=3)
                battery, _latency_ms = query_battery(self.client, ip, timeout=3, retries=1)
                statuses.append(DroneStatus(ip=ip, ok=True, battery=battery))
            except Exception as error:
                statuses.append(DroneStatus(ip=ip, ok=False, error=str(error)))

        print_status_table(statuses)
        return statuses

    def battery(self) -> None:
        self.status()

    def spin(self, seconds: int = DEFAULT_SPIN_SECONDS) -> None:
        self._require_reachable()
        print(f"Spinning motors on all drones for {seconds} seconds.")
        self.client.send_all(self.ips, "motoron")
        self.motors_on = True
        time.sleep(seconds)
        self.motoroff()

    def motoron(self) -> None:
        self._require_reachable()
        print("Turning motors on for all drones.")
        self.client.send_all(self.ips, "motoron")
        self.motors_on = True

    def motoroff(self) -> None:
        print("Turning motors off for all drones.")
        self.client.send_all(self.ips, "motoroff", timeout=3)
        self.motors_on = False

    def takeoff(self) -> None:
        self._require_flight_battery()
        print("Taking off all drones.")
        self.client.send_all(self.ips, "takeoff")
        self.has_taken_off = True

    def land(self) -> None:
        print("Landing all drones.")
        self.client.send_all(self.ips, "land")
        self.has_taken_off = False

    def emergency(self) -> None:
        print("Sending emergency stop to all drones.")
        self.client.send_all(self.ips, "emergency", timeout=3)
        self.has_taken_off = False
        self.motors_on = False

    def raw(self, command: str) -> None:
        if not command:
            print("Missing raw command.")
            return
        print(f"Sending raw command to all drones: {command}")
        self.client.send_all(self.ips, command)

    def raw_one(self, ip: str, command: str) -> None:
        if not command:
            print("Missing raw command.")
            return
        if ip not in self.ips:
            print(f"Unknown drone IP: {ip}")
            return
        self.client.send_one(ip, command)

    def safe_shutdown(self) -> None:
        if self.motors_on:
            try:
                self.motoroff()
            except Exception as error:
                print(f"Motor shutdown failed: {error}")

        if self.has_taken_off:
            try:
                self.land()
            except Exception as error:
                print(f"Landing failed: {error}")

    def _require_reachable(self) -> None:
        statuses = self.status()
        offline = [status.ip for status in statuses if not status.ok]
        if offline:
            raise RuntimeError(f"Offline drone(s): {', '.join(offline)}")

    def _require_flight_battery(self) -> None:
        statuses = self.status()
        failures = [
            status.ip
            for status in statuses
            if not status.ok or status.battery is None or status.battery < MIN_FLIGHT_BATTERY_PERCENT
        ]
        if failures:
            raise RuntimeError(
                f"Cannot take off. Offline or low-battery drone(s): {', '.join(failures)}"
            )


def print_status_table(statuses: list[DroneStatus]) -> None:
    print("")
    print("Drone status")
    print("------------")
    for index, status in enumerate(statuses, start=1):
        if status.ok:
            print(f"{index}. {status.ip}  ok  battery={status.battery}%")
        else:
            print(f"{index}. {status.ip}  error  {status.error}")
    print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy interactive station-mode Tello swarm controller.")
    parser.add_argument(
        "ips",
        nargs="*",
        help="Drone IP address(es). If omitted, uses scripts/swarm/config/drone_ips.txt.",
    )
    return parser.parse_args()


def print_help() -> None:
    print(
        """
Commands:
  help                         Show this help
  status                       Enter SDK mode and print battery for each drone
  battery                      Alias for status
  spin [seconds]               Spin all motors briefly without takeoff
  motoron                      Turn motors on for all drones
  motoroff                     Turn motors off for all drones
  takeoff                      Take off all drones, after battery checks
  land                         Land all drones
  emergency                    Stop all motors immediately
  raw <sdk command>            Send raw SDK command to all drones
  one <ip> <sdk command>       Send raw SDK command to one drone
  quit                         Land/stop if needed and exit
"""
    )


def parse_seconds(parts: list[str], default: int) -> int:
    if len(parts) < 2:
        return default
    seconds = int(parts[1])
    if seconds <= 0 or seconds > 30:
        raise ValueError("seconds must be between 1 and 30")
    return seconds


def run_repl(controller: SwarmFlightController) -> None:
    print_help()
    while True:
        try:
            line = input("swarm> ").strip()
        except EOFError:
            print("")
            return

        if not line:
            continue

        try:
            parts = shlex.split(line)
            command = parts[0].lower()

            if command in {"quit", "exit"}:
                return
            if command == "help":
                print_help()
            elif command in {"status", "battery"}:
                controller.status()
            elif command == "spin":
                controller.spin(parse_seconds(parts, DEFAULT_SPIN_SECONDS))
            elif command == "motoron":
                controller.motoron()
            elif command == "motoroff":
                controller.motoroff()
            elif command == "takeoff":
                controller.takeoff()
            elif command == "land":
                controller.land()
            elif command == "emergency":
                controller.emergency()
            elif command == "raw":
                controller.raw(" ".join(parts[1:]))
            elif command == "one":
                if len(parts) < 3:
                    print("Usage: one <ip> <sdk command>")
                else:
                    controller.raw_one(parts[1], " ".join(parts[2:]))
            else:
                print(f"Unknown command: {command}. Type 'help'.")
        except Exception as error:
            print(f"Command failed: {error}")


def main() -> None:
    args = parse_args()
    ips = args.ips or load_registered_ips()
    if not ips:
        print(f"No registered drones found in {DEFAULT_DRONE_IPS_FILE}.")
        print("Run scripts/swarm/provisioning/09_setup_new_drone.py for each drone first.")
        return

    controller = SwarmFlightController(ips)
    try:
        controller.initialize()
        run_repl(controller)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        controller.safe_shutdown()
        controller.close()


if __name__ == "__main__":
    main()
