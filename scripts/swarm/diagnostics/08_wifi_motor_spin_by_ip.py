"""Bench-only motor spin test for one Tello already connected to your router."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import time


SPIN_SECONDS = 3
MIN_BATTERY_PERCENT = 20
TELLO_PORT = 8889
COMMAND_TIMEOUT_SECONDS = 7
COMMAND_RETRIES = 3
ROOT_DIR = Path(__file__).resolve().parents[3]
DRONE_IPS_FILE = ROOT_DIR / "scripts" / "swarm" / "config" / "drone_ips.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spin motors on one station-mode Tello by IP.")
    parser.add_argument(
        "ip",
        nargs="?",
        help="Drone IP address on your router/hotspot. If omitted, uses scripts/swarm/config/drone_ips.txt.",
    )
    return parser.parse_args()


def create_sdk_socket() -> socket.socket:
    """Create a UDP socket with a stable local SDK source port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # In station mode, TT/Tello can ignore repeated scripts if each run uses a
    # different ephemeral source port. Binding to 8889 keeps the SDK client stable.
    sock.bind(("", TELLO_PORT))
    return sock


def drain_socket(sock: socket.socket) -> None:
    """Clear delayed UDP responses left from previous commands."""
    previous_timeout = sock.gettimeout()
    sock.settimeout(0.05)
    try:
        while True:
            sock.recvfrom(1024)
    except TimeoutError:
        pass
    finally:
        sock.settimeout(previous_timeout)


def send_sdk_command(
    sock: socket.socket,
    ip: str,
    command: str,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
    retries: int = COMMAND_RETRIES,
) -> str:
    """Send one SDK command and return the raw response."""
    sock.settimeout(timeout)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            drain_socket(sock)
            print(f">> {ip}: {command}")
            sock.sendto(command.encode("utf-8"), (ip, TELLO_PORT))
            response, _addr = sock.recvfrom(1024)
            decoded = response.decode(errors="replace").strip()
            print(f"<< {ip}: {decoded}")
            return decoded
        except TimeoutError as error:
            last_error = error
            print(f"No response for '{command}' on attempt {attempt}/{retries}.")
            time.sleep(1)

    raise TimeoutError(f"Command '{command}' timed out after {retries} attempts") from last_error


def send_sdk_command_to_many(sock: socket.socket, ips: list[str], command: str, timeout: float = COMMAND_TIMEOUT_SECONDS) -> None:
    """Send one command to all drones first, then collect their responses."""
    drain_socket(sock)

    for ip in ips:
        print(f">> {ip}: {command}")
        sock.sendto(command.encode("utf-8"), (ip, TELLO_PORT))

    pending = set(ips)
    deadline = time.time() + timeout

    while pending and time.time() < deadline:
        sock.settimeout(max(0.05, deadline - time.time()))
        try:
            response, addr = sock.recvfrom(1024)
        except TimeoutError:
            break

        source_ip = addr[0]
        decoded = response.decode(errors="replace").strip()
        print(f"<< {source_ip}: {decoded}")
        pending.discard(source_ip)

    if pending:
        raise TimeoutError(f"Command '{command}' timed out for: {', '.join(sorted(pending))}")


def load_registered_ips() -> list[str]:
    if not DRONE_IPS_FILE.exists():
        return []

    ips: list[str] = []
    for line in DRONE_IPS_FILE.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            ips.append(clean)
    return ips


def check_battery(sock: socket.socket, ip: str) -> int:
    print(f"Connecting to drone at {ip}...")
    send_sdk_command(sock, ip, "command")
    battery = int(send_sdk_command(sock, ip, "battery?"))
    print(f"Battery(query) for {ip}: {battery}%")
    return battery


def spin_one_drone(sock: socket.socket, ip: str) -> None:
    motors_on = False

    try:
        battery = check_battery(sock, ip)
        if battery < MIN_BATTERY_PERCENT:
            print(f"Battery too low for motor test. Charge to at least {MIN_BATTERY_PERCENT}%.")
            return

        print("WARNING: The propellers will spin without taking off.")
        print("Keep hands, hair, clothing, cables, and loose objects away from the drone.")
        print(f"Spinning motors for {SPIN_SECONDS} seconds...")

        send_sdk_command(sock, ip, "motoron")
        motors_on = True
        time.sleep(SPIN_SECONDS)

        print("Stopping motors...")
        send_sdk_command(sock, ip, "motoroff")
        motors_on = False

        # Leave the drone responsive on the same SDK session before closing.
        send_sdk_command(sock, ip, "battery?", timeout=3, retries=1)
        print("Wi-Fi motor spin test completed.")
    finally:
        if motors_on:
            try:
                send_sdk_command(sock, ip, "motoroff", timeout=3)
            except Exception as stop_error:
                print(f"Could not stop motors cleanly: {stop_error}")


def spin_registered_drones(sock: socket.socket, ips: list[str]) -> None:
    motors_on_ips: list[str] = []

    try:
        batteries = {ip: check_battery(sock, ip) for ip in ips}
        low_batteries = [
            f"{ip} ({battery}%)"
            for ip, battery in batteries.items()
            if battery < MIN_BATTERY_PERCENT
        ]
        if low_batteries:
            print(f"Battery too low for motor test: {', '.join(low_batteries)}")
            print(f"Charge every drone to at least {MIN_BATTERY_PERCENT}%.")
            return

        print("WARNING: The propellers on every registered drone will spin without taking off.")
        print("Keep hands, hair, clothing, cables, and loose objects away from every drone.")
        print(f"Spinning motors for {SPIN_SECONDS} seconds...")

        send_sdk_command_to_many(sock, ips, "motoron")
        motors_on_ips = list(ips)

        time.sleep(SPIN_SECONDS)

        print("Stopping motors...")
        send_sdk_command_to_many(sock, motors_on_ips, "motoroff")
        motors_on_ips.clear()

        for ip in ips:
            send_sdk_command(sock, ip, "battery?", timeout=3, retries=1)

        print("Registered-drone motor spin test completed.")
    finally:
        for ip in list(motors_on_ips):
            try:
                send_sdk_command(sock, ip, "motoroff", timeout=3)
            except Exception as stop_error:
                print(f"Could not stop motors cleanly for {ip}: {stop_error}")


def main() -> None:
    args = parse_args()
    ips = [args.ip] if args.ip else load_registered_ips()

    if not ips:
        print(f"No drone IP provided and no registered drones found in {DRONE_IPS_FILE}.")
        print("Run scripts/swarm/provisioning/09_setup_new_drone.py first, or pass an IP explicitly.")
        return

    sock = create_sdk_socket()
    try:
        if args.ip is None:
            print(f"No IP argument provided. Running motor test for {len(ips)} registered drone(s).")
            spin_registered_drones(sock, ips)
        else:
            spin_one_drone(sock, ips[0])

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Wi-Fi motor spin test failed: {error}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
