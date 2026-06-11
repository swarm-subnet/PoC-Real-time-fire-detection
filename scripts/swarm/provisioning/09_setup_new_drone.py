"""Configure one new TT/Tello drone for Wi-Fi and register its station-mode IP."""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import socket
import sys
import time


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_TELLO_AP_IP = "192.168.10.1"
TELLO_PORT = 8889
COMMAND_TIMEOUT_SECONDS = 7
DRONE_IPS_FILE = ROOT_DIR / "scripts" / "swarm" / "config" / "drone_ips.txt"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_utils import load_dotenv_if_present


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect Windows to one new drone's TELLO-* Wi-Fi first. This script sends the ap command, "
            "waits for reboot, scans the router network, and appends detected IPs to config/drone_ips.txt."
        )
    )
    parser.add_argument(
        "--ssid",
        default=os.getenv("TELLO_TARGET_WIFI_SSID") or os.getenv("TELLO_TARGET_SSID") or "",
        help="Router/hotspot Wi-Fi name. Defaults to TELLO_TARGET_WIFI_SSID or TELLO_TARGET_SSID.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("TELLO_TARGET_WIFI_PASSWORD") or os.getenv("TELLO_TARGET_PASSWORD") or "",
        help="Router/hotspot Wi-Fi password. Defaults to TELLO_TARGET_WIFI_PASSWORD or TELLO_TARGET_PASSWORD.",
    )
    parser.add_argument("--host", default=DEFAULT_TELLO_AP_IP, help="Drone IP while connected to its own Wi-Fi.")
    parser.add_argument("--subnet", default=None, help="Router subnet to scan, for example 192.168.1.0/24.")
    parser.add_argument("--reboot-wait", type=int, default=20, help="Seconds to wait after the ap command.")
    parser.add_argument("--poll-seconds", type=int, default=90, help="Seconds to keep scanning for the drone.")
    parser.add_argument("--scan-timeout", type=float, default=0.35, help="Per-IP UDP timeout during scans.")
    parser.add_argument("--skip-ap", action="store_true", help="Skip sending ap and only scan/register drones.")
    return parser.parse_args()


def create_sdk_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", TELLO_PORT))
    return sock


def drain_socket(sock: socket.socket) -> None:
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
    verbose: bool = True,
) -> str:
    sock.settimeout(timeout)
    drain_socket(sock)
    if verbose:
        print(f">> {ip}: {command}")
    sock.sendto(command.encode("utf-8"), (ip, TELLO_PORT))
    response, _addr = sock.recvfrom(1024)
    decoded = response.decode(errors="replace").strip()
    if verbose:
        print(f"<< {ip}: {decoded}")
    return decoded


def guess_local_subnet() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
    except OSError:
        return "192.168.1.0/24"
    finally:
        sock.close()

    return str(ipaddress.ip_network(f"{local_ip}/24", strict=False))


def probe_ip(sock: socket.socket, ip: str, timeout: float) -> tuple[str, str] | None:
    try:
        response = send_sdk_command(sock, ip, "command", timeout=timeout, verbose=False)
        if response.lower() != "ok":
            return None
        battery = send_sdk_command(sock, ip, "battery?", timeout=timeout, verbose=False)
        return ip, battery
    except OSError:
        return None


def scan_subnet(subnet: str, timeout: float) -> list[tuple[str, str]]:
    network = ipaddress.ip_network(subnet, strict=False)
    found: list[tuple[str, str]] = []

    with create_sdk_socket() as sock:
        for ip in network.hosts():
            result = probe_ip(sock, str(ip), timeout)
            if result is not None:
                found.append(result)
                print(f"Found Tello at {result[0]}, battery: {result[1]}%")

    return found


def load_registered_ips() -> list[str]:
    if not DRONE_IPS_FILE.exists():
        return []

    ips: list[str] = []
    for line in DRONE_IPS_FILE.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            ips.append(clean)
    return ips


def save_registered_ips(ips: list[str]) -> None:
    unique_ips = sorted(set(ips), key=lambda value: tuple(int(part) for part in value.split(".")))
    DRONE_IPS_FILE.write_text("\n".join(unique_ips) + "\n", encoding="utf-8")


def wait_for_reboot(seconds: int) -> None:
    print(f"Waiting {seconds} seconds for the drone to reboot and join Wi-Fi...")
    for remaining in range(seconds, 0, -1):
        if remaining % 5 == 0 or remaining <= 3:
            print(f"  {remaining}s remaining")
        time.sleep(1)


def main() -> None:
    load_dotenv_if_present(ROOT_DIR / ".env")
    args = parse_args()

    try:
        if not args.skip_ap and (not args.ssid or not args.password):
            print("Missing Wi-Fi credentials.")
            print("Use --ssid and --password, or set TELLO_TARGET_WIFI_SSID and TELLO_TARGET_WIFI_PASSWORD.")
            return

        if not args.skip_ap:
            with create_sdk_socket() as sock:
                print(f"Connecting to new drone at {args.host}...")
                send_sdk_command(sock, args.host, "command")
                battery = send_sdk_command(sock, args.host, "battery?")
                print(f"Battery(query): {battery}%")

                print(f"Configuring drone to join Wi-Fi SSID: {args.ssid}")
                response = send_sdk_command(sock, args.host, f"ap {args.ssid} {args.password}", timeout=15)
                print(f"Drone response: {response}")

            wait_for_reboot(args.reboot_wait)

        target_network = args.ssid or "the router/hotspot Wi-Fi"
        print(f"Connect Windows to {target_network} now if it is not already connected.")

        known_ips = load_registered_ips()
        print(f"Currently registered IPs: {known_ips or 'none'}")

        deadline = time.time() + args.poll_seconds
        while time.time() < deadline:
            subnet = args.subnet or guess_local_subnet()
            print(f"Scanning {subnet}...")
            found = scan_subnet(subnet, args.scan_timeout)
            found_ips = [ip for ip, _battery in found]
            new_ips = [ip for ip in found_ips if ip not in known_ips]

            if new_ips:
                save_registered_ips(known_ips + new_ips)
                print(f"Registered new drone IP(s): {', '.join(new_ips)}")
                print(f"Updated list: {DRONE_IPS_FILE}")
                return

            if found_ips:
                print("Found only already-registered drones. Still polling for a new IP...")
            else:
                print("No drones found yet. Confirm Windows is connected to the router Wi-Fi.")

            time.sleep(5)

        print("Timed out without registering a new drone.")
        print("Check the router DHCP list, then rerun with --skip-ap if the drone already joined Wi-Fi.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Drone setup failed: {error}")


if __name__ == "__main__":
    main()
