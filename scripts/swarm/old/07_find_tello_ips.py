"""Find Tello drones on the current local network using safe SDK queries."""

from __future__ import annotations

import argparse
import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed


TELLO_PORT = 8889


def guess_local_subnet() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
    finally:
        sock.close()

    network = ipaddress.ip_network(f"{local_ip}/24", strict=False)
    return str(network)


def probe_ip(ip: str, timeout: float) -> tuple[str, str] | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)

    try:
        sock.sendto(b"command", (ip, TELLO_PORT))
        response, _ = sock.recvfrom(1024)
        if response.decode(errors="replace").strip().lower() != "ok":
            return None

        sock.sendto(b"battery?", (ip, TELLO_PORT))
        battery, _ = sock.recvfrom(1024)
        return ip, battery.decode(errors="replace").strip()
    except OSError:
        return None
    finally:
        sock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan a subnet for Tello drones in station mode.")
    parser.add_argument(
        "--subnet",
        default=None,
        help="Subnet to scan, for example 192.168.1.0/24. Defaults to your current /24.",
    )
    parser.add_argument("--timeout", type=float, default=1.0, help="Seconds to wait for each UDP response.")
    parser.add_argument("--workers", type=int, default=64, help="Parallel probe workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subnet = args.subnet or guess_local_subnet()
    network = ipaddress.ip_network(subnet, strict=False)

    print(f"Scanning {network} for Tello drones...")
    print("This sends the safe SDK commands 'command' and 'battery?' to each IP.")

    found: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(probe_ip, str(ip), args.timeout) for ip in network.hosts()]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                found.append(result)
                ip, battery = result
                print(f"Found Tello at {ip}, battery: {battery}%")

    if not found:
        print("No Tello drones found.")
        print("Confirm your laptop is on the same router/hotspot Wi-Fi as the drones.")
        return

    print("Use these IPs in the single-drone or swarm scripts:")
    for ip, _battery in sorted(found):
        print(f'  "{ip}",')


if __name__ == "__main__":
    main()
