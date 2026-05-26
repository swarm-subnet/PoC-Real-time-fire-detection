"""Relay Tello UDP state/video packets from Windows into a WSL2 distro.

Run this with native Windows Python while your Linux-side script runs in WSL2.
It listens on the Windows host for Tello UDP packets and forwards them to the
current WSL2 IP address on the same ports.
"""

from __future__ import annotations

import argparse
import select
import socket
import subprocess
import sys
from typing import Iterable


DEFAULT_PORTS = (8890, 11111)


def detect_wsl_ip(distro: str | None) -> str:
    """Return the first IPv4 address reported by the target WSL distro."""
    command = ["wsl.exe"]
    if distro:
        command.extend(["-d", distro])
    command.extend(["hostname", "-I"])

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    candidates = result.stdout.strip().split()

    for candidate in candidates:
        if candidate.count(".") == 3:
            return candidate

    raise RuntimeError(
        f"Could not detect a WSL IPv4 address from output: {result.stdout!r}"
    )


def build_listener(port: int, listen_host: str) -> socket.socket:
    """Create a UDP listener for a specific port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((listen_host, port))
    return sock


def relay_forever(wsl_ip: str, listen_host: str, ports: Iterable[int]) -> None:
    """Forward every packet received on Windows to the WSL guest IP."""
    listeners = [build_listener(port, listen_host) for port in ports]
    forwarders = {listener: socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for listener in listeners}
    packets_seen = {listener: 0 for listener in listeners}

    print(f"Forwarding Tello UDP packets to WSL IP {wsl_ip}")
    for listener in listeners:
        print(f"  Listening on {listen_host}:{listener.getsockname()[1]} -> {wsl_ip}:{listener.getsockname()[1]}")
    print("Press Ctrl+C to stop the relay.")

    try:
        while True:
            readable, _, _ = select.select(listeners, [], [])
            for listener in readable:
                data, source = listener.recvfrom(65535)
                port = listener.getsockname()[1]
                forwarders[listener].sendto(data, (wsl_ip, port))
                packets_seen[listener] += 1

                # Print the first few packets and then every 100th packet per port.
                count = packets_seen[listener]
                if count <= 3 or count % 100 == 0:
                    print(f"[udp:{port}] packet #{count} from {source[0]}:{source[1]} ({len(data)} bytes)")
    finally:
        for listener in listeners:
            listener.close()
            forwarders[listener].close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wsl-ip",
        help="WSL2 guest IPv4 address. If omitted, the script queries wsl.exe hostname -I.",
    )
    parser.add_argument(
        "--distro",
        help="Optional WSL distro name for auto-detection, for example 'Ubuntu'.",
    )
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="Windows host address to bind. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--ports",
        nargs="+",
        type=int,
        default=list(DEFAULT_PORTS),
        help="UDP ports to relay. Default: 8890 11111",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        wsl_ip = args.wsl_ip or detect_wsl_ip(args.distro)
        relay_forever(wsl_ip=wsl_ip, listen_host=args.listen_host, ports=args.ports)
    except KeyboardInterrupt:
        print("\nRelay stopped.")
        return 0
    except Exception as error:
        print(f"Relay failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
