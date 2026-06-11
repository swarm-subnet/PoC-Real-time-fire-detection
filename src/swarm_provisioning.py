"""Provisioning helpers for moving Tellos from AP mode to station mode."""

from __future__ import annotations

import ipaddress
import socket
import time

from swarm_utils import TelloUdpClient, query_battery, timestamp


DEFAULT_TELLO_AP_IP = "192.168.10.1"


def guess_local_subnet(default: str = "192.168.1.0/24") -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
    except OSError:
        return default
    finally:
        sock.close()

    return str(ipaddress.ip_network(f"{local_ip}/24", strict=False))


def wait_for_sdk(
    client: TelloUdpClient,
    ip: str = DEFAULT_TELLO_AP_IP,
    timeout_seconds: float = 30,
    retries_per_probe: int = 1,
) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = client.send_one(ip, "command", timeout=3, retries=retries_per_probe, verbose=False)
            if response.text.lower() == "ok":
                return True
        except Exception:
            time.sleep(1)
    return False


def configure_tello_ap(
    client: TelloUdpClient,
    target_ssid: str,
    target_password: str,
    host: str = DEFAULT_TELLO_AP_IP,
    command_timeout: float = 3,
    command_retries: int = 3,
) -> tuple[int | None, str]:
    print(f"[{timestamp()}] Connecting to Tello AP at {host}...")
    client.send_one(host, "command", timeout=command_timeout, retries=command_retries)

    battery: int | None = None
    try:
        battery, _latency_ms = query_battery(client, host, timeout=command_timeout, retries=2)
        print(f"[{timestamp()}] Battery(query): {battery}%")
    except Exception as error:
        print(f"[{timestamp()}] Could not read battery before AP setup: {error}")

    print(f"[{timestamp()}] >> {host}: ap {target_ssid} <hidden>")
    response = client.send_one(host, f"ap {target_ssid} {target_password}", timeout=15, retries=1, verbose=False)
    print(f"[{timestamp()}] << {host}: {response.text}")
    return battery, response.text


def wait_for_reboot(seconds: int) -> None:
    print(f"[{timestamp()}] Waiting {seconds}s for the drone to reboot and join Wi-Fi...")
    for remaining in range(seconds, 0, -1):
        if remaining % 5 == 0 or remaining <= 3:
            print(f"  {remaining}s remaining")
        time.sleep(1)


def scan_tello_ips(
    subnet: str,
    timeout: float = 0.35,
    exclude_ips: set[str] | None = None,
    stop_after_first: bool = False,
) -> dict[str, int | None]:
    network = ipaddress.ip_network(subnet, strict=False)
    exclude = exclude_ips or set()
    found: dict[str, int | None] = {}
    client = TelloUdpClient()
    try:
        for ip in network.hosts():
            ip_text = str(ip)
            if ip_text in exclude:
                continue
            try:
                response = client.send_one(ip_text, "command", timeout=timeout, retries=1, verbose=False)
                if response.text.lower() != "ok":
                    continue
                battery, _latency_ms = query_battery(client, ip_text, timeout=timeout, retries=1, verbose=False)
                found[ip_text] = battery
                print(f"[{timestamp()}] Found Tello at {ip_text}, battery: {battery}%")
                if stop_after_first:
                    return found
            except Exception:
                continue
    finally:
        client.close()

    return found


def poll_for_tellos(
    subnet: str,
    poll_seconds: float,
    scan_timeout: float,
    exclude_ips: set[str] | None = None,
    stop_after_first: bool = False,
) -> dict[str, int | None]:
    deadline = time.time() + poll_seconds
    last_found: dict[str, int | None] = {}
    while time.time() < deadline:
        print(f"[{timestamp()}] Scanning {subnet}...")
        found = scan_tello_ips(
            subnet,
            timeout=scan_timeout,
            exclude_ips=exclude_ips,
            stop_after_first=stop_after_first,
        )
        if found:
            last_found.update(found)
            return last_found
        print(f"[{timestamp()}] No Tellos found yet.")
        time.sleep(5)
    return last_found
