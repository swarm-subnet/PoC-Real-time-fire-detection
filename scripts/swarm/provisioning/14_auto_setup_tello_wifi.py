"""Automatically move visible TELLO-* drones onto a target Wi-Fi network.

The script scans local Wi-Fi networks, connects to each Tello AP, sends the
Tello SDK ``ap <ssid> <password>`` command, reconnects to the target Wi-Fi, and
then scans the local subnet for station-mode Tello IPs.

Run this from Windows PowerShell for Wi-Fi switching. WSL generally cannot
control the Windows Wi-Fi adapter directly.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import sys
import time


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_REBOOT_WAIT_SECONDS = 25
DEFAULT_SCAN_SECONDS = 75
DEFAULT_EMPTY_SCAN_RETRIES = 5
DEFAULT_EMPTY_SCAN_SLEEP_SECONDS = 30
DEFAULT_TELLO_AP_SETTLE_SECONDS = 6
DEFAULT_VERIFY_MOTOR_SPIN_SECONDS = 1.0
DEFAULT_SUBNET_DETECT_TIMEOUT_SECONDS = 45
DEFAULT_SUBNET_DETECT_INTERVAL_SECONDS = 2
DEFAULT_MAX_AP_SDK_FAILURES = 3

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_utils import load_dotenv_if_present  # noqa: E402
from swarm_provisioning import (  # noqa: E402
    DEFAULT_TELLO_AP_IP,
    configure_tello_ap,
    guess_local_subnet,
    poll_for_tellos,
    wait_for_reboot,
)
from swarm_utils import (  # noqa: E402
    DEFAULT_DRONE_IPS_FILE,
    TelloUdpClient,
    load_registered_ips,
    save_registered_ips,
    timestamp,
    unique_ips,
)
from wifi_manager import WifiManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find visible TELLO-* Wi-Fi APs and configure each drone to join the target Wi-Fi."
    )
    parser.add_argument(
        "--ssid",
        default=os.getenv("TELLO_TARGET_WIFI_SSID") or os.getenv("TELLO_TARGET_SSID") or "",
        help="Target router/hotspot Wi-Fi SSID. Defaults to TELLO_TARGET_WIFI_SSID or TELLO_TARGET_SSID.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("TELLO_TARGET_WIFI_PASSWORD") or os.getenv("TELLO_TARGET_PASSWORD") or "",
        help="Target Wi-Fi password. Defaults to TELLO_TARGET_WIFI_PASSWORD or TELLO_TARGET_PASSWORD.",
    )
    parser.add_argument("--tello-prefix", default="TELLO-", help="SSID prefix used to identify drone APs.")
    parser.add_argument("--host", default=DEFAULT_TELLO_AP_IP, help="Tello AP-mode SDK IP.")
    parser.add_argument("--subnet", default="", help="Target Wi-Fi subnet to scan, for example 192.168.1.0/24.")
    parser.add_argument("--reboot-wait", type=int, default=DEFAULT_REBOOT_WAIT_SECONDS, help="Seconds to wait after ap command.")
    parser.add_argument("--scan-seconds", type=int, default=DEFAULT_SCAN_SECONDS, help="Seconds to scan after each drone setup.")
    parser.add_argument("--scan-timeout", type=float, default=0.35, help="Per-IP UDP scan timeout.")
    parser.add_argument("--connect-timeout", type=float, default=45, help="Seconds to wait for Wi-Fi connects.")
    parser.add_argument(
        "--subnet-detect-timeout",
        type=int,
        default=DEFAULT_SUBNET_DETECT_TIMEOUT_SECONDS,
        help="Seconds to wait for Windows to expose the target Wi-Fi IPv4 subnet after reconnecting.",
    )
    parser.add_argument(
        "--subnet-detect-interval",
        type=int,
        default=DEFAULT_SUBNET_DETECT_INTERVAL_SECONDS,
        help="Seconds between target Wi-Fi subnet detection attempts.",
    )
    parser.add_argument(
        "--tello-ap-settle",
        type=int,
        default=DEFAULT_TELLO_AP_SETTLE_SECONDS,
        help="Seconds to wait after joining a Tello AP before probing SDK at 192.168.10.1.",
    )
    parser.add_argument(
        "--max-ap-sdk-failures",
        type=int,
        default=DEFAULT_MAX_AP_SDK_FAILURES,
        help="Skip a visible Tello AP after this many SDK probe failures in one run.",
    )
    parser.add_argument(
        "--empty-scan-retries",
        type=int,
        default=DEFAULT_EMPTY_SCAN_RETRIES,
        help="Exit after this many consecutive scans with no unprocessed Tello AP visible.",
    )
    parser.add_argument(
        "--empty-scan-sleep",
        type=int,
        default=DEFAULT_EMPTY_SCAN_SLEEP_SECONDS,
        help="Seconds to wait between scans when no unprocessed Tello AP is visible.",
    )
    parser.add_argument("--max-drones", type=int, default=0, help="Maximum drones to configure. 0 means no explicit limit.")
    parser.add_argument(
        "--scan-existing-first",
        action="store_true",
        help="Before waiting for TELLO-* APs, scan the target Wi-Fi subnet and register already-configured Tellos.",
    )
    parser.add_argument(
        "--verify-motor-spin-seconds",
        type=float,
        default=DEFAULT_VERIFY_MOTOR_SPIN_SECONDS,
        help="Spin propellers for this many seconds after each newly found station-mode drone. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-verify-motor-spin",
        action="store_true",
        help="Disable the post-registration propeller spin check.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan and print planned actions without sending ap commands.")
    return parser.parse_args()


def find_tello_ssids(wifi: WifiManager, prefix: str) -> list[str]:
    networks = wifi.scan()
    return sorted({network.ssid for network in networks if network.ssid.startswith(prefix)})


def _reconnect_to_target_wifi(
    wifi: WifiManager,
    args: argparse.Namespace,
) -> bool:
    try:
        print(f"[{timestamp()}] Reconnecting laptop Wi-Fi to target SSID: {args.ssid}")
        wifi.connect_wpa(args.ssid, args.password, timeout_seconds=args.connect_timeout)
        _print_wifi_state(wifi, "After target Wi-Fi reconnect")
        return True
    except Exception as error:
        print(f"[{timestamp()}] Could not reconnect to target Wi-Fi: {error}")
        print("Reconnect manually, then rerun the script.")
        return False


def _scan_subnet_for_target_wifi(wifi: WifiManager, args: argparse.Namespace) -> str:
    if args.subnet:
        return args.subnet

    deadline = time.time() + args.subnet_detect_timeout
    attempt = 0
    while time.time() <= deadline:
        attempt += 1
        detected = wifi.current_ipv4_network(ssid=args.ssid)
        if detected:
            print(f"[{timestamp()}] Detected active Wi-Fi subnet: {detected}")
            return detected

        remaining = max(0, int(deadline - time.time()))
        print(
            f"[{timestamp()}] Waiting for {args.ssid} IPv4 subnet "
            f"({attempt}, {remaining}s remaining)..."
        )
        time.sleep(args.subnet_detect_interval)

    fallback = guess_local_subnet()
    print(
        f"[{timestamp()}] Could not detect active Wi-Fi subnet after "
        f"{args.subnet_detect_timeout}s; falling back to {fallback}. "
        "If this is wrong, rerun with --subnet."
    )
    return fallback


def _print_wifi_state(wifi: WifiManager, label: str) -> None:
    try:
        current_ssid = wifi.current_ssid()
    except Exception as error:
        print(f"[{timestamp()}] {label}: could not read current Wi-Fi SSID: {error}")
        return

    current_network = None
    if current_ssid:
        try:
            current_network = wifi.current_ipv4_network(ssid=current_ssid)
        except Exception:
            current_network = None

    print(
        f"[{timestamp()}] {label}: Windows reports Wi-Fi SSID="
        f"{current_ssid or 'unknown'}, IPv4 subnet={current_network or 'unknown'}"
    )


def _expected_tello_ap_subnet(host: str) -> str:
    return str(ipaddress.ip_network(f"{host}/24", strict=False))


def _ensure_tello_ap_ipv4(
    wifi: WifiManager,
    args: argparse.Namespace,
    tello_ssid: str,
) -> bool:
    expected_subnet = _expected_tello_ap_subnet(args.host)
    current_subnet = wifi.current_ipv4_network(ssid=tello_ssid)
    if current_subnet == expected_subnet:
        return False

    print(
        f"[{timestamp()}] Tello AP is connected but Windows has no usable "
        f"{expected_subnet} IPv4 route."
    )
    print(f"[{timestamp()}] This usually means Windows assigned a 169.254.x.x fallback IP.")
    raise RuntimeError(
        f"Connected to {tello_ssid}, but Windows did not get a {expected_subnet} IPv4 route. "
        "This script will not modify adapter IP settings. Repair Windows Wi-Fi/DHCP first."
    )


def _save_found_ips(known_ips: list[str], found_station_ips: dict[str, int | None]) -> None:
    if not found_station_ips:
        return

    # Queue provisioning must be append-safe: a drone that was already saved
    # should never disappear because a later scan only found one new IP.
    current_file_ips = load_registered_ips()
    final_ips = unique_ips(current_file_ips + known_ips + sorted(found_station_ips))
    save_registered_ips(final_ips)


def _verify_motor_spin(ip: str, seconds: float) -> bool:
    if seconds <= 0:
        return True

    print(f"[{timestamp()}] Verifying {ip}: spinning propellers for {seconds:.1f}s.")
    motor_started = False
    with TelloUdpClient() as client:
        try:
            command_response = client.send_text(ip, "command", timeout=5, retries=3)
            if command_response.lower() != "ok":
                print(f"[{timestamp()}] {ip}: SDK command mode returned {command_response!r}; skipping motor check.")
                return False

            motoron_response = client.send_text(ip, "motoron", timeout=5, retries=2)
            if motoron_response.lower() != "ok":
                print(f"[{timestamp()}] {ip}: motoron returned {motoron_response!r}; skipping motor check.")
                return False

            motor_started = True
            time.sleep(seconds)
            return True
        except Exception as error:
            print(f"[{timestamp()}] {ip}: motor check failed: {error}")
            return False
        finally:
            if motor_started:
                try:
                    client.send_text(ip, "motoroff", timeout=5, retries=3)
                except Exception as error:
                    print(f"[{timestamp()}] {ip}: motoroff failed: {error}")


def _verify_new_motor_spins(
    args: argparse.Namespace,
    new_ips: set[str],
    motor_verified_ips: set[str],
) -> None:
    if args.no_verify_motor_spin:
        return

    for ip in sorted(new_ips):
        if ip in motor_verified_ips:
            continue
        if _verify_motor_spin(ip, args.verify_motor_spin_seconds):
            motor_verified_ips.add(ip)


def main() -> None:
    load_dotenv_if_present(ROOT_DIR / ".env")
    args = parse_args()

    if not args.ssid or not args.password:
        print("Missing target Wi-Fi credentials.")
        print("Use --ssid/--password, or set TELLO_TARGET_WIFI_SSID and TELLO_TARGET_WIFI_PASSWORD in .env.")
        return
    if args.max_drones < 0:
        raise ValueError("--max-drones must be 0 or greater")
    if args.empty_scan_retries < 1:
        raise ValueError("--empty-scan-retries must be at least 1")
    if args.empty_scan_sleep < 0:
        raise ValueError("--empty-scan-sleep must be 0 or greater")
    if args.tello_ap_settle < 0:
        raise ValueError("--tello-ap-settle must be 0 or greater")
    if args.max_ap_sdk_failures < 1:
        raise ValueError("--max-ap-sdk-failures must be at least 1")
    if args.verify_motor_spin_seconds < 0:
        raise ValueError("--verify-motor-spin-seconds must be 0 or greater")
    if args.subnet_detect_timeout < 1:
        raise ValueError("--subnet-detect-timeout must be at least 1")
    if args.subnet_detect_interval < 1:
        raise ValueError("--subnet-detect-interval must be at least 1")
    wifi = WifiManager()
    wifi.ensure_supported()

    known_ips = load_registered_ips()
    found_station_ips: dict[str, int | None] = {}
    processed_ssids: set[str] = set()
    skipped_ssids: set[str] = set()
    sdk_failure_counts: dict[str, int] = {}
    motor_verified_ips: set[str] = set()
    empty_scan_count = 0

    print(f"[{timestamp()}] Wi-Fi backend: {wifi.backend}")
    print(f"[{timestamp()}] Target Wi-Fi SSID: {args.ssid}")
    print(f"[{timestamp()}] Existing registered IPs: {known_ips or 'none'}")
    print("[INFO] Router admin credentials are not needed; IP detection uses Tello SDK subnet probing.")

    if args.scan_existing_first:
        if wifi.current_ssid() != args.ssid and not _reconnect_to_target_wifi(wifi, args):
            return
        subnet = _scan_subnet_for_target_wifi(wifi, args)
        existing_found = poll_for_tellos(
            subnet,
            poll_seconds=args.scan_seconds,
            scan_timeout=args.scan_timeout,
            exclude_ips=set(known_ips),
            stop_after_first=False,
        )
        found_station_ips.update(existing_found)
        _save_found_ips(known_ips, found_station_ips)
        _verify_new_motor_spins(args, set(existing_found), motor_verified_ips)
        print(f"[{timestamp()}] Existing station-mode Tellos found: {sorted(existing_found) or 'none'}")

    while True:
        if args.max_drones and len(processed_ssids) >= args.max_drones:
            print(f"[{timestamp()}] Reached --max-drones={args.max_drones}.")
            break

        visible_tellos = [
            ssid
            for ssid in find_tello_ssids(wifi, args.tello_prefix)
            if ssid not in processed_ssids and ssid not in skipped_ssids
        ]
        if not visible_tellos:
            if args.dry_run:
                print(f"[{timestamp()}] No unprocessed {args.tello_prefix}* networks visible.")
                break

            if empty_scan_count >= args.empty_scan_retries:
                print(
                    f"[{timestamp()}] No unprocessed {args.tello_prefix}* networks visible "
                    f"after {args.empty_scan_retries} retry attempt(s). Exiting."
                )
                break

            empty_scan_count += 1
            print(
                f"[{timestamp()}] No unprocessed {args.tello_prefix}* networks visible "
                f"({empty_scan_count}/{args.empty_scan_retries}). "
                f"Retrying in {args.empty_scan_sleep}s..."
            )
            time.sleep(args.empty_scan_sleep)
            continue

        tello_ssid = visible_tellos[0]
        print("")
        print(f"[{timestamp()}] Processing drone Wi-Fi: {tello_ssid}")

        if args.dry_run:
            print(f"[DRY RUN] Would connect to {tello_ssid}, send ap command, reconnect to {args.ssid}, and scan for IP.")
            processed_ssids.add(tello_ssid)
            continue

        connected_to_tello_ap = False
        try:
            print(f"[{timestamp()}] Connecting laptop Wi-Fi to drone AP: {tello_ssid}")
            wifi.connect_open(tello_ssid, timeout_seconds=args.connect_timeout)
            connected_to_tello_ap = True
            _print_wifi_state(wifi, "After Tello AP connect")
            _ensure_tello_ap_ipv4(wifi, args, tello_ssid)
            if args.tello_ap_settle:
                print(f"[{timestamp()}] Waiting {args.tello_ap_settle}s for Tello AP routing to settle...")
                time.sleep(args.tello_ap_settle)

            with TelloUdpClient() as client:
                configure_tello_ap(client, args.ssid, args.password, host=args.host)
                processed_ssids.add(tello_ssid)
                sdk_failure_counts.pop(tello_ssid, None)
                empty_scan_count = 0
        except Exception as error:
            failures = sdk_failure_counts.get(tello_ssid, 0) + 1
            sdk_failure_counts[tello_ssid] = failures
            print(f"[{timestamp()}] Failed while configuring {tello_ssid}: {error}")
            if failures >= args.max_ap_sdk_failures:
                skipped_ssids.add(tello_ssid)
                print(f"[{timestamp()}] Skipping {tello_ssid} for this run after {failures} failure(s).")
            if connected_to_tello_ap and not _reconnect_to_target_wifi(wifi, args):
                break
            time.sleep(5)
            continue

        wait_for_reboot(args.reboot_wait)

        if not _reconnect_to_target_wifi(wifi, args):
            break

        subnet = _scan_subnet_for_target_wifi(wifi, args)
        already_known_ips = set(known_ips) | set(found_station_ips)
        if already_known_ips:
            print(f"[{timestamp()}] Ignoring already-known IPs while scanning: {sorted(already_known_ips)}")
        found = poll_for_tellos(
            subnet,
            poll_seconds=args.scan_seconds,
            scan_timeout=args.scan_timeout,
            exclude_ips=already_known_ips,
            stop_after_first=True,
        )
        if not found:
            print(f"[{timestamp()}] No station-mode Tello found after configuring {tello_ssid}.")
        found_station_ips.update(found)
        _save_found_ips(known_ips, found_station_ips)
        _verify_new_motor_spins(args, set(found), motor_verified_ips)

        print(f"[{timestamp()}] Found station-mode IPs so far: {sorted(found_station_ips) or 'none'}")
        time.sleep(2)

    if args.dry_run:
        print("[DRY RUN] No changes were made.")
        return

    final_ips = sorted(found_station_ips)
    if not final_ips:
        print(f"[{timestamp()}] No station-mode IPs found; not updating {DEFAULT_DRONE_IPS_FILE}.")
        return

    _save_found_ips(known_ips, found_station_ips)
    print(f"[{timestamp()}] Current registered IPs: {load_registered_ips()}")

    print(f"[{timestamp()}] Updated list: {DEFAULT_DRONE_IPS_FILE}")


if __name__ == "__main__":
    main()
