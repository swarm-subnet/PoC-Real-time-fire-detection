"""Short hover battery test for a station-mode Tello / RoboMaster TT swarm.

This is a flight test. It sends ``takeoff`` to selected drones, lets them hover
for a fixed duration, sends ``land``, and saves per-drone battery drop stats.
Before takeoff it runs repeated preflight status rounds and aborts if any
selected drone is not reachable or below the configured battery threshold.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
COMMAND_TIMEOUT_SECONDS = 10
COMMAND_RETRIES = 3
DEFAULT_HOVER_SECONDS = 60
DEFAULT_MIN_BATTERY_PERCENT = 50
DEFAULT_PREFLIGHT_ROUNDS = 3
DEFAULT_PREFLIGHT_RETRIES = 5
DEFAULT_PREFLIGHT_PAUSE_SECONDS = 2.0
DEFAULT_OUTPUT_DIR = ROOT_DIR / "captures" / "swarm_hover_tests"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swarm_utils import (  # noqa: E402
    DEFAULT_DRONE_IPS_FILE,
    TelloUdpClient,
    load_registered_ips,
    log_timestamp,
    query_battery as query_swarm_battery,
    unique_ips,
)
from swarm_diagnostics import (  # noqa: E402
    HoverStats,
    print_hover_summary as print_summary,
    write_hover_event as write_event,
    write_hover_summary_csv as write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Take off selected Tellos, hover briefly, land, and log battery drop."
    )
    parser.add_argument(
        "ips",
        nargs="*",
        help="Drone IP address(es). If omitted, uses every IP from scripts/swarm/config/drone_ips.txt.",
    )
    parser.add_argument(
        "--hover-seconds",
        type=float,
        default=DEFAULT_HOVER_SECONDS,
        help=f"Seconds to hover before landing. Default: {DEFAULT_HOVER_SECONDS}.",
    )
    parser.add_argument(
        "--min-battery",
        type=int,
        default=DEFAULT_MIN_BATTERY_PERCENT,
        help=f"Refuse takeoff below this battery percent. Default: {DEFAULT_MIN_BATTERY_PERCENT}.",
    )
    parser.add_argument(
        "--allow-low-battery",
        action="store_true",
        help="Allow takeoff even when a drone is below --min-battery.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen and query batteries, but do not take off.",
    )
    parser.add_argument(
        "--preflight-rounds",
        type=int,
        default=DEFAULT_PREFLIGHT_ROUNDS,
        help=f"Full status rounds before takeoff. Default: {DEFAULT_PREFLIGHT_ROUNDS}.",
    )
    parser.add_argument(
        "--preflight-retries",
        type=int,
        default=DEFAULT_PREFLIGHT_RETRIES,
        help=f"Retries per preflight command. Default: {DEFAULT_PREFLIGHT_RETRIES}.",
    )
    parser.add_argument(
        "--preflight-pause",
        type=float,
        default=DEFAULT_PREFLIGHT_PAUSE_SECONDS,
        help=f"Seconds between preflight rounds. Default: {DEFAULT_PREFLIGHT_PAUSE_SECONDS}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV logs. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def query_battery(client: TelloUdpClient, ip: str, retries: int = COMMAND_RETRIES) -> int:
    battery, _latency_ms = query_swarm_battery(client, ip, timeout=3, retries=retries)
    return battery


def run_preflight(
    client: TelloUdpClient,
    ips: list[str],
    stats: dict[str, HoverStats],
    writer: csv.DictWriter,
    rounds: int,
    retries: int,
    pause_seconds: float,
    min_battery: int,
    allow_low_battery: bool,
) -> bool:
    """Run repeated status checks and return True only if the final round passes."""
    final_round_ok = False

    for round_index in range(1, rounds + 1):
        print(f"Preflight round {round_index}/{rounds}: checking every selected drone.")
        round_failures: list[str] = []

        for ip in ips:
            phase = f"preflight_{round_index}"
            try:
                response = client.send_one(ip, "command", timeout=3, retries=retries)
                stats[ip].command_ok = True
                battery = query_battery(client, ip, retries=retries)
                stats[ip].initial_battery = battery
                stats[ip].error = None
                write_event(writer, ip, phase, battery, response.text)

                if battery < min_battery and not allow_low_battery:
                    note = f"below minimum flight battery {min_battery}%"
                    stats[ip].error = note
                    write_event(writer, ip, phase, battery, response.text, note)
                    round_failures.append(f"{ip} ({battery}% < {min_battery}%)")
            except Exception as error:
                note = str(error)
                stats[ip].error = note
                write_event(writer, ip, phase, None, None, note)
                round_failures.append(f"{ip} ({note})")

        if round_failures:
            final_round_ok = False
            print(f"Preflight round {round_index} failed: {', '.join(round_failures)}")
            if round_index < rounds:
                print(f"Retrying full preflight in {pause_seconds:.1f}s...")
                time.sleep(pause_seconds)
            continue

        final_round_ok = True
        print(f"Preflight round {round_index} passed for all selected drones.")
        if round_index < rounds:
            time.sleep(pause_seconds)

    if not final_round_ok:
        print("Refusing takeoff. Final preflight round did not pass for every selected drone.")

    return final_round_ok


def run_test(args: argparse.Namespace) -> None:
    ips = unique_ips(args.ips or load_registered_ips())
    if not ips:
        print(f"No IPs provided and no registered drones found in {DEFAULT_DRONE_IPS_FILE}.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = log_timestamp()
    label = "swarm" if len(ips) > 1 else ips[0].replace(".", "-")
    events_path = args.output_dir / f"{run_id}_{label}_hover_events.csv"
    summary_path = args.output_dir / f"{run_id}_{label}_hover_summary.csv"
    stats = {ip: HoverStats(ip=ip) for ip in ips}
    airborne_ips: list[str] = []
    client = TelloUdpClient()

    print(f"Testing {len(ips)} drone(s): {', '.join(ips)}")
    print(f"Hover duration: {args.hover_seconds:.0f} seconds")
    print(f"Logging events to: {events_path}")
    print(f"Logging summary to: {summary_path}")

    with events_path.open("w", newline="", encoding="utf-8") as events_file:
        writer = csv.DictWriter(
            events_file,
            fieldnames=["timestamp", "ip", "phase", "battery_percent", "response", "note"],
        )
        writer.writeheader()

        try:
            if not run_preflight(
                client=client,
                ips=ips,
                stats=stats,
                writer=writer,
                rounds=args.preflight_rounds,
                retries=args.preflight_retries,
                pause_seconds=args.preflight_pause,
                min_battery=args.min_battery,
                allow_low_battery=args.allow_low_battery,
            ):
                return

            low = [
                f"{ip} ({stats[ip].initial_battery}%)"
                for ip in ips
                if stats[ip].initial_battery is None or stats[ip].initial_battery < args.min_battery
            ]
            if low and not args.allow_low_battery:
                print(f"Refusing takeoff. Below {args.min_battery}%: {', '.join(low)}")
                print("Charge first, or pass --allow-low-battery if you intentionally want to continue.")
                return

            if args.dry_run:
                print("DRY RUN mode. No takeoff or landing commands will be sent.")
                return

            print("WARNING: this will take off every selected drone.")
            print("Keep drones well spaced and keep the area clear.")

            takeoff_responses = client.send_all(ips, "takeoff", timeout=12)
            airborne_ips = list(ips)
            for ip in ips:
                response = takeoff_responses.get(ip)
                response_text = response.text if response is not None else "timeout"
                stats[ip].takeoff_response = response_text
                write_event(writer, ip, "takeoff", None, response_text, "timeout" if response is None else "")

            time.sleep(5)
            for ip in ips:
                try:
                    battery = query_battery(client, ip)
                    stats[ip].post_takeoff_battery = battery
                    write_event(writer, ip, "post_takeoff", battery, "ok")
                except Exception as error:
                    write_event(writer, ip, "post_takeoff", None, None, str(error))

            print(f"Hovering for {args.hover_seconds:.0f} seconds...")
            time.sleep(args.hover_seconds)

        except KeyboardInterrupt:
            print("Interrupted by user.")
            for ip in ips:
                write_event(writer, ip, "interrupted", None, None, "keyboard interrupt")
        finally:
            if airborne_ips:
                print("Landing drones...")
                land_responses = client.send_all(airborne_ips, "land", timeout=12)
                for ip in airborne_ips:
                    response = land_responses.get(ip)
                    response_text = response.text if response is not None else "timeout"
                    stats[ip].land_response = response_text
                    write_event(writer, ip, "land", None, response_text, "timeout" if response is None else "")
                time.sleep(5)

            for ip in ips:
                if not stats[ip].command_ok:
                    continue
                try:
                    battery = query_battery(client, ip)
                    stats[ip].final_battery = battery
                    write_event(writer, ip, "final", battery, "ok")
                except Exception as error:
                    write_event(writer, ip, "final", None, None, str(error))

            client.close()
            write_summary_csv(summary_path, stats, args.hover_seconds)
            print_summary(stats)
            print(f"Saved event CSV: {events_path}")
            print(f"Saved summary CSV: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.hover_seconds <= 0:
        raise ValueError("--hover-seconds must be greater than 0")
    if args.preflight_rounds <= 0:
        raise ValueError("--preflight-rounds must be greater than 0")
    if args.preflight_retries <= 0:
        raise ValueError("--preflight-retries must be greater than 0")
    if args.preflight_pause < 0:
        raise ValueError("--preflight-pause must be 0 or greater")
    run_test(args)


if __name__ == "__main__":
    main()
