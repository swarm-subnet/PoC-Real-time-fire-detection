"""Bench battery drain diagnostic for station-mode Tello / RoboMaster TT drones.

The script keeps drones on the ground, optionally spins the propellers with
``motoron``, and logs timestamped battery readings. It can test one drone, a
subset of drones, or every IP in ``scripts/swarm/drone_ips.txt``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
COMMAND_TIMEOUT_SECONDS = 5
COMMAND_RETRIES = 3
DEFAULT_DURATION_SECONDS = 180
DEFAULT_SAMPLE_EVERY_SECONDS = 2.0
DEFAULT_MIN_BATTERY_PERCENT = 20
DEFAULT_STOP_BATTERY_PERCENT = 8
DEFAULT_OUTPUT_DIR = ROOT_DIR / "captures" / "swarm_battery_tests"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swarm_utils import (  # noqa: E402
    DEFAULT_DRONE_IPS_FILE,
    TelloUdpClient,
    load_registered_ips,
    log_timestamp,
    query_battery as query_swarm_battery,
    timestamp,
    unique_ips,
)
from swarm_diagnostics import (  # noqa: E402
    DroneBatteryStats,
    add_battery_sample as add_sample,
    add_battery_timeout as add_timeout,
    print_battery_reading as print_reading,
    print_battery_summary as print_summary,
    write_battery_row as write_row,
    write_battery_summary_csv as write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Spin one or more Tello motors on the bench while logging battery "
            "with timestamps and per-drone statistics."
        )
    )
    parser.add_argument(
        "ips",
        nargs="*",
        help=(
            "Drone IP address(es). If omitted, uses every IP from "
            "scripts/swarm/drone_ips.txt."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help=f"Motor-test duration in seconds. Default: {DEFAULT_DURATION_SECONDS}.",
    )
    parser.add_argument(
        "--sample-every",
        type=float,
        default=DEFAULT_SAMPLE_EVERY_SECONDS,
        help=f"Seconds between battery sweeps. Default: {DEFAULT_SAMPLE_EVERY_SECONDS}.",
    )
    parser.add_argument(
        "--min-battery",
        type=int,
        default=DEFAULT_MIN_BATTERY_PERCENT,
        help=f"Skip motor spin below this battery percent. Default: {DEFAULT_MIN_BATTERY_PERCENT}.",
    )
    parser.add_argument(
        "--stop-battery",
        type=int,
        default=DEFAULT_STOP_BATTERY_PERCENT,
        help=f"Stop every motor if any drone reaches this percent. Default: {DEFAULT_STOP_BATTERY_PERCENT}.",
    )
    parser.add_argument(
        "--allow-low-battery",
        action="store_true",
        help="Allow motor spin even when battery is below --min-battery.",
    )
    parser.add_argument(
        "--no-motor",
        action="store_true",
        help="Only query battery over time. Do not run motoron.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV logs. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def send_sdk_command(
    client: TelloUdpClient,
    ip: str,
    command: str,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
    retries: int = COMMAND_RETRIES,
    verbose: bool = True,
) -> tuple[str, int]:
    response = client.send_one(
        ip,
        command,
        timeout=timeout,
        retries=retries,
        retry_pause=1.0,
        verbose=verbose,
    )
    return response.text, response.latency_ms


def send_sdk_command_to_many(
    client: TelloUdpClient,
    ips: list[str],
    command: str,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
) -> dict[str, tuple[str, int]]:
    responses = client.send_all(ips, command, timeout=timeout)
    return {ip: (response.text, response.latency_ms) for ip, response in responses.items()}


def query_battery(client: TelloUdpClient, ip: str) -> tuple[int, int]:
    return query_swarm_battery(client, ip, timeout=3, retries=1, verbose=False)


def run_test(args: argparse.Namespace) -> None:
    ips = unique_ips(args.ips or load_registered_ips())
    if not ips:
        print(f"No IPs provided and no registered drones found in {DEFAULT_DRONE_IPS_FILE}.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = log_timestamp()
    label = "swarm" if len(ips) > 1 else ips[0].replace(".", "-")
    csv_path = args.output_dir / f"{run_id}_{label}_battery_motor.csv"
    summary_path = args.output_dir / f"{run_id}_{label}_battery_summary.csv"

    stats = {ip: DroneBatteryStats(ip=ip) for ip in ips}
    motors_on_ips: list[str] = []
    start_time = time.perf_counter()
    client = TelloUdpClient()

    print(f"Testing {len(ips)} drone(s): {', '.join(ips)}")
    print(f"Motor-test duration: {args.duration:.0f} seconds")
    print(f"Logging battery readings to: {csv_path}")
    print(f"Logging summary to: {summary_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["timestamp", "ip", "elapsed_s", "phase", "battery_percent", "latency_ms", "note"],
        )
        writer.writeheader()

        try:
            reachable_ips: list[str] = []
            for ip in ips:
                try:
                    send_sdk_command(client, ip, "command")
                    stats[ip].command_ok = True
                    battery, latency_ms = query_battery(client, ip)
                    elapsed_s = time.perf_counter() - start_time
                    add_sample(stats, writer, ip, elapsed_s, "initial", battery, latency_ms)
                    reachable_ips.append(ip)
                except Exception as error:
                    elapsed_s = time.perf_counter() - start_time
                    note = str(error)
                    stats[ip].error = note
                    write_row(writer, ip, elapsed_s, "initial", None, None, note)
                    print_reading(ip, elapsed_s, "initial", None, None, note)

            if not reachable_ips:
                print("No drones responded. Nothing to test.")
                return

            test_ips = list(reachable_ips)
            if not args.no_motor and not args.allow_low_battery:
                eligible_ips: list[str] = []
                for ip in test_ips:
                    first_battery = stats[ip].first_battery
                    if first_battery is None or first_battery < args.min_battery:
                        note = f"below min battery {args.min_battery}%; skipping motoron"
                        stats[ip].error = note
                        elapsed_s = time.perf_counter() - start_time
                        write_row(writer, ip, elapsed_s, "skipped", first_battery, None, note)
                        print_reading(ip, elapsed_s, "skipped", first_battery, None, note)
                    else:
                        eligible_ips.append(ip)
                test_ips = eligible_ips

            if not test_ips:
                print("No drones eligible for the selected test.")
                return

            if args.no_motor:
                print("Monitoring battery only. Motors will stay off.")
            else:
                print("WARNING: propellers will spin on every selected drone.")
                print("Keep hands, cables, loose objects, and drones clear of each other.")
                motoron_responses = send_sdk_command_to_many(client, test_ips, "motoron")
                # If a command was sent, keep the IP in cleanup even if its response timed out.
                motors_on_ips = list(test_ips)
                for ip in test_ips:
                    response = motoron_responses.get(ip)
                    if response is None:
                        stats[ip].motoron_response = "timeout"
                    else:
                        stats[ip].motoron_response = response[0]

            phase = "monitor" if args.no_motor else "motoron"
            next_sample_at = time.perf_counter()
            end_at = time.perf_counter() + max(0.0, args.duration)
            stop_reason: str | None = None

            while time.perf_counter() < end_at:
                now = time.perf_counter()
                if now < next_sample_at:
                    time.sleep(min(0.1, next_sample_at - now))
                    continue

                for ip in test_ips:
                    elapsed_s = time.perf_counter() - start_time
                    try:
                        battery, latency_ms = query_battery(client, ip)
                        add_sample(stats, writer, ip, elapsed_s, phase, battery, latency_ms)
                        if battery <= args.stop_battery:
                            stop_reason = f"{ip} reached stop battery {battery}% <= {args.stop_battery}%"
                            break
                    except Exception as error:
                        note = str(error)
                        add_timeout(stats, writer, ip, elapsed_s, phase, note)
                        if stats[ip].consecutive_timeouts >= 2:
                            stop_reason = f"{ip} missed two consecutive battery samples"
                            break

                csv_file.flush()

                if stop_reason:
                    elapsed_s = time.perf_counter() - start_time
                    print(f"[{timestamp()}] Stopping test: {stop_reason}.")
                    for ip in test_ips:
                        write_row(writer, ip, elapsed_s, "stopping", stats[ip].last_battery, None, stop_reason)
                    break

                next_sample_at += max(0.5, args.sample_every)

        except KeyboardInterrupt:
            elapsed_s = time.perf_counter() - start_time
            print("Interrupted by user.")
            for ip in ips:
                write_row(writer, ip, elapsed_s, "interrupted", stats[ip].last_battery, None, "keyboard interrupt")
        finally:
            if motors_on_ips:
                print("Stopping motors...")
                motoroff_responses = send_sdk_command_to_many(client, motors_on_ips, "motoroff", timeout=3)
                for ip in motors_on_ips:
                    response = motoroff_responses.get(ip)
                    stats[ip].motoroff_response = "timeout" if response is None else response[0]

            for ip in ips:
                if not stats[ip].command_ok:
                    continue
                try:
                    elapsed_s = time.perf_counter() - start_time
                    battery, latency_ms = query_battery(client, ip)
                    add_sample(stats, writer, ip, elapsed_s, "final", battery, latency_ms)
                except Exception as error:
                    elapsed_s = time.perf_counter() - start_time
                    note = str(error)
                    add_timeout(stats, writer, ip, elapsed_s, "final", note)

            client.close()
            write_summary_csv(summary_path, stats)
            print_summary(stats)
            print(f"Saved CSV log: {csv_path}")
            print(f"Saved summary CSV: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be greater than 0")
    if args.sample_every <= 0:
        raise ValueError("--sample-every must be greater than 0")
    run_test(args)


if __name__ == "__main__":
    main()
