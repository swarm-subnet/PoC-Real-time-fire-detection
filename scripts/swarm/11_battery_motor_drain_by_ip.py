"""Bench battery drain diagnostic for station-mode Tello / RoboMaster TT drones.

The script keeps drones on the ground, optionally spins the propellers with
``motoron``, and logs timestamped battery readings. It can test one drone, a
subset of drones, or every IP in ``scripts/swarm/drone_ips.txt``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import socket
import time


ROOT_DIR = Path(__file__).resolve().parents[2]
DRONE_IPS_FILE = Path(__file__).with_name("drone_ips.txt")
TELLO_PORT = 8889
COMMAND_TIMEOUT_SECONDS = 5
COMMAND_RETRIES = 3
DEFAULT_DURATION_SECONDS = 180
DEFAULT_SAMPLE_EVERY_SECONDS = 2.0
DEFAULT_MIN_BATTERY_PERCENT = 20
DEFAULT_STOP_BATTERY_PERCENT = 8
DEFAULT_OUTPUT_DIR = ROOT_DIR / "captures" / "swarm_battery_tests"


@dataclass
class DroneRunStats:
    ip: str
    samples: list[tuple[float, int]] = field(default_factory=list)
    latencies_ms: list[int] = field(default_factory=list)
    timeouts: int = 0
    consecutive_timeouts: int = 0
    command_ok: bool = False
    motoron_response: str | None = None
    motoroff_response: str | None = None
    error: str | None = None

    def add_sample(self, elapsed_s: float, battery: int, latency_ms: int | None) -> None:
        self.samples.append((elapsed_s, battery))
        self.consecutive_timeouts = 0
        if latency_ms is not None:
            self.latencies_ms.append(latency_ms)

    def add_timeout(self) -> None:
        self.timeouts += 1
        self.consecutive_timeouts += 1

    @property
    def first_battery(self) -> int | None:
        return self.samples[0][1] if self.samples else None

    @property
    def last_battery(self) -> int | None:
        return self.samples[-1][1] if self.samples else None

    @property
    def drop_percent(self) -> int | None:
        if self.first_battery is None or self.last_battery is None:
            return None
        return self.first_battery - self.last_battery

    @property
    def sample_span_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return max(0.0, self.samples[-1][0] - self.samples[0][0])

    @property
    def drop_per_minute(self) -> float | None:
        if self.drop_percent is None or self.sample_span_s <= 0:
            return None
        return self.drop_percent / (self.sample_span_s / 60.0)

    @property
    def average_latency_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        return sum(self.latencies_ms) / len(self.latencies_ms)


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


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_registered_ips() -> list[str]:
    if not DRONE_IPS_FILE.exists():
        return []

    ips: list[str] = []
    for line in DRONE_IPS_FILE.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            ips.append(clean)
    return ips


def unique_ips(ips: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ip in ips:
        if ip not in seen:
            seen.add(ip)
            out.append(ip)
    return out


def create_sdk_socket() -> socket.socket:
    """Create a UDP socket with a stable local SDK source port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # In station mode, TT/Tello can ignore repeated scripts if each run uses a
    # different ephemeral source port. Binding to 8889 keeps the SDK client stable.
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
    retries: int = COMMAND_RETRIES,
    verbose: bool = True,
) -> tuple[str, int]:
    """Send one SDK command and return (response, latency_ms)."""
    sock.settimeout(timeout)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            drain_socket(sock)
            if verbose:
                print(f"[{timestamp()}] >> {ip}: {command}")
            started = time.perf_counter()
            sock.sendto(command.encode("utf-8"), (ip, TELLO_PORT))
            response, _addr = sock.recvfrom(1024)
            latency_ms = int((time.perf_counter() - started) * 1000)
            decoded = response.decode(errors="replace").strip()
            if verbose:
                print(f"[{timestamp()}] << {ip}: {decoded} ({latency_ms} ms)")
            return decoded, latency_ms
        except TimeoutError as error:
            last_error = error
            print(f"[{timestamp()}] No response for '{command}' on {ip} attempt {attempt}/{retries}.")
            time.sleep(1)

    raise TimeoutError(f"Command '{command}' timed out for {ip} after {retries} attempts") from last_error


def send_sdk_command_to_many(
    sock: socket.socket,
    ips: list[str],
    command: str,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
) -> dict[str, tuple[str, int]]:
    """Send one command to all drones first, then collect responses."""
    drain_socket(sock)
    responses: dict[str, tuple[str, int]] = {}
    pending = set(ips)
    sent_at: dict[str, float] = {}

    for ip in ips:
        print(f"[{timestamp()}] >> {ip}: {command}")
        sent_at[ip] = time.perf_counter()
        sock.sendto(command.encode("utf-8"), (ip, TELLO_PORT))

    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        sock.settimeout(max(0.05, deadline - time.time()))
        try:
            response, addr = sock.recvfrom(1024)
        except TimeoutError:
            break

        source_ip = addr[0]
        decoded = response.decode(errors="replace").strip()
        latency_ms = int((time.perf_counter() - sent_at.get(source_ip, time.perf_counter())) * 1000)
        responses[source_ip] = (decoded, latency_ms)
        pending.discard(source_ip)
        print(f"[{timestamp()}] << {source_ip}: {decoded} ({latency_ms} ms)")

    for ip in sorted(pending):
        print(f"[{timestamp()}] !! {ip}: timeout waiting for '{command}'")

    return responses


def query_battery(sock: socket.socket, ip: str) -> tuple[int, int]:
    response, latency_ms = send_sdk_command(sock, ip, "battery?", timeout=3, retries=1, verbose=False)
    return int(response), latency_ms


def write_row(
    writer: csv.DictWriter,
    ip: str,
    elapsed_s: float,
    phase: str,
    battery: int | None,
    latency_ms: int | None,
    note: str,
) -> None:
    writer.writerow(
        {
            "timestamp": timestamp(),
            "ip": ip,
            "elapsed_s": f"{elapsed_s:.1f}",
            "phase": phase,
            "battery_percent": "" if battery is None else battery,
            "latency_ms": "" if latency_ms is None else latency_ms,
            "note": note,
        }
    )


def print_reading(
    ip: str,
    elapsed_s: float,
    phase: str,
    battery: int | None,
    latency_ms: int | None,
    note: str = "",
) -> None:
    battery_text = "timeout" if battery is None else f"{battery}%"
    latency_text = "" if latency_ms is None else f" response={latency_ms}ms"
    note_text = "" if not note else f" {note}"
    print(f"[{timestamp()}] {ip} t={elapsed_s:6.1f}s phase={phase:<9} battery={battery_text}{latency_text}{note_text}")


def add_sample(
    stats: dict[str, DroneRunStats],
    writer: csv.DictWriter,
    ip: str,
    elapsed_s: float,
    phase: str,
    battery: int,
    latency_ms: int,
    note: str = "",
) -> None:
    stats[ip].add_sample(elapsed_s, battery, latency_ms)
    write_row(writer, ip, elapsed_s, phase, battery, latency_ms, note)
    print_reading(ip, elapsed_s, phase, battery, latency_ms, note)


def add_timeout(
    stats: dict[str, DroneRunStats],
    writer: csv.DictWriter,
    ip: str,
    elapsed_s: float,
    phase: str,
    note: str,
) -> None:
    stats[ip].add_timeout()
    write_row(writer, ip, elapsed_s, phase, None, None, note)
    print_reading(ip, elapsed_s, phase, None, None, note)


def print_summary(stats: dict[str, DroneRunStats]) -> None:
    print("")
    print("Battery drain summary")
    print("---------------------")
    for ip in sorted(stats):
        row = stats[ip]
        first = "n/a" if row.first_battery is None else f"{row.first_battery}%"
        last = "n/a" if row.last_battery is None else f"{row.last_battery}%"
        drop = "n/a" if row.drop_percent is None else f"{row.drop_percent}%"
        rate = "n/a" if row.drop_per_minute is None else f"{row.drop_per_minute:.2f}%/min"
        latency = "n/a" if row.average_latency_ms is None else f"{row.average_latency_ms:.0f}ms"
        status = row.error or "ok"
        print(
            f"{ip}: start={first} end={last} drop={drop} "
            f"rate={rate} samples={len(row.samples)} timeouts={row.timeouts} "
            f"avg_response={latency} status={status}"
        )
    print("")


def write_summary_csv(path: Path, stats: dict[str, DroneRunStats]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "ip",
                "start_battery_percent",
                "end_battery_percent",
                "drop_percent",
                "sample_span_s",
                "drop_per_minute",
                "samples",
                "timeouts",
                "avg_latency_ms",
                "command_ok",
                "motoron_response",
                "motoroff_response",
                "error",
            ],
        )
        writer.writeheader()
        for ip in sorted(stats):
            row = stats[ip]
            writer.writerow(
                {
                    "ip": ip,
                    "start_battery_percent": "" if row.first_battery is None else row.first_battery,
                    "end_battery_percent": "" if row.last_battery is None else row.last_battery,
                    "drop_percent": "" if row.drop_percent is None else row.drop_percent,
                    "sample_span_s": f"{row.sample_span_s:.1f}",
                    "drop_per_minute": "" if row.drop_per_minute is None else f"{row.drop_per_minute:.3f}",
                    "samples": len(row.samples),
                    "timeouts": row.timeouts,
                    "avg_latency_ms": "" if row.average_latency_ms is None else f"{row.average_latency_ms:.1f}",
                    "command_ok": row.command_ok,
                    "motoron_response": row.motoron_response or "",
                    "motoroff_response": row.motoroff_response or "",
                    "error": row.error or "",
                }
            )


def run_test(args: argparse.Namespace) -> None:
    ips = unique_ips(args.ips or load_registered_ips())
    if not ips:
        print(f"No IPs provided and no registered drones found in {DRONE_IPS_FILE}.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = log_timestamp()
    label = "swarm" if len(ips) > 1 else ips[0].replace(".", "-")
    csv_path = args.output_dir / f"{run_id}_{label}_battery_motor.csv"
    summary_path = args.output_dir / f"{run_id}_{label}_battery_summary.csv"

    stats = {ip: DroneRunStats(ip=ip) for ip in ips}
    motors_on_ips: list[str] = []
    start_time = time.perf_counter()
    sock = create_sdk_socket()

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
                    send_sdk_command(sock, ip, "command")
                    stats[ip].command_ok = True
                    battery, latency_ms = query_battery(sock, ip)
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
                motoron_responses = send_sdk_command_to_many(sock, test_ips, "motoron")
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
                        battery, latency_ms = query_battery(sock, ip)
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
                motoroff_responses = send_sdk_command_to_many(sock, motors_on_ips, "motoroff", timeout=3)
                for ip in motors_on_ips:
                    response = motoroff_responses.get(ip)
                    stats[ip].motoroff_response = "timeout" if response is None else response[0]

            for ip in ips:
                if not stats[ip].command_ok:
                    continue
                try:
                    elapsed_s = time.perf_counter() - start_time
                    battery, latency_ms = query_battery(sock, ip)
                    add_sample(stats, writer, ip, elapsed_s, "final", battery, latency_ms)
                except Exception as error:
                    elapsed_s = time.perf_counter() - start_time
                    note = str(error)
                    add_timeout(stats, writer, ip, elapsed_s, "final", note)

            sock.close()
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
