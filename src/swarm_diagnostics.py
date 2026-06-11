"""Shared reporting helpers for swarm battery and hover diagnostics."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from swarm_utils import timestamp


@dataclass
class DroneBatteryStats:
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


@dataclass
class HoverStats:
    ip: str
    initial_battery: int | None = None
    post_takeoff_battery: int | None = None
    final_battery: int | None = None
    command_ok: bool = False
    takeoff_response: str | None = None
    land_response: str | None = None
    error: str | None = None

    @property
    def drop_percent(self) -> int | None:
        if self.initial_battery is None or self.final_battery is None:
            return None
        return self.initial_battery - self.final_battery

    @property
    def hover_drop_percent(self) -> int | None:
        if self.post_takeoff_battery is None or self.final_battery is None:
            return None
        return self.post_takeoff_battery - self.final_battery


def write_battery_row(
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


def print_battery_reading(
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


def add_battery_sample(
    stats: dict[str, DroneBatteryStats],
    writer: csv.DictWriter,
    ip: str,
    elapsed_s: float,
    phase: str,
    battery: int,
    latency_ms: int,
    note: str = "",
) -> None:
    stats[ip].add_sample(elapsed_s, battery, latency_ms)
    write_battery_row(writer, ip, elapsed_s, phase, battery, latency_ms, note)
    print_battery_reading(ip, elapsed_s, phase, battery, latency_ms, note)


def add_battery_timeout(
    stats: dict[str, DroneBatteryStats],
    writer: csv.DictWriter,
    ip: str,
    elapsed_s: float,
    phase: str,
    note: str,
) -> None:
    stats[ip].add_timeout()
    write_battery_row(writer, ip, elapsed_s, phase, None, None, note)
    print_battery_reading(ip, elapsed_s, phase, None, None, note)


def print_battery_summary(stats: dict[str, DroneBatteryStats]) -> None:
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


def write_battery_summary_csv(path: Path, stats: dict[str, DroneBatteryStats]) -> None:
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


def write_hover_event(
    writer: csv.DictWriter,
    ip: str,
    phase: str,
    battery: int | None,
    response: str | None,
    note: str = "",
) -> None:
    writer.writerow(
        {
            "timestamp": timestamp(),
            "ip": ip,
            "phase": phase,
            "battery_percent": "" if battery is None else battery,
            "response": response or "",
            "note": note,
        }
    )


def write_hover_summary_csv(path: Path, stats: dict[str, HoverStats], hover_seconds: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "ip",
                "hover_seconds",
                "initial_battery_percent",
                "post_takeoff_battery_percent",
                "final_battery_percent",
                "total_drop_percent",
                "hover_drop_percent",
                "command_ok",
                "takeoff_response",
                "land_response",
                "error",
            ],
        )
        writer.writeheader()
        for ip in sorted(stats):
            row = stats[ip]
            writer.writerow(
                {
                    "ip": ip,
                    "hover_seconds": f"{hover_seconds:.1f}",
                    "initial_battery_percent": "" if row.initial_battery is None else row.initial_battery,
                    "post_takeoff_battery_percent": "" if row.post_takeoff_battery is None else row.post_takeoff_battery,
                    "final_battery_percent": "" if row.final_battery is None else row.final_battery,
                    "total_drop_percent": "" if row.drop_percent is None else row.drop_percent,
                    "hover_drop_percent": "" if row.hover_drop_percent is None else row.hover_drop_percent,
                    "command_ok": row.command_ok,
                    "takeoff_response": row.takeoff_response or "",
                    "land_response": row.land_response or "",
                    "error": row.error or "",
                }
            )


def print_hover_summary(stats: dict[str, HoverStats]) -> None:
    print("")
    print("Hover battery summary")
    print("---------------------")
    for ip in sorted(stats):
        row = stats[ip]
        initial = "n/a" if row.initial_battery is None else f"{row.initial_battery}%"
        post_takeoff = "n/a" if row.post_takeoff_battery is None else f"{row.post_takeoff_battery}%"
        final = "n/a" if row.final_battery is None else f"{row.final_battery}%"
        total_drop = "n/a" if row.drop_percent is None else f"{row.drop_percent}%"
        hover_drop = "n/a" if row.hover_drop_percent is None else f"{row.hover_drop_percent}%"
        status = row.error or "ok"
        print(
            f"{ip}: initial={initial} post_takeoff={post_takeoff} "
            f"final={final} total_drop={total_drop} hover_drop={hover_drop} status={status}"
        )
    print("")
