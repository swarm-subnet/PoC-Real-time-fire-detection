"""Runtime state models for swarm control and dashboard rendering."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class DroneRuntimeState:
    ip: str
    label: str
    slot_index: int
    online: bool = False
    airborne: bool = False
    battery: int | None = None
    command_state: str = "idle"
    last_command: str = ""
    last_response: str = ""
    last_error: str = ""
    last_latency_ms: int | None = None
    last_seen_monotonic: float | None = None
    temperature_low_c: int | None = None
    temperature_high_c: int | None = None
    height_cm: int | None = None
    tof_cm: int | None = None
    logical_x_cm: float = 0.0
    logical_y_cm: float = 0.0
    logical_z_cm: float = 0.0

    @property
    def short_id(self) -> str:
        return self.ip.rsplit(".", 1)[-1]

    @property
    def age_seconds(self) -> float | None:
        if self.last_seen_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self.last_seen_monotonic)


def build_initial_states(ips: list[str]) -> dict[str, DroneRuntimeState]:
    states: dict[str, DroneRuntimeState] = {}
    for index, ip in enumerate(ips, start=1):
        states[ip] = DroneRuntimeState(ip=ip, label=f"swarm_drone_{ip.rsplit('.', 1)[-1]}", slot_index=index)
    apply_logical_line_layout(states)
    return states


def apply_logical_line_layout(states: dict[str, DroneRuntimeState], spacing_cm: float = 80.0) -> None:
    """Set a logical screen layout only. This is not real localization."""
    ips = sorted(states)
    center = (len(ips) - 1) / 2.0
    for index, ip in enumerate(ips):
        state = states[ip]
        state.logical_x_cm = (index - center) * spacing_cm
        state.logical_y_cm = 0.0
        state.logical_z_cm = 0.0
