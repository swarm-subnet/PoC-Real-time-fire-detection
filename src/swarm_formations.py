"""Relative swarm choreographies for Tellos without shared localization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormationStep:
    name: str
    commands: dict[str, str]
    settle_seconds: float = 2.0


@dataclass(frozen=True)
class FormationPlan:
    name: str
    description: str
    requires_translation: bool
    steps: list[FormationStep]
    visual_offsets_cm: dict[str, tuple[float, float]]


def build_yaw_wave(ips: list[str], degrees: int = 30) -> FormationPlan:
    """Sequential yaw wave. No position change, so it is the safest visible pattern."""
    steps: list[FormationStep] = []
    for ip in ips:
        steps.append(FormationStep(name=f"{ip} cw {degrees}", commands={ip: f"cw {degrees}"}, settle_seconds=1.2))
        steps.append(FormationStep(name=f"{ip} ccw {degrees}", commands={ip: f"ccw {degrees}"}, settle_seconds=1.2))

    return FormationPlan(
        name="yaw_wave",
        description="Each drone turns clockwise and back, one after another.",
        requires_translation=False,
        steps=steps,
        visual_offsets_cm=_line_offsets(ips, spacing_cm=80.0),
    )


def build_line_spread(ips: list[str], spacing_cm: int = 50) -> FormationPlan:
    """Spread drones left/right relative to their own yaw, then return.

    This assumes drones start already well separated and facing the same direction.
    It is choreography, not localization-corrected formation control.
    """
    spread_commands: dict[str, str] = {}
    return_commands: dict[str, str] = {}
    visual_offsets: dict[str, tuple[float, float]] = {}
    center = (len(ips) - 1) / 2.0

    for index, ip in enumerate(ips):
        offset_steps = index - center
        offset_cm = int(round(offset_steps * spacing_cm))
        visual_offsets[ip] = (float(offset_cm), 0.0)
        if abs(offset_cm) < 20:
            continue
        if offset_cm < 0:
            spread_commands[ip] = f"left {abs(offset_cm)}"
            return_commands[ip] = f"right {abs(offset_cm)}"
        else:
            spread_commands[ip] = f"right {abs(offset_cm)}"
            return_commands[ip] = f"left {abs(offset_cm)}"

    return FormationPlan(
        name="line_spread",
        description="Drones spread left/right relative to their current yaw, hold, then return.",
        requires_translation=True,
        steps=[
            FormationStep(name="spread", commands=spread_commands, settle_seconds=5.0),
            FormationStep(name="return", commands=return_commands, settle_seconds=5.0),
        ],
        visual_offsets_cm=visual_offsets,
    )


def build_forward_wave(ips: list[str], distance_cm: int = 40) -> FormationPlan:
    """Small sequential forward/back wave."""
    steps: list[FormationStep] = []
    for ip in ips:
        steps.append(FormationStep(name=f"{ip} forward {distance_cm}", commands={ip: f"forward {distance_cm}"}, settle_seconds=2.0))
        steps.append(FormationStep(name=f"{ip} back {distance_cm}", commands={ip: f"back {distance_cm}"}, settle_seconds=2.0))

    return FormationPlan(
        name="forward_wave",
        description="Each drone moves forward and back in sequence.",
        requires_translation=True,
        steps=steps,
        visual_offsets_cm=_line_offsets(ips, spacing_cm=80.0),
    )


def build_formation(name: str, ips: list[str]) -> FormationPlan:
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "yaw_wave":
        return build_yaw_wave(ips)
    if normalized == "line_spread":
        return build_line_spread(ips)
    if normalized == "forward_wave":
        return build_forward_wave(ips)
    raise ValueError(f"Unknown formation: {name}")


def available_formations() -> list[str]:
    return ["yaw_wave", "line_spread", "forward_wave"]


def _line_offsets(ips: list[str], spacing_cm: float) -> dict[str, tuple[float, float]]:
    center = (len(ips) - 1) / 2.0
    return {ip: ((index - center) * spacing_cm, 0.0) for index, ip in enumerate(ips)}
