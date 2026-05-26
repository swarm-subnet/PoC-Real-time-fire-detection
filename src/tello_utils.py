"""Helpers for simple RoboMaster TT / Tello starter scripts."""

from __future__ import annotations

from djitellopy import Tello


def connect_tello(wait_for_state: bool = True) -> Tello:
    """Create a Tello instance and connect to the drone."""
    tello = Tello()
    print("Connecting to drone...")
    tello.connect(wait_for_state=wait_for_state)
    return tello


def print_basic_status(tello: Tello) -> None:
    """Print a small, beginner-friendly status snapshot."""
    battery = tello.get_battery()
    print(f"Battery: {battery}%")

    try:
        state = tello.get_current_state()
    except Exception as error:
        print(f"State packet unavailable: {error}")
        return

    if not state:
        print("No state packet received yet.")
        return

    print("State snapshot:")

    status_fields = [
        ("Height", state.get("h"), "cm"),
        ("TOF Distance", state.get("tof"), "cm"),
        ("Flight Time", state.get("time"), "s"),
        ("Temp Low", state.get("templ"), "C"),
        ("Temp High", state.get("temph"), "C"),
        ("Mission Pad ID", state.get("mid"), ""),
    ]

    for label, value, unit in status_fields:
        if value is None:
            continue

        suffix = f" {unit}" if unit else ""
        print(f"  - {label}: {value}{suffix}")


def safe_land(tello: Tello | None) -> bool:
    """Attempt to land only if the drone appears to be flying."""
    if tello is None:
        return False

    if not getattr(tello, "is_flying", False):
        return False

    print("Attempting emergency safe landing...")

    try:
        tello.land()
        return True
    except Exception as error:
        print(f"Landing failed: {error}")
        return False
