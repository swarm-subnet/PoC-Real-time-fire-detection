"""Shared helpers for connecting to a Tello video stream."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from djitellopy import Tello


DEFAULT_DIRECT_TELLO_IP = "192.168.10.1"


def resolve_path(root_dir: Path, path_text: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root_dir / path


def load_ips_from_file(path: Path) -> list[str]:
    """Read one drone IP per line, ignoring blank lines and comments."""
    if not path.exists():
        return []

    ips: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            ips.append(clean)
    return ips


def dedupe_preserving_order(values: list[str]) -> list[str]:
    """Return unique values while preserving the first occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_candidate_ips(
    root_dir: Path,
    requested_ips: list[str] | None,
    ip_file: str | Path,
    include_direct_fallback: bool,
    direct_tello_ip: str = DEFAULT_DIRECT_TELLO_IP,
) -> list[str]:
    """Build the ordered drone IP list used by live camera scripts."""
    candidates = list(requested_ips or [])
    if not candidates:
        candidates.extend(load_ips_from_file(resolve_path(root_dir, ip_file)))
    if include_direct_fallback:
        candidates.append(direct_tello_ip)
    return dedupe_preserving_order(candidates)


def stop_streaming_tello(tello: Tello | None, frame_read: Any) -> None:
    """Stop frame reading, stream, and socket resources as best effort."""
    if frame_read is not None:
        try:
            frame_read.stop()
        except Exception:
            pass

    if tello is not None:
        if getattr(tello, "stream_on", False):
            try:
                tello.streamoff()
            except Exception:
                pass
            finally:
                tello.stream_on = False
        try:
            tello.end()
        except Exception:
            pass


def wait_for_first_frame(frame_read: Any, timeout_seconds: float):
    """Wait until djitellopy exposes the first video frame."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        frame = frame_read.frame
        if frame is not None:
            return frame
        time.sleep(0.05)
    return None


def connect_first_streaming_drone(
    candidate_ips: list[str],
    frame_timeout: float,
    usage_label: str = "",
) -> tuple[Tello, Any, str, int | None]:
    """Connect to the first IP that responds and provides a video frame."""
    if not candidate_ips:
        raise RuntimeError("No candidate drone IPs found.")

    print("Trying drone IPs in this order:")
    for ip in candidate_ips:
        print(f"  - {ip}")

    for ip in candidate_ips:
        tello = None
        frame_read = None
        try:
            print(f"\nConnecting to drone at {ip}...")
            tello = Tello(host=ip)
            tello.connect(wait_for_state=False)

            battery: int | None = None
            try:
                battery = int(tello.query_battery())
                print(f"Battery(query) for {ip}: {battery}%")
            except Exception as battery_error:
                print(f"Could not read battery from {ip}: {battery_error}")

            try:
                tello.streamoff()
            except Exception:
                pass

            print(f"Starting video stream from {ip}...")
            tello.streamon()
            frame_read = tello.get_frame_read()
            if wait_for_first_frame(frame_read, frame_timeout) is None:
                raise TimeoutError(f"No video frame received from {ip} after {frame_timeout:.1f}s")

            suffix = f" {usage_label}" if usage_label else ""
            print(f"Using drone {ip}{suffix}.")
            return tello, frame_read, ip, battery
        except Exception as error:
            print(f"Skipping {ip}: {error}")
            stop_streaming_tello(tello, frame_read)

    raise RuntimeError("No candidate drone produced a usable video stream.")
