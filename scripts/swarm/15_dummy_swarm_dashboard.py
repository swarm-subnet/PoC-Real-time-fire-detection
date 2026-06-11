"""Dummy five-drone dashboard for iterating on swarm visualization."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as error:
    missing = error.name or "required package"
    print(f"Missing Python package: {missing}")
    print("Run this dashboard with the project Windows venv:")
    print(r"  .\venv-win\Scripts\python.exe scripts\swarm\15_dummy_swarm_dashboard.py")
    print("Or activate/install dependencies first:")
    print(r"  .\venv-win\Scripts\Activate.ps1")
    print("  python -m pip install -r requirements.txt")
    raise SystemExit(1) from error


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
WINDOW_NAME = "Swarm Dashboard - Dummy Data"
DEFAULT_DUMMY_IPS = [
    "192.168.100.89",
    "192.168.100.90",
    "192.168.100.91",
    "192.168.100.92",
    "192.168.100.93",
]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swarm_dashboard import DashboardViewState, SwarmDashboardRenderer  # noqa: E402
from swarm_state import DroneRuntimeState, build_initial_states  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the swarm dashboard with simulated drone data.")
    parser.add_argument("--width", type=int, default=1480, help="Dashboard window width.")
    parser.add_argument("--height", type=int, default=860, help="Dashboard window height.")
    parser.add_argument("--fps", type=float, default=20.0, help="Dashboard refresh rate.")
    parser.add_argument(
        "--ips",
        nargs="*",
        default=DEFAULT_DUMMY_IPS,
        help="Dummy drone IPs to display. Defaults to five station-mode-like IPs.",
    )
    return parser.parse_args()


class DummySwarmData:
    formations = ("line", "v_shape", "arc")

    def __init__(self, ips: list[str]) -> None:
        self.ips = ips
        self.states = build_initial_states(ips)
        self.started_at = time.monotonic()
        self.airborne = False
        self.formation_index = 0
        self.selected_index = 0
        self._base_colors = [
            (70, 150, 240),
            (72, 212, 132),
            (240, 190, 76),
            (232, 92, 92),
            (164, 132, 238),
        ]

    @property
    def selected_ip(self) -> str:
        return self.ips[self.selected_index % len(self.ips)]

    @property
    def formation_name(self) -> str:
        return self.formations[self.formation_index % len(self.formations)]

    def toggle_airborne(self) -> None:
        self.airborne = not self.airborne

    def next_formation(self) -> None:
        self.formation_index = (self.formation_index + 1) % len(self.formations)

    def next_selected(self) -> None:
        self.selected_index = (self.selected_index + 1) % len(self.ips)

    def snapshot(self) -> tuple[list[DroneRuntimeState], dict[str, np.ndarray], str]:
        elapsed = time.monotonic() - self.started_at
        self._update_states(elapsed)
        frames = {
            ip: self._build_camera_frame(ip, index, elapsed)
            for index, ip in enumerate(self.ips)
        }
        banner = (
            "dummy swarm airborne; all feeds simulated"
            if self.airborne
            else "dummy swarm ready; all feeds simulated"
        )
        return [DroneRuntimeState(**self.states[ip].__dict__) for ip in self.ips], frames, banner

    def _update_states(self, elapsed: float) -> None:
        self._apply_formation()
        for index, ip in enumerate(self.ips):
            state = self.states[ip]
            state.online = True
            state.airborne = self.airborne
            state.battery = int(max(0, min(100, 96 - index * 3 - elapsed * 0.025)))
            state.command_state = "hovering" if self.airborne else "streaming"
            state.last_command = "video"
            state.last_response = f"{28 + index * 4}ms camera"
            state.last_error = ""
            state.last_latency_ms = 28 + index * 4
            state.last_seen_monotonic = time.monotonic()
            state.logical_z_cm = 80.0 if self.airborne else 0.0

    def _apply_formation(self) -> None:
        count = len(self.ips)
        center = (count - 1) / 2.0
        name = self.formation_name
        for index, ip in enumerate(self.ips):
            state = self.states[ip]
            offset = index - center
            if name == "line":
                state.logical_x_cm = offset * 85.0
                state.logical_y_cm = 0.0
            elif name == "v_shape":
                state.logical_x_cm = offset * 70.0
                state.logical_y_cm = -abs(offset) * 45.0
            else:
                angle = math.radians(210 - index * (240 / max(1, count - 1)))
                state.logical_x_cm = math.cos(angle) * 125.0
                state.logical_y_cm = math.sin(angle) * 90.0 + 60.0

    def _build_camera_frame(self, ip: str, index: int, elapsed: float) -> np.ndarray:
        height, width = 240, 360
        color = self._base_colors[index % len(self._base_colors)]
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        horizontal = np.linspace(0.35, 1.0, width, dtype=np.float32)
        vertical = np.linspace(0.25, 1.0, height, dtype=np.float32)[:, None]
        for channel, value in enumerate(color):
            frame[:, :, channel] = np.clip(value * horizontal * vertical, 0, 255).astype(np.uint8)

        phase = int((elapsed * 32 + index * 24) % width)
        for x in range(-width, width * 2, 48):
            x0 = x + phase
            cv2.line(frame, (x0, 0), (x0 - 80, height), (255, 255, 255), 1, cv2.LINE_AA)

        horizon = int(height * (0.48 + 0.04 * math.sin(elapsed * 0.8 + index)))
        cv2.line(frame, (0, horizon), (width, horizon), (245, 245, 245), 2, cv2.LINE_AA)
        cv2.circle(frame, (width // 2, height // 2), 34, (20, 22, 24), 2, cv2.LINE_AA)
        cv2.line(frame, (width // 2 - 52, height // 2), (width // 2 + 52, height // 2), (20, 22, 24), 1)
        cv2.line(frame, (width // 2, height // 2 - 42), (width // 2, height // 2 + 42), (20, 22, 24), 1)

        moving_x = int((width * 0.2) + ((elapsed * 35 + index * 37) % (width * 0.6)))
        moving_y = int(height * 0.68 + 12 * math.sin(elapsed + index))
        cv2.rectangle(frame, (moving_x - 18, moving_y - 24), (moving_x + 18, moving_y + 24), (30, 32, 36), -1)
        cv2.rectangle(frame, (moving_x - 18, moving_y - 24), (moving_x + 18, moving_y + 24), (255, 255, 255), 1)

        label = f"DRONE {index + 1}"
        battery = self.states[ip].battery
        cv2.rectangle(frame, (0, 0), (width, 34), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(frame, ip, (132, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"BAT {battery}%", (270, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 220, 220), 1, cv2.LINE_AA)

        status = "AIRBORNE" if self.airborne else "BENCH"
        cv2.putText(frame, status, (12, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
        return frame


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if not args.ips:
        raise ValueError("At least one dummy IP is required")

    source = DummySwarmData(args.ips)
    renderer = SwarmDashboardRenderer(width=args.width, height=args.height)
    view = DashboardViewState(
        selected_ip=source.selected_ip,
        flight_enabled=False,
        translation_enabled=False,
        formation_name=source.formation_name,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)
    frame_delay_ms = max(1, int(1000 / args.fps))

    try:
        while True:
            states, frames, banner = source.snapshot()
            view.selected_ip = source.selected_ip
            view.formation_name = source.formation_name
            view.banner = banner
            view.camera_frames = frames
            canvas = renderer.render(states, view)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                source.toggle_airborne()
            elif key == ord("f"):
                source.next_formation()
            elif key == ord("c"):
                source.next_selected()
    finally:
        cv2.destroyWindow(WINDOW_NAME)


if __name__ == "__main__":
    main()
