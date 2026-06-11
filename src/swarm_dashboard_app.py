"""Runtime app loop for the Tello swarm bench dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import Callable

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2

from swarm_camera import DEFAULT_LOCAL_VIDEO_PORT, SwarmCameraWall
from swarm_control import SwarmController, console_status
from swarm_dashboard import DashboardViewState, SwarmDashboardRenderer
from swarm_state import DroneRuntimeState
from swarm_telemetry import TelloTelemetry, TelloTelemetryListener


DEFAULT_STATUS_EVERY_SECONDS = 5.0
DEFAULT_THERMAL_WARNING_C = 75
DEFAULT_THERMAL_COOLING_START_C = 78
DEFAULT_THERMAL_COOLING_STOP_C = 75
DEFAULT_THERMAL_COOLING_MIN_SECONDS = 45.0
DEFAULT_THERMAL_VIDEO_STOP_C = 82
WINDOW_NAME = "Swarm Bench Dashboard"


@dataclass(frozen=True)
class SwarmDashboardConfig:
    ips: list[str]
    status_retries: int = 2
    motor_spin_seconds: float = 2.0
    camera_wall_enabled: bool = True
    video_port_start: int = DEFAULT_LOCAL_VIDEO_PORT
    status_every: float = DEFAULT_STATUS_EVERY_SECONDS
    width: int = 1480
    height: int = 860
    thermal_warning_c: int = DEFAULT_THERMAL_WARNING_C
    thermal_cooling_start_c: int = DEFAULT_THERMAL_COOLING_START_C
    thermal_cooling_stop_c: int = DEFAULT_THERMAL_COOLING_STOP_C
    thermal_cooling_min_seconds: float = DEFAULT_THERMAL_COOLING_MIN_SECONDS
    thermal_video_stop_c: int = DEFAULT_THERMAL_VIDEO_STOP_C
    auto_stop_video_on_heat: bool = True


class DashboardApp:
    """OpenCV dashboard app for bench-only swarm checks.

    This app intentionally exposes no flight commands. It can refresh status,
    restart video streams, and run the low-speed `motoron`/`motoroff` bench test.
    """

    def __init__(self, config: SwarmDashboardConfig) -> None:
        self.config = config
        self.controller = SwarmController(config.ips)
        self.camera = SwarmCameraWall(
            self.controller.client,
            config.ips,
            video_port_start=config.video_port_start,
        )
        self.telemetry = TelloTelemetryListener(config.ips, self._handle_telemetry)
        self.renderer = SwarmDashboardRenderer(width=config.width, height=config.height)
        self.view = DashboardViewState(
            selected_ip=config.ips[0],
            banner="starting dashboard",
            thermal_warning_c=config.thermal_warning_c,
            thermal_cooling_start_c=config.thermal_cooling_start_c,
            thermal_cooling_stop_c=config.thermal_cooling_stop_c,
            thermal_video_stop_c=config.thermal_video_stop_c,
        )
        self._view_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._action_thread: threading.Thread | None = None
        self._status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self._thermal_stop_thread: threading.Thread | None = None
        self._thermal_video_stop_started = False
        self._cooling_command_thread: threading.Thread | None = None
        self._cooling_ips: set[str] = set()
        self._cooling_started_at: dict[str, float] = {}
        self._cooling_lock = threading.RLock()
        self._manual_spin_thread: threading.Thread | None = None
        self._manual_spin_deadline = 0.0
        self._manual_spin_retry_requested = False
        self._manual_spin_lock = threading.RLock()

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.config.width, self.config.height)
        cv2.setMouseCallback(WINDOW_NAME, self._handle_mouse)

        self.telemetry.start()
        if self.config.camera_wall_enabled:
            self.camera.start()
        self._status_thread.start()
        self._set_banner("dashboard ready")

        try:
            while not self._stop_event.is_set():
                with self._view_lock:
                    self.view.action_busy = self._action_thread is not None and self._action_thread.is_alive()
                    self.view.manual_spin_active = self._manual_spin_is_active()
                    self.view.camera_enabled = self.camera.running
                    self.view.camera_frames = self.camera.get_frames() if self.config.camera_wall_enabled else None
                    states = self.controller.snapshot()
                    self._maybe_handle_thermal(states)
                    frame = self.renderer.render(states, self.view)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(50) & 0xFF
                if key != 255:
                    self._handle_key(key)
        finally:
            self._stop_event.set()
            self.camera.stop()
            self._wait_for_manual_spin()
            self._wait_for_cooling_command()
            self._stop_cooling_motors()
            self.telemetry.stop()
            self.controller.close()
            cv2.destroyWindow(WINDOW_NAME)

    def _status_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._action_thread and self._action_thread.is_alive():
                    self._stop_event.wait(max(1.0, self.config.status_every))
                    continue
                self.controller.refresh_all(retries=self.config.status_retries, status=self._set_banner)
            except Exception as error:
                self._set_banner(f"status failed: {error}")
            self._stop_event.wait(max(1.0, self.config.status_every))

    def _handle_key(self, key: int) -> None:
        char = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27 or char == "q":
            self._set_banner("quitting dashboard")
            self._stop_event.set()
        elif char == "s":
            self._start_action(
                "status",
                lambda: self.controller.refresh_all(
                    retries=self.config.status_retries,
                    status=self._set_banner,
                ),
            )
        elif char == "m":
            self._request_manual_motor_spin()
        elif char == "c":
            self._cycle_selected_drone()
        elif char == "v":
            self._restart_camera_wall()

    def _handle_telemetry(self, telemetry: TelloTelemetry) -> None:
        self.controller.update_telemetry(
            telemetry.ip,
            battery=telemetry.battery,
            temperature_low_c=telemetry.temperature_low_c,
            temperature_high_c=telemetry.temperature_high_c,
            height_cm=telemetry.height_cm,
            tof_cm=telemetry.tof_cm,
        )

    def _handle_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self.renderer.hit_test_motor_button(x, y):
            self._request_manual_motor_spin()

    def _request_manual_motor_spin(self) -> None:
        with self._manual_spin_lock:
            self._manual_spin_deadline = time.monotonic() + self.config.motor_spin_seconds
            self._manual_spin_retry_requested = True
            if self._manual_spin_thread and self._manual_spin_thread.is_alive():
                self._set_banner(f"motor spin extended {self.config.motor_spin_seconds:.1f}s")
                return
            self._manual_spin_thread = threading.Thread(target=self._manual_motor_spin_loop, daemon=True)
            self._manual_spin_thread.start()

    def _manual_motor_spin_loop(self) -> None:
        active_ips: set[str] = set()
        try:
            while not self._stop_event.is_set():
                self._wait_for_cooling_command()
                with self._manual_spin_lock:
                    retry_requested = self._manual_spin_retry_requested
                    self._manual_spin_retry_requested = False
                    deadline = self._manual_spin_deadline

                if retry_requested or not active_ips:
                    with self._cooling_lock:
                        cooling_ips = set(self._cooling_ips)
                    spin_targets = [ip for ip in self.controller.ips if ip not in cooling_ips]
                    started = []
                    if spin_targets:
                        started = self.controller.motoron_ips(
                            spin_targets,
                            status=self._set_banner,
                            command_state=f"motor spin {self.config.motor_spin_seconds:.1f}s",
                        )
                    active_ips.update(started)
                    if not started and not active_ips and not cooling_ips:
                        self._set_banner("motor spin failed: no drone accepted motoron")
                        return
                    if spin_targets:
                        self._set_banner(f"motors spinning until {self.config.motor_spin_seconds:.1f}s after last press")
                    else:
                        self._set_banner("motors already spinning for thermal cooling")

                while not self._stop_event.is_set():
                    with self._manual_spin_lock:
                        deadline = self._manual_spin_deadline
                        retry_requested = self._manual_spin_retry_requested
                    if retry_requested:
                        break

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._stop_event.wait(min(0.05, remaining))

                if self._stop_event.is_set():
                    break

                with self._manual_spin_lock:
                    deadline = self._manual_spin_deadline
                    retry_requested = self._manual_spin_retry_requested
                if retry_requested or time.monotonic() < deadline:
                    continue

                if active_ips:
                    self.controller.motoroff_ips(sorted(active_ips), status=self._set_banner)
                    active_ips.clear()

                with self._manual_spin_lock:
                    if self._manual_spin_retry_requested or time.monotonic() < self._manual_spin_deadline:
                        continue
                return
        finally:
            if active_ips:
                self.controller.motoroff_ips(sorted(active_ips), status=self._set_banner)
            with self._view_lock:
                self.view.manual_spin_active = False

    def _cycle_selected_drone(self) -> None:
        ips = self.controller.ips
        current = self.view.selected_ip
        next_index = 0
        if current in ips:
            next_index = (ips.index(current) + 1) % len(ips)
        with self._view_lock:
            self.view.selected_ip = ips[next_index]
        self._set_banner(f"selected {ips[next_index]}")

    def _restart_camera_wall(self) -> None:
        if not self.config.camera_wall_enabled:
            self._set_banner("camera wall disabled: restart without --no-camera-wall")
            return
        try:
            self._thermal_video_stop_started = False
            with self._view_lock:
                self.view.camera_auto_stopped = False
            self.camera.restart()
            self._set_banner("camera wall restarted")
        except Exception as error:
            self._set_banner(f"camera wall failed: {error}")

    def _maybe_handle_thermal(self, states: list[DroneRuntimeState]) -> None:
        self._maybe_cool_with_motors(states)
        self._maybe_stop_video_for_heat(states)

    def _maybe_stop_video_for_heat(self, states: list[DroneRuntimeState]) -> None:
        if not self.config.auto_stop_video_on_heat:
            return
        if not self.config.camera_wall_enabled or not self.camera.running:
            return
        if self._thermal_video_stop_started:
            return

        hot = [
            state
            for state in states
            if state.temperature_high_c is not None
            and state.temperature_high_c >= self.config.thermal_video_stop_c
        ]
        if not hot:
            return

        self._thermal_video_stop_started = True
        hot_text = ", ".join(f"{state.short_id}:{state.temperature_high_c}C" for state in hot)
        self._set_banner(f"thermal video stop: {hot_text}")
        self._thermal_stop_thread = threading.Thread(target=self._stop_camera_wall_for_heat, daemon=True)
        self._thermal_stop_thread.start()

    def _maybe_cool_with_motors(self, states: list[DroneRuntimeState]) -> None:
        if self._manual_spin_is_active():
            return
        if self._cooling_command_thread and self._cooling_command_thread.is_alive():
            return

        state_by_ip = {state.ip: state for state in states}
        with self._cooling_lock:
            cooling_now = set(self._cooling_ips)
            to_stop = [
                ip
                for ip in cooling_now
                if self._cooling_should_stop(state_by_ip.get(ip))
            ]
            to_start = [
                state.ip
                for state in states
                if self._cooling_should_start(state)
                and state.ip not in cooling_now
            ]
            if not to_start and not to_stop:
                return

        self._cooling_command_thread = threading.Thread(
            target=self._run_cooling_commands,
            args=(to_start, to_stop),
            daemon=True,
        )
        self._cooling_command_thread.start()

    def _cooling_should_start(self, state: DroneRuntimeState) -> bool:
        return (
            not state.airborne
            and state.online
            and state.temperature_high_c is not None
            and state.temperature_high_c >= self.config.thermal_cooling_start_c
        )

    def _cooling_should_stop(self, state: DroneRuntimeState | None) -> bool:
        if state is None:
            return True
        if state.airborne or not state.online:
            return True
        if state.age_seconds is not None and state.age_seconds > 15:
            return True
        started_at = self._cooling_started_at.get(state.ip)
        if started_at is not None and time.monotonic() - started_at < self.config.thermal_cooling_min_seconds:
            return False
        return (
            state.temperature_high_c is not None
            and state.temperature_high_c <= self.config.thermal_cooling_stop_c
        )

    def _run_cooling_commands(self, to_start: list[str], to_stop: list[str]) -> None:
        stopped: list[str] = []
        started: list[str] = []
        try:
            if to_stop:
                stopped = self.controller.motoroff_ips(to_stop, status=self._set_banner)
            if to_start:
                started = self.controller.motoron_ips(to_start, status=self._set_banner)
        finally:
            now = time.monotonic()
            with self._cooling_lock:
                self._cooling_ips.difference_update(stopped)
                self._cooling_ips.update(started)
                for ip in stopped:
                    self._cooling_started_at.pop(ip, None)
                for ip in started:
                    self._cooling_started_at[ip] = now
                cooling_ips = tuple(sorted(self._cooling_ips))
            with self._view_lock:
                self.view.cooling_ips = cooling_ips

    def _stop_cooling_motors(self) -> None:
        with self._cooling_lock:
            ips = sorted(self._cooling_ips)
            self._cooling_ips.clear()
            self._cooling_started_at.clear()
        if not ips:
            return
        try:
            self.controller.motoroff_ips(ips, status=self._set_banner)
        except Exception as error:
            self._set_banner(f"cooling motoroff failed: {error}")
        with self._view_lock:
            self.view.cooling_ips = ()

    def _wait_for_cooling_command(self) -> None:
        thread = self._cooling_command_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)

    def _manual_spin_is_active(self) -> bool:
        thread = self._manual_spin_thread
        return thread is not None and thread.is_alive()

    def _wait_for_manual_spin(self) -> None:
        thread = self._manual_spin_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)

    def _stop_camera_wall_for_heat(self) -> None:
        self.camera.stop()
        with self._view_lock:
            self.view.camera_auto_stopped = True
        self._set_banner("camera stopped for cooling; press V to restart")

    def _start_action(self, name: str, action: Callable[[], object]) -> None:
        if self._action_thread and self._action_thread.is_alive():
            return

        def run_action() -> None:
            try:
                self._set_banner(f"{name} started")
                action()
            except Exception as error:
                self._set_banner(f"{name} failed: {error}")

        self._action_thread = threading.Thread(target=run_action, daemon=True)
        self._action_thread.start()

    def _set_banner(self, message: str) -> None:
        console_status(message)
        with self._view_lock:
            self.view.banner = message
