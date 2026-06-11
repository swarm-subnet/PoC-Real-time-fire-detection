"""OpenCV dashboard rendering for Tello swarm state."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from swarm_state import DroneRuntimeState


@dataclass
class DashboardViewState:
    selected_ip: str | None = None
    banner: str = "idle"
    camera_enabled: bool = False
    camera_frames: dict[str, np.ndarray] | None = None
    action_busy: bool = False
    manual_spin_active: bool = False
    camera_auto_stopped: bool = False
    cooling_ips: tuple[str, ...] = ()
    thermal_warning_c: int = 75
    thermal_cooling_start_c: int = 78
    thermal_cooling_stop_c: int = 75
    thermal_video_stop_c: int = 82


class SwarmDashboardRenderer:
    def __init__(self, width: int = 1480, height: int = 860) -> None:
        self.width = width
        self.height = height
        self.bg = (17, 20, 23)
        self.panel = (30, 35, 39)
        self.panel_alt = (38, 44, 48)
        self.text = (238, 244, 240)
        self.muted = (155, 166, 166)
        self.green = (79, 217, 122)
        self.yellow = (238, 190, 78)
        self.red = (231, 88, 88)
        self.blue = (88, 170, 255)
        self.motor_button_rect = (1086, 648, 1426, 698)

    def render(self, states: list[DroneRuntimeState], view: DashboardViewState) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), self.bg, dtype=np.uint8)
        self._draw_header(canvas, states, view)
        self._draw_cards(canvas, states, view, x0=24, y0=88, w=self.width - 48, h=90)
        self._draw_camera_wall(canvas, view, x0=24, y0=198, w=self.width - 48, h=self.height - 290)
        self._draw_compact_footer(canvas, view, x0=24, y0=self.height - 76, w=self.width - 48, h=56)
        return canvas

    def _draw_header(self, img: np.ndarray, states: list[DroneRuntimeState], view: DashboardViewState) -> None:
        online = sum(1 for state in states if state.online)
        cv2.putText(img, "Swarm Bench Dashboard", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.text, 2, cv2.LINE_AA)
        motor = "SPINNING" if view.manual_spin_active else "READY"
        if view.action_busy:
            motor = "BUSY"
        max_temp = max((state.temperature_high_c for state in states if state.temperature_high_c is not None), default=None)
        temp_text = "temp=n/a" if max_temp is None else f"max temp={max_temp}C"
        video_text = "video=OFF(heat)" if view.camera_auto_stopped else ("video=ON" if view.camera_enabled else "video=OFF")
        cooling_text = "cooling=none" if not view.cooling_ips else "cooling=" + ",".join(ip.rsplit(".", 1)[-1] for ip in view.cooling_ips)
        summary = f"{online}/{len(states)} online   {temp_text}   {video_text}   {cooling_text}   motor spin={motor}"
        cv2.putText(img, summary, (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.muted, 1, cv2.LINE_AA)

        banner_color = self.green if "passed" in view.banner or view.banner == "idle" else self.yellow
        if "failed" in view.banner or "blocked" in view.banner or "timeout" in view.banner:
            banner_color = self.red
        self._rounded_rect(img, (820, 20), (1450, 72), self.panel_alt, radius=14)
        cv2.putText(img, view.banner[:68], (842, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_color, 1, cv2.LINE_AA)

    def _draw_cards(
        self,
        img: np.ndarray,
        states: list[DroneRuntimeState],
        view: DashboardViewState,
        x0: int,
        y0: int,
        w: int,
        h: int,
    ) -> None:
        count = max(1, len(states))
        gap = 14
        card_w = int((w - gap * (count - 1)) / count)
        card_h = h
        for idx, state in enumerate(states):
            x = x0 + idx * (card_w + gap)
            y = y0
            selected = state.ip == view.selected_ip
            fill = self.panel_alt if selected else self.panel
            self._rounded_rect(img, (x, y), (x + card_w, y + card_h), fill, radius=16)
            if selected:
                cv2.rectangle(img, (x + 3, y + 3), (x + card_w - 3, y + card_h - 3), self.blue, 1)
            status_color = self.green if state.online else self.red
            if state.online and state.battery is not None and state.battery < 35:
                status_color = self.yellow
            cv2.circle(img, (x + 18, y + 25), 7, status_color, -1)
            cv2.putText(img, state.short_id, (x + 34, y + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.text, 1, cv2.LINE_AA)
            cv2.putText(img, state.ip, (x + 16, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.muted, 1, cv2.LINE_AA)
            battery = "n/a" if state.battery is None else f"{state.battery}%"
            temp = "n/a" if state.temperature_high_c is None else f"{state.temperature_high_c}C"
            temp_color = self.text
            if state.temperature_high_c is not None:
                if state.temperature_high_c >= view.thermal_video_stop_c:
                    temp_color = self.red
                elif state.temperature_high_c >= view.thermal_cooling_start_c:
                    temp_color = self.blue
                elif state.temperature_high_c >= view.thermal_warning_c:
                    temp_color = self.yellow
            cv2.putText(img, f"bat {battery}", (x + 16, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.text, 1, cv2.LINE_AA)
            cv2.putText(img, f"temp {temp}", (x + max(92, card_w - 100), y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.46, temp_color, 1, cv2.LINE_AA)

    def _draw_camera_wall(self, img: np.ndarray, view: DashboardViewState, x0: int, y0: int, w: int, h: int) -> None:
        self._rounded_rect(img, (x0, y0), (x0 + w, y0 + h), self.panel, radius=18)
        cv2.putText(img, "Live Cameras", (x0 + 22, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.76, self.text, 1, cv2.LINE_AA)

        frames = view.camera_frames or {}
        ips = sorted(frames)
        if not ips:
            cv2.putText(img, "no camera frames", (x0 + 30, y0 + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.muted, 1, cv2.LINE_AA)
            return

        cols = 3 if len(ips) > 4 else 2
        rows = int(np.ceil(len(ips) / cols))
        gap = 12
        pad_x = 22
        pad_top = 56
        pad_bottom = 22
        tile_w = int((w - 2 * pad_x - gap * (cols - 1)) / cols)
        tile_h = int((h - pad_top - pad_bottom - gap * (rows - 1)) / rows)

        for index, ip in enumerate(ips):
            row = index // cols
            col = index % cols
            tx = x0 + pad_x + col * (tile_w + gap)
            ty = y0 + pad_top + row * (tile_h + gap)
            selected = ip == view.selected_ip
            border = self.blue if selected else (68, 76, 80)
            cv2.rectangle(img, (tx - 2, ty - 2), (tx + tile_w + 2, ty + tile_h + 2), border, 2)
            frame = self._fit_frame(frames[ip], tile_w, tile_h)
            img[ty : ty + tile_h, tx : tx + tile_w] = frame
            cv2.rectangle(img, (tx, ty), (tx + tile_w, ty + 24), (0, 0, 0), -1)
            cv2.putText(img, ip, (tx + 8, ty + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.text, 1, cv2.LINE_AA)

    def _draw_compact_footer(self, img: np.ndarray, view: DashboardViewState, x0: int, y0: int, w: int, h: int) -> None:
        self._rounded_rect(img, (x0, y0), (x0 + w, y0 + h), self.panel, radius=18)
        enabled = True
        bx1, by1 = x0 + w - 250, y0 + 9
        bx2, by2 = x0 + w - 18, y0 + h - 9
        self.motor_button_rect = (bx1, by1, bx2, by2)
        button_fill = (55, 77, 65) if enabled else (58, 58, 58)
        button_text = self.green if enabled else self.muted
        self._rounded_rect(img, (bx1, by1), (bx2, by2), button_fill, radius=14)
        label = "SPINNING..." if view.manual_spin_active else "MOTOR SPIN"
        cv2.putText(img, label, (bx1 + 42, by1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, button_text, 2, cv2.LINE_AA)
        line = "S status   M spin props   V restart video   C select   Q quit"
        cv2.putText(img, line, (x0 + 24, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.text, 1, cv2.LINE_AA)

    def hit_test_motor_button(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.motor_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _rounded_rect(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[int, int, int], radius: int = 12) -> None:
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
