"""OpenCV dashboard rendering for Tello swarm state."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from swarm_detector import PersonDetection, PersonDetectorStats
from swarm_mission import MissionSnapshot
from swarm_state import DroneRuntimeState


@dataclass
class DashboardViewState:
    selected_ip: str | None = None
    banner: str = "idle"
    camera_enabled: bool = False
    camera_frames: dict[str, np.ndarray] | None = None
    person_detections: dict[str, list[PersonDetection]] | None = None
    person_detector_stats: PersonDetectorStats | None = None
    mission_snapshot: MissionSnapshot | None = None
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
        self.preflight_button_rect = (0, 0, 0, 0)
        self.search_button_rect = (0, 0, 0, 0)
        self.hold_button_rect = (0, 0, 0, 0)
        self.land_button_rect = (0, 0, 0, 0)
        self.abort_button_rect = (0, 0, 0, 0)

    def render(self, states: list[DroneRuntimeState], view: DashboardViewState) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), self.bg, dtype=np.uint8)
        self._draw_header(canvas, states, view)
        self._draw_cards(canvas, states, view, x0=24, y0=88, w=self.width - 48, h=106)
        self._draw_camera_wall(canvas, view, x0=24, y0=214, w=self.width - 48, h=self.height - 306)
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
        detector_text = self._detector_summary(view)
        mission_text = self._mission_summary(view)
        summary = f"{online}/{len(states)} online   {temp_text}   {video_text}   {cooling_text}   {detector_text}   {mission_text}   motor={motor.lower()}"
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
            mission = view.mission_snapshot
            if mission is not None:
                drone_state = mission.drone_states.get(state.ip)
                if drone_state is not None:
                    cv2.putText(img, drone_state.value[:11], (x + 16, y + 103), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.blue, 1, cv2.LINE_AA)

    def _draw_camera_wall(self, img: np.ndarray, view: DashboardViewState, x0: int, y0: int, w: int, h: int) -> None:
        self._rounded_rect(img, (x0, y0), (x0 + w, y0 + h), self.panel, radius=18)
        cv2.putText(img, "Live Cameras", (x0 + 22, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.76, self.text, 1, cv2.LINE_AA)

        frames = view.camera_frames or {}
        detections_by_ip = view.person_detections or {}
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
            source_frame = frames[ip]
            frame = self._fit_frame(source_frame, tile_w, tile_h)
            if detections_by_ip.get(ip):
                self._draw_person_detections(frame, detections_by_ip[ip])
            img[ty : ty + tile_h, tx : tx + tile_w] = frame
            cv2.rectangle(img, (tx, ty), (tx + tile_w, ty + 24), (0, 0, 0), -1)
            person_count = len(detections_by_ip.get(ip, []))
            label = ip if person_count == 0 else f"{ip}  person x{person_count}"
            cv2.putText(img, label, (tx + 8, ty + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.text, 1, cv2.LINE_AA)

    def _draw_compact_footer(self, img: np.ndarray, view: DashboardViewState, x0: int, y0: int, w: int, h: int) -> None:
        self._rounded_rect(img, (x0, y0), (x0 + w, y0 + h), self.panel, radius=18)
        button_w = 132
        button_h = h - 18
        gap = 10
        by1 = y0 + 9
        by2 = by1 + button_h
        x = x0 + w - 18 - button_w
        self.abort_button_rect = (x, by1, x + button_w, by2)
        x -= button_w + gap
        self.land_button_rect = (x, by1, x + button_w, by2)
        x -= button_w + gap
        self.hold_button_rect = (x, by1, x + button_w, by2)
        x -= button_w + gap
        self.search_button_rect = (x, by1, x + button_w, by2)
        x -= button_w + gap
        self.preflight_button_rect = (x, by1, x + button_w, by2)
        x -= button_w + gap
        self.motor_button_rect = (x, by1, x + button_w, by2)

        self._draw_button(img, self.motor_button_rect, "SPIN" if not view.manual_spin_active else "SPINNING", (55, 77, 65), self.green)
        self._draw_button(img, self.preflight_button_rect, "PREFLIGHT", (45, 65, 86), self.blue)
        self._draw_button(img, self.search_button_rect, "SEARCH", (52, 70, 55), self.green)
        self._draw_button(img, self.hold_button_rect, "HOLD", (66, 62, 48), self.yellow)
        self._draw_button(img, self.land_button_rect, "LAND", (78, 58, 45), self.yellow)
        self._draw_button(img, self.abort_button_rect, "ABORT", (82, 45, 45), self.red)

        line = "S status   M spin   P preflight   G search   H hold   L land   A abort   V video   Q quit"
        cv2.putText(img, line, (x0 + 24, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.text, 1, cv2.LINE_AA)

    def hit_test_motor_button(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.motor_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def hit_test_preflight_button(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.preflight_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def hit_test_search_button(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.search_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def hit_test_hold_button(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.hold_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def hit_test_land_button(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.land_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def hit_test_abort_button(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.abort_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _draw_button(
        self,
        img: np.ndarray,
        rect: tuple[int, int, int, int],
        label: str,
        fill: tuple[int, int, int],
        text_color: tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = rect
        self._rounded_rect(img, (x1, y1), (x2, y2), fill, radius=14)
        (text_w, text_h), _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
        tx = x1 + max(6, (x2 - x1 - text_w) // 2)
        ty = y1 + max(text_h + 6, (y2 - y1 + text_h) // 2)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_color, 2, cv2.LINE_AA)

    def _detector_summary(self, view: DashboardViewState) -> str:
        stats = view.person_detector_stats
        if stats is None:
            return "person=off"
        if stats.status == "error":
            return f"person=error:{stats.last_error[:20]}"
        if not stats.loaded:
            return f"person={stats.status}"
        last_ms = "n/a" if stats.last_inference_ms is None else f"{stats.last_inference_ms:.0f}ms"
        hits = sum(len(value) for value in (view.person_detections or {}).values())
        return f"people={hits} infer={last_ms}"

    def _mission_summary(self, view: DashboardViewState) -> str:
        mission = view.mission_snapshot
        if mission is None:
            return "mission=off"
        tracker = "-" if mission.tracker_ip is None else mission.tracker_ip.rsplit(".", 1)[-1]
        return f"mission={mission.state.value} tracker={tracker} steps={mission.approach_steps}"

    def _draw_person_detections(self, frame: np.ndarray, detections: list[PersonDetection]) -> None:
        for detection in detections:
            source_h, source_w = detection.frame_shape
            target_h, target_w = frame.shape[:2]
            if source_h <= 0 or source_w <= 0 or target_h <= 0 or target_w <= 0:
                continue
            coords = np.asarray(detection.xyxy, dtype=np.float32).reshape(-1)
            if coords.shape[0] != 4 or not np.all(np.isfinite(coords)):
                continue
            sx = target_w / max(1, source_w)
            sy = target_h / max(1, source_h)
            x1, y1, x2, y2 = (
                int(round(float(coords[0]) * sx)),
                int(round(float(coords[1]) * sy)),
                int(round(float(coords[2]) * sx)),
                int(round(float(coords[3]) * sy)),
            )
            x1 = int(np.clip(x1, 0, target_w - 1))
            x2 = int(np.clip(x2, 0, target_w - 1))
            y1 = int(np.clip(y1, 0, target_h - 1))
            y2 = int(np.clip(y2, 0, target_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.green, 3)
            label = f"person {detection.confidence:.2f}"
            cv2.putText(frame, label, (x1 + 5, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, self.green, 2, cv2.LINE_AA)

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
