"""Single-drone stationary person search validation run."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

import cv2
import numpy as np

from swarm_camera import DEFAULT_LOCAL_VIDEO_PORT, SwarmCameraWall
from swarm_control import SwarmController
from swarm_detector import PersonDetectorConfig, SwarmPersonDetector
from swarm_mission import DetectionAggregator, DetectionLike, MissionCommand, SearchMissionConfig, TrackerCandidate


StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class SingleDroneSearchConfig:
    min_battery: int = 20
    status_retries: int = 2
    max_search_seconds: float = 90.0
    yaw_step_degrees: int = 20
    yaw_interval_seconds: float = 1.2
    takeoff_settle_seconds: float = 3.0
    video_port_start: int = DEFAULT_LOCAL_VIDEO_PORT
    confirmation_detections: int = 5
    confirmation_window_seconds: float = 3.0
    max_detection_age_seconds: float = 1.5
    min_confidence: float = 0.2
    preview_enabled: bool = True
    preview_window_name: str = "Single Drone Person Search"
    detection_hold_seconds: float = 1.5


@dataclass(frozen=True)
class SingleDroneSearchResult:
    detected: bool
    landed: bool
    reason: str
    candidate: TrackerCandidate | None = None


class SingleDroneSearchRunner:
    """Take off one drone, yaw-search in place, land after confirmed person detection."""

    def __init__(
        self,
        ip: str,
        detector_config: PersonDetectorConfig,
        config: SingleDroneSearchConfig | None = None,
        controller: SwarmController | None = None,
        camera: SwarmCameraWall | None = None,
        detector: SwarmPersonDetector | None = None,
    ) -> None:
        self.ip = ip
        self.config = config or SingleDroneSearchConfig()
        self.controller = controller or SwarmController([ip])
        self.camera = camera or SwarmCameraWall(
            self.controller.client,
            [ip],
            video_port_start=self.config.video_port_start,
        )
        self.detector = detector or SwarmPersonDetector([ip], detector_config)
        self._preview_window_created = False
        self._last_status_message = "starting"
        self.aggregator = DetectionAggregator(
            [ip],
            SearchMissionConfig(
                confirmation_detections=self.config.confirmation_detections,
                confirmation_window_seconds=self.config.confirmation_window_seconds,
                max_detection_age_seconds=self.config.max_detection_age_seconds,
                min_confidence=self.config.min_confidence,
                yaw_step_degrees=self.config.yaw_step_degrees,
                yaw_interval_seconds=self.config.yaw_interval_seconds,
            ),
        )

    def run(self, status: StatusCallback | None = None) -> SingleDroneSearchResult:
        landed = False
        try:
            self._status(status, "loading GPU person detector")
            self.detector.ensure_ready()

            self._status(status, "starting video stream")
            self.camera.start()
            self.detector.start()

            self._status(status, "preflight")
            ready = self.controller.check_all_ready(
                retries=self.config.status_retries,
                min_battery=self.config.min_battery,
                status=status,
            )
            if not ready:
                return SingleDroneSearchResult(False, False, "preflight failed")

            self._status(status, "takeoff")
            if not self.controller.takeoff_sequential(
                settle_seconds=self.config.takeoff_settle_seconds,
                status=status,
            ):
                self._land(status)
                return SingleDroneSearchResult(False, True, "takeoff failed")

            result = self._search_until_detection(status)
            landed = self._land(status)
            return SingleDroneSearchResult(result.detected, landed, result.reason, result.candidate)
        except KeyboardInterrupt:
            landed = self._land(status)
            return SingleDroneSearchResult(False, landed, "interrupted")
        finally:
            if self.controller.any_airborne() and not landed:
                self._land(status)
            self.detector.stop()
            self.camera.stop()
            self._close_preview()
            self.controller.close()

    def _search_until_detection(self, status: StatusCallback | None) -> SingleDroneSearchResult:
        started = time.monotonic()
        next_yaw_at = started
        deadline = started + self.config.max_search_seconds
        self._status(status, "stationary yaw search started")

        while time.monotonic() < deadline:
            now = time.monotonic()
            snapshot = self.camera.get_snapshot(copy=False)
            self.detector.update_frames(snapshot.frames, snapshot.frame_versions)
            detections = self.detector.get_detections()
            if not self._show_preview(snapshot.frames.get(self.ip), detections.get(self.ip, [])):
                raise KeyboardInterrupt
            self.aggregator.update(detections, now=now)
            candidate = self.aggregator.best_confirmed_candidate(now=now)
            if candidate is not None:
                self._status(
                    status,
                    f"PERSON DETECTED by {self.ip}: conf={candidate.latest_confidence:.2f} "
                    f"center_error={candidate.center_error_ratio:.2f}; landing",
                )
                self._send_mission_command(MissionCommand(self.ip, "stop", "person detected"), status)
                self._hold_detection_preview(snapshot.frames.get(self.ip), detections.get(self.ip, []))
                return SingleDroneSearchResult(True, False, "person detected", candidate)

            if now >= next_yaw_at:
                command = MissionCommand(
                    self.ip,
                    f"cw {self.config.yaw_step_degrees}",
                    "stationary search yaw",
                )
                self._send_mission_command(command, status)
                next_yaw_at = now + self.config.yaw_interval_seconds

            time.sleep(0.03)

        self._status(status, "search timed out; landing")
        self._send_mission_command(MissionCommand(self.ip, "stop", "search timeout"), status)
        return SingleDroneSearchResult(False, False, "search timed out")

    def _show_preview(self, frame: np.ndarray | None, detections: list[DetectionLike]) -> bool:
        key = self._render_preview(frame, detections)
        return key not in (27, ord("q"), ord("Q"))

    def _render_preview(self, frame: np.ndarray | None, detections: list[DetectionLike]) -> int | None:
        if not self.config.preview_enabled:
            return None
        if frame is None:
            return None
        if not self._preview_window_created:
            cv2.namedWindow(self.config.preview_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.config.preview_window_name, 960, 720)
            self._preview_window_created = True

        canvas = frame.copy()
        self._draw_detections(canvas, detections)
        self._draw_status(canvas)
        cv2.imshow(self.config.preview_window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        return None if key == 255 else key

    def _hold_detection_preview(self, frame: np.ndarray | None, detections: list[DetectionLike]) -> None:
        if not self.config.preview_enabled or self.config.detection_hold_seconds <= 0:
            return
        deadline = time.monotonic() + self.config.detection_hold_seconds
        self._last_status_message = "PERSON DETECTED - landing"
        while time.monotonic() < deadline:
            if not self._show_preview(frame, detections):
                break
            time.sleep(0.03)

    def _draw_status(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        overlay_h = 58
        cv2.rectangle(frame, (0, 0), (width, overlay_h), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{self.ip} | {self._last_status_message[:80]}",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 230, 120),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "P preflight | G start search | L land | Q/Esc quit-land | no forward/sideways",
            (14, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 235, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.line(frame, (width // 2, overlay_h), (width // 2, height), (80, 160, 255), 1)

    @staticmethod
    def _draw_detections(frame: np.ndarray, detections: list[DetectionLike]) -> None:
        target_h, target_w = frame.shape[:2]
        for detection in detections:
            try:
                source_h, source_w = detection.frame_shape
                coords = np.asarray(detection.xyxy, dtype=np.float32).reshape(-1)
                confidence = float(detection.confidence)
            except Exception:
                continue
            if coords.shape[0] != 4 or source_h <= 0 or source_w <= 0 or not np.all(np.isfinite(coords)):
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 230, 120), 3)
            label = f"person {confidence:.2f}"
            cv2.rectangle(frame, (x1, max(0, y1 - 28)), (min(target_w - 1, x1 + 130), y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 5, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 230, 120), 2, cv2.LINE_AA)

    def _close_preview(self) -> None:
        if not self._preview_window_created:
            return
        try:
            cv2.destroyWindow(self.config.preview_window_name)
        except Exception:
            pass
        self._preview_window_created = False

    def _send_mission_command(self, command: MissionCommand, status: StatusCallback | None) -> bool:
        return self.controller.send_single_command(
            command.ip,
            command.command,
            timeout=8,
            retries=1,
            status=status,
            command_state=command.reason[:24],
        )

    def _land(self, status: StatusCallback | None) -> bool:
        self._status(status, "landing")
        return self.controller.land_all(status=status)

    def _status(self, status: StatusCallback | None, message: str) -> None:
        self._last_status_message = message
        if status is not None:
            status(message)


class SingleDroneSearchPreviewApp(SingleDroneSearchRunner):
    """Preview-first one-drone app: video/detection first, flight only after G."""

    def __init__(
        self,
        ip: str,
        detector_config: PersonDetectorConfig,
        config: SingleDroneSearchConfig | None = None,
        controller: SwarmController | None = None,
        camera: SwarmCameraWall | None = None,
        detector: SwarmPersonDetector | None = None,
    ) -> None:
        super().__init__(ip, detector_config, config, controller, camera, detector)
        self._stop_event = threading.Event()
        self._action_lock = threading.RLock()
        self._action_thread: threading.Thread | None = None
        self._last_action_result: SingleDroneSearchResult | None = None

    def run(self, status: StatusCallback | None = None) -> SingleDroneSearchResult:
        result = SingleDroneSearchResult(False, False, "quit before autonomous run")
        try:
            self._status(status, "loading GPU person detector")
            self.detector.ensure_ready()

            self._status(status, "starting live preview")
            self.camera.start()
            self.detector.start()
            self._status(status, "preview ready: verify boxes, press G to fly")

            while not self._stop_event.is_set():
                snapshot = self.camera.get_snapshot(copy=False)
                self.detector.update_frames(snapshot.frames, snapshot.frame_versions)
                detections = self.detector.get_detections()
                key = self._render_preview(snapshot.frames.get(self.ip), detections.get(self.ip, []))
                if key is not None:
                    self._handle_preview_key(key, status)

                thread = self._action_thread
                if thread is not None and not thread.is_alive():
                    thread.join(timeout=0)
                    self._action_thread = None
                time.sleep(0.01)

            self._wait_for_action(timeout=10)
            if self._last_action_result is not None:
                return self._last_action_result
            if self.controller.any_airborne():
                landed = self._land(status)
                return SingleDroneSearchResult(False, landed, "operator quit")
            return result
        except KeyboardInterrupt:
            self._status(status, "interrupted; landing")
            landed = self._land(status)
            return SingleDroneSearchResult(False, landed, "interrupted")
        finally:
            self._stop_event.set()
            self._wait_for_action(timeout=10)
            if self.controller.any_airborne():
                self._land(status)
            self.detector.stop()
            self.camera.stop()
            self._close_preview()
            self.controller.close()

    def _handle_preview_key(self, key: int, status: StatusCallback | None) -> None:
        char = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27 or char == "q":
            self._status(status, "quit requested; landing if airborne")
            self._stop_event.set()
        elif char == "p":
            self._start_action("preflight", lambda: self._preflight(status), status)
        elif char == "g":
            self._start_action("autonomous search", lambda: self._autonomous_run(status), status)
        elif char == "l":
            self._status(status, "land requested")
            self._stop_event.set()
            self._start_action("land", lambda: self._land(status), status)

    def _start_action(self, name: str, target: Callable[[], object], status: StatusCallback | None) -> None:
        with self._action_lock:
            if self._action_thread is not None and self._action_thread.is_alive():
                self._status(status, f"{name} ignored: action already running")
                return

            def run_action() -> None:
                try:
                    self._status(status, f"{name} started")
                    action_result = target()
                    if isinstance(action_result, SingleDroneSearchResult):
                        self._last_action_result = action_result
                except Exception as error:
                    self._status(status, f"{name} failed: {error}")

            self._action_thread = threading.Thread(target=run_action, daemon=True)
            self._action_thread.start()

    def _preflight(self, status: StatusCallback | None) -> bool:
        return self.controller.check_all_ready(
            retries=self.config.status_retries,
            min_battery=self.config.min_battery,
            status=status,
        )

    def _autonomous_run(self, status: StatusCallback | None) -> SingleDroneSearchResult:
        if self.controller.any_airborne():
            self._status(status, "already airborne; starting yaw search")
        elif not self._preflight(status):
            return SingleDroneSearchResult(False, False, "preflight failed")
        else:
            self._status(status, "takeoff")
            if not self.controller.takeoff_sequential(
                settle_seconds=self.config.takeoff_settle_seconds,
                status=status,
            ):
                landed = self._land(status)
                return SingleDroneSearchResult(False, landed, "takeoff failed")

        self._reset_detection_history()
        result = self._search_from_preview_detector(status)
        landed = self._land(status)
        final_result = SingleDroneSearchResult(result.detected, landed, result.reason, result.candidate)
        self._status(status, f"autonomous complete: {final_result.reason}")
        self._stop_event.set()
        return final_result

    def _search_from_preview_detector(self, status: StatusCallback | None) -> SingleDroneSearchResult:
        started = time.monotonic()
        next_yaw_at = started
        deadline = started + self.config.max_search_seconds
        self._status(status, "stationary yaw search started")

        while time.monotonic() < deadline and not self._stop_event.is_set():
            now = time.monotonic()
            detections = self.detector.get_detections()
            self.aggregator.update(detections, now=now)
            candidate = self.aggregator.best_confirmed_candidate(now=now)
            if candidate is not None:
                self._status(
                    status,
                    f"PERSON DETECTED by {self.ip}: conf={candidate.latest_confidence:.2f} "
                    f"center_error={candidate.center_error_ratio:.2f}; landing",
                )
                self._send_mission_command(MissionCommand(self.ip, "stop", "person detected"), status)
                return SingleDroneSearchResult(True, False, "person detected", candidate)

            if now >= next_yaw_at:
                self._send_mission_command(
                    MissionCommand(self.ip, f"cw {self.config.yaw_step_degrees}", "stationary search yaw"),
                    status,
                )
                next_yaw_at = now + self.config.yaw_interval_seconds
            time.sleep(0.03)

        if self._stop_event.is_set():
            self._send_mission_command(MissionCommand(self.ip, "stop", "operator stop"), status)
            return SingleDroneSearchResult(False, False, "operator stop")

        self._status(status, "search timed out; landing")
        self._send_mission_command(MissionCommand(self.ip, "stop", "search timeout"), status)
        return SingleDroneSearchResult(False, False, "search timed out")

    def _reset_detection_history(self) -> None:
        self.aggregator = DetectionAggregator(
            [self.ip],
            SearchMissionConfig(
                confirmation_detections=self.config.confirmation_detections,
                confirmation_window_seconds=self.config.confirmation_window_seconds,
                max_detection_age_seconds=self.config.max_detection_age_seconds,
                min_confidence=self.config.min_confidence,
                yaw_step_degrees=self.config.yaw_step_degrees,
                yaw_interval_seconds=self.config.yaw_interval_seconds,
            ),
        )

    def _wait_for_action(self, timeout: float) -> None:
        thread = self._action_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
