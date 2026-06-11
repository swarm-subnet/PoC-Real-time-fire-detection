"""Single-drone stationary person search validation run."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from swarm_camera import DEFAULT_LOCAL_VIDEO_PORT, SwarmCameraWall
from swarm_control import SwarmController
from swarm_detector import PersonDetectorConfig, SwarmPersonDetector
from swarm_mission import DetectionAggregator, MissionCommand, SearchMissionConfig, TrackerCandidate


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

    @staticmethod
    def _status(status: StatusCallback | None, message: str) -> None:
        if status is not None:
            status(message)
