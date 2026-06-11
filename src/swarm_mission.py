"""Pure mission logic for stationary Tello swarm person search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Protocol


class MissionState(str, Enum):
    IDLE = "idle"
    PREFLIGHT = "preflight"
    TAKEOFF_SEQUENCE = "takeoff_sequence"
    STATIONARY_YAW_SEARCH = "stationary_yaw_search"
    TRACKER_ACQUIRE = "tracker_acquire"
    TRACKER_CENTER = "tracker_center"
    TRACKER_APPROACH_STEP = "tracker_approach_step"
    TRACKER_REASSESS = "tracker_reassess"
    TARGET_LOST = "target_lost"
    HOLD_ALL = "hold_all"
    LAND_SEQUENCE = "land_sequence"
    ABORT = "abort"


class DroneMissionState(str, Enum):
    IDLE = "idle"
    READY = "ready"
    TAKING_OFF = "taking_off"
    HOVERING = "hovering"
    SCANNING = "scanning"
    TRACKING = "tracking"
    CENTERING = "centering"
    APPROACHING = "approaching"
    HOLDING = "holding"
    LANDING = "landing"
    FAILED = "failed"


class DetectionLike(Protocol):
    ip: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    frame_shape: tuple[int, int]
    detected_at_monotonic: float


@dataclass(frozen=True)
class SearchMissionConfig:
    confirmation_detections: int = 5
    confirmation_window_seconds: float = 3.0
    max_detection_age_seconds: float = 1.5
    min_confidence: float = 0.2
    center_tolerance_ratio: float = 0.15
    lost_target_seconds: float = 2.0
    yaw_step_degrees: int = 20
    yaw_interval_seconds: float = 1.2
    center_yaw_interval_seconds: float = 0.8
    approach_distance_cm: int = 20
    approach_reassess_seconds: float = 0.8
    max_approach_steps: int = 8
    max_approach_seconds: float = 30.0


@dataclass(frozen=True)
class DetectionObservation:
    ip: str
    confidence: float
    bbox_center_x: float
    frame_width: int
    frame_height: int
    detected_at_monotonic: float

    @property
    def center_error_ratio(self) -> float:
        if self.frame_width <= 0:
            return 0.0
        image_center = self.frame_width * 0.5
        half_width = max(1.0, image_center)
        return (self.bbox_center_x - image_center) / half_width


@dataclass(frozen=True)
class TrackerCandidate:
    ip: str
    detections_in_window: int
    latest_confidence: float
    center_error_ratio: float
    centered: bool
    last_seen_age_seconds: float


@dataclass(frozen=True)
class MissionCommand:
    ip: str
    command: str
    reason: str
    horizontal_motion: bool = False
    followup_stop: bool = False


@dataclass(frozen=True)
class MissionSnapshot:
    state: MissionState
    drone_states: dict[str, DroneMissionState]
    tracker_ip: str | None
    candidate_ip: str | None
    latest_confidence_by_ip: dict[str, float]
    center_error_by_ip: dict[str, float]
    detections_by_ip: dict[str, int]
    approach_steps: int
    safety_blockers: tuple[str, ...]
    movement_lock_ip: str | None


@dataclass
class DetectionAggregator:
    ips: list[str]
    config: SearchMissionConfig
    _history: dict[str, list[DetectionObservation]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for ip in self.ips:
            self._history.setdefault(ip, [])

    def update(self, detections_by_ip: dict[str, list[DetectionLike]], now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        for ip in self.ips:
            existing = self._history.setdefault(ip, [])
            for detection in detections_by_ip.get(ip, []):
                observation = observation_from_detection(detection)
                if observation is None:
                    continue
                if observation.confidence < self.config.min_confidence:
                    continue
                existing.append(observation)
            cutoff = now - self.config.confirmation_window_seconds
            self._history[ip] = [
                item
                for item in existing
                if item.detected_at_monotonic >= cutoff
            ]

    def latest(self, ip: str, now: float | None = None) -> TrackerCandidate | None:
        now = time.monotonic() if now is None else now
        history = self._history.get(ip, [])
        if not history:
            return None
        latest = max(history, key=lambda item: item.detected_at_monotonic)
        age = now - latest.detected_at_monotonic
        centered = abs(latest.center_error_ratio) <= self.config.center_tolerance_ratio
        return TrackerCandidate(
            ip=ip,
            detections_in_window=len(history),
            latest_confidence=latest.confidence,
            center_error_ratio=latest.center_error_ratio,
            centered=centered,
            last_seen_age_seconds=age,
        )

    def best_confirmed_candidate(self, now: float | None = None) -> TrackerCandidate | None:
        now = time.monotonic() if now is None else now
        candidates: list[TrackerCandidate] = []
        for ip in self.ips:
            candidate = self.latest(ip, now=now)
            if candidate is None:
                continue
            if candidate.last_seen_age_seconds > self.config.max_detection_age_seconds:
                continue
            if candidate.detections_in_window < self.config.confirmation_detections:
                continue
            candidates.append(candidate)
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.detections_in_window, item.latest_confidence))

    def counts(self) -> dict[str, int]:
        return {ip: len(self._history.get(ip, [])) for ip in self.ips}

    def latest_confidences(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ip in self.ips:
            history = self._history.get(ip, [])
            if history:
                out[ip] = max(history, key=lambda item: item.detected_at_monotonic).confidence
        return out

    def latest_center_errors(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ip in self.ips:
            history = self._history.get(ip, [])
            if history:
                out[ip] = max(history, key=lambda item: item.detected_at_monotonic).center_error_ratio
        return out


class SearchMission:
    """State machine for stationary yaw search and constrained tracker approach."""

    def __init__(self, ips: list[str], config: SearchMissionConfig | None = None) -> None:
        self.ips = list(ips)
        self.config = config or SearchMissionConfig()
        self.aggregator = DetectionAggregator(self.ips, self.config)
        self.state = MissionState.IDLE
        self.drone_states = {ip: DroneMissionState.IDLE for ip in self.ips}
        self.tracker_ip: str | None = None
        self.candidate_ip: str | None = None
        self.approach_steps = 0
        self._approach_started_at: float | None = None
        self._next_yaw_at = {ip: 0.0 for ip in self.ips}
        self._next_approach_at = 0.0
        self._movement_lock_ip: str | None = None
        self._safety_blockers: list[str] = []

    def mark_preflight(self) -> None:
        self.state = MissionState.PREFLIGHT
        self._set_all(DroneMissionState.READY)

    def mark_takeoff_sequence(self) -> None:
        self.state = MissionState.TAKEOFF_SEQUENCE
        self._set_all(DroneMissionState.TAKING_OFF)

    def start_search(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.state = MissionState.STATIONARY_YAW_SEARCH
        self.tracker_ip = None
        self.candidate_ip = None
        self.approach_steps = 0
        self._movement_lock_ip = None
        self._approach_started_at = None
        self._next_approach_at = now
        for ip in self.ips:
            self.drone_states[ip] = DroneMissionState.SCANNING
            self._next_yaw_at[ip] = now

    def hold(self) -> None:
        self.state = MissionState.HOLD_ALL
        self._movement_lock_ip = None
        for ip in self.ips:
            self.drone_states[ip] = DroneMissionState.HOLDING

    def mark_landing(self) -> None:
        self.state = MissionState.LAND_SEQUENCE
        self._movement_lock_ip = None
        self._set_all(DroneMissionState.LANDING)

    def abort(self) -> None:
        self.state = MissionState.ABORT
        self._movement_lock_ip = None
        self._set_all(DroneMissionState.HOLDING)

    def mark_command_complete(self, ip: str, command: str, ok: bool, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if not ok:
            self.drone_states[ip] = DroneMissionState.FAILED
            self._safety_blockers.append(f"{ip} command failed: {command}")
        if self._movement_lock_ip == ip:
            self._movement_lock_ip = None
            self._next_approach_at = now + self.config.approach_reassess_seconds
            if ok and self.state == MissionState.TRACKER_APPROACH_STEP:
                self.state = MissionState.TRACKER_REASSESS
                self.drone_states[ip] = DroneMissionState.TRACKING

    def tick(
        self,
        detections_by_ip: dict[str, list[DetectionLike]],
        now: float | None = None,
    ) -> list[MissionCommand]:
        now = time.monotonic() if now is None else now
        self.aggregator.update(detections_by_ip, now=now)

        if self.state not in {
            MissionState.STATIONARY_YAW_SEARCH,
            MissionState.TRACKER_ACQUIRE,
            MissionState.TRACKER_CENTER,
            MissionState.TRACKER_REASSESS,
            MissionState.TARGET_LOST,
        }:
            return []

        if self.tracker_ip is None:
            candidate = self.aggregator.best_confirmed_candidate(now=now)
            if candidate is not None:
                self._select_tracker(candidate, now=now)
                return [MissionCommand(candidate.ip, "stop", "tracker selected: stop before centering")]
            return self._stationary_yaw_scan(now)

        candidate = self.aggregator.latest(self.tracker_ip, now=now)
        if candidate is None or candidate.last_seen_age_seconds > self.config.lost_target_seconds:
            self.state = MissionState.TARGET_LOST
            self.drone_states[self.tracker_ip] = DroneMissionState.SCANNING
            return self._tracker_local_search(now)

        self.candidate_ip = self.tracker_ip
        if not candidate.centered:
            return self._center_tracker(candidate, now)
        return self._approach_tracker(candidate, now)

    def snapshot(self) -> MissionSnapshot:
        return MissionSnapshot(
            state=self.state,
            drone_states=dict(self.drone_states),
            tracker_ip=self.tracker_ip,
            candidate_ip=self.candidate_ip,
            latest_confidence_by_ip=self.aggregator.latest_confidences(),
            center_error_by_ip=self.aggregator.latest_center_errors(),
            detections_by_ip=self.aggregator.counts(),
            approach_steps=self.approach_steps,
            safety_blockers=tuple(self._safety_blockers[-5:]),
            movement_lock_ip=self._movement_lock_ip,
        )

    def _select_tracker(self, candidate: TrackerCandidate, now: float) -> None:
        self.state = MissionState.TRACKER_ACQUIRE
        self.tracker_ip = candidate.ip
        self.candidate_ip = candidate.ip
        self.approach_steps = 0
        self._approach_started_at = now
        self._movement_lock_ip = None
        for ip in self.ips:
            self.drone_states[ip] = DroneMissionState.HOLDING
        self.drone_states[candidate.ip] = DroneMissionState.TRACKING

    def _stationary_yaw_scan(self, now: float) -> list[MissionCommand]:
        commands: list[MissionCommand] = []
        self.state = MissionState.STATIONARY_YAW_SEARCH
        for ip in self.ips:
            self.drone_states[ip] = DroneMissionState.SCANNING
            if now < self._next_yaw_at[ip]:
                continue
            self._next_yaw_at[ip] = now + self.config.yaw_interval_seconds
            commands.append(MissionCommand(ip, f"cw {self.config.yaw_step_degrees}", "stationary search yaw"))
        return commands

    def _tracker_local_search(self, now: float) -> list[MissionCommand]:
        if self.tracker_ip is None:
            return []
        ip = self.tracker_ip
        if now < self._next_yaw_at[ip]:
            return []
        self._next_yaw_at[ip] = now + self.config.yaw_interval_seconds
        return [MissionCommand(ip, f"cw {self.config.yaw_step_degrees}", "target lost local yaw")]

    def _center_tracker(self, candidate: TrackerCandidate, now: float) -> list[MissionCommand]:
        ip = candidate.ip
        self.state = MissionState.TRACKER_CENTER
        self.drone_states[ip] = DroneMissionState.CENTERING
        if now < self._next_yaw_at[ip]:
            return []
        self._next_yaw_at[ip] = now + self.config.center_yaw_interval_seconds
        command = f"cw {self.config.yaw_step_degrees}"
        if candidate.center_error_ratio < 0:
            command = f"ccw {self.config.yaw_step_degrees}"
        return [MissionCommand(ip, command, "center person before approach")]

    def _approach_tracker(self, candidate: TrackerCandidate, now: float) -> list[MissionCommand]:
        ip = candidate.ip
        if self._movement_lock_ip is not None:
            return []
        if self.approach_steps >= self.config.max_approach_steps:
            self.hold()
            self._safety_blockers.append("max approach steps reached")
            return [MissionCommand(ip, "stop", "max approach steps reached")]
        if self._approach_started_at is not None and now - self._approach_started_at > self.config.max_approach_seconds:
            self.hold()
            self._safety_blockers.append("max approach time reached")
            return [MissionCommand(ip, "stop", "max approach time reached")]
        if now < self._next_approach_at:
            self.state = MissionState.TRACKER_REASSESS
            return []

        self.state = MissionState.TRACKER_APPROACH_STEP
        self.drone_states[ip] = DroneMissionState.APPROACHING
        self.approach_steps += 1
        self._movement_lock_ip = ip
        return [
            MissionCommand(
                ip=ip,
                command=f"forward {self.config.approach_distance_cm}",
                reason="person centered: constrained approach step",
                horizontal_motion=True,
                followup_stop=True,
            )
        ]

    def _set_all(self, state: DroneMissionState) -> None:
        for ip in self.ips:
            self.drone_states[ip] = state


def observation_from_detection(detection: DetectionLike) -> DetectionObservation | None:
    try:
        x1, _y1, x2, _y2 = detection.xyxy
        frame_height, frame_width = detection.frame_shape
        confidence = float(detection.confidence)
        detected_at = float(detection.detected_at_monotonic)
    except Exception:
        return None
    if frame_width <= 0 or frame_height <= 0:
        return None
    center_x = (float(x1) + float(x2)) * 0.5
    return DetectionObservation(
        ip=detection.ip,
        confidence=confidence,
        bbox_center_x=center_x,
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        detected_at_monotonic=detected_at,
    )
