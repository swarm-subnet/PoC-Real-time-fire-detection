from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_mission import (  # noqa: E402
    MissionState,
    SearchMission,
    SearchMissionConfig,
    observation_from_detection,
)


@dataclass(frozen=True)
class FakeDetection:
    ip: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    frame_shape: tuple[int, int]
    detected_at_monotonic: float


def detection(
    ip: str,
    center_x: float,
    now: float,
    confidence: float = 0.8,
    width: float = 40.0,
    height: float = 80.0,
) -> FakeDetection:
    return FakeDetection(
        ip=ip,
        xyxy=(center_x - width * 0.5, 40, center_x + width * 0.5, 40 + height),
        confidence=confidence,
        frame_shape=(240, 320),
        detected_at_monotonic=now,
    )


class SearchMissionTests(unittest.TestCase):
    def config(self) -> SearchMissionConfig:
        return SearchMissionConfig(
            confirmation_detections=3,
            confirmation_window_seconds=3.0,
            max_detection_age_seconds=1.0,
            center_tolerance_ratio=0.10,
            lost_target_seconds=1.5,
            yaw_step_degrees=20,
            yaw_interval_seconds=1.0,
            center_yaw_interval_seconds=0.5,
            approach_distance_cm=20,
            approach_reassess_seconds=0.5,
            max_approach_steps=3,
        )

    def test_center_error_calculation(self) -> None:
        obs = observation_from_detection(detection("1", center_x=240, now=1.0))
        self.assertIsNotNone(obs)
        self.assertAlmostEqual(obs.center_error_ratio, 0.5)

    def test_repeated_detection_selects_tracker(self) -> None:
        mission = SearchMission(["1", "2"], self.config())
        mission.start_search(now=0.0)
        for now in (0.0, 0.5):
            mission.tick({"1": [detection("1", 160, now)]}, now=now)
        commands = mission.tick({"1": [detection("1", 160, 1.0)]}, now=1.0)
        self.assertEqual(mission.snapshot().tracker_ip, "1")
        self.assertEqual(mission.snapshot().state, MissionState.TRACKER_ACQUIRE)
        self.assertEqual([command.command for command in commands], ["stop"])

    def test_small_detection_is_not_confirmed_as_tracker(self) -> None:
        config = SearchMissionConfig(
            confirmation_detections=2,
            confirmation_window_seconds=3.0,
            max_detection_age_seconds=1.0,
            min_confidence=0.55,
            min_detection_area_ratio=0.025,
            min_detection_height_ratio=0.20,
            yaw_interval_seconds=10.0,
        )
        mission = SearchMission(["1"], config)
        mission.start_search(now=0.0)
        mission.tick({"1": [detection("1", 160, 0.0, confidence=0.9, width=24, height=30)]}, now=0.0)
        mission.tick({"1": [detection("1", 160, 0.5, confidence=0.9, width=24, height=30)]}, now=0.5)

        self.assertIsNone(mission.snapshot().tracker_ip)
        self.assertEqual(mission.snapshot().detections_by_ip["1"], 0)

    def test_off_center_target_yaws_before_forward(self) -> None:
        mission = SearchMission(["1"], self.config())
        mission.start_search(now=0.0)
        for now in (0.0, 0.4, 0.8):
            mission.tick({"1": [detection("1", 60, now)]}, now=now)
        commands = mission.tick({"1": [detection("1", 60, 1.2)]}, now=1.2)
        self.assertEqual([command.command for command in commands], ["ccw 20"])
        self.assertFalse(any(command.horizontal_motion for command in commands))

    def test_centered_target_moves_forward_once_and_locks(self) -> None:
        mission = SearchMission(["1"], self.config())
        mission.start_search(now=0.0)
        for now in (0.0, 0.4, 0.8):
            mission.tick({"1": [detection("1", 160, now)]}, now=now)
        commands = mission.tick({"1": [detection("1", 160, 1.2)]}, now=1.2)
        self.assertEqual([command.command for command in commands], ["forward 20"])
        self.assertTrue(commands[0].horizontal_motion)
        self.assertTrue(commands[0].followup_stop)

        locked = mission.tick({"1": [detection("1", 160, 1.3)]}, now=1.3)
        self.assertEqual(locked, [])

        mission.mark_command_complete("1", "forward 20", True, now=1.4)
        reassess_wait = mission.tick({"1": [detection("1", 160, 1.5)]}, now=1.5)
        self.assertEqual(reassess_wait, [])
        next_step = mission.tick({"1": [detection("1", 160, 2.0)]}, now=2.0)
        self.assertEqual([command.command for command in next_step], ["forward 20"])

    def test_only_one_drone_can_get_horizontal_motion(self) -> None:
        mission = SearchMission(["1", "2"], self.config())
        mission.start_search(now=0.0)
        for now in (0.0, 0.4, 0.8):
            mission.tick(
                {
                    "1": [detection("1", 160, now, confidence=0.9)],
                    "2": [detection("2", 160, now, confidence=0.8)],
                },
                now=now,
            )

        commands = mission.tick(
            {
                "1": [detection("1", 160, 1.2, confidence=0.9)],
                "2": [detection("2", 160, 1.2, confidence=0.8)],
            },
            now=1.2,
        )
        horizontal = [command for command in commands if command.horizontal_motion]
        self.assertLessEqual(len(horizontal), 1)
        self.assertEqual(mission.snapshot().tracker_ip, "1")

    def test_lost_target_enters_local_search(self) -> None:
        mission = SearchMission(["1"], self.config())
        mission.start_search(now=0.0)
        for now in (0.0, 0.4, 0.8):
            mission.tick({"1": [detection("1", 160, now)]}, now=now)
        mission.tick({"1": [detection("1", 160, 1.2)]}, now=1.2)
        mission.mark_command_complete("1", "forward 20", True, now=1.3)

        commands = mission.tick({}, now=3.5)
        self.assertEqual(mission.snapshot().state, MissionState.TARGET_LOST)
        self.assertEqual([command.command for command in commands], ["cw 20"])


if __name__ == "__main__":
    unittest.main()
