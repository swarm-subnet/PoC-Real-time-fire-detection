from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_detector import PersonDetectorConfig  # noqa: E402
from swarm_single_search import SingleDroneSearchConfig, SingleDroneSearchRunner  # noqa: E402


SCRIPT_DIR = ROOT / "scripts" / "swarm"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import importlib  # noqa: E402


single_search_script = importlib.import_module("16_single_drone_person_search_land")


@dataclass(frozen=True)
class FakeDetection:
    ip: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    frame_shape: tuple[int, int]
    detected_at_monotonic: float


@dataclass(frozen=True)
class FakeSnapshot:
    frames: dict[str, np.ndarray]
    frame_versions: dict[str, int]


class FakeController:
    def __init__(self, ip: str) -> None:
        self.ips = [ip]
        self.commands: list[str] = []
        self.airborne = False

    def check_all_ready(self, **_kwargs) -> bool:
        self.commands.append("ready")
        return True

    def takeoff_sequential(self, **_kwargs) -> bool:
        self.commands.append("takeoff")
        self.airborne = True
        return True

    def send_single_command(self, _ip: str, command: str, **_kwargs) -> bool:
        self.commands.append(command)
        return True

    def land_all(self, **_kwargs) -> bool:
        self.commands.append("land")
        self.airborne = False
        return True

    def any_airborne(self) -> bool:
        return self.airborne

    def close(self) -> None:
        self.commands.append("close")


class FakeCamera:
    def __init__(self, ip: str) -> None:
        self.ip = ip
        self.version = 0
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def get_snapshot(self, copy: bool = False) -> FakeSnapshot:
        self.version += 1
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        return FakeSnapshot(frames={self.ip: frame}, frame_versions={self.ip: self.version})


class FakeDetector:
    def __init__(self, ip: str) -> None:
        self.ip = ip
        self.calls = 0

    def ensure_ready(self) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def update_frames(self, _frames, _versions) -> None:
        self.calls += 1

    def get_detections(self):
        if self.calls < 3:
            return {}
        now = time.monotonic()
        return {
            self.ip: [
                FakeDetection(
                    ip=self.ip,
                    xyxy=(140.0, 40.0, 180.0, 120.0),
                    confidence=0.9,
                    frame_shape=(240, 320),
                    detected_at_monotonic=now,
                )
            ]
        }


class SingleDroneSearchRunnerTests(unittest.TestCase):
    def test_detects_person_stops_and_lands_without_forward_motion(self) -> None:
        ip = "192.168.100.89"
        controller = FakeController(ip)
        config = SingleDroneSearchConfig(
            max_search_seconds=2.0,
            yaw_step_degrees=20,
            yaw_interval_seconds=0.01,
            takeoff_settle_seconds=0,
            confirmation_detections=2,
            confirmation_window_seconds=3.0,
            max_detection_age_seconds=10.0,
            min_confidence=0.2,
            preview_enabled=False,
        )
        runner = SingleDroneSearchRunner(
            ip,
            PersonDetectorConfig(),
            config,
            controller=controller,  # type: ignore[arg-type]
            camera=FakeCamera(ip),  # type: ignore[arg-type]
            detector=FakeDetector(ip),  # type: ignore[arg-type]
        )

        result = runner.run()

        self.assertTrue(result.detected)
        self.assertTrue(result.landed)
        self.assertIn("cw 20", controller.commands)
        self.assertIn("stop", controller.commands)
        self.assertIn("land", controller.commands)
        self.assertNotIn("forward 20", controller.commands)

    def test_first_reachable_registered_ip_skips_offline_drone(self) -> None:
        class FakeProbeClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                pass

            def send_one(self, ip: str, command: str, **_kwargs):
                if ip.endswith(".89"):
                    raise TimeoutError("offline")
                return object()

        def fake_load_registered_ips():
            return ["192.168.100.89", "192.168.100.90"]

        def fake_query_battery(_client, ip: str, **_kwargs):
            return (88, 3)

        original_client = single_search_script.TelloUdpClient
        original_load = single_search_script.load_registered_ips
        original_query = single_search_script.query_battery
        try:
            single_search_script.TelloUdpClient = FakeProbeClient
            single_search_script.load_registered_ips = fake_load_registered_ips
            single_search_script.query_battery = fake_query_battery
            selected = single_search_script._first_reachable_registered_ip(timeout=0.01, retries=1)
        finally:
            single_search_script.TelloUdpClient = original_client
            single_search_script.load_registered_ips = original_load
            single_search_script.query_battery = original_query

        self.assertEqual(selected, "192.168.100.90")


if __name__ == "__main__":
    unittest.main()
