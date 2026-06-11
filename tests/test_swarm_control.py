from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_control import SwarmController  # noqa: E402
from swarm_utils import SdkResponse  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.sent_one: list[tuple[str, str]] = []
        self.sent_all: list[tuple[tuple[str, ...], str]] = []

    def send_one(self, ip: str, command: str, **_kwargs) -> SdkResponse:
        self.sent_one.append((ip, command))
        if command == "battery?":
            return SdkResponse(ip=ip, text="80", latency_ms=1)
        return SdkResponse(ip=ip, text="ok", latency_ms=1)

    def send_all(self, ips: list[str], command: str, **_kwargs) -> dict[str, SdkResponse]:
        self.sent_all.append((tuple(ips), command))
        responses: dict[str, SdkResponse] = {}
        for ip in ips:
            if ip.endswith(".2") and command == "land":
                continue
            responses[ip] = SdkResponse(ip=ip, text="ok", latency_ms=1)
        return responses

    def close(self) -> None:
        pass


class SwarmControllerTests(unittest.TestCase):
    def test_takeoff_sequence_sends_one_drone_at_a_time(self) -> None:
        client = FakeClient()
        controller = SwarmController(["192.168.0.1", "192.168.0.2"], client=client)

        ok = controller.takeoff_sequential(settle_seconds=0)

        self.assertTrue(ok)
        self.assertEqual(
            client.sent_one,
            [
                ("192.168.0.1", "takeoff"),
                ("192.168.0.2", "takeoff"),
            ],
        )
        self.assertTrue(controller.states["192.168.0.1"].airborne)
        self.assertTrue(controller.states["192.168.0.2"].airborne)

    def test_ready_check_queries_command_and_battery_before_flight(self) -> None:
        client = FakeClient()
        controller = SwarmController(["192.168.0.1", "192.168.0.2"], client=client)

        ok = controller.check_all_ready(retries=1, min_battery=20)

        self.assertTrue(ok)
        self.assertEqual(
            client.sent_one,
            [
                ("192.168.0.1", "command"),
                ("192.168.0.1", "battery?"),
                ("192.168.0.2", "command"),
                ("192.168.0.2", "battery?"),
            ],
        )

    def test_land_all_is_best_effort(self) -> None:
        client = FakeClient()
        controller = SwarmController(["192.168.0.1", "192.168.0.2"], client=client)
        controller.states["192.168.0.1"].airborne = True
        controller.states["192.168.0.2"].airborne = True

        ok = controller.land_all()

        self.assertFalse(ok)
        self.assertEqual(client.sent_all, [(("192.168.0.1", "192.168.0.2"), "land")])
        self.assertFalse(controller.states["192.168.0.1"].airborne)
        self.assertTrue(controller.states["192.168.0.2"].airborne)


if __name__ == "__main__":
    unittest.main()
