"""Tello state telemetry listener for station-mode swarm dashboards."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import threading
from typing import Callable


DEFAULT_TELLO_STATE_PORT = 8890


@dataclass(frozen=True)
class TelloTelemetry:
    ip: str
    battery: int | None = None
    temperature_low_c: int | None = None
    temperature_high_c: int | None = None
    height_cm: int | None = None
    tof_cm: int | None = None


TelemetryCallback = Callable[[TelloTelemetry], None]


def parse_tello_state_packet(ip: str, payload: bytes) -> TelloTelemetry | None:
    """Parse a semicolon-delimited Tello state packet.

    Example packet:
    ``mid:-1;x:0;y:0;z:0;pitch:0;roll:0;yaw:0;templ:47;temph:52;bat:89;...``
    """
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return None

    values: dict[str, str] = {}
    for part in text.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        values[key.strip()] = value.strip()

    return TelloTelemetry(
        ip=ip,
        battery=_parse_int(values.get("bat")),
        temperature_low_c=_parse_int(values.get("templ")),
        temperature_high_c=_parse_int(values.get("temph")),
        height_cm=_parse_int(values.get("h")),
        tof_cm=_parse_int(values.get("tof")),
    )


class TelloTelemetryListener:
    """Receive Tello state packets on UDP 8890 and dispatch parsed telemetry."""

    def __init__(
        self,
        source_ips: list[str],
        on_telemetry: TelemetryCallback,
        port: int = DEFAULT_TELLO_STATE_PORT,
    ) -> None:
        self.source_ips = set(source_ips)
        self.on_telemetry = on_telemetry
        self.port = port
        self._stop_event = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.stop()
        self._stop_event.clear()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        sock.settimeout(0.2)
        self._socket = sock

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._socket = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._thread = None

    def _run(self) -> None:
        sock = self._socket
        if sock is None:
            return

        while not self._stop_event.is_set():
            try:
                packet, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except TimeoutError:
                continue
            except OSError:
                break

            source_ip = addr[0]
            if source_ip not in self.source_ips:
                continue

            telemetry = parse_tello_state_packet(source_ip, packet)
            if telemetry is not None:
                self.on_telemetry(telemetry)


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None
