"""Video transport primitives for multi-Tello camera streams."""

from __future__ import annotations

import os
import socket
import threading
from typing import Callable

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2


DEFAULT_TELLO_VIDEO_PORT = 11111
DEFAULT_LOCAL_VIDEO_PORT = 12111
LOW_LATENCY_FFMPEG_OPTIONS = "fflags;nobuffer|flags;low_delay|probesize;32768|analyzeduration;0"


PacketCallback = Callable[[str], None]


class UdpVideoDemuxer:
    """Forward Tello video packets from one shared UDP port to per-drone local ports.

    Tello streams normally arrive on UDP 11111. With several station-mode drones,
    all packets can arrive on that same laptop port. The demuxer separates them by
    source IP and forwards each stream to a local-only port for OpenCV.
    """

    def __init__(
        self,
        source_ips: list[str],
        local_ports: dict[str, int],
        tello_video_port: int = DEFAULT_TELLO_VIDEO_PORT,
        on_packet: PacketCallback | None = None,
    ) -> None:
        self.source_ips = set(source_ips)
        self.local_ports = local_ports
        self.tello_video_port = tello_video_port
        self.on_packet = on_packet
        self._stop_event = threading.Event()
        self._receive_socket: socket.socket | None = None
        self._forward_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.stop()
        self._stop_event.clear()

        receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receive_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        receive_socket.bind(("", self.tello_video_port))
        receive_socket.settimeout(0.2)

        self._receive_socket = receive_socket
        self._forward_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for sock in (self._receive_socket, self._forward_socket):
            if sock is None:
                continue
            try:
                sock.close()
            except OSError:
                pass
        self._receive_socket = None
        self._forward_socket = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._thread = None

    def _run(self) -> None:
        receive_socket = self._receive_socket
        forward_socket = self._forward_socket
        if receive_socket is None or forward_socket is None:
            return

        while not self._stop_event.is_set():
            try:
                packet, addr = receive_socket.recvfrom(65535)
            except socket.timeout:
                continue
            except TimeoutError:
                continue
            except OSError:
                break

            source_ip = addr[0]
            if source_ip not in self.source_ips:
                continue
            local_port = self.local_ports.get(source_ip)
            if local_port is None:
                continue

            if self.on_packet is not None:
                self.on_packet(source_ip)
            try:
                forward_socket.sendto(packet, ("127.0.0.1", local_port))
            except OSError:
                break


def open_udp_capture(port: int) -> cv2.VideoCapture | None:
    low_latency_urls = [
        f"udp://@127.0.0.1:{port}?fifo_size=1000000&overrun_nonfatal=1&buffer_size=65536",
        f"udp://@0.0.0.0:{port}?fifo_size=1000000&overrun_nonfatal=1&buffer_size=65536",
        f"udp://@:{port}?fifo_size=1000000&overrun_nonfatal=1&buffer_size=65536",
    ]
    fallback_urls = [
        f"udp://@127.0.0.1:{port}?fifo_size=5000000&overrun_nonfatal=1",
        f"udp://@0.0.0.0:{port}?fifo_size=5000000&overrun_nonfatal=1",
        f"udp://@:{port}?fifo_size=5000000&overrun_nonfatal=1",
        f"udp://@127.0.0.1:{port}",
        f"udp://@0.0.0.0:{port}",
        f"udp://0.0.0.0:{port}",
        f"udp://@:{port}",
    ]

    cap = _try_open_urls(low_latency_urls, ffmpeg_options=LOW_LATENCY_FFMPEG_OPTIONS)
    if cap is not None:
        return cap
    return _try_open_urls(fallback_urls, ffmpeg_options=None)


def _try_open_urls(urls: list[str], ffmpeg_options: str | None) -> cv2.VideoCapture | None:
    previous_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    if ffmpeg_options is not None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_options

    for url in urls:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            _apply_low_latency_capture_settings(cap)
            if previous_options is None and ffmpeg_options is not None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            elif previous_options is not None:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous_options
            return cap
        cap.release()
    if previous_options is None and ffmpeg_options is not None:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    elif previous_options is not None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous_options
    return None


def _apply_low_latency_capture_settings(cap: cv2.VideoCapture) -> None:
    for prop, value in (
        (cv2.CAP_PROP_BUFFERSIZE, 1),
        (cv2.CAP_PROP_FPS, 30),
    ):
        try:
            cap.set(prop, value)
        except Exception:
            pass
