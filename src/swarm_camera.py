"""Camera preview helpers for the swarm dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np

from swarm_utils import TelloUdpClient
from swarm_video import (
    DEFAULT_LOCAL_VIDEO_PORT,
    DEFAULT_TELLO_VIDEO_PORT,
    UdpVideoDemuxer,
    open_udp_capture,
)


STREAM_RETRY_SECONDS = 3.0
STREAM_STALL_SECONDS = 8.0
STREAM_OPEN_TIMEOUT_SECONDS = 12.0
STREAM_DECODER_WARMUP_SECONDS = 0.2
DEFAULT_STREAM_RESOLUTION = "low"
DEFAULT_STREAM_FPS = "low"
DEFAULT_STREAM_BITRATE = 1


@dataclass(frozen=True)
class CameraSnapshot:
    frames: dict[str, np.ndarray]
    frame_versions: dict[str, int]
    frame_timestamps: dict[str, float]


class SwarmCameraWall:
    """Read one live video stream per drone using unique UDP video ports."""

    def __init__(
        self,
        client: TelloUdpClient,
        ips: list[str],
        video_port_start: int = DEFAULT_LOCAL_VIDEO_PORT,
        tello_video_port: int = DEFAULT_TELLO_VIDEO_PORT,
        tile_size: tuple[int, int] = (360, 240),
    ) -> None:
        self.client = client
        self.ips = ips
        self.tello_video_port = tello_video_port
        self.video_ports = {
            ip: video_port_start + index
            for index, ip in enumerate(ips)
        }
        self.tile_size = tile_size
        self.frames: dict[str, np.ndarray] = {}
        self.frame_versions: dict[str, int] = {}
        self.frame_timestamps: dict[str, float] = {}
        self.errors: dict[str, str] = {}
        self.packet_counts: dict[str, int] = {}
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._demuxer: UdpVideoDemuxer | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    def start(self) -> None:
        self.stop()
        self._stop_event.clear()
        with self._lock:
            self.frames = {ip: self._placeholder(ip, "starting stream") for ip in self.ips}
            self.frame_versions = {ip: 0 for ip in self.ips}
            self.frame_timestamps = {ip: time.monotonic() for ip in self.ips}
            self.errors = {}
            self.packet_counts = {ip: 0 for ip in self.ips}

        try:
            self._demuxer = UdpVideoDemuxer(
                source_ips=self.ips,
                local_ports=self.video_ports,
                tello_video_port=self.tello_video_port,
                on_packet=self._mark_packet,
            )
            self._demuxer.start()
        except OSError as error:
            with self._lock:
                for ip in self.ips:
                    self.frames[ip] = self._placeholder(ip, f"video demux failed: {error}")
                    self.errors[ip] = str(error)
            return

        for ip in self.ips:
            thread = threading.Thread(target=self._stream_loop, args=(ip,), daemon=True)
            self._threads[ip] = thread
            thread.start()

    def stop(self) -> None:
        had_streams = bool(self._threads) or bool(self._caps)
        self._stop_event.set()

        if self._demuxer is not None:
            self._demuxer.stop()
            self._demuxer = None

        for cap in list(self._caps.values()):
            try:
                cap.release()
            except Exception:
                pass
        self._caps.clear()

        for thread in list(self._threads.values()):
            if thread.is_alive():
                thread.join(timeout=0.5)
        self._threads.clear()

        if had_streams:
            for ip in self.ips:
                self._stop_drone_stream(ip, wait_response=False)

    def restart(self) -> None:
        self.start()

    def get_frames(self, copy: bool = True) -> dict[str, np.ndarray]:
        with self._lock:
            frames = dict(self.frames)
            for ip in self.ips:
                frames.setdefault(ip, self._placeholder(ip, self.errors.get(ip, "no frame")))
            if copy:
                return {ip: frame.copy() for ip, frame in frames.items()}
            return frames

    def get_snapshot(self, copy: bool = True) -> CameraSnapshot:
        with self._lock:
            frames = dict(self.frames)
            for ip in self.ips:
                frames.setdefault(ip, self._placeholder(ip, self.errors.get(ip, "no frame")))
            if copy:
                frames = {ip: frame.copy() for ip, frame in frames.items()}
            return CameraSnapshot(
                frames=frames,
                frame_versions=dict(self.frame_versions),
                frame_timestamps=dict(self.frame_timestamps),
            )

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads.values())

    def _stream_loop(self, ip: str) -> None:
        video_port = self.video_ports[ip]

        while not self._stop_event.is_set():
            cap: cv2.VideoCapture | None = None
            try:
                self._set_placeholder(ip, f"video {video_port}: configuring")
                self._configure_drone_stream(ip)
                self._set_placeholder(ip, f"video {video_port}: opening decoder")
                cap = self._open_capture_for_new_stream(ip, video_port)
                if cap is None:
                    raise RuntimeError(f"could not open local video port {video_port}")
                with self._lock:
                    self._caps[ip] = cap

                last_frame_at = time.monotonic()
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    now = time.monotonic()
                    if not ok or frame is None:
                        if now - last_frame_at > STREAM_STALL_SECONDS:
                            raise RuntimeError("video stalled; reconnecting")
                        if now - last_frame_at > 3:
                            self._set_placeholder(ip, "waiting for video")
                        time.sleep(0.03)
                        continue

                    last_frame_at = now
                    with self._lock:
                        self.frames[ip] = frame
                        self.frame_versions[ip] = self.frame_versions.get(ip, 0) + 1
                        self.frame_timestamps[ip] = now
                        self.errors.pop(ip, None)
            except Exception as error:
                if self._stop_event.is_set():
                    break
                self._set_placeholder(ip, f"{str(error)[:34]}; retrying")
                self._stop_drone_stream(ip)
                self._stop_event.wait(STREAM_RETRY_SECONDS)
            finally:
                if cap is not None:
                    cap.release()
                with self._lock:
                    if self._caps.get(ip) is cap:
                        self._caps.pop(ip, None)

        self._stop_drone_stream(ip, wait_response=False)

    def _open_capture_for_new_stream(self, ip: str, video_port: int) -> cv2.VideoCapture | None:
        """Open the decoder before streamon so FFmpeg sees the initial SPS/PPS packets."""
        result: dict[str, cv2.VideoCapture | None] = {"cap": None}
        done = threading.Event()
        abandoned = threading.Event()

        def open_capture() -> None:
            try:
                cap = open_udp_capture(video_port)
                if abandoned.is_set() and cap is not None:
                    cap.release()
                    return
                result["cap"] = cap
            finally:
                done.set()

        thread = threading.Thread(target=open_capture, daemon=True)
        thread.start()
        self._stop_event.wait(STREAM_DECODER_WARMUP_SECONDS)
        self._start_drone_stream(ip)

        deadline = time.monotonic() + STREAM_OPEN_TIMEOUT_SECONDS
        while not self._stop_event.is_set():
            if done.wait(0.05):
                return result.get("cap")
            if time.monotonic() >= deadline:
                abandoned.set()
                self._stop_drone_stream(ip, wait_response=False)
                return None
        abandoned.set()
        return None

    def _mark_packet(self, ip: str) -> None:
        with self._lock:
            count = self.packet_counts.get(ip, 0) + 1
            self.packet_counts[ip] = count
            if count == 1 or (count % 300 == 0 and ip in self.errors):
                self.errors[ip] = "receiving h264"
                self.frames[ip] = self._placeholder(ip, "receiving h264")
                self.frame_timestamps[ip] = time.monotonic()

    def _stop_drone_stream(self, ip: str, wait_response: bool = True, timeout: float = 3) -> None:
        try:
            if wait_response:
                self.client.send_one(ip, "streamoff", timeout=timeout, retries=1, retry_pause=0, verbose=False)
            else:
                self.client.send_no_wait(ip, "streamoff", verbose=False)
        except Exception:
            pass

    def _configure_drone_stream(self, ip: str) -> None:
        self.client.send_one(ip, "command", timeout=4, retries=3, verbose=False)
        self._stop_drone_stream(ip, wait_response=True, timeout=1)
        self._try_send(ip, f"setresolution {DEFAULT_STREAM_RESOLUTION}")
        self._try_send(ip, f"setfps {DEFAULT_STREAM_FPS}")
        self._try_send(ip, f"setbitrate {DEFAULT_STREAM_BITRATE}")

    def _start_drone_stream(self, ip: str) -> None:
        self.client.send_one(ip, "streamon", timeout=5, retries=3, verbose=False)

    def _try_send(self, ip: str, command: str) -> None:
        try:
            self.client.send_one(ip, command, timeout=2, retries=1, verbose=False)
        except Exception:
            pass

    def _set_placeholder(self, ip: str, message: str) -> None:
        with self._lock:
            self.errors[ip] = message
            self.frames[ip] = self._placeholder(ip, message)
            self.frame_timestamps[ip] = time.monotonic()

    def _placeholder(self, ip: str, message: str) -> np.ndarray:
        width, height = self.tile_size
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (22, 25, 28)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (72, 82, 88), 2)
        cv2.putText(frame, ip, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 240, 236), 1, cv2.LINE_AA)
        cv2.putText(frame, message[:42], (14, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 182, 182), 1, cv2.LINE_AA)
        cv2.putText(frame, "live stream pending", (14, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 130), 1, cv2.LINE_AA)
        return frame
