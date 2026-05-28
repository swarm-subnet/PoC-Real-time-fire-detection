"""Shared image and video recording helpers for live camera scripts."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoRecording:
    writer: cv2.VideoWriter
    final_path: Path
    writer_path: Path
    frame_size: tuple[int, int]
    fps: float
    copy_to_final: bool
    frames_written: int = 0
    last_write_time: float | None = None


def save_frame_once_per_second(
    frame_bgr,
    save_dir: Path,
    drone_ip: str,
    saved_second: int | None,
) -> tuple[int | None, int]:
    """Save one annotated image per wall-clock second."""
    current_second = int(time.time())
    if current_second == saved_second:
        return saved_second, 0

    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{drone_ip.replace('.', '-')}.jpg"
    if not cv2.imwrite(str(output_path), frame_bgr):
        raise RuntimeError(f"Could not save frame to {output_path}")
    print(f"[{time.strftime('%H:%M:%S')}] Saved {output_path}")
    return current_second, 1


def is_windows_unc_path(path: Path) -> bool:
    return os.name == "nt" and str(path).startswith("\\\\")


def choose_video_writer_path(final_path: Path, temp_subdir: str) -> tuple[Path, bool]:
    """Avoid streaming MP4 writes directly to WSL/network UNC paths on Windows."""
    if not is_windows_unc_path(final_path):
        return final_path, False

    temp_dir = Path(tempfile.gettempdir()) / "tello_drone" / temp_subdir
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / final_path.name, True


def open_cv_video_writer(path: Path, fps: float, frame_size: tuple[int, int], codec: str) -> cv2.VideoWriter | None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), max(1.0, fps), frame_size)
    if writer.isOpened():
        return writer

    writer.release()
    return None


def create_video_recording(
    video_dir: Path,
    drone_ip: str,
    frame_bgr,
    fps: float,
    temp_subdir: str,
) -> VideoRecording:
    """Create a video writer for annotated frames."""
    video_dir.mkdir(parents=True, exist_ok=True)
    height, width = frame_bgr.shape[:2]
    final_path = video_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{drone_ip.replace('.', '-')}.mp4"
    writer_path, copy_to_final = choose_video_writer_path(final_path, temp_subdir)
    writer_path.parent.mkdir(parents=True, exist_ok=True)
    frame_size = (width, height)

    writer = open_cv_video_writer(writer_path, fps, frame_size, "mp4v")
    if writer is None:
        final_path = final_path.with_suffix(".avi")
        writer_path = writer_path.with_suffix(".avi")
        writer = open_cv_video_writer(writer_path, fps, frame_size, "MJPG")

    if writer is None:
        raise RuntimeError(f"Could not create video writer: {final_path}")

    if copy_to_final:
        print(f"Windows UNC video path detected. Recording to local temp first: {writer_path}")

    return VideoRecording(
        writer=writer,
        final_path=final_path,
        writer_path=writer_path,
        frame_size=frame_size,
        fps=max(1.0, fps),
        copy_to_final=copy_to_final,
    )


def write_video_frame(recording: VideoRecording, frame_bgr) -> None:
    """Write frames with real-time pacing so saved video does not play too fast."""
    expected_width, expected_height = recording.frame_size
    if frame_bgr.shape[1] != expected_width or frame_bgr.shape[0] != expected_height:
        frame_bgr = cv2.resize(frame_bgr, recording.frame_size)

    now = time.monotonic()
    frame_interval = 1.0 / recording.fps

    if recording.last_write_time is None:
        frames_to_write = 1
        recording.last_write_time = now
    else:
        elapsed = now - recording.last_write_time
        if elapsed < frame_interval:
            return

        frames_to_write = max(1, int(round(elapsed / frame_interval)))
        frames_to_write = min(frames_to_write, 5)
        recording.last_write_time += frames_to_write * frame_interval

    for _ in range(frames_to_write):
        recording.writer.write(frame_bgr)
        recording.frames_written += 1


def close_video_recording(recording: VideoRecording) -> None:
    """Close a recording and copy it back from temp storage if needed."""
    recording.writer.release()

    if recording.copy_to_final:
        try:
            recording.final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(recording.writer_path, recording.final_path)
            recording.writer_path.unlink(missing_ok=True)
        except Exception as error:
            print(f"Could not copy video to {recording.final_path}: {error}")
            print(f"Temporary video kept at: {recording.writer_path}")
            return

    print(f"Saved annotated video: {recording.final_path} ({recording.frames_written} frames)")
