"""Fire-following Tello agent with conservative flight limits.

Default behavior is a dry run: it opens the camera, detects fire, draws
the chosen target, records annotated video, and prints the command it would run.
Pass --enable-flight only after the dry-run behavior looks correct.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from djitellopy import Tello


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_IP_FILE = ROOT_DIR / "scripts" / "swarm" / "drone_ips.txt"
DEFAULT_SAVE_DIR = ROOT_DIR / "captures" / "fire_agent"
DEFAULT_VIDEO_DIR = ROOT_DIR / "captures" / "fire_agent_videos"
DEFAULT_RECORD_FPS = 20.0
DEFAULT_DETECTION_PROFILE = "candle"
DEFAULT_DETECT_EVERY_FRAMES = 5
DEFAULT_HOLD_SECONDS = 2.0
DEFAULT_CENTER_CROP = 0.70
DEFAULT_DECISION_EVERY_SECONDS = 3.0
DEFAULT_FORWARD_STEP_CM = 50
DEFAULT_MAX_FORWARD_CM = 400
DEFAULT_MAX_NO_FIRE_FORWARDS = 2
DEFAULT_LOST_FIRE_LAND_SECONDS = 5.0
DEFAULT_YAW_STEP_DEGREES = 10
DEFAULT_CENTER_TOLERANCE_PERCENT = 28.0
DEFAULT_TARGET_COVERAGE_PERCENT = 5.0
MIN_BATTERY_PERCENT = 30

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_utils import (  # noqa: E402
    FIRE_MODEL_REPO_ID,
    FireOnnxDetector,
    get_fire_model_path,
    keep_fire_detections,
    predict_center_zoom,
)
from chutes_agent import (  # noqa: E402
    DEFAULT_CHUTES_MODEL,
    AgentDecision,
    get_chutes_api_key,
    load_dotenv_if_present,
    request_fire_command_from_chutes,
)
from media_utils import (  # noqa: E402
    VideoRecording,
    close_video_recording,
    create_video_recording,
    save_frame_once_per_second,
    write_video_frame,
)
from tello_stream import (  # noqa: E402
    build_candidate_ips as build_stream_candidate_ips,
    connect_first_streaming_drone,
    resolve_path as resolve_root_path,
    stop_streaming_tello,
)
from detection_utils import Detection, draw_detections  # noqa: E402


WINDOW_NAME = "Tello Fire Agent"


@dataclass
class FireTargetContext:
    target: Detection | None
    hazard_found: bool
    fire_count: int
    smoke_count: int
    center_offset_x_percent: float = 0.0
    center_offset_y_percent: float = 0.0
    coverage_percent: float = 0.0
    confidence: float = 0.0
    label: str = ""
    target_age_seconds: float = 0.0


@dataclass
class FireCommandDecision:
    command: str
    reason: str
    mission_complete: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect fire and cautiously fly a Tello toward it with a hard forward-distance cap."
    )
    parser.add_argument(
        "--ip",
        action="append",
        help="Drone IP to try. Can be passed multiple times. If omitted, uses scripts/swarm/drone_ips.txt.",
    )
    parser.add_argument("--ip-file", default=str(DEFAULT_IP_FILE), help="Text file with one drone IP per line.")
    parser.add_argument("--no-direct-fallback", action="store_true", help="Do not try 192.168.10.1 as fallback.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Local .onnx model path. If omitted, uses the cached/downloaded SuperBitDev/fire1 ONNX model.",
    )
    parser.add_argument("--repo-id", default=FIRE_MODEL_REPO_ID, help="Hugging Face model repo ID.")
    parser.add_argument(
        "--profile",
        choices=("candle", "miner"),
        default=DEFAULT_DETECTION_PROFILE,
        help="'candle' is more sensitive for small flames; 'miner' matches the published miner thresholds.",
    )
    parser.add_argument("--include-smoke", action="store_true", help="Also allow smoke boxes as fallback targets.")
    parser.add_argument(
        "--detect-every",
        type=int,
        default=DEFAULT_DETECT_EVERY_FRAMES,
        help="Run fire inference every N frames. Default: 5.",
    )
    parser.add_argument(
        "--center-zoom",
        action="store_true",
        help="Run inference on the center crop instead of the full frame.",
    )
    parser.add_argument(
        "--center-crop",
        type=float,
        default=DEFAULT_CENTER_CROP,
        help="Fraction of frame width/height used when --center-zoom is enabled. Default: 0.70.",
    )
    parser.add_argument(
        "--center-include-full-frame",
        action="store_true",
        help="Also run a full-frame pass when center-zoom is enabled. Slower but safer if the target may leave center.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=DEFAULT_HOLD_SECONDS,
        help="Keep using the last positive detection briefly to reduce flicker. Default: 2.0.",
    )
    parser.add_argument(
        "--decision-every",
        type=float,
        default=DEFAULT_DECISION_EVERY_SECONDS,
        help="Seconds between Chutes movement decisions. Default: 3.",
    )
    parser.add_argument(
        "--chutes-model",
        default=DEFAULT_CHUTES_MODEL,
        help="Chutes LLM model ID.",
    )
    parser.add_argument(
        "--chutes-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for one Chutes response. Default: 60.",
    )
    parser.add_argument(
        "--forward-step-cm",
        type=int,
        default=DEFAULT_FORWARD_STEP_CM,
        help="Forward movement per centered decision. Tello minimum is 20 cm. Default: 50.",
    )
    parser.add_argument(
        "--max-forward-cm",
        type=int,
        default=DEFAULT_MAX_FORWARD_CM,
        help="Hard cap for total forward movement before landing. Default: 400 cm.",
    )
    parser.add_argument(
        "--max-no-fire-forwards",
        type=int,
        default=DEFAULT_MAX_NO_FIRE_FORWARDS,
        help="Land if no fire is found after this many forward search moves. Default: 2.",
    )
    parser.add_argument(
        "--lost-fire-land-seconds",
        type=float,
        default=DEFAULT_LOST_FIRE_LAND_SECONDS,
        help="After fire was seen, land if it stays out of view for this many seconds. Default: 5.",
    )
    parser.add_argument(
        "--yaw-step",
        type=int,
        default=DEFAULT_YAW_STEP_DEGREES,
        help="Yaw correction in degrees when the fire is left/right of center. Default: 10.",
    )
    parser.add_argument(
        "--center-tolerance",
        type=float,
        default=DEFAULT_CENTER_TOLERANCE_PERCENT,
        help="Horizontal offset percentage considered centered. Default: 28.",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=DEFAULT_TARGET_COVERAGE_PERCENT,
        help="Land early when the selected hazard box covers at least this percentage of the frame. Default: 5.",
    )
    parser.add_argument("--min-battery", type=int, default=MIN_BATTERY_PERCENT, help="Minimum battery for flight.")
    parser.add_argument("--enable-flight", action="store_true", help="Actually take off and execute movement commands.")
    parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_SAVE_DIR),
        help="Directory where annotated frames are saved once per second.",
    )
    parser.add_argument(
        "--video-dir",
        default=str(DEFAULT_VIDEO_DIR),
        help="Directory where the annotated MP4 recording is saved.",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=DEFAULT_RECORD_FPS,
        help="FPS to write into the annotated video file. Default: 20.",
    )
    parser.add_argument("--no-record-video", action="store_true", help="Disable annotated video recording.")
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for video frames before trying the next IP.",
    )
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    return resolve_root_path(ROOT_DIR, path_text)


def build_candidate_ips(args: argparse.Namespace) -> list[str]:
    return build_stream_candidate_ips(
        ROOT_DIR,
        args.ip,
        args.ip_file,
        include_direct_fallback=not args.no_direct_fallback,
    )


def detect_fire(
    model: FireOnnxDetector,
    frame_bgr,
    include_smoke: bool,
    center_zoom: bool,
    center_crop: float,
    center_include_full_frame: bool,
) -> list[Detection]:
    if center_zoom:
        detections = predict_center_zoom(
            model,
            frame_bgr,
            crop_fraction=center_crop,
            include_full_frame=center_include_full_frame,
        )
    else:
        detections = model.predict(frame_bgr)
    return keep_fire_detections(detections, include_smoke=include_smoke)


def run_fire_detection(
    model,
    frame_bgr,
    include_smoke: bool,
    center_zoom: bool,
    center_crop: float,
    center_include_full_frame: bool,
) -> tuple[list[Detection], float]:
    inference_start = time.perf_counter()
    detections = detect_fire(
        model,
        frame_bgr,
        include_smoke=include_smoke,
        center_zoom=center_zoom,
        center_crop=center_crop,
        center_include_full_frame=center_include_full_frame,
    )
    inference_ms = (time.perf_counter() - inference_start) * 1000
    return detections, inference_ms


def count_hazards(detections: list[Detection]) -> tuple[int, int]:
    fire_count = sum(1 for detection in detections if detection.label.lower() == "fire")
    smoke_count = sum(1 for detection in detections if detection.label.lower() == "smoke")
    return fire_count, smoke_count


def detection_area(detection: Detection) -> int:
    x1, y1, x2, y2 = detection.xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)


def select_target_hazard(detections: list[Detection]) -> Detection | None:
    """Prefer fire, then pick the largest/most confident box."""
    if not detections:
        return None

    return max(
        detections,
        key=lambda detection: (
            1 if detection.label.lower() == "fire" else 0,
            detection_area(detection),
            detection.confidence,
        ),
    )


def build_fire_context(frame_bgr, detections: list[Detection], target_age_seconds: float) -> FireTargetContext:
    frame_height, frame_width = frame_bgr.shape[:2]
    fire_count, smoke_count = count_hazards(detections)
    target = select_target_hazard(detections)
    context = FireTargetContext(
        target=target,
        hazard_found=target is not None,
        fire_count=fire_count,
        smoke_count=smoke_count,
        target_age_seconds=target_age_seconds,
    )
    if target is None:
        return context

    x1, y1, x2, y2 = target.xyxy
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    box_area = box_width * box_height
    frame_area = frame_width * frame_height
    center_x = x1 + box_width / 2
    center_y = y1 + box_height / 2

    context.center_offset_x_percent = ((center_x - frame_width / 2) / (frame_width / 2)) * 100
    context.center_offset_y_percent = ((center_y - frame_height / 2) / (frame_height / 2)) * 100
    context.coverage_percent = (box_area / frame_area) * 100 if frame_area else 0.0
    context.confidence = target.confidence
    context.label = target.label
    return context


def build_fire_agent_context(
    frame_bgr,
    context: FireTargetContext,
    total_forward_cm: int,
    max_forward_cm: int,
    no_fire_forward_count: int,
    max_no_fire_forwards: int,
    target_coverage_percent: float,
    center_tolerance_percent: float,
    has_taken_off: bool,
    last_command: str,
) -> dict[str, Any]:
    frame_height, frame_width = frame_bgr.shape[:2]
    remaining_forward_cm = max(0, max_forward_cm - total_forward_cm)
    payload: dict[str, Any] = {
        "target": "fire",
        "frame_width": frame_width,
        "frame_height": frame_height,
        "airborne": has_taken_off,
        "last_command": last_command,
        "hazard_found": context.hazard_found,
        "fire_count": context.fire_count,
        "smoke_count": context.smoke_count,
        "target_coverage_threshold_percent": target_coverage_percent,
        "center_deadband_percent": center_tolerance_percent,
        "total_forward_cm": total_forward_cm,
        "max_forward_cm": max_forward_cm,
        "remaining_forward_cm": remaining_forward_cm,
        "no_fire_forward_count": no_fire_forward_count,
        "max_no_fire_forwards": max_no_fire_forwards,
        "no_fire_forward_attempts_remaining": max(0, max_no_fire_forwards - no_fire_forward_count),
    }

    if context.target is None:
        return payload

    x1, y1, x2, y2 = context.target.xyxy
    payload.update(
        {
            "target_label": context.label,
            "target_confidence": round(context.confidence, 3),
            "target_coordinate": {
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
            },
            "target_center": {
                "x": round((x1 + x2) / 2, 1),
                "y": round((y1 + y2) / 2, 1),
            },
            "center_offset_x_percent": round(context.center_offset_x_percent, 1),
            "center_offset_y_percent": round(context.center_offset_y_percent, 1),
            "object_coverage_percentage": round(context.coverage_percent, 2),
            "near_enough": context.coverage_percent >= target_coverage_percent,
            "target_age_seconds": round(context.target_age_seconds, 2),
        }
    )
    return payload


def enforce_fire_safety_limits(
    command: str,
    hazard_found: bool,
    total_forward_cm: int,
    max_forward_cm: int,
    no_fire_forward_count: int,
    max_no_fire_forwards: int,
    forward_step_cm: int,
    yaw_step_degrees: int,
) -> FireCommandDecision:
    """Clamp Chutes output to local movement budget and landing rules."""
    if total_forward_cm >= max_forward_cm:
        return FireCommandDecision("land", "forward budget reached", mission_complete=True)

    remaining_forward = max_forward_cm - total_forward_cm
    local_forward_step = max(20, min(50, forward_step_cm))
    if not hazard_found:
        if no_fire_forward_count >= max_no_fire_forwards:
            return FireCommandDecision("land", "no fire after forward search limit", mission_complete=True)
        if remaining_forward < 20:
            return FireCommandDecision("land", "remaining forward budget below Tello minimum", mission_complete=True)

        step = max(20, min(local_forward_step, remaining_forward))
        return FireCommandDecision(f"forward {step}", "no fire detected; searching forward")

    parts = command.split()
    if len(parts) == 2 and parts[0] in {"cw", "ccw"}:
        yaw_step_degrees = max(1, min(30, yaw_step_degrees))
        if command != f"{parts[0]} {yaw_step_degrees}":
            return FireCommandDecision(f"{parts[0]} {yaw_step_degrees}", "Chutes yaw direction accepted; angle set locally")

    if len(parts) == 2 and parts[0] == "back":
        return FireCommandDecision("stop", "back is disabled for fire approach")

    if len(parts) == 2 and parts[0] == "forward":
        if remaining_forward < 20:
            return FireCommandDecision("land", "remaining forward budget below Tello minimum", mission_complete=True)

        requested = parse_forward_distance_cm(command)
        clamped = max(20, min(50, requested, local_forward_step, remaining_forward))
        if clamped != requested:
            return FireCommandDecision(f"forward {clamped}", f"Chutes forward clamped to remaining budget {remaining_forward} cm")

    return FireCommandDecision(command, "Chutes command accepted")


def parse_forward_distance_cm(command: str) -> int:
    parts = command.split()
    if len(parts) == 2 and parts[0] == "forward":
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


def draw_target_marker(frame_bgr, target: Detection | None) -> None:
    if target is None:
        return

    x1, y1, x2, y2 = target.xyxy
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(
        frame_bgr,
        "TRACKING FIRE TARGET",
        (x1, min(frame_bgr.shape[0] - 10, y2 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_status_overlay(
    frame_bgr,
    drone_ip: str,
    battery: int | None,
    context: FireTargetContext,
    command: str,
    reason: str,
    total_forward_cm: int,
    max_forward_cm: int,
    no_fire_forward_count: int,
    max_no_fire_forwards: int,
    inference_ms: float | None,
    agent_latency_ms: float | None,
    enable_flight: bool,
) -> None:
    drone_name = f"swarm_drone_{drone_ip.rsplit('.', 1)[-1]}"
    mode = "FOLLOW_FIRE" if enable_flight else "DRY_RUN FOLLOW_FIRE"
    inference_text = f"{inference_ms:.0f} ms" if inference_ms is not None else "waiting"
    chutes_text = f"{agent_latency_ms:.0f} ms" if agent_latency_ms is not None else "waiting"
    target_text = (
        f"{context.label} conf={context.confidence:.2f} "
        f"x={context.center_offset_x_percent:.1f}% size={context.coverage_percent:.2f}%"
        if context.hazard_found
        else "no fresh target"
    )

    overlay_height = 128
    y0 = max(0, frame_bgr.shape[0] - overlay_height)
    cv2.rectangle(frame_bgr, (0, y0), (frame_bgr.shape[1], frame_bgr.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        f"Mode: {mode}  Chutes Command: {command}",
        (12, y0 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"Drone: {drone_name}  Battery: {battery if battery is not None else '?'}%  ONNX: {inference_text}  Chutes: {chutes_text}",
        (12, y0 + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"Fire: {context.fire_count}  Forward: {total_forward_cm}/{max_forward_cm} cm",
        (12, y0 + 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"{target_text}  Search: {no_fire_forward_count}/{max_no_fire_forwards}  Reason: {reason}  Press q to quit",
        (12, y0 + 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def execute_control_command(tello: Tello, command: str) -> str:
    print(f"[{time.strftime('%H:%M:%S')}] Executing: {command}")
    tello.send_control_command(command)
    return command


def finish_fire_mission(tello: Tello) -> str:
    print(f"[{time.strftime('%H:%M:%S')}] Fire target reached. Stopping before landing...")
    try:
        tello.send_control_command("stop", timeout=3)
    except Exception as stop_error:
        print(f"Stop command failed before landing: {stop_error}")

    print(f"[{time.strftime('%H:%M:%S')}] Landing after fire approach...")
    tello.land()
    return "landed"


def stop_motion_if_flying(tello: Tello | None, has_taken_off: bool) -> None:
    if tello is None or not has_taken_off:
        return

    try:
        print("Stopping drone motion...")
        tello.send_control_command("stop", timeout=3)
    except Exception as stop_error:
        print(f"Stop command failed: {stop_error}")


def print_context_summary(context: FireTargetContext, decision: FireCommandDecision, total_forward_cm: int) -> None:
    if not context.hazard_found:
        print(f"[{time.strftime('%H:%M:%S')}] No fresh fire target. command={decision.command!r}")
        return

    print(
        f"[{time.strftime('%H:%M:%S')}] Target={context.label} conf={context.confidence:.2f} "
        f"x_offset={context.center_offset_x_percent:.1f}% coverage={context.coverage_percent:.2f}% "
        f"forward={total_forward_cm}cm command={decision.command!r} reason={decision.reason}"
    )


def main() -> None:
    args = parse_args()
    load_dotenv_if_present(ROOT_DIR / ".env")
    if args.chutes_model == DEFAULT_CHUTES_MODEL:
        args.chutes_model = os.environ.get("CHUTES_MODEL", args.chutes_model)
    api_key = get_chutes_api_key()

    candidate_ips = build_candidate_ips(args)
    save_dir = resolve_path(args.save_dir)
    video_dir = resolve_path(args.video_dir)
    model_path = resolve_path(args.model) if args.model is not None else get_fire_model_path(args.repo_id)
    if model_path.suffix.lower() != ".onnx":
        raise ValueError("The fire PoC expects an ONNX model. Use weights.onnx from SuperBitDev/fire1.")
    center_zoom = args.center_zoom

    tello = None
    frame_read = None
    has_taken_off = False
    latest_detections: list[Detection] = []
    last_positive_detections: list[Detection] = []
    last_positive_time = 0.0
    latest_context = FireTargetContext(None, False, 0, 0)
    latest_inference_ms: float | None = None
    latest_agent_latency_ms: float | None = None
    latest_raw_response = ""
    latest_command = "none"
    latest_reason = "waiting"
    last_executed_command = ""
    total_forward_cm = 0
    no_fire_forward_count = 0
    frame_count = 0
    saved_second: int | None = None
    saved_count = 0
    next_decision_time = 0.0
    has_inference_result = False
    has_seen_fire = False
    fire_lost_since: float | None = None
    mission_completed = False
    finish_after_record = False
    user_requested_stop = False
    pending_future: concurrent.futures.Future[tuple[list[Detection], float]] | None = None
    agent_future: concurrent.futures.Future[AgentDecision] | None = None
    movement_future: concurrent.futures.Future[str] | None = None
    movement_kind = ""
    movement_was_no_fire_search = False
    video_recording: VideoRecording | None = None

    detector_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    command_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        print(f"Loading fire model: {model_path}")
        model = FireOnnxDetector(model_path, profile=args.profile)
        print(f"Model labels: {model.class_names}")
        print(f"Detection profile: {model.profile}")

        print(f"Using Chutes model: {args.chutes_model}")
        tello, frame_read, drone_ip, battery = connect_first_streaming_drone(
            candidate_ips,
            frame_timeout=args.frame_timeout,
            usage_label="for fire agent",
        )

        print("Fire-agent mode. Start with dry run; pass --enable-flight only after the video and commands look correct.")
        detection_mode = f"center-zoom crop={args.center_crop:.2f}" if center_zoom else "full-frame"
        print(
            f"Controller: detection={detection_mode}, decision-every={args.decision_every:.1f}s, "
            f"forward-step={args.forward_step_cm}cm, max-forward={args.max_forward_cm}cm, "
            f"max-no-fire-forwards={args.max_no_fire_forwards}."
        )

        if args.enable_flight:
            if battery is None:
                raise RuntimeError("Battery status unavailable. Refusing to fly.")
            if battery < args.min_battery:
                raise RuntimeError(f"Battery too low for flight: {battery}%. Need at least {args.min_battery}%.")

            print("Flight enabled. Taking off...")
            tello.takeoff()
            has_taken_off = True
            time.sleep(2)
        else:
            print("DRY RUN mode. The script will NOT take off or execute movement commands.")

        print(f"Saving annotated frames to: {save_dir}")
        if args.no_record_video:
            print("Annotated video recording is disabled.")
        else:
            print(f"Saving annotated video to: {video_dir}")
        print("Press 'q' in the video window to quit.")

        while True:
            if movement_future is not None and movement_future.done():
                try:
                    completed_command = movement_future.result()
                    if movement_kind == "finish":
                        has_taken_off = False
                        print(f"[{time.strftime('%H:%M:%S')}] Landing completed.")
                        break

                    moved_forward = parse_forward_distance_cm(completed_command)
                    total_forward_cm += moved_forward
                    if movement_was_no_fire_search and moved_forward > 0:
                        no_fire_forward_count += 1
                    last_executed_command = completed_command
                    print(f"[{time.strftime('%H:%M:%S')}] Command completed: {completed_command}")
                    next_decision_time = time.time() + args.decision_every
                except Exception as movement_error:
                    print(f"[{time.strftime('%H:%M:%S')}] Drone command failed: {movement_error}")
                    user_requested_stop = True
                    break
                finally:
                    movement_future = None
                    movement_kind = ""
                    movement_was_no_fire_search = False

            if pending_future is not None and pending_future.done():
                try:
                    latest_detections, latest_inference_ms = pending_future.result()
                    has_inference_result = True
                    if latest_detections:
                        has_seen_fire = True
                        fire_lost_since = None
                        last_positive_detections = latest_detections
                        last_positive_time = time.monotonic()
                    elif has_seen_fire and fire_lost_since is None:
                        fire_lost_since = time.monotonic()
                    fire_count, smoke_count = count_hazards(latest_detections)
                    if args.include_smoke:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] Fire: {fire_count}  Smoke: {smoke_count}  "
                            f"ONNX: {latest_inference_ms:.0f} ms"
                        )
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] Fire: {fire_count}  ONNX: {latest_inference_ms:.0f} ms")
                except Exception as error:
                    print(f"[{time.strftime('%H:%M:%S')}] Fire inference failed: {error}")
                finally:
                    pending_future = None

            if agent_future is not None and agent_future.done():
                try:
                    agent_decision = agent_future.result()
                    latest_raw_response = agent_decision.raw_response
                    latest_agent_latency_ms = agent_decision.latency_ms
                    safety_decision = enforce_fire_safety_limits(
                        agent_decision.command,
                        hazard_found=latest_context.hazard_found,
                        total_forward_cm=total_forward_cm,
                        max_forward_cm=args.max_forward_cm,
                        no_fire_forward_count=no_fire_forward_count,
                        max_no_fire_forwards=args.max_no_fire_forwards,
                        forward_step_cm=args.forward_step_cm,
                        yaw_step_degrees=args.yaw_step,
                    )
                    latest_command = safety_decision.command
                    latest_reason = safety_decision.reason
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Chutes raw={latest_raw_response!r} "
                        f"validated={agent_decision.command!r} final={latest_command!r} "
                        f"latency={latest_agent_latency_ms:.0f} ms"
                    )

                    if safety_decision.mission_complete:
                        mission_completed = True
                        finish_after_record = True
                    elif args.enable_flight:
                        movement_future = command_executor.submit(execute_control_command, tello, latest_command)
                        movement_kind = "command"
                        movement_was_no_fire_search = not latest_context.hazard_found and parse_forward_distance_cm(latest_command) > 0
                    else:
                        moved_forward = parse_forward_distance_cm(latest_command)
                        total_forward_cm += moved_forward
                        if not latest_context.hazard_found and moved_forward > 0:
                            no_fire_forward_count += 1
                        last_executed_command = latest_command
                        print(f"[{time.strftime('%H:%M:%S')}] DRY RUN command: {latest_command}")
                        next_decision_time = time.time() + args.decision_every
                except Exception as error:
                    latest_command = "stop"
                    latest_raw_response = f"agent error: {error}"
                    latest_reason = "Chutes failed; stopping"
                    print(f"[{time.strftime('%H:%M:%S')}] Chutes failed: {error}")
                    if args.enable_flight and movement_future is None:
                        movement_future = command_executor.submit(execute_control_command, tello, "stop")
                        movement_kind = "command"
                        movement_was_no_fire_search = False
                    else:
                        next_decision_time = time.time() + args.decision_every
                finally:
                    agent_future = None

            frame_rgb = frame_read.frame
            if frame_rgb is None:
                time.sleep(0.05)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_count += 1

            if pending_future is None and frame_count % max(1, args.detect_every) == 0:
                pending_future = detector_executor.submit(
                    run_fire_detection,
                    model,
                    frame_bgr.copy(),
                    args.include_smoke,
                    center_zoom,
                    args.center_crop,
                    args.center_include_full_frame,
                )

            use_held_detection = not has_seen_fire
            visible_detections = latest_detections
            target_age = 0.0
            if use_held_detection and not visible_detections and time.monotonic() - last_positive_time <= args.hold_seconds:
                visible_detections = last_positive_detections
                target_age = time.monotonic() - last_positive_time

            latest_context = build_fire_context(frame_bgr, visible_detections, target_age)
            lost_fire_seconds = time.monotonic() - fire_lost_since if fire_lost_since is not None else 0.0
            if latest_context.hazard_found and no_fire_forward_count:
                print(f"[{time.strftime('%H:%M:%S')}] Fire found. Resetting no-fire search counter.")
                no_fire_forward_count = 0
            if latest_context.hazard_found:
                fire_lost_since = None
                lost_fire_seconds = 0.0
            elif has_seen_fire and fire_lost_since is None:
                fire_lost_since = time.monotonic()
                lost_fire_seconds = 0.0
            now = time.time()

            if (
                not mission_completed
                and movement_future is None
                and agent_future is None
                and now >= next_decision_time
                and has_inference_result
            ):
                if latest_context.coverage_percent >= args.target_coverage:
                    decision = FireCommandDecision("land", "target visual size reached", mission_complete=True)
                    latest_command = decision.command
                    latest_reason = decision.reason
                    print_context_summary(latest_context, decision, total_forward_cm)
                    mission_completed = True
                    finish_after_record = True
                elif total_forward_cm >= args.max_forward_cm:
                    decision = FireCommandDecision("land", "forward budget reached", mission_complete=True)
                    latest_command = decision.command
                    latest_reason = decision.reason
                    print_context_summary(latest_context, decision, total_forward_cm)
                    mission_completed = True
                    finish_after_record = True
                elif has_seen_fire and not latest_context.hazard_found and lost_fire_seconds >= args.lost_fire_land_seconds:
                    decision = FireCommandDecision(
                        "land",
                        f"fire lost for {lost_fire_seconds:.1f}s",
                        mission_complete=True,
                    )
                    latest_command = decision.command
                    latest_reason = decision.reason
                    print_context_summary(latest_context, decision, total_forward_cm)
                    mission_completed = True
                    finish_after_record = True
                elif has_seen_fire and not latest_context.hazard_found:
                    latest_command = "stop"
                    latest_reason = f"fire lost {lost_fire_seconds:.1f}/{args.lost_fire_land_seconds:.1f}s"
                    print(f"[{time.strftime('%H:%M:%S')}] {latest_reason}; waiting before landing.")
                    if args.enable_flight and movement_future is None:
                        movement_future = command_executor.submit(execute_control_command, tello, "stop")
                        movement_kind = "command"
                        movement_was_no_fire_search = False
                    else:
                        next_decision_time = time.time() + 1.0
                elif not latest_context.hazard_found and no_fire_forward_count >= args.max_no_fire_forwards:
                    decision = FireCommandDecision("land", "no fire after forward search limit", mission_complete=True)
                    latest_command = decision.command
                    latest_reason = decision.reason
                    print_context_summary(latest_context, decision, total_forward_cm)
                    mission_completed = True
                    finish_after_record = True
                else:
                    agent_context = build_fire_agent_context(
                        frame_bgr,
                        latest_context,
                        total_forward_cm=total_forward_cm,
                        max_forward_cm=args.max_forward_cm,
                        no_fire_forward_count=no_fire_forward_count,
                        max_no_fire_forwards=args.max_no_fire_forwards,
                        target_coverage_percent=args.target_coverage,
                        center_tolerance_percent=args.center_tolerance,
                        has_taken_off=has_taken_off,
                        last_command=last_executed_command,
                    )
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Asking Chutes: "
                        f"target={latest_context.label or 'none'} "
                        f"x_offset={latest_context.center_offset_x_percent:.1f}% "
                        f"coverage={latest_context.coverage_percent:.2f}% "
                        f"remaining={args.max_forward_cm - total_forward_cm}cm "
                        f"search={no_fire_forward_count}/{args.max_no_fire_forwards}"
                    )
                    agent_future = command_executor.submit(
                        request_fire_command_from_chutes,
                        agent_context,
                        api_key,
                        args.chutes_model,
                        args.chutes_timeout,
                    )

            display_frame = draw_detections(frame_bgr, visible_detections)
            draw_target_marker(display_frame, latest_context.target)
            draw_status_overlay(
                display_frame,
                drone_ip,
                battery,
                latest_context,
                latest_command,
                latest_reason,
                total_forward_cm,
                args.max_forward_cm,
                no_fire_forward_count,
                args.max_no_fire_forwards,
                latest_inference_ms,
                latest_agent_latency_ms,
                args.enable_flight,
            )

            if not args.no_record_video:
                if video_recording is None:
                    video_recording = create_video_recording(
                        video_dir,
                        drone_ip,
                        display_frame,
                        args.record_fps,
                        temp_subdir="fire_agent_videos",
                    )
                    print(f"Recording annotated video: {video_recording.final_path}")
                write_video_frame(video_recording, display_frame)

            saved_second, did_save = save_frame_once_per_second(display_frame, save_dir, drone_ip, saved_second)
            saved_count += did_save

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User requested stop with q.")
                user_requested_stop = True
                break

            if finish_after_record:
                if args.enable_flight:
                    if movement_future is None:
                        movement_future = command_executor.submit(finish_fire_mission, tello)
                        movement_kind = "finish"
                        finish_after_record = False
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] DRY RUN target reached: would land now.")
                    break

    except KeyboardInterrupt:
        print("Interrupted by user.")
        user_requested_stop = True
    except Exception as error:
        print(f"Fire agent failed: {error}")
    finally:
        if pending_future is not None:
            pending_future.cancel()
        if agent_future is not None:
            agent_future.cancel()

        if movement_future is not None and not movement_future.done():
            print("Waiting for active drone command to finish before shutdown...")
            try:
                movement_result = movement_future.result(timeout=10)
                if movement_kind == "finish":
                    has_taken_off = False
                print(f"Active drone command finished: {movement_result}")
            except concurrent.futures.TimeoutError:
                print("Active drone command did not finish within 10 seconds. Attempting safe shutdown anyway.")
            except Exception as movement_error:
                print(f"Active drone command failed during shutdown: {movement_error}")

        if user_requested_stop or has_taken_off:
            stop_motion_if_flying(tello, has_taken_off)

        if has_taken_off:
            print("Attempting safe landing...")
            try:
                tello.land()
                has_taken_off = False
            except Exception as landing_error:
                print(f"Landing failed: {landing_error}")

        stop_streaming_tello(tello, frame_read)
        if video_recording is not None:
            close_video_recording(video_recording)
        detector_executor.shutdown(wait=False, cancel_futures=True)
        command_executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
