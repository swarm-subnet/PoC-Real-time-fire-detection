"""YOLO + Chutes agent loop for cautious human-following experiments.

By default this is a dry-run: it detects people, asks Chutes for one Tello SDK
command, displays that command, and saves frames. It only flies if you pass
--enable-flight.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from djitellopy import Tello


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_IP_FILE = ROOT_DIR / "scripts" / "swarm" / "drone_ips.txt"
DEFAULT_SAVE_DIR = ROOT_DIR / "captures" / "human_agent"
DEFAULT_VIDEO_DIR = ROOT_DIR / "captures" / "human_agent_videos"
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_STOP_COVERAGE_PERCENT = 40.0
DEFAULT_FINISH_YAW_DEGREES = 180
DEFAULT_RECORD_FPS = 20.0
MIN_BATTERY_PERCENT = 30

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chutes_agent import (
    DEFAULT_CHUTES_MODEL,
    AgentDecision,
    add_control_state,
    get_chutes_api_key,
    load_dotenv_if_present,
    request_tello_command_from_chutes,
)
from media_utils import (
    VideoRecording,
    close_video_recording,
    create_video_recording,
    save_frame_once_per_second,
    write_video_frame,
)
from tello_stream import (
    build_candidate_ips as build_stream_candidate_ips,
    connect_first_streaming_drone,
    resolve_path as resolve_root_path,
    stop_streaming_tello,
)
from yolo_utils import Detection, DEFAULT_MODEL_NAME, detect_people_in_image, draw_detections, load_yolo_model


WINDOW_NAME = "Tello Human Agent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect humans with YOLO and ask Chutes for cautious Tello movement commands."
    )
    parser.add_argument(
        "--ip",
        action="append",
        help="Drone IP to try. Can be passed multiple times. If omitted, uses scripts/swarm/drone_ips.txt.",
    )
    parser.add_argument("--ip-file", default=str(DEFAULT_IP_FILE), help="Text file with one drone IP per line.")
    parser.add_argument("--no-direct-fallback", action="store_true", help="Do not try 192.168.10.1 as fallback.")
    parser.add_argument("--yolo-model", default=DEFAULT_MODEL_NAME, help="YOLO model file/name.")
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="YOLO person confidence threshold. Default: 0.80.",
    )
    parser.add_argument(
        "--chutes-model",
        default=DEFAULT_CHUTES_MODEL,
        help="Chutes LLM model ID.",
    )
    parser.add_argument(
        "--agent-every",
        type=float,
        default=3.0,
        help="Seconds between LLM movement decisions.",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=3,
        help="Run YOLO every N frames to reduce CPU load.",
    )
    parser.add_argument(
        "--stop-coverage",
        type=float,
        default=DEFAULT_STOP_COVERAGE_PERCENT,
        help="If the selected person covers at least this frame percentage, command stop.",
    )
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
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Disable annotated video recording.",
    )
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for video frames before trying the next IP.",
    )
    parser.add_argument(
        "--enable-flight",
        action="store_true",
        help="Actually take off and execute validated agent commands. Default is dry-run.",
    )
    parser.add_argument(
        "--finish-yaw",
        type=int,
        default=DEFAULT_FINISH_YAW_DEGREES,
        help="Clockwise rotation before landing after target is reached. Default: 180.",
    )
    parser.add_argument("--min-battery", type=int, default=MIN_BATTERY_PERCENT, help="Minimum battery for flight.")
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


def select_target_person(detections: list[Detection]) -> Detection | None:
    """Select the largest person box; this usually means the nearest visible human."""
    if not detections:
        return None

    return max(
        detections,
        key=lambda detection: (
            (detection.xyxy[2] - detection.xyxy[0]) * (detection.xyxy[3] - detection.xyxy[1]),
            detection.confidence,
        ),
    )


def build_detection_context(
    frame_bgr,
    target: Detection | None,
    stop_coverage_percent: float,
    airborne: bool,
    last_command: str,
) -> dict[str, Any]:
    frame_height, frame_width = frame_bgr.shape[:2]

    context: dict[str, Any] = {
        "target": "person",
        "frame_width": frame_width,
        "frame_height": frame_height,
        "airborne": airborne,
        "last_command": last_command,
        "stop_coverage_threshold_percent": stop_coverage_percent,
        "person_found": target is not None,
    }

    if target is None:
        return add_control_state(context)

    x1, y1, x2, y2 = target.xyxy
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    box_area = box_width * box_height
    frame_area = frame_width * frame_height
    center_x = x1 + box_width / 2
    center_y = y1 + box_height / 2
    coverage_percent = (box_area / frame_area) * 100 if frame_area else 0

    context.update(
        {
            "person_confidence": round(target.confidence, 3),
            "person_coordinate": {
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
            },
            "person_center": {
                "x": round(center_x, 1),
                "y": round(center_y, 1),
            },
            "center_offset_x_percent": round(((center_x - frame_width / 2) / (frame_width / 2)) * 100, 1),
            "center_offset_y_percent": round(((center_y - frame_height / 2) / (frame_height / 2)) * 100, 1),
            "object_coverage_percentage": round(coverage_percent, 2),
            "near_enough": coverage_percent >= stop_coverage_percent,
        }
    )
    return add_control_state(context)


def draw_target_marker(frame_bgr, target: Detection | None) -> None:
    if target is None:
        return

    x1, y1, x2, y2 = target.xyxy
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(
        frame_bgr,
        "TRACKING TARGET",
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
    command: str,
    agent_latency_ms: float | None,
) -> None:
    drone_name = f"swarm_drone_{drone_ip.rsplit('.', 1)[-1]}"
    latency_text = f"{agent_latency_ms:.0f} ms" if agent_latency_ms is not None else "waiting"

    overlay_height = 104
    y0 = max(0, frame_bgr.shape[0] - overlay_height)
    cv2.rectangle(frame_bgr, (0, y0), (frame_bgr.shape[1], frame_bgr.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        f"Mode: FOLLOW_HUMAN  Chutes Command: {command}",
        (12, y0 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"Drone: {drone_name}  Battery: {battery if battery is not None else '?'}%  Chutes: {latency_text}",
        (12, y0 + 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        "Press q to quit",
        (12, y0 + 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def execute_safe_command(tello: Tello, command: str) -> str:
    print(f"[{time.strftime('%H:%M:%S')}] Executing: {command}")
    tello.send_control_command(command)
    return command


def finish_target_reached(tello: Tello, yaw_degrees: int) -> str:
    """Visible finish behavior once the human target is close enough."""
    yaw_degrees = max(1, min(360, yaw_degrees))
    print(f"[{time.strftime('%H:%M:%S')}] Target reached. Rotating clockwise {yaw_degrees} degrees...")
    tello.rotate_clockwise(yaw_degrees)

    print(f"[{time.strftime('%H:%M:%S')}] Landing after target reached...")
    tello.land()
    return "landed"


def stop_motion_if_flying(tello: Tello | None, has_taken_off: bool) -> None:
    """Ask the drone to hover before landing during user aborts/errors."""
    if tello is None or not has_taken_off:
        return

    try:
        print("Stopping drone motion...")
        tello.send_control_command("stop", timeout=3)
    except Exception as stop_error:
        print(f"Stop command failed: {stop_error}")


def print_context_summary(context: dict[str, Any]) -> None:
    if not context["person_found"]:
        print(f"[{time.strftime('%H:%M:%S')}] Person not found.")
        return

    print(
        f"[{time.strftime('%H:%M:%S')}] Person conf={context['person_confidence']:.2f} "
        f"x_offset={context['center_offset_x_percent']}% coverage={context['object_coverage_percentage']}%"
    )


def main() -> None:
    args = parse_args()
    load_dotenv_if_present(ROOT_DIR / ".env")
    if args.chutes_model == DEFAULT_CHUTES_MODEL:
        args.chutes_model = os.environ.get("CHUTES_MODEL", args.chutes_model)
    api_key = get_chutes_api_key()

    tello = None
    frame_read = None
    has_taken_off = False
    candidate_ips = build_candidate_ips(args)
    save_dir = resolve_path(args.save_dir)
    video_dir = resolve_path(args.video_dir)
    latest_detections: list[Detection] = []
    selected_target: Detection | None = None
    latest_context: dict[str, Any] | None = None
    latest_command = "none"
    latest_raw_response = ""
    latest_agent_latency_ms: float | None = None
    next_agent_request_time = 0.0
    last_executed_command = ""
    mission_completed = False
    finish_after_record = False
    saved_second: int | None = None
    saved_count = 0
    frame_count = 0
    pending_future: concurrent.futures.Future[AgentDecision] | None = None
    movement_future: concurrent.futures.Future[str] | None = None
    movement_kind = ""
    video_recording: VideoRecording | None = None
    user_requested_stop = False

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    command_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        print(f"Loading YOLO model: {args.yolo_model}")
        yolo_model = load_yolo_model(args.yolo_model)

        print(f"Using Chutes model: {args.chutes_model}")
        tello, frame_read, drone_ip, battery = connect_first_streaming_drone(
            candidate_ips,
            frame_timeout=args.frame_timeout,
            usage_label="for human agent",
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
                    completed_result = movement_future.result()
                    if movement_kind == "finish":
                        has_taken_off = False
                        print(f"[{time.strftime('%H:%M:%S')}] Finish maneuver completed.")
                        break

                    last_executed_command = completed_result
                    print(f"[{time.strftime('%H:%M:%S')}] Command completed: {completed_result}")
                    next_agent_request_time = time.time() + args.agent_every
                except Exception as movement_error:
                    print(f"[{time.strftime('%H:%M:%S')}] Drone command failed: {movement_error}")
                    user_requested_stop = True
                    break
                finally:
                    movement_future = None
                    movement_kind = ""

            frame_rgb = frame_read.frame
            if frame_rgb is None:
                time.sleep(0.05)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_count += 1

            if frame_count % max(1, args.detect_every) == 0:
                latest_detections = detect_people_in_image(
                    yolo_model,
                    frame_bgr,
                    confidence_threshold=args.conf,
                )
                selected_target = select_target_person(latest_detections)
                latest_context = build_detection_context(
                    frame_bgr,
                    selected_target,
                    stop_coverage_percent=args.stop_coverage,
                    airborne=has_taken_off,
                    last_command=last_executed_command,
                )

                if latest_context.get("near_enough") and not mission_completed:
                    latest_command = f"target reached -> cw {args.finish_yaw} -> land"
                    print_context_summary(latest_context)
                    mission_completed = True
                    finish_after_record = True

            now = time.time()

            if (
                not mission_completed
                and latest_context is not None
                and pending_future is None
                and movement_future is None
                and now >= next_agent_request_time
            ):
                print_context_summary(latest_context)
                pending_future = executor.submit(
                    request_tello_command_from_chutes,
                    latest_context,
                    api_key,
                    args.chutes_model,
                )

            if pending_future is not None and pending_future.done():
                try:
                    decision = pending_future.result()
                    if mission_completed:
                        print(f"[{time.strftime('%H:%M:%S')}] Ignoring Chutes response because target is already reached.")
                        pending_future = None
                        continue

                    latest_command = decision.command
                    latest_raw_response = decision.raw_response
                    latest_agent_latency_ms = decision.latency_ms
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Chutes raw={latest_raw_response!r} "
                        f"validated={latest_command!r} latency={latest_agent_latency_ms:.0f} ms"
                    )
                except Exception as error:
                    latest_command = "stop"
                    latest_raw_response = f"agent error: {error}"
                    print(f"[{time.strftime('%H:%M:%S')}] Agent failed: {error}")
                finally:
                    pending_future = None

                if args.enable_flight:
                    if movement_future is None:
                        movement_future = command_executor.submit(execute_safe_command, tello, latest_command)
                        movement_kind = "command"
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] Drone is still executing a command; skipping {latest_command!r}.")
                elif not args.enable_flight:
                    print(f"[{time.strftime('%H:%M:%S')}] DRY RUN command: {latest_command}")
                    next_agent_request_time = time.time() + args.agent_every

            display_frame = draw_detections(frame_bgr, latest_detections)
            draw_target_marker(display_frame, selected_target)
            draw_status_overlay(
                display_frame,
                drone_ip,
                battery,
                latest_command,
                latest_agent_latency_ms,
            )

            if not args.no_record_video:
                if video_recording is None:
                    video_recording = create_video_recording(
                        video_dir,
                        drone_ip,
                        display_frame,
                        args.record_fps,
                        temp_subdir="human_agent_videos",
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
                        movement_future = command_executor.submit(finish_target_reached, tello, args.finish_yaw)
                        movement_kind = "finish"
                        finish_after_record = False
                else:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] DRY RUN target reached: "
                        f"would rotate clockwise {args.finish_yaw} degrees and land."
                    )
                    break

    except KeyboardInterrupt:
        print("Interrupted by user.")
        user_requested_stop = True
    except Exception as error:
        print(f"Human agent failed: {error}")
    finally:
        if pending_future is not None:
            pending_future.cancel()

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
        executor.shutdown(wait=False, cancel_futures=True)
        command_executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
