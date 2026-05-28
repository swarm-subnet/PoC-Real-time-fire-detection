"""Chutes LLM helper for safe, single-command Tello agent experiments."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from env_utils import load_dotenv_if_present


CHUTES_CHAT_COMPLETIONS_URL = "https://llm.chutes.ai/v1/chat/completions"
DEFAULT_CHUTES_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct-TEE"

SAFE_COMMAND_RANGES = {
    "cw": (1, 30),
    "ccw": (1, 30),
    "forward": (20, 50),
    "back": (20, 40),
}
SAFE_EXACT_COMMANDS = {"stop"}


AGENT_SYSTEM_PROMPT = """You are a cautious DJI Tello flight assistant.

You receive YOLO person-detection data from the drone camera.
Return exactly one Tello SDK command and nothing else.

Allowed commands:
- stop
- cw 1..30
- ccw 1..30
- forward 20..50

Rules:
- The target is always a human. Never output land.
- Do not output takeoff. The Python script handles takeoff/landing.
- Use the derived control_state fields first. Do not reinterpret small raw offsets.
- Apply these rules in order:
  1. If person_found is false, output cw 20.
  2. Else if distance_status is reached, output stop.
  3. Else if horizontal_position is left, output ccw 15.
  4. Else if horizontal_position is right, output cw 15.
  5. Else if horizontal_position is centered and distance_status is too_far, output forward 50.
  6. Else output stop.
- If uncertain, output stop.
- Do not use up/down, left/right, flip, go, curve, rc, emergency, or any other command.
- Do not explain your decision.
- Do not repeat these rules.
- Your entire response must be exactly one command, for example: forward 50"""


FIRE_AGENT_SYSTEM_PROMPT = """You are a cautious DJI Tello flight assistant.

You receive fire detection data from the drone camera.
Return exactly one Tello SDK command and nothing else.

Allowed commands:
- stop
- cw 1..30
- ccw 1..30
- forward 20..50

Rules:
- The target is fire when visible. Never output land.
- Do not output takeoff. The Python script handles takeoff/landing.
- Use the derived control_state fields first. Do not reinterpret small raw offsets.
- Apply these rules in order:
  1. If hazard_found is false and search_forward_status is available, output forward 50.
  2. If hazard_found is false and search_forward_status is exhausted, output stop.
  3. Else if distance_status is reached, output stop.
  4. Else if forward_budget_status is exhausted, output stop.
  5. Else if horizontal_position is left, output ccw 10.
  6. Else if horizontal_position is right, output cw 10.
  7. Else if horizontal_position is centered and distance_status is too_far, output forward 50.
  8. Else output stop.
- If uncertain, output stop.
- Do not use back, up/down, left/right, flip, go, curve, rc, emergency, land, or any other command.
- Do not explain your decision.
- Do not repeat these rules.
- Your entire response must be exactly one command, for example: forward 50"""


@dataclass
class AgentDecision:
    command: str
    raw_response: str
    model: str
    latency_ms: float


def get_chutes_api_key() -> str:
    api_key = os.environ.get("CHUTES_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CHUTES_API_KEY is not set. Set it in your shell or in an ignored .env file.")
    return api_key


def request_tello_command_from_chutes(
    detection_context: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_CHUTES_MODEL,
    timeout_seconds: float = 60.0,
) -> AgentDecision:
    """Ask Chutes for one safe Tello command and validate the returned text."""
    return request_command_from_chutes(
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_message=build_chutes_user_message(detection_context),
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def request_fire_command_from_chutes(
    detection_context: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_CHUTES_MODEL,
    timeout_seconds: float = 60.0,
) -> AgentDecision:
    """Ask Chutes for one safe fire-following Tello command."""
    return request_command_from_chutes(
        system_prompt=FIRE_AGENT_SYSTEM_PROMPT,
        user_message=build_fire_chutes_user_message(detection_context),
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def request_command_from_chutes(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str = DEFAULT_CHUTES_MODEL,
    timeout_seconds: float = 60.0,
) -> AgentDecision:
    """Ask Chutes for one safe Tello command and validate the returned text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "max_tokens": 64,
    }

    started = time.perf_counter()
    request = urllib.request.Request(
        CHUTES_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chutes HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Chutes request failed: {error}") from error

    latency_ms = (time.perf_counter() - started) * 1000
    try:
        message = data["choices"][0]["message"]
        raw_response = (message.get("content") or message.get("reasoning_content") or "").strip()
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Unexpected Chutes response: {data}") from error

    if not raw_response:
        raise RuntimeError(f"Chutes returned an empty message: {data}")

    command = parse_safe_tello_command(raw_response)
    return AgentDecision(
        command=command,
        raw_response=raw_response,
        model=model,
        latency_ms=latency_ms,
    )


def parse_safe_tello_command(raw_text: str) -> str:
    """Extract one allowed command. Fall back to stop if output is invalid."""
    normalized = raw_text.strip().lower().replace("`", "").replace('"', "").replace("'", "")
    candidates = [line.strip() for line in normalized.splitlines() if line.strip()]

    command_tag_matches = re.findall(r"\bcommand\s*:\s*(stop|cw\s+-?\d+|ccw\s+-?\d+|forward\s+-?\d+|back\s+-?\d+)\b", normalized)
    candidates.extend(command_tag_matches)

    if normalized in SAFE_EXACT_COMMANDS or re.fullmatch(r"(cw|ccw|forward|back)\s+-?\d+", normalized):
        candidates.append(normalized)

    for candidate in candidates:
        if candidate in SAFE_EXACT_COMMANDS:
            return candidate

        match = re.fullmatch(r"(cw|ccw|forward|back)\s+(-?\d+)", candidate)
        if not match:
            continue

        action = match.group(1)
        value = int(match.group(2))
        min_value, max_value = SAFE_COMMAND_RANGES[action]
        if min_value <= value <= max_value:
            return f"{action} {value}"

    return "stop"


def add_control_state(detection_context: dict[str, Any]) -> dict[str, Any]:
    """Add derived control labels so the LLM does not infer thresholds itself."""
    if not detection_context.get("person_found"):
        detection_context["control_state"] = {
            "horizontal_position": "unknown",
            "distance_status": "searching",
        }
        return detection_context

    if detection_context.get("near_enough"):
        distance_status = "reached"
    else:
        distance_status = "too_far"

    try:
        x_offset = float(detection_context.get("center_offset_x_percent", 0))
    except (TypeError, ValueError):
        x_offset = 0

    if x_offset < -15:
        horizontal_position = "left"
    elif x_offset > 15:
        horizontal_position = "right"
    else:
        horizontal_position = "centered"

    detection_context["control_state"] = {
        "horizontal_position": horizontal_position,
        "distance_status": distance_status,
        "center_deadband_percent": 15,
    }
    return detection_context


def build_chutes_user_message(detection_context: dict[str, Any]) -> str:
    """Build a concise, unambiguous command-selection message."""
    context = add_control_state(dict(detection_context))
    control_state = context.get("control_state", {})

    decision_input = {
        "person_found": context.get("person_found"),
        "horizontal_position": control_state.get("horizontal_position"),
        "distance_status": control_state.get("distance_status"),
        "center_offset_x_percent": context.get("center_offset_x_percent"),
        "object_coverage_percentage": context.get("object_coverage_percentage"),
        "near_enough": context.get("near_enough"),
    }

    return (
        "Choose exactly one Tello SDK command from the rules.\n"
        "Use DECISION_INPUT first; FULL_DETECTION_JSON is only supporting context.\n"
        "DECISION_INPUT:\n"
        + json.dumps(decision_input, separators=(",", ":"))
        + "\nFULL_DETECTION_JSON:\n"
        + json.dumps(context, separators=(",", ":"))
    )


def add_fire_control_state(detection_context: dict[str, Any]) -> dict[str, Any]:
    """Add derived fire-control labels so the LLM follows explicit thresholds."""
    try:
        remaining_forward = float(detection_context.get("remaining_forward_cm", 0))
    except (TypeError, ValueError):
        remaining_forward = 0

    try:
        no_fire_forward_count = int(detection_context.get("no_fire_forward_count", 0))
    except (TypeError, ValueError):
        no_fire_forward_count = 0

    try:
        max_no_fire_forwards = int(detection_context.get("max_no_fire_forwards", 2))
    except (TypeError, ValueError):
        max_no_fire_forwards = 2

    search_forward_status = (
        "available"
        if no_fire_forward_count < max_no_fire_forwards and remaining_forward >= 20
        else "exhausted"
    )

    if not detection_context.get("hazard_found"):
        detection_context["control_state"] = {
            "horizontal_position": "unknown",
            "distance_status": "searching",
            "forward_budget_status": "available" if remaining_forward >= 20 else "exhausted",
            "search_forward_status": search_forward_status,
        }
        return detection_context

    if detection_context.get("near_enough"):
        distance_status = "reached"
    else:
        distance_status = "too_far"

    try:
        x_offset = float(detection_context.get("center_offset_x_percent", 0))
    except (TypeError, ValueError):
        x_offset = 0

    try:
        center_deadband = float(detection_context.get("center_deadband_percent", 18))
    except (TypeError, ValueError):
        center_deadband = 18

    if x_offset < -center_deadband:
        horizontal_position = "left"
    elif x_offset > center_deadband:
        horizontal_position = "right"
    else:
        horizontal_position = "centered"

    detection_context["control_state"] = {
        "horizontal_position": horizontal_position,
        "distance_status": distance_status,
        "forward_budget_status": "available" if remaining_forward >= 20 else "exhausted",
        "search_forward_status": "not_needed",
        "center_deadband_percent": center_deadband,
    }
    return detection_context


def build_fire_chutes_user_message(detection_context: dict[str, Any]) -> str:
    """Build a concise fire command-selection message."""
    context = add_fire_control_state(dict(detection_context))
    control_state = context.get("control_state", {})

    decision_input = {
        "hazard_found": context.get("hazard_found"),
        "target_label": context.get("target_label"),
        "horizontal_position": control_state.get("horizontal_position"),
        "distance_status": control_state.get("distance_status"),
        "forward_budget_status": control_state.get("forward_budget_status"),
        "search_forward_status": control_state.get("search_forward_status"),
        "center_offset_x_percent": context.get("center_offset_x_percent"),
        "object_coverage_percentage": context.get("object_coverage_percentage"),
        "near_enough": context.get("near_enough"),
        "remaining_forward_cm": context.get("remaining_forward_cm"),
        "no_fire_forward_count": context.get("no_fire_forward_count"),
        "max_no_fire_forwards": context.get("max_no_fire_forwards"),
    }

    return (
        "Choose exactly one Tello SDK command from the rules.\n"
        "Use DECISION_INPUT first; FULL_DETECTION_JSON is only supporting context.\n"
        "DECISION_INPUT:\n"
        + json.dumps(decision_input, separators=(",", ":"))
        + "\nFULL_DETECTION_JSON:\n"
        + json.dumps(context, separators=(",", ":"))
    )
