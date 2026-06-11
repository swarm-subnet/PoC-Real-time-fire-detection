"""High-level safe control layer for a station-mode Tello swarm."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

from swarm_formations import FormationPlan
from swarm_state import DroneRuntimeState, apply_logical_line_layout, build_initial_states
from swarm_utils import TelloUdpClient, query_battery, timestamp


StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class PreflightConfig:
    min_battery: int = 50
    rounds: int = 3
    retries: int = 5
    pause_seconds: float = 2.0
    allow_low_battery: bool = False


class SwarmController:
    def __init__(self, ips: list[str], client: TelloUdpClient | None = None) -> None:
        self.ips = ips
        self.client = client or TelloUdpClient()
        self.states = build_initial_states(ips)
        self.lock = threading.RLock()
        self.last_preflight_ok = False
        self.active_action = "idle"

    def close(self) -> None:
        self.client.close()

    def snapshot(self) -> list[DroneRuntimeState]:
        with self.lock:
            return [self._copy_state(self.states[ip]) for ip in self.ips]

    def any_airborne(self) -> bool:
        with self.lock:
            return any(state.airborne for state in self.states.values())

    def all_airborne(self) -> bool:
        with self.lock:
            return all(state.airborne for state in self.states.values())

    def refresh_all(self, retries: int = 1, status: StatusCallback | None = None) -> bool:
        ok = True
        self._set_action("status")
        try:
            for ip in self.ips:
                if not self.refresh_one(ip, retries=retries):
                    ok = False
            if status:
                if ok:
                    status("status refresh complete")
                else:
                    with self.lock:
                        offline = [ip for ip in self.ips if not self.states[ip].online]
                    status("status refresh missing: " + ", ".join(offline))
            return ok
        finally:
            self._set_action("idle")

    def require_all_ready(
        self,
        config: PreflightConfig,
        status: StatusCallback | None = None,
    ) -> bool:
        """Refresh all drones with retries and require every configured IP online."""
        if not self.preflight(config, status=status):
            return False

        with self.lock:
            offline = [ip for ip in self.ips if not self.states[ip].online]
        if offline:
            if status:
                status("blocked: offline drone(s): " + ", ".join(offline))
            return False
        return True

    def check_all_ready(
        self,
        retries: int = 2,
        min_battery: int | None = None,
        status: StatusCallback | None = None,
    ) -> bool:
        """Require every configured drone to answer before a swarm action starts."""
        self._set_action("ready_check")
        failures: list[str] = []
        try:
            for ip in self.ips:
                passed = self.refresh_one(ip, retries=retries)
                state = self.states[ip]
                battery = state.battery
                if not passed:
                    failures.append(f"{ip}: offline")
                elif min_battery is not None and battery is None:
                    failures.append(f"{ip}: no battery")
                elif min_battery is not None and battery < min_battery:
                    failures.append(f"{ip}: {battery}% < {min_battery}%")

            if failures:
                if status:
                    status("ready check failed: " + "; ".join(failures))
                return False
            if status:
                status("ready check passed")
            return True
        finally:
            self._set_action("idle")

    def refresh_one(self, ip: str, retries: int = 1) -> bool:
        with self.lock:
            self._update_state(ip, command_state="checking", last_command="battery?")

        try:
            command_response = self.client.send_one(ip, "command", timeout=3, retries=retries)
            battery, latency_ms = query_battery(self.client, ip, timeout=3, retries=retries)
            with self.lock:
                self._update_state(
                    ip,
                    online=True,
                    battery=battery,
                    command_state="idle",
                    last_response=f"command={command_response.text}; battery={battery}%",
                    last_error="",
                    last_latency_ms=latency_ms,
                    seen=True,
                )
            return True
        except Exception as error:
            with self.lock:
                self._update_state(
                    ip,
                    online=False,
                    command_state="error",
                    last_error=str(error),
                    last_response="",
                )
            return False

    def update_telemetry(
        self,
        ip: str,
        battery: int | None = None,
        temperature_low_c: int | None = None,
        temperature_high_c: int | None = None,
        height_cm: int | None = None,
        tof_cm: int | None = None,
    ) -> None:
        if ip not in self.states:
            return

        updates = {
            "online": True,
            "last_error": "",
        }
        if battery is not None:
            updates["battery"] = battery
        if temperature_low_c is not None:
            updates["temperature_low_c"] = temperature_low_c
        if temperature_high_c is not None:
            updates["temperature_high_c"] = temperature_high_c
        if height_cm is not None:
            updates["height_cm"] = height_cm
        if tof_cm is not None:
            updates["tof_cm"] = tof_cm

        with self.lock:
            self._update_state(ip, seen=True, **updates)

    def preflight(self, config: PreflightConfig, status: StatusCallback | None = None) -> bool:
        self._set_action("preflight")
        final_ok = False
        try:
            for round_index in range(1, config.rounds + 1):
                failures: list[str] = []
                if status:
                    status(f"preflight round {round_index}/{config.rounds}")

                for ip in self.ips:
                    passed = self.refresh_one(ip, retries=config.retries)
                    state = self.states[ip]
                    battery = state.battery
                    if not passed:
                        failures.append(f"{ip}: offline")
                    elif battery is None:
                        failures.append(f"{ip}: no battery")
                    elif battery < config.min_battery and not config.allow_low_battery:
                        failures.append(f"{ip}: {battery}% < {config.min_battery}%")

                if failures:
                    final_ok = False
                    if status:
                        status("preflight failed: " + "; ".join(failures))
                    if round_index < config.rounds:
                        time.sleep(config.pause_seconds)
                    continue

                final_ok = True
                if status:
                    status(f"preflight round {round_index} passed")
                if round_index < config.rounds:
                    time.sleep(config.pause_seconds)

            self.last_preflight_ok = final_ok
            if not final_ok and status:
                status("preflight blocked takeoff")
            return final_ok
        finally:
            self._set_action("idle")

    def takeoff_all(self, status: StatusCallback | None = None) -> bool:
        self._set_action("takeoff")
        try:
            self._mark_command(self.ips, "takeoff")
            responses = self.client.send_all(self.ips, "takeoff", timeout=12)
            failed: list[str] = []
            with self.lock:
                for ip in self.ips:
                    response = responses.get(ip)
                    ok = self._response_ok(response)
                    if not ok:
                        failed.append(ip)
                    self._update_state(
                        ip,
                        airborne=self.states[ip].airborne or ok,
                        command_state="hovering" if ok else "error",
                        last_command="takeoff",
                        last_response=response.text if response else "timeout",
                        last_error="" if ok else "takeoff failed or timed out",
                        last_latency_ms=response.latency_ms if response else None,
                        seen=response is not None,
                    )
            if status:
                if failed:
                    status("takeoff incomplete: " + ", ".join(failed))
                else:
                    status("takeoff complete")
            return not failed
        finally:
            self._set_action("idle")

    def takeoff_sequential(
        self,
        settle_seconds: float = 2.0,
        status: StatusCallback | None = None,
    ) -> bool:
        """Take off one drone at a time to reduce launch risk."""
        self._set_action("takeoff_sequence")
        failed: list[str] = []
        try:
            for index, ip in enumerate(self.ips, start=1):
                if status:
                    status(f"takeoff {index}/{len(self.ips)}: {ip}")
                self._mark_command([ip], "takeoff")
                try:
                    response = self.client.send_one(ip, "takeoff", timeout=12, retries=1)
                except Exception:
                    response = None
                ok = self._response_ok(response)
                if not ok:
                    failed.append(ip)
                with self.lock:
                    self._update_state(
                        ip,
                        airborne=self.states[ip].airborne or ok,
                        command_state="hovering" if ok else "error",
                        last_command="takeoff",
                        last_response=response.text if response else "timeout",
                        last_error="" if ok else "takeoff failed or timed out",
                        last_latency_ms=response.latency_ms if response else None,
                        seen=response is not None,
                    )
                if ok and settle_seconds > 0:
                    time.sleep(settle_seconds)

            if status:
                if failed:
                    status("takeoff sequence incomplete: " + ", ".join(failed))
                else:
                    status("takeoff sequence complete")
            return not failed
        finally:
            self._set_action("idle")

    def land_all(self, status: StatusCallback | None = None) -> bool:
        self._set_action("land")
        try:
            self._mark_command(self.ips, "land")
            responses = self.client.send_all(self.ips, "land", timeout=12)
            failed: list[str] = []
            with self.lock:
                for ip in self.ips:
                    response = responses.get(ip)
                    ok = self._response_ok(response)
                    if not ok:
                        failed.append(ip)
                    self._update_state(
                        ip,
                        airborne=False if ok else self.states[ip].airborne,
                        command_state="idle" if ok else "error",
                        last_command="land",
                        last_response=response.text if response else "timeout",
                        last_error="" if ok else "land failed or timed out",
                        last_latency_ms=response.latency_ms if response else None,
                        seen=response is not None,
                    )
            apply_logical_line_layout(self.states)
            if status:
                if failed:
                    status("land incomplete: " + ", ".join(failed))
                else:
                    status("land complete")
            return not failed
        finally:
            self._set_action("idle")

    def motor_spin_all(
        self,
        seconds: float,
        status: StatusCallback | None = None,
    ) -> None:
        if seconds <= 0:
            raise ValueError("motor spin seconds must be greater than 0")

        self._set_action("motor_spin")
        motor_started = False
        try:
            self._mark_command(self.ips, "motoron")
            responses = self.client.send_all(self.ips, "motoron", timeout=8)
            motor_started = any(
                self._response_ok(response)
                for response in responses.values()
            )
            missing = [ip for ip in self.ips if not self._response_ok(responses.get(ip))]
            if missing:
                with self.lock:
                    for ip, response in responses.items():
                        self._update_state(
                            ip,
                            command_state="commanding",
                            last_command="motoron",
                            last_response=response.text,
                            last_error="" if self._response_ok(response) else "motoron failed",
                            last_latency_ms=response.latency_ms,
                            seen=True,
                        )
                    for ip in missing:
                        self._update_state(
                            ip,
                            command_state="error",
                            last_command="motoron",
                            last_error="motoron failed or timed out",
                        )
                raise RuntimeError("motoron failed for: " + ", ".join(missing))

            with self.lock:
                for ip, response in responses.items():
                    self._update_state(
                        ip,
                        command_state=f"motor spin {seconds:.1f}s",
                        last_command="motoron",
                        last_response=response.text,
                        last_error="",
                        last_latency_ms=response.latency_ms,
                        seen=True,
                    )
            if status:
                status(f"motors spinning for {seconds:.1f}s")
            time.sleep(seconds)
        finally:
            if motor_started:
                self._mark_command(self.ips, "motoroff")
                responses = self.client.send_all(self.ips, "motoroff", timeout=8)
                with self.lock:
                    for ip in self.ips:
                        response = responses.get(ip)
                        self._update_state(
                            ip,
                            command_state="idle",
                            last_command="motoroff",
                            last_response=response.text if response else "timeout",
                            last_error="" if response else "motoroff response timeout",
                            last_latency_ms=response.latency_ms if response else None,
                            seen=response is not None,
                        )
                if status:
                    status("motor spin complete")
            self._set_action("idle")

    def motoron_ips(
        self,
        ips: list[str],
        status: StatusCallback | None = None,
        command_state: str = "cooling",
    ) -> list[str]:
        selected = [ip for ip in ips if ip in self.states]
        if not selected:
            return []

        self._mark_command(selected, "motoron")
        responses = self.client.send_all(selected, "motoron", timeout=8)
        ok_ips: list[str] = []
        failed: list[str] = []
        with self.lock:
            for ip in selected:
                response = responses.get(ip)
                ok = self._response_ok(response)
                if ok:
                    ok_ips.append(ip)
                else:
                    failed.append(ip)
                self._update_state(
                    ip,
                    command_state=command_state if ok else "error",
                    last_command="motoron",
                    last_response=response.text if response else "timeout",
                    last_error="" if ok else "motoron failed or timed out",
                    last_latency_ms=response.latency_ms if response else None,
                    seen=response is not None,
                )
        if status:
            if ok_ips:
                status("cooling motoron: " + ", ".join(ok_ips))
            if failed:
                status("cooling motoron failed: " + ", ".join(failed))
        return ok_ips

    def motoroff_ips(
        self,
        ips: list[str],
        status: StatusCallback | None = None,
    ) -> list[str]:
        selected = [ip for ip in ips if ip in self.states]
        if not selected:
            return []

        self._mark_command(selected, "motoroff")
        responses = self.client.send_all(selected, "motoroff", timeout=8)
        ok_ips: list[str] = []
        failed: list[str] = []
        with self.lock:
            for ip in selected:
                response = responses.get(ip)
                ok = self._response_ok(response)
                if ok:
                    ok_ips.append(ip)
                else:
                    failed.append(ip)
                self._update_state(
                    ip,
                    command_state="idle" if ok else "error",
                    last_command="motoroff",
                    last_response=response.text if response else "timeout",
                    last_error="" if ok else "motoroff failed or timed out",
                    last_latency_ms=response.latency_ms if response else None,
                    seen=response is not None,
                )
        if status:
            if ok_ips:
                status("cooling motoroff: " + ", ".join(ok_ips))
            if failed:
                status("cooling motoroff failed: " + ", ".join(failed))
        return ok_ips

    def send_single_command(
        self,
        ip: str,
        command: str,
        timeout: float = 8,
        retries: int = 1,
        status: StatusCallback | None = None,
        command_state: str = "mission",
    ) -> bool:
        if ip not in self.states:
            if status:
                status(f"mission command skipped for unknown drone: {ip}")
            return False
        self._mark_command([ip], command)
        try:
            response = self.client.send_one(ip, command, timeout=timeout, retries=retries)
        except Exception:
            response = None
        ok = self._response_ok(response)
        with self.lock:
            self._update_state(
                ip,
                command_state=command_state if ok else "error",
                last_command=command,
                last_response=response.text if response else "timeout",
                last_error="" if ok else f"{command} failed or timed out",
                last_latency_ms=response.latency_ms if response else None,
                seen=response is not None,
            )
        if status:
            status(f"{command} {'ok' if ok else 'failed'}: {ip}")
        return ok

    def stop_ips(self, ips: list[str], status: StatusCallback | None = None) -> list[str]:
        stopped: list[str] = []
        for ip in ips:
            if self.send_single_command(ip, "stop", timeout=4, retries=1, status=status, command_state="holding"):
                stopped.append(ip)
        return stopped

    def emergency_all(self, status: StatusCallback | None = None) -> None:
        self._set_action("emergency")
        try:
            self._mark_command(self.ips, "emergency")
            responses = self.client.send_all(self.ips, "emergency", timeout=3)
            with self.lock:
                for ip in self.ips:
                    response = responses.get(ip)
                    self._update_state(
                        ip,
                        airborne=False,
                        command_state="emergency",
                        last_command="emergency",
                        last_response=response.text if response else "timeout",
                        last_error="" if response else "emergency response timeout",
                        last_latency_ms=response.latency_ms if response else None,
                        seen=response is not None,
                    )
            if status:
                status("emergency command sent")
        finally:
            self._set_action("idle")

    def hover_then_land(
        self,
        hover_seconds: float,
        preflight_config: PreflightConfig,
        status: StatusCallback | None = None,
    ) -> bool:
        if not self.preflight(preflight_config, status=status):
            return False

        if not self.takeoff_all(status=status):
            if status:
                status("takeoff incomplete; landing all drones")
            self.land_all(status=status)
            return False
        deadline = time.monotonic() + hover_seconds
        while time.monotonic() < deadline:
            remaining = max(0, int(round(deadline - time.monotonic())))
            with self.lock:
                for state in self.states.values():
                    if state.airborne:
                        state.command_state = f"hover {remaining}s"
            if status:
                status(f"hovering: {remaining}s remaining")
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

        self.land_all(status=status)
        return True

    def execute_formation(
        self,
        plan: FormationPlan,
        allow_translation: bool,
        execute: bool,
        status: StatusCallback | None = None,
    ) -> bool:
        if plan.requires_translation and not allow_translation:
            self.apply_visual_offsets(plan)
            if status:
                status(f"{plan.name} preview only: translation formations are disabled")
            return False

        self.apply_visual_offsets(plan)
        if not execute:
            if status:
                status(f"{plan.name} preview only: flight execution disabled")
            return False

        self._set_action(f"formation:{plan.name}")
        try:
            for step in plan.steps:
                if status:
                    status(f"{plan.name}: {step.name}")
                if not self._execute_step(step.commands, status=status):
                    if status:
                        status(f"{plan.name} stopped: {step.name} failed")
                    return False
                time.sleep(step.settle_seconds)
            if status:
                status(f"{plan.name} complete")
            return True
        finally:
            self._set_action("idle")

    def apply_visual_offsets(self, plan: FormationPlan) -> None:
        with self.lock:
            for ip, (x_cm, y_cm) in plan.visual_offsets_cm.items():
                if ip not in self.states:
                    continue
                self.states[ip].logical_x_cm = x_cm
                self.states[ip].logical_y_cm = y_cm

    def reset_visual_map(self) -> None:
        with self.lock:
            apply_logical_line_layout(self.states)

    def safe_shutdown(self, status: StatusCallback | None = None) -> None:
        if self.any_airborne():
            self.land_all(status=status)

    def _execute_step(self, commands: dict[str, str], status: StatusCallback | None = None) -> bool:
        grouped: dict[str, list[str]] = {}
        for ip, command in commands.items():
            grouped.setdefault(command, []).append(ip)

        step_ok = True
        for command, ips in grouped.items():
            self._mark_command(ips, command)
            responses = self.client.send_all(ips, command, timeout=10)
            failed: list[str] = []
            with self.lock:
                for ip in ips:
                    response = responses.get(ip)
                    ok = self._response_ok(response)
                    if not ok:
                        failed.append(ip)
                        step_ok = False
                    self._update_state(
                        ip,
                        command_state=("hovering" if self.states[ip].airborne else "idle") if ok else "error",
                        last_command=command,
                        last_response=response.text if response else "timeout",
                        last_error="" if ok else f"{command} failed or timed out",
                        last_latency_ms=response.latency_ms if response else None,
                        seen=response is not None,
                    )
            if failed and status:
                status(f"{command} failed for: " + ", ".join(failed))
        return step_ok

    def _mark_command(self, ips: list[str], command: str) -> None:
        with self.lock:
            for ip in ips:
                self._update_state(ip, command_state="commanding", last_command=command)

    def _set_action(self, action: str) -> None:
        with self.lock:
            self.active_action = action

    def _update_state(self, ip: str, seen: bool = False, **updates) -> None:
        state = self.states[ip]
        for name, value in updates.items():
            setattr(state, name, value)
        if seen:
            state.last_seen_monotonic = time.monotonic()

    @staticmethod
    def _response_ok(response) -> bool:
        return response is not None and response.text.strip().lower() == "ok"

    @staticmethod
    def _copy_state(state: DroneRuntimeState) -> DroneRuntimeState:
        return DroneRuntimeState(**state.__dict__)


def console_status(message: str) -> None:
    print(f"[{timestamp()}] {message}")
