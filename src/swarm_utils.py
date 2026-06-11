"""Shared helpers for station-mode Tello / RoboMaster TT swarm scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import socket
import threading
import time


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DRONE_IPS_FILE = ROOT_DIR / "scripts" / "swarm" / "drone_ips.txt"
TELLO_PORT = 8889
DEFAULT_COMMAND_TIMEOUT_SECONDS = 7


@dataclass(frozen=True)
class SdkResponse:
    ip: str
    text: str
    latency_ms: int


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def short_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_registered_ips(path: Path = DEFAULT_DRONE_IPS_FILE) -> list[str]:
    if not path.exists():
        return []

    ips: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            ips.append(clean)
    return ips


def save_registered_ips(ips: list[str], path: Path = DEFAULT_DRONE_IPS_FILE, sort_ips: bool = True) -> None:
    values = unique_ips(ips)
    if sort_ips:
        values = sorted(values, key=_ip_sort_key)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def unique_ips(ips: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ip in ips:
        if ip not in seen:
            seen.add(ip)
            out.append(ip)
    return out


def _ip_sort_key(value: str) -> tuple[object, ...]:
    try:
        return (0, *tuple(int(part) for part in value.split(".")))
    except ValueError:
        return (1, value)


class TelloUdpClient:
    """Small SDK transport that keeps a stable local UDP source port."""

    def __init__(self, port: int = TELLO_PORT) -> None:
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # In station mode, TT/Tello can ignore repeated scripts if each run uses
        # a different ephemeral source port. Binding to 8889 keeps it stable.
        self.sock.bind(("", port))
        self._io_lock = threading.RLock()

    def __enter__(self) -> "TelloUdpClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.sock.close()

    def drain(self) -> None:
        previous_timeout = self.sock.gettimeout()
        self.sock.settimeout(0.05)
        try:
            while True:
                self.sock.recvfrom(1024)
        except TimeoutError:
            pass
        finally:
            self.sock.settimeout(previous_timeout)

    def send_one(
        self,
        ip: str,
        command: str,
        timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        retries: int = 1,
        retry_pause: float = 0.5,
        verbose: bool = True,
    ) -> SdkResponse:
        last_error: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                with self._io_lock:
                    self.drain()
                    self.sock.settimeout(timeout)
                    if verbose:
                        print(f"[{timestamp()}] >> {ip}: {command}")
                    started = time.perf_counter()
                    self.sock.sendto(command.encode("utf-8"), (ip, self.port))
                    deadline = time.monotonic() + timeout
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError
                        self.sock.settimeout(max(0.05, remaining))
                        response, addr = self.sock.recvfrom(1024)
                        source_ip = addr[0]
                        decoded = response.decode(errors="replace").strip()
                        if source_ip != ip:
                            if verbose:
                                print(f"[{timestamp()}] !! ignored response from {source_ip}: {decoded}")
                            continue
                        latency_ms = int((time.perf_counter() - started) * 1000)
                        if verbose:
                            print(f"[{timestamp()}] << {source_ip}: {decoded} ({latency_ms} ms)")
                        return SdkResponse(ip=source_ip, text=decoded, latency_ms=latency_ms)
            except TimeoutError as error:
                last_error = error
                if verbose:
                    print(f"[{timestamp()}] !! {ip}: timeout waiting for '{command}' attempt {attempt}/{retries}")
                if attempt < retries and retry_pause > 0:
                    time.sleep(retry_pause)

        raise TimeoutError(f"Command '{command}' timed out for {ip} after {retries} attempt(s)") from last_error

    def send_no_wait(self, ip: str, command: str, verbose: bool = True) -> bool:
        """Best-effort command send without waiting for an SDK response."""
        acquired = self._io_lock.acquire(blocking=False)
        if not acquired:
            return False
        try:
            if verbose:
                print(f"[{timestamp()}] >> {ip}: {command} (no-wait)")
            self.sock.sendto(command.encode("utf-8"), (ip, self.port))
            return True
        except OSError:
            return False
        finally:
            self._io_lock.release()

    def send_text(
        self,
        ip: str,
        command: str,
        timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        retries: int = 1,
        retry_pause: float = 0.5,
        verbose: bool = True,
    ) -> str:
        return self.send_one(
            ip,
            command,
            timeout=timeout,
            retries=retries,
            retry_pause=retry_pause,
            verbose=verbose,
        ).text

    def send_all(
        self,
        ips: list[str],
        command: str,
        timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        verbose: bool = True,
    ) -> dict[str, SdkResponse]:
        """Fan out a command to all drones first, then collect responses."""
        with self._io_lock:
            self.drain()
            responses: dict[str, SdkResponse] = {}
            pending = set(ips)
            sent_at: dict[str, float] = {}

            for ip in ips:
                if verbose:
                    print(f"[{timestamp()}] >> {ip}: {command}")
                sent_at[ip] = time.perf_counter()
                self.sock.sendto(command.encode("utf-8"), (ip, self.port))

            deadline = time.monotonic() + timeout
            while pending and time.monotonic() < deadline:
                self.sock.settimeout(max(0.05, deadline - time.monotonic()))
                try:
                    response, addr = self.sock.recvfrom(1024)
                except TimeoutError:
                    break

                source_ip = addr[0]
                decoded = response.decode(errors="replace").strip()
                if source_ip not in pending:
                    if verbose:
                        print(f"[{timestamp()}] !! ignored response from {source_ip}: {decoded}")
                    continue
                sent_time = sent_at.get(source_ip, time.perf_counter())
                latency_ms = int((time.perf_counter() - sent_time) * 1000)
                responses[source_ip] = SdkResponse(ip=source_ip, text=decoded, latency_ms=latency_ms)
                pending.discard(source_ip)
                if verbose:
                    print(f"[{timestamp()}] << {source_ip}: {decoded} ({latency_ms} ms)")

            if verbose:
                for ip in sorted(pending):
                    print(f"[{timestamp()}] !! {ip}: timeout waiting for '{command}'")

            return responses


def query_battery(
    client: TelloUdpClient,
    ip: str,
    timeout: float = 3,
    retries: int = 3,
    verbose: bool = True,
) -> tuple[int, int]:
    response = client.send_one(ip, "battery?", timeout=timeout, retries=retries, verbose=verbose)
    return int(response.text), response.latency_ms
