"""Small cross-platform Wi-Fi control helpers for Tello provisioning."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import xml.sax.saxutils


@dataclass(frozen=True)
class WifiNetwork:
    ssid: str


class WifiManager:
    def __init__(self) -> None:
        if os.name == "nt":
            self.backend = "windows"
        elif shutil.which("nmcli"):
            self.backend = "nmcli"
        else:
            self.backend = "unsupported"

    def ensure_supported(self) -> None:
        if self.backend == "unsupported":
            raise RuntimeError(
                "No supported Wi-Fi backend found. Run this from Windows PowerShell, "
                "or from Linux with NetworkManager/nmcli available."
            )

    def scan(self) -> list[WifiNetwork]:
        self.ensure_supported()
        if self.backend == "windows":
            return self._scan_windows()
        return self._scan_nmcli()

    def current_ssid(self) -> str | None:
        self.ensure_supported()
        if self.backend == "windows":
            output = self._run(["netsh", "wlan", "show", "interfaces"], timeout=15).stdout
            for line in output.splitlines():
                clean = line.strip()
                if clean.lower().startswith("ssid") and "bssid" not in clean.lower() and ":" in clean:
                    return clean.split(":", 1)[1].strip() or None
            return None

        output = self._run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"], timeout=15).stdout
        for line in output.splitlines():
            active, _, ssid = line.partition(":")
            if active == "yes":
                return ssid or None
        return None

    def current_ipv4_network(self, ssid: str | None = None) -> str | None:
        self.ensure_supported()
        if self.backend == "windows":
            return self._current_windows_ipv4_network(ssid=ssid)
        return self._current_nmcli_ipv4_network()

    def connect_open(self, ssid: str, timeout_seconds: float = 30) -> None:
        self.ensure_supported()
        if self.backend == "windows":
            self._disconnect_windows()
            self._delete_windows_profile(ssid)
            self._add_windows_open_profile(ssid)
            self._run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"], timeout=20)
        else:
            self._run(["nmcli", "dev", "wifi", "connect", ssid], timeout=30)
        self.wait_until_connected(ssid, timeout_seconds=timeout_seconds)

    def connect_wpa(self, ssid: str, password: str, timeout_seconds: float = 45) -> None:
        self.ensure_supported()
        if self.backend == "windows":
            self._disconnect_windows()
            self._add_windows_wpa_profile(ssid, password)
            self._run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"], timeout=20)
        else:
            self._run(["nmcli", "dev", "wifi", "connect", ssid, "password", password], timeout=45)
        self.wait_until_connected(ssid, timeout_seconds=timeout_seconds)

    def wait_until_connected(self, ssid: str, timeout_seconds: float) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            current = self.current_ssid()
            if current == ssid:
                return
            time.sleep(1)
        raise TimeoutError(f"Timed out waiting to connect to Wi-Fi SSID {ssid!r}")

    def restore_windows_wifi_dhcp(self) -> None:
        if self.backend != "windows":
            return

        interface_name = self._current_windows_wifi_interface_name()
        if not interface_name:
            return

        self._run(
            [
                "netsh",
                "interface",
                "ipv4",
                "set",
                "address",
                f"name={interface_name}",
                "source=dhcp",
            ],
            timeout=20,
        )
        self._run(
            [
                "netsh",
                "interface",
                "ipv4",
                "set",
                "dnsservers",
                f"name={interface_name}",
                "source=dhcp",
            ],
            timeout=20,
        )

    def _scan_windows(self) -> list[WifiNetwork]:
        output = self._run(["netsh", "wlan", "show", "networks", "mode=bssid"], timeout=20).stdout
        ssids: list[str] = []
        for line in output.splitlines():
            match = re.match(r"^\s*SSID\s+\d+\s*:\s*(.*)\s*$", line)
            if not match:
                continue
            ssid = match.group(1).strip()
            if ssid:
                ssids.append(ssid)
        return [WifiNetwork(ssid) for ssid in _unique(ssids)]

    def _scan_nmcli(self) -> list[WifiNetwork]:
        output = self._run(["nmcli", "-t", "-f", "ssid", "dev", "wifi", "list", "--rescan", "yes"], timeout=30).stdout
        ssids = [line.strip() for line in output.splitlines() if line.strip()]
        return [WifiNetwork(ssid) for ssid in _unique(ssids)]

    def _current_windows_ipv4_network(self, ssid: str | None = None) -> str | None:
        if ssid and self.current_ssid() != ssid:
            return None

        profile_filter = ""
        if ssid:
            escaped_ssid = _powershell_single_quote(ssid)
            profile_filter = f"| Where-Object {{ $_.Name -eq '{escaped_ssid}' }} "

        script = (
            "$profile = Get-NetConnectionProfile "
            f"{profile_filter}"
            "| Select-Object -First 1; "
            "if (-not $profile) { "
            "$profile = Get-NetConnectionProfile "
            "| Where-Object { $_.InterfaceAlias -match 'Wi-Fi|Wireless|WLAN|802.11' } "
            "| Select-Object -First 1 "
            "}; "
            "if ($profile) { "
            "$ip = Get-NetIPAddress -InterfaceIndex $profile.InterfaceIndex -AddressFamily IPv4 "
            "| Where-Object { $_.IPAddress -notlike '169.254*' } "
            "| Select-Object -First 1; "
            "if ($ip) { '{0}/{1}' -f $ip.IPAddress,$ip.PrefixLength } "
            "}"
        )
        try:
            output = self._run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                timeout=20,
            ).stdout
        except Exception:
            output = ""

        detected = _network_from_ip_prefix_output(output)
        if detected:
            return detected

        interface_name = self._current_windows_wifi_interface_name()
        if not interface_name:
            return None

        escaped_interface_name = _powershell_single_quote(interface_name)
        interface_script = (
            f"$ip = Get-NetIPAddress -InterfaceAlias '{escaped_interface_name}' -AddressFamily IPv4 "
            "| Where-Object { $_.IPAddress -notlike '169.254*' } "
            "| Select-Object -First 1; "
            "if ($ip) { '{0}/{1}' -f $ip.IPAddress,$ip.PrefixLength }"
        )
        try:
            output = self._run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", interface_script],
                timeout=20,
            ).stdout
        except Exception:
            return None
        return _network_from_ip_prefix_output(output)

    def _current_windows_wifi_interface_name(self) -> str | None:
        output = self._run(["netsh", "wlan", "show", "interfaces"], timeout=15).stdout
        for line in output.splitlines():
            clean = line.strip()
            if clean.lower().startswith("name") and ":" in clean:
                return clean.split(":", 1)[1].strip() or None
        return None

    def _current_nmcli_ipv4_network(self) -> str | None:
        try:
            output = self._run(["nmcli", "-t", "-f", "IP4.ADDRESS", "dev", "show"], timeout=15).stdout
        except Exception:
            return None
        return _network_from_ip_prefix_output(output)

    def _add_windows_open_profile(self, ssid: str) -> None:
        escaped = xml.sax.saxutils.escape(ssid)
        ssid_hex = _ssid_hex(ssid)
        profile_xml = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{escaped}</name>
  <SSIDConfig><SSID><hex>{ssid_hex}</hex><name>{escaped}</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>manual</connectionMode>
  <MSM><security><authEncryption><authentication>open</authentication><encryption>none</encryption><useOneX>false</useOneX></authEncryption></security></MSM>
</WLANProfile>
"""
        self._add_windows_profile_xml(profile_xml, ssid)

    def _add_windows_wpa_profile(self, ssid: str, password: str) -> None:
        escaped_ssid = xml.sax.saxutils.escape(ssid)
        escaped_password = xml.sax.saxutils.escape(password)
        ssid_hex = _ssid_hex(ssid)
        profile_xml = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{escaped_ssid}</name>
  <SSIDConfig><SSID><hex>{ssid_hex}</hex><name>{escaped_ssid}</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>manual</connectionMode>
  <MSM>
    <security>
      <authEncryption><authentication>WPA2PSK</authentication><encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
      <sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>{escaped_password}</keyMaterial></sharedKey>
    </security>
  </MSM>
</WLANProfile>
"""
        self._add_windows_profile_xml(profile_xml, ssid)

    def _add_windows_profile_xml(self, xml: str, ssid: str) -> None:
        with tempfile.TemporaryDirectory(prefix="tello_wifi_profile_") as temp_dir:
            profile_path = Path(temp_dir) / f"{_safe_filename(ssid)}.xml"
            profile_path.write_text(xml, encoding="utf-8")
            try:
                self._run(["netsh", "wlan", "add", "profile", f"filename={profile_path}", "user=current"], timeout=20)
            except RuntimeError as error:
                if _is_existing_windows_profile_error(str(error)):
                    return
                raise

    def _disconnect_windows(self) -> None:
        self._run_optional(["netsh", "wlan", "disconnect"], timeout=10)
        time.sleep(1)

    def _delete_windows_profile(self, ssid: str) -> None:
        self._run_optional(["netsh", "wlan", "delete", "profile", f"name={ssid}"], timeout=10)

    @staticmethod
    def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            output = "\n".join(
                part.strip()
                for part in (result.stdout, result.stderr)
                if part and part.strip()
            )
            detail = f": {output}" if output else ""
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}: "
                f"{_format_command(command)}{detail}"
            )
        return result

    @staticmethod
    def _run_optional(command: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except Exception:
            return None


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value) or "wifi_profile"


def _ssid_hex(value: str) -> str:
    return value.encode("utf-8").hex().upper()


def _format_command(command: list[str]) -> str:
    safe: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            safe.append("<hidden>")
            redact_next = False
            continue
        safe.append(part)
        if part.lower() == "password":
            redact_next = True
    return " ".join(safe)


def _is_existing_windows_profile_error(message: str) -> bool:
    lower = message.lower()
    return (
        "profile with this name already exists" in lower
        or "cannot be overwritten" in lower
    )


def _network_from_ip_prefix_output(output: str) -> str | None:
    for line in output.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if ":" in clean:
            clean = clean.split(":", 1)[1].strip()
        try:
            return str(ipaddress.ip_network(clean, strict=False))
        except ValueError:
            continue
    return None


def _powershell_single_quote(value: str) -> str:
    return value.replace("'", "''")
