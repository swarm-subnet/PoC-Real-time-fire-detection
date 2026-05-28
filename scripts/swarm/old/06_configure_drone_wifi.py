"""Configure one RoboMaster TT / Tello EDU drone to join your Wi-Fi network."""

from __future__ import annotations

import argparse
import os


DEFAULT_TELLO_AP_IP = "192.168.10.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to one drone's own Wi-Fi first, then run this script to "
            "make that drone join your router/hotspot."
        )
    )
    parser.add_argument(
        "--ssid",
        default=os.getenv("TELLO_TARGET_WIFI_SSID") or os.getenv("TELLO_TARGET_SSID") or "",
        help="Router/hotspot Wi-Fi name.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("TELLO_TARGET_WIFI_PASSWORD") or os.getenv("TELLO_TARGET_PASSWORD") or "",
        help="Router/hotspot Wi-Fi password.",
    )
    parser.add_argument("--host", default=DEFAULT_TELLO_AP_IP, help="Drone IP while connected to its own Wi-Fi.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ssid or not args.password:
        print("Missing Wi-Fi credentials.")
        print("Use --ssid and --password, or set TELLO_TARGET_WIFI_SSID and TELLO_TARGET_WIFI_PASSWORD.")
        return

    from djitellopy import Tello

    tello = Tello(host=args.host)

    try:
        print(f"Connecting to drone at {args.host}...")
        tello.connect(wait_for_state=False)

        battery = tello.query_battery()
        print(f"Battery(query): {battery}%")

        print(f"Configuring drone to join Wi-Fi SSID: {args.ssid}")
        print("The password will not be printed.")
        response = tello.send_command_with_return(f"ap {args.ssid} {args.password}", timeout=15)
        print(f"Drone response: {response}")

        print("Power-cycle the drone now.")
        print("After reboot, connect your laptop to the router/hotspot and run 07_find_tello_ips.py.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Wi-Fi configuration failed: {error}")
    finally:
        try:
            tello.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
