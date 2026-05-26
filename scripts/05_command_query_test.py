"""Run a few non-flight SDK queries and print the raw responses."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tello_utils import connect_tello


RAW_QUERIES = [
    ("Battery", "battery?"),
    ("Flight Time", "time?"),
    ("Height", "height?"),
    ("Temperature", "temp?"),
    ("TOF Distance", "tof?"),
    ("Attitude", "attitude?"),
    ("Wi-Fi SNR", "wifi?"),
    ("SDK Version", "sdk?"),
    ("Serial Number", "sn?"),
]


def main() -> None:
    tello = None

    try:
        tello = connect_tello(wait_for_state=False)

        print("Running raw SDK queries. These do not start motors or take off.")
        for label, command in RAW_QUERIES:
            response = tello.send_read_command(command).strip()
            print(f"{label}: {response}")

        print("Query test completed.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Query test failed: {error}")
    finally:
        if tello is not None:
            try:
                tello.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
