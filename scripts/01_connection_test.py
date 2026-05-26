"""Minimal connection and status check. This script does not take off."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tello_utils import connect_tello, print_basic_status


def main() -> None:
    tello = None

    try:
        tello = connect_tello()
        print_basic_status(tello)
        print("Connection test completed. Motors were not started.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"Connection test failed: {error}")
    finally:
        if tello is not None:
            try:
                tello.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
