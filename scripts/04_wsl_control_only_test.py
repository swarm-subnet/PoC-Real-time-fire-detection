"""WSL-focused control-channel test that does not require Tello state packets."""

from __future__ import annotations

from djitellopy import Tello


def query_raw(tello: Tello, command: str) -> str:
    """Run a raw SDK query and return the unparsed response."""
    return tello.send_read_command(command).strip()


def main() -> None:
    tello = Tello()

    try:
        print("Connecting to drone with wait_for_state=False...")
        tello.connect(wait_for_state=False)

        print("Running SDK query commands over the control channel...")
        print(f"Battery(query raw): {query_raw(tello, 'battery?')}")
        print(f"Height(query raw): {query_raw(tello, 'height?')}")
        print(f"Temp(query raw): {query_raw(tello, 'temp?')}")
        print("Control-only WSL test completed.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as error:
        print(f"WSL control-only test failed: {error}")
    finally:
        try:
            tello.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
