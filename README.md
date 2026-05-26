# robomaster-tt-python-starter

Starter repository for programming a DJI RoboMaster TT / Tello Talent drone with Python and `djitellopy`.

This project is meant for safe first steps:

- verify that your computer can connect to the drone
- read battery and state information
- run a tiny first-flight test
- optionally test mission pad detection

The repository assumes your computer is already connected to the drone's Wi-Fi network before you run any script.

## What This Project Is For

RoboMaster TT is compatible with Tello EDU SDK-style commands, so it works well with `djitellopy`. This starter keeps the code simple and readable while still including the safety basics you want for a real first flight.

Repository layout:

```text
robomaster-tt-python-starter/
  README.md
  requirements.txt
  .gitignore
  src/
    tello_utils.py
  scripts/
    01_connection_test.py
    02_first_flight.py
    03_mission_pad_test.py
    04_wsl_control_only_test.py
    05_command_query_test.py
    06_tt_led_test.py
    07_motor_spin_test.py
```

## Hardware Required

- DJI RoboMaster TT / Tello Talent drone
- Charged flight battery
- Computer with Wi-Fi
- Python 3.10 or newer
- Mission pad(s) for the optional mission pad script

## Python Setup

Create a virtual environment and install the dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Connect To The Drone Wi-Fi

1. Power on the drone.
2. Wait for the drone Wi-Fi network to appear.
3. Connect your computer to the drone's Wi-Fi network.
4. Disconnect from VPNs or other network tools if they interfere with local UDP traffic.
5. Close the official Tello or RoboMaster app before running Python scripts, so only one controller is talking to the drone.

## Run The Scripts

From the repository root:

```bash
python scripts/01_connection_test.py
python scripts/02_first_flight.py
python scripts/03_mission_pad_test.py
python scripts/05_command_query_test.py
python scripts/06_tt_led_test.py
python scripts/07_motor_spin_test.py
```

What each script does:

- `01_connection_test.py`: connects, prints battery and state, then exits. It does not start motors or take off.
- `02_first_flight.py`: checks battery, takes off, hovers for 2 seconds, performs a very small test movement, rotates a little, and lands.
- `03_mission_pad_test.py`: enables mission pad detection, points detection downward, and prints detected mission pad IDs for about 10 seconds. It does not take off unless you change `TAKEOFF_FOR_TEST = False` to `True`.
- `04_wsl_control_only_test.py`: WSL-focused command-channel probe that skips the state-packet wait and prints raw query results.
- `05_command_query_test.py`: runs several non-flight SDK query commands like `battery?`, `height?`, `attitude?`, `sdk?`, and `sn?`.
- `06_tt_led_test.py`: cycles the RoboMaster TT / Tello Talent top LED without taking off.
- `07_motor_spin_test.py`: optional bench-only propeller spin test using `motoron` / `motoroff`. It does not take off, but it does spin the props and is disabled by default.

## Safe First-Flight Notes

- Run `01_connection_test.py` first. Do not skip directly to flight.
- Fly indoors only in a clear area for your first test.
- Remove fragile objects from the flight zone.
- Keep people, pets, loose clothing, and hair away from the propellers.
- Turn off fans and avoid strong airflow.
- Start with a fully charged battery. The flight script refuses to fly below 30%.
- Place the drone on a flat surface with overhead clearance.
- Stay ready to press `Ctrl+C` if the test needs to stop.
- In `scripts/02_first_flight.py`, set `ALLOW_MOVE_UP_TEST = False` if your ceiling is low or you want an even more conservative first flight.
- `scripts/07_motor_spin_test.py` can spin the propellers without takeoff on supported firmware, but it is still hazardous and should be treated like a live prop test.

## Indoor Safety Checklist

- Use a large uncluttered room with good lighting.
- Keep at least a few meters of clear space around the drone.
- Confirm the battery is seated properly.
- Make sure the propellers are unobstructed and in good condition.
- Put the drone on the floor or another stable flat surface before takeoff.
- Do not hand-launch or hand-catch during these starter tests.
- Keep the mission pad flat and well lit if you plan to test pad detection.

## Troubleshooting

### Cannot Connect To Drone Wi-Fi

- Confirm the drone is powered on and broadcasting its Wi-Fi network.
- Reconnect your computer directly to the drone Wi-Fi.
- Forget and rejoin the network if your computer switches back to another saved network.
- Temporarily disable software that may block local UDP traffic.

### Command Timeout

- Make sure your computer is connected to the drone Wi-Fi and not another network.
- Wait a few seconds after joining Wi-Fi before running the script.
- Close other apps that might still be talking to the drone.
- Power-cycle the drone and try the connection test again.

### Battery Too Low

- Charge the battery before using `02_first_flight.py`.
- The first-flight script will refuse to take off below 30%.

### App Still Connected To Drone

- Close the Tello or RoboMaster mobile app on phones or tablets.
- Disconnect any other computer or controller that may still be using the drone.

### Mission Pad Not Detected

- Confirm you are using a compatible TT / Tello EDU-style drone.
- Make sure mission pad detection is enabled by the script.
- Keep the pad flat, centered, and well lit.
- Downward detection usually works best when the drone is above the pad instead of sitting directly on it.
- If ground testing does not detect the pad, set `TAKEOFF_FOR_TEST = True` and test carefully in a controlled indoor space.

### LED Or Expansion Commands Do Not Work

- `06_tt_led_test.py` relies on the TT / Talent expansion board command set.
- If you are using a plain Tello model, expansion commands may fail.

### Motor Spin Test Does Not Work

- `motoron` / `motoroff` support can vary by firmware and model.
- Confirm you are using compatible Tello EDU / RoboMaster TT firmware.
- If the command is rejected, do not force it with repeated retries while near the drone.

## Script Behavior And Safety

The flight-oriented scripts are written to attempt a landing if something fails after takeoff. That does not remove the need for supervision, but it does reduce the chance of leaving the drone flying after an exception.

## Next Steps

After these scripts work reliably, you can extend the project with:

- keyboard control
- video streaming
- mission pad navigation
- structured logging
- autonomous routines
