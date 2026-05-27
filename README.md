# robomaster-tt-python-starter

Starter repository for programming a DJI RoboMaster TT / Tello Talent drone with Python and `djitellopy`.

This version keeps the project intentionally small. It covers four things:

- verify that the laptop can connect to the drone
- spin the motors on the bench without takeoff
- run one tiny first-flight script
- preview the live camera feed on the laptop

The repository assumes your computer is already connected to the drone's Wi-Fi network before you run any script.

## What This Project Is For

RoboMaster TT is compatible with Tello EDU SDK-style commands, so it works well with `djitellopy`. This repo is a simple starter for first hardware tests, not a full autonomy stack.

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
    02_motor_spin_test.py
    03_first_flight.py
    04_camera_preview.py
```

## Hardware Required

- DJI RoboMaster TT / Tello Talent drone
- Charged flight battery
- Computer with Wi-Fi
- Python 3.10 or newer

## Python Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

`opencv-python` is included because the camera preview script uses OpenCV to show the video stream in a desktop window.

## Connect To The Drone Wi-Fi

1. Power on the drone.
2. Wait for the drone Wi-Fi network to appear.
3. Connect your computer to the drone's Wi-Fi network.
4. Close the official Tello or RoboMaster app before running Python scripts.
5. If Windows keeps assigning a `169.254.x.x` address instead of `192.168.10.x`, fix the adapter IP first before troubleshooting Python.

## Run The Scripts

From the repository root:

```bash
python scripts/01_connection_test.py
python scripts/02_motor_spin_test.py
python scripts/03_first_flight.py
python scripts/04_camera_preview.py
python scripts/swarm/08_wifi_motor_spin_by_ip.py 192.168.1.101
python scripts/swarm/08_wifi_motor_spin_by_ip.py
python scripts/swarm/09_setup_new_drone.py
```

What each script does:

- `01_connection_test.py`: connects, prints battery and current state, then exits. It does not start motors.
- `02_motor_spin_test.py`: spins the propellers on the bench for a few seconds without takeoff.
- `03_first_flight.py`: checks battery, takes off, hovers briefly, performs a tiny movement test, rotates a little, and lands.
- `04_camera_preview.py`: opens the live video stream in an OpenCV window. Press `q` to quit.
- `swarm/08_wifi_motor_spin_by_ip.py`: tests one station-mode drone by IP, or every registered drone when no IP is passed.
- `swarm/09_setup_new_drone.py`: configures one new drone, scans for it on the router, and adds its IP to `swarm/drone_ips.txt`.
- `swarm/old/`: older step-by-step swarm setup helpers kept for reference.

## Two-Drone Wi-Fi Workflow

Use this only with Tello EDU / RoboMaster TT / Tello Talent drones that support the SDK `ap` command.

1. Power on only the new drone.
2. Connect Windows to that drone's `TELLO-*` Wi-Fi.
3. Run `python scripts/swarm/09_setup_new_drone.py`.
4. When the drone reboots, make sure Windows reconnects to `DIGIFIBRA-HU4H`.
5. The script scans the router network and appends any new drone IP to `scripts/swarm/drone_ips.txt`.

After drones are registered in `drone_ips.txt`, this command runs the motor test for every registered drone:

```bash
python scripts/swarm/08_wifi_motor_spin_by_ip.py
```

### Station-Mode UDP Note

When a TT / Tello is connected through a router, it can behave like it remembers the active SDK client by UDP source port. If one script talks to the drone from a random local port and the next script uses a different local port, the drone may ignore `command` for a while even though `ping` still works.

For this reason, `scripts/swarm/08_wifi_motor_spin_by_ip.py` binds its local UDP socket to port `8889` before sending SDK commands. That makes repeated runs look like the same SDK client:

```text
laptop:8889 -> drone:8889
```

If you write more station-mode scripts, use the same pattern: bind the local UDP socket to `8889`, drain delayed responses before each command, and retry commands instead of relying on a fresh ephemeral source port every run.

## Safety Notes

- Run `01_connection_test.py` first.
- Do not run the motor spin script while holding the drone.
- Do not run the first-flight script in a room with a low ceiling.
- `move_up(20)` is a blind command. The drone does not know where the ceiling is.
- Start with a fully charged battery. The first-flight script refuses to fly below 30%.
- Stay ready to press `Ctrl+C` if you need to stop a script.
- The first-flight script attempts to land in `finally` if something fails after takeoff.

## Indoor Checklist

- Use a clear room with good lighting.
- Remove fragile objects from the area.
- Keep people, pets, cables, clothing, and hair away from the propellers.
- Put the drone on a flat stable surface before takeoff or motor tests.
- Turn off fans and avoid drafts.

## Troubleshooting

### Cannot Connect To Drone Wi-Fi

- Confirm the drone is powered on and broadcasting its Wi-Fi network.
- Reconnect directly to the drone Wi-Fi.
- Forget and rejoin the network if the laptop switches back to another saved network.
- Close VPNs or network tools that may interfere with local UDP traffic.

### Command Timeout

- Make sure the laptop is connected to the drone Wi-Fi and not another network.
- Wait a few seconds after joining Wi-Fi before running the script.
- Close the Tello or RoboMaster mobile app.
- Power-cycle the drone and retry.

### Battery Too Low

- Charge the battery before using `03_first_flight.py`.
- The first-flight script refuses to take off below 30%.

### Motor Spin Test Does Not Work

- `motoron` / `motoroff` support can vary by firmware and model.
- Confirm you are using compatible Tello EDU / RoboMaster TT firmware.
- If the command is rejected, do not keep retrying while near the drone.

### Camera Window Does Not Open

- Make sure `opencv-python` installed successfully.
- Run the camera script on native Windows if your WSL setup has UDP/video limitations.
- Close the mobile app so the stream is not busy elsewhere.

## Next Steps

After these four scripts work reliably, you can add:

- mission pad tests
- keyboard control
- structured logging
- autonomous control loops
