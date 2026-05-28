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
python scripts/swarm/10_swarm_controller.py
python scripts/yolo/01_detect_person_image.py captures/example.jpg
python scripts/yolo/02_capture_tello_images.py
python scripts/yolo/03_live_person_detection.py
python scripts/yolo/04_live_person_detection_save_by_ip.py
python scripts/yolo/05_human_agent_go_to_person.py --ip 192.168.1.132
python scripts/yolo/06_chutes_frame_drill.py --stop-coverage 40
python scripts/fire/01_detect_fire_image.py
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132
```

What each script does:

- `01_connection_test.py`: connects, prints battery and current state, then exits. It does not start motors.
- `02_motor_spin_test.py`: spins the propellers on the bench for a few seconds without takeoff.
- `03_first_flight.py`: checks battery, takes off, hovers briefly, performs a tiny movement test, rotates a little, and lands.
- `04_camera_preview.py`: opens the live video stream in an OpenCV window. Press `q` to quit.
- `swarm/08_wifi_motor_spin_by_ip.py`: tests one station-mode drone by IP, or every registered drone when no IP is passed.
- `swarm/09_setup_new_drone.py`: configures one new drone, scans for it on the router, and adds its IP to `swarm/drone_ips.txt`.
- `swarm/10_swarm_controller.py`: starts an interactive long-running controller for all registered drones.
- `swarm/old/`: older step-by-step swarm setup helpers kept for reference.
- `yolo/01_detect_person_image.py`: runs YOLO person detection on one local image.
- `yolo/02_capture_tello_images.py`: saves still images from the Tello camera for offline testing.
- `yolo/03_live_person_detection.py`: runs live YOLO person detection on the Tello camera stream.
- `yolo/04_live_person_detection_save_by_ip.py`: tries registered drone IPs, opens the first working video stream, draws person boxes, and saves one annotated frame per second.
- `yolo/05_human_agent_go_to_person.py`: detects humans with YOLO, asks Chutes for one safe Tello command, and runs in dry-run unless `--enable-flight` is passed.
- `yolo/06_chutes_frame_drill.py`: runs one saved-frame Chutes command-generation drill without connecting to the drone.
- `fire/01_detect_fire_image.py`: downloads the SuperBitDev/fire1 model and a sample fire image, then writes an annotated fire-detection result.
- `fire/02_live_fire_detection_save_by_ip.py`: tries registered drone IPs, opens the first working video stream, draws fire boxes, and saves one annotated frame per second. It does not fly.
- `fire/03_fire_agent_go_to_fire.py`: detects fire, asks Chutes for one safe Tello command, records annotated video, and runs in dry-run unless `--enable-flight` is passed.

## YOLO Person Detection Workflow

Start with offline detection before trying real-time video. The first run downloads the YOLO model, so it needs internet access.

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Test YOLO on one local image:

```bash
python scripts/yolo/01_detect_person_image.py samples/yolo/person_crosswalk.jpg
```

This writes an annotated image to `captures/yolo_<image-name>`.

3. Connect Windows to the drone Wi-Fi and capture sample images:

```bash
python scripts/yolo/02_capture_tello_images.py --count 5
```

This saves images in `captures/`.

4. Run YOLO on one captured image:

```bash
python scripts/yolo/01_detect_person_image.py captures/tello_capture_XXXX_01.jpg
```

5. If people are detected correctly in still images, run live detection:

```bash
python scripts/yolo/03_live_person_detection.py
```

Press `q` in the video window to quit.

6. If the drones are in station mode and registered in `scripts/swarm/drone_ips.txt`, run the IP-scanning live detector:

```bash
python scripts/yolo/04_live_person_detection_save_by_ip.py
```

This tries each registered IP, uses the first drone that responds with video, draws person boxes on screen, and saves one annotated frame per second in `captures/yolo_live/`.
For this live-save script, the default confidence threshold is `0.80`, so the UI only shows a person as present when YOLO is at least 80% confident.

The default model is `yolo11n.pt`, which is small and usually the right first test for laptop + Tello video. You can change model or confidence threshold:

```bash
python scripts/yolo/03_live_person_detection.py --model yolo11s.pt --conf 0.45 --every 5
python scripts/yolo/04_live_person_detection_save_by_ip.py --model yolo11s.pt --conf 0.45 --every 5
```

`--every` controls how often YOLO runs. Higher values reduce CPU load and latency pressure.

## YOLO Fire Detection Workflow

The fire scripts use the Hugging Face model `SuperBitDev/fire1`. The first run needs internet access because it downloads the ONNX weight into `models/fire1/`. The model file is ignored by git. The ONNX post-processing follows the published miner structure: class remap, per-class thresholds/rescue, sanity-box filtering, per-class NMS, cross-class dedup, and horizontal-flip TTA.

Start with the offline image test:

```bash
python scripts/fire/01_detect_fire_image.py
```

If no image path is passed, the script downloads a Wikimedia Commons fire sample into `samples/fire/`, runs the fire model, and writes the annotated result to `captures/fire/`.

To test your own image:

```bash
python scripts/fire/01_detect_fire_image.py path/to/fire_image.jpg
python scripts/fire/01_detect_fire_image.py path/to/fire_image.jpg --profile miner
```

Then test the Tello camera stream without flying:

```bash
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132
```

This opens the first working drone camera, runs fire detection, draws boxes, saves one annotated frame per second to `captures/fire_live/`, and records annotated video to `captures/fire_videos/`. Press `q` to quit.

The live fire script detects `fire` by default. It uses asynchronous inference so ONNX processing does not block the camera preview. It also keeps the last positive detection visible for one second by default, which reduces frame-to-frame flicker when confidence bounces around the threshold. Smoke detection is disabled by default; add `--include-smoke` only if you explicitly want smoke boxes too.

Useful tuning options:

```bash
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --profile candle --hold-seconds 1.5 --every 5
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --profile candle --center-zoom --center-crop 0.70 --every 5 --hold-seconds 2
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --profile candle --tiled --every 8 --hold-seconds 2
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --profile miner
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --include-smoke
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --no-record-video
```

Use `--profile miner` for the exact published threshold behavior. Use `--profile candle` for small flames, which keeps the same miner-style pipeline but lowers fire thresholds and admits smaller boxes. A candle flame can be below the miner's normal fire threshold and may occupy only a few pixels in the Tello stream; move closer, improve lighting/contrast, or show a larger flame image on a screen if you are testing safely.

Use `--center-zoom --center-crop 0.70` when the fire is expected in the central 70% of the camera view. This runs inference on that center crop and maps boxes back to the full frame. It is usually faster than tiled mode because it runs one cropped inference instead of four or five passes.

Use `--tiled` for small-object search when the flame is present but too small in the full frame. Tiled mode runs the detector on the full frame plus four overlapping 2x2 tiles, maps tile detections back onto the full image, then merges duplicates. It is more sensitive to small fires but costs roughly 5x more inference work, so combine it with a larger `--every` value such as `--every 8`.

Safety constraint: do not fly a Tello/RoboMaster TT near real fire, heat, smoke, candles, fireplaces, or people. These scripts are for model testing and visual detection only.

## Fire Approach Agent

The fire approach script uses the same agent pattern as the human-follow script: detection is local, command selection is done by Chutes, and Python enforces hard safety limits. It selects the largest visible fire box, sends that context to Chutes, and validates the response. Smoke is ignored by default.

Chutes is allowed to output only:

- `stop`
- `cw 1..30`
- `ccw 1..30`
- `forward 20..50`

The Python script still owns the dangerous decisions:

- it never asks Chutes to take off or land
- it lands locally when the target looks close enough
- it lands locally when the total forward movement reaches 400 cm
- if no fire is detected, Chutes should command `forward 50`
- if no fire is still detected after two forward search moves, Python lands locally
- once fire has been detected at least once, losing it for 5 continuous seconds makes Python land locally
- it clamps Chutes forward commands to `--forward-step-cm` and the remaining movement budget
- it clamps Chutes yaw direction to the local `--yaw-step`

Run the dry-run first:

```bash
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132
```

The default detection settings are `--profile candle`, full-frame inference, `--detect-every 5`, and `--hold-seconds 2`. The script also reads `CHUTES_API_KEY` and optional `CHUTES_MODEL` from `.env`. Add `--center-zoom --center-crop 0.70` only if you want to use the central crop again.

Only after the dry-run commands look correct, use flight mode:

```bash
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --enable-flight
```

Flight mode takes off, asks Chutes for one command every few seconds, sends at most 400 cm of total forward movement by default, then lands. The landing command runs in a background worker, so the preview and annotated MP4 recording continue during landing. Press `q` to stop; the script will stop motion and attempt to land.

Safer tuning options:

```bash
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --max-forward-cm 200 --forward-step-cm 30
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --max-no-fire-forwards 1
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --lost-fire-land-seconds 3
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --target-coverage 3 --center-tolerance 35
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --center-zoom --center-include-full-frame
```

Do not test this against a real flame. Use a screen showing a fire image/video or another safe visual target first.

## YOLO + Chutes Human Agent

The human-agent script is similar to the `tello-agent` idea, but it keeps the detector local and uses YOLO only for humans. Chutes is used only to choose the next small Tello command from the latest detection JSON.

Create a local `.env` file from `.env.example` and add your key:

```bash
cp .env.example .env
```

```env
CHUTES_API_KEY=your_chutes_api_key_here
CHUTES_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-TEE
TELLO_TARGET_WIFI_SSID=your_router_or_hotspot_ssid
TELLO_TARGET_WIFI_PASSWORD=your_router_or_hotspot_password
```

Run the safe dry-run first:

```bash
python scripts/yolo/05_human_agent_go_to_person.py --ip 192.168.1.132
```

Dry-run mode does not take off and does not send movement commands. It opens the camera, detects people, asks Chutes for the next command, draws the selected target, saves annotated frames to `captures/human_agent/`, and records annotated video to `captures/human_agent_videos/`.

Only after the dry-run commands look sane, enable flight:

```bash
python scripts/yolo/05_human_agent_go_to_person.py --ip 192.168.1.132 --enable-flight
```

The script takes off, asks Chutes for one command every few seconds, validates that command against a small safe list, and lands on exit. By default, a human is considered close enough when the person bounding box covers at least 40% of the image; tune this with `--stop-coverage`.

- `stop`
- `cw 1..30`
- `ccw 1..30`
- `forward 20..50`
- `back 20..40`

The approach loop is intentionally simple: ask Chutes for one command, execute it, wait `--agent-every` seconds, then ask again. The default wait is 3 seconds. Movement commands run in a separate single-command worker so the camera preview and video recording continue while the drone is moving.

Because the target is a human, the model is not allowed to land on the target. When the person fills enough of the image, the script itself rotates clockwise 180 degrees as a visible success signal, then lands. Tune the visible finish turn with `--finish-yaw`.

The MP4 recording includes the same annotations shown in the UI: person bounding boxes, selected target marker, `FOLLOW_HUMAN` mode, latest Chutes command, drone label, battery, and Chutes latency. It records at 20 FPS by default; tune this with `--record-fps`. The recorder uses real-time pacing so occasional slow processing frames do not make the saved video play back too fast. When Windows Python is running from a `\\wsl.localhost\...` repo path, the script records to a local temp file first and copies the finished video back to `captures/human_agent_videos/`; this avoids OpenCV/FFmpeg frame-write warnings on WSL network paths. Disable video recording with `--no-record-video` if needed.

Press `q` in the video window to stop the run. In flight mode, `q` cancels pending agent work, sends `stop`, attempts to land, stops the video stream, saves the MP4, and closes the window. `Ctrl+C` follows the same safe-landing path.

To test Chutes on one saved frame without connecting to the drone:

```bash
python scripts/yolo/06_chutes_frame_drill.py --stop-coverage 40
```

With the sample close-person frame, `40` means the person is close enough, so the expected result is the target-reached path instead of another `forward 50`.

## Two-Drone Wi-Fi Workflow

Use this only with Tello EDU / RoboMaster TT / Tello Talent drones that support the SDK `ap` command.

1. Power on only the new drone.
2. Connect Windows to that drone's `TELLO-*` Wi-Fi.
3. Put your router/hotspot credentials in `.env`, or pass them with `--ssid` and `--password`.
4. Run `python scripts/swarm/09_setup_new_drone.py`.
5. When the drone reboots, make sure Windows reconnects to your router/hotspot Wi-Fi.
6. The script scans the router network and appends any new drone IP to `scripts/swarm/drone_ips.txt`.

After drones are registered in `drone_ips.txt`, this command runs the motor test for every registered drone:

```bash
python scripts/swarm/08_wifi_motor_spin_by_ip.py
```

For repeated testing and flight work, use the interactive swarm controller:

```bash
python scripts/swarm/10_swarm_controller.py
```

Controller commands:

- `help`: show available commands.
- `status` or `battery`: enter SDK mode and print battery/status for each registered drone.
- `spin [seconds]`: spin all motors without takeoff, defaulting to 3 seconds.
- `motoron`: turn motors on for all drones.
- `motoroff`: turn motors off for all drones.
- `takeoff`: take off all drones after checking every battery is at least 30%.
- `land`: land all drones.
- `emergency`: send emergency motor stop to all drones.
- `raw <sdk command>`: send a raw Tello SDK command to all drones.
- `one <ip> <sdk command>`: send a raw Tello SDK command to one drone.
- `quit`: stop/land if needed and exit.

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
