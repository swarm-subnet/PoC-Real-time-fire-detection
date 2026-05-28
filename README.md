<a id="readme-top"></a>

# Swarm Tello Fire Detection PoC

<p align="center">
  <b>Fire detection from a DJI RoboMaster TT / Tello camera, with an optional Chutes-powered flight agent.</b><br/>
  A small proof of concept for turning a drone camera feed into fire awareness, annotated evidence, and cautious autonomous movement commands.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Live%20Video-5C3EE8?style=flat-square&logo=opencv&logoColor=white" />
  <img alt="ONNX Runtime" src="https://img.shields.io/badge/ONNX%20Runtime-Fire%20Model-111111?style=flat-square" />
  <img alt="Chutes" src="https://img.shields.io/badge/Chutes-LLM%20Agent-111111?style=flat-square" />
  <img alt="Tello" src="https://img.shields.io/badge/DJI%20Tello-SDK%20Control-222222?style=flat-square" />
</p>

## What This Is

This repository is a practical fire-detection proof of concept built around the RoboMaster TT / Tello Talent drone. The main experiment is:

- stream the drone camera to a laptop
- detect fire locally with an ONNX fire model
- draw bounding boxes and status overlays in real time
- save annotated frames and videos for review
- optionally ask Chutes for a single safe movement command
- optionally fly toward the detected fire target under strict local safety limits

The Tello scripts are still included, but they are not the main point of the project anymore. They are support tools for validating connectivity, camera streaming, station-mode networking, and safe bench tests before running the fire PoC.

> [!CAUTION]
> This is an experimental robotics proof of concept. Do not fly near real fire, heat, smoke, candles, people, pets, or fragile objects. For flight tests, use a controlled visual target such as a fire image or fire video displayed on a screen.

## Why It Matters

The goal is to make aerial perception experiments understandable and reproducible. Instead of hiding the autonomy loop inside a black box, this repo keeps the pipeline explicit:

```text
Tello camera
  -> OpenCV frame
  -> fire detector
  -> target geometry
  -> Chutes command suggestion
  -> local safety validator
  -> Tello SDK command
  -> annotated video evidence
```

That means you can inspect each step independently: model output, Chutes prompt context, validated command, drone action, and recorded result.

## Built With

- `djitellopy` for Tello / RoboMaster TT SDK control
- `opencv-python` for live camera display, annotation, and video recording
- `onnxruntime` for the fire model inference path
- `ultralytics` for YOLO person-detection experiments
- Chutes LLM API for command selection in the agent scripts

## Fire Model

The fire detector uses the current top Score / Manako fire miner model:

```text
manak0/Detect-fire
```

Model page:

```text
https://console.scorevision.io/elements/manak0%2FDetect-fire
```

Thanks to Manako and Score for making this model available. This PoC uses it as the perception layer: the model looks at each camera frame and returns boxes around visual fire-like regions so the rest of the system can decide what to display, record, or do next.

The code downloads and caches the compatible ONNX weights from:

```text
SuperBitDev/fire1
```

Cached model path:

```text
models/fire1/weights.onnx
```

In simple terms, [src/fire_utils.py](src/fire_utils.py) does four things:

- prepares the drone image in the format the model expects
- runs the fire model locally on the laptop
- filters out weak or obviously bad boxes
- returns clean `fire` detections that can be drawn on the video or sent to the agent

By default, the live scripts focus on `fire` only. Smoke is disabled unless you explicitly pass `--include-smoke`.

Two profiles are available:

- `miner`: closer to the published miner thresholds.
- `candle`: more sensitive for small flames and controlled indoor tests.

## Chutes Integration

The flight-agent scripts do not let the LLM directly fly the drone. Chutes receives a compact JSON context derived from the current detection:

- whether fire is visible
- selected target label and confidence
- bounding box position
- horizontal offset from frame center
- object coverage percentage
- remaining forward movement budget
- whether search moves are still allowed
- current airborne state and last command

Chutes must return exactly one small Tello SDK command:

```text
stop
cw 1..30
ccw 1..30
forward 20..50
```

Python then validates and clamps the result locally. The Python code, not Chutes, owns the dangerous decisions:

- takeoff is only done when `--enable-flight` is passed
- landing is handled locally
- total forward movement is capped
- yaw angle is clamped locally
- if no fire is found after two search-forward moves, the drone lands
- if fire was seen and then lost for several seconds, the drone lands
- `q` or `Ctrl+C` attempts to stop motion, land, save video, and close cleanly

The Chutes helper lives in [src/chutes_agent.py](src/chutes_agent.py).

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/swarm-subnet/swarm-tello-drone.git
cd swarm-tello-drone

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv venv-win
.\venv-win\Scripts\Activate.ps1
pip install -r requirements.txt
```

Native Windows Python is recommended for video display and Tello UDP streaming. WSL can work for some control-only tests, but Windows networking is usually simpler for camera work.

### 2. Configure Chutes

Create a local `.env` file:

```bash
cp .env.example .env
```

Fill in:

```env
CHUTES_API_KEY=your_chutes_api_key_here
CHUTES_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-TEE
TELLO_TARGET_WIFI_SSID=your_router_or_hotspot_ssid
TELLO_TARGET_WIFI_PASSWORD=your_router_or_hotspot_password
```

`.env` is ignored by git. Do not commit API keys or Wi-Fi passwords.

### 3. Connect the drone

For one direct drone:

1. Power on the RoboMaster TT / Tello.
2. Connect the laptop to the `TELLO-*` Wi-Fi network.
3. Close the Tello / RoboMaster mobile app.
4. Run a connection or camera script.

For station-mode / multi-drone tests:

1. Power on one new drone.
2. Connect Windows to that drone's `TELLO-*` Wi-Fi.
3. Put router credentials in `.env`, or pass `--ssid` and `--password`.
4. Run:

```bash
python scripts/swarm/09_setup_new_drone.py
```

The script sends the Tello `ap` command, waits for reboot, scans the router network, and appends the new IP to [scripts/swarm/drone_ips.txt](scripts/swarm/drone_ips.txt).

## Fire PoC Workflow

### Step 1: Offline image test

This downloads the fire model if needed, downloads a sample fire image if no image is provided, and writes an annotated result.

```bash
python scripts/fire/01_detect_fire_image.py
```

Test your own image:

```bash
python scripts/fire/01_detect_fire_image.py path/to/fire_image.jpg
python scripts/fire/01_detect_fire_image.py path/to/fire_image.jpg --profile miner
```

Outputs go to:

```text
captures/fire/
```

### Step 2: Live fire detection, no flight

Use this before any autonomous movement.

```bash
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132
```

If `--ip` is omitted, the script tries IPs from [scripts/swarm/drone_ips.txt](scripts/swarm/drone_ips.txt), then falls back to `192.168.10.1`.

What it does:

- opens the first usable drone video stream
- runs fire detection asynchronously
- draws fire bounding boxes and status text
- saves one annotated frame per second
- records annotated video
- exits with `q`

Outputs:

```text
captures/fire_live/
captures/fire_videos/
```

Useful options:

```bash
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --profile candle --every 5 --hold-seconds 2
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --profile miner
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --center-zoom --center-crop 0.70
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --tiled --every 8
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --no-record-video
```

Use `--center-zoom` when the target is expected near the image center. Use `--tiled` when the fire is very small, but expect slower inference because it runs multiple crops per detection cycle.

### Step 3: Fire agent dry run

Dry run opens the camera, detects fire, asks Chutes what command it would run, records video, but does not take off and does not move.

```bash
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132
```

Expected behavior:

- if fire is centered and far, Chutes should usually suggest `forward 50`
- if fire is left, Chutes should suggest a small `ccw`
- if fire is right, Chutes should suggest a small `cw`
- if fire is close enough, Python enters the target-reached landing path
- if no fire is found, Python allows at most two forward-search commands before landing logic triggers

Outputs:

```text
captures/fire_agent/
captures/fire_agent_videos/
```

### Step 4: Fire agent flight mode

Run this only after dry-run commands and video overlays look correct.

```bash
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --enable-flight
```

Conservative test options:

```bash
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --enable-flight --max-forward-cm 200 --forward-step-cm 30
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --enable-flight --max-no-fire-forwards 1
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --enable-flight --lost-fire-land-seconds 3
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --enable-flight --target-coverage 3 --center-tolerance 35
```

In flight mode, pressing `q` in the OpenCV window requests shutdown. The script attempts to stop motion, land, stop the video stream, save the recording, and close the window.

## Supporting Scripts

### Tello validation

Use these to validate hardware and networking before the fire PoC:

```bash
python scripts/01_connection_test.py
python scripts/02_motor_spin_test.py
python scripts/03_first_flight.py
python scripts/04_camera_preview.py
```

### Human detection experiments

The human scripts are kept as a parallel reference implementation for YOLO + Chutes control:

```bash
python scripts/yolo/01_detect_person_image.py samples/yolo/person_crosswalk.jpg
python scripts/yolo/04_live_person_detection_save_by_ip.py --ip 192.168.1.132
python scripts/yolo/05_human_agent_go_to_person.py --ip 192.168.1.132
```

### Swarm utilities

After multiple drones are registered in [scripts/swarm/drone_ips.txt](scripts/swarm/drone_ips.txt):

```bash
python scripts/swarm/08_wifi_motor_spin_by_ip.py
python scripts/swarm/10_swarm_controller.py
```

The swarm motor script binds local UDP port `8889`, which avoids the station-mode issue where a drone may ignore commands from changing UDP source ports.

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- scripts/
|   |-- 01_connection_test.py
|   |-- 02_motor_spin_test.py
|   |-- 03_first_flight.py
|   |-- 04_camera_preview.py
|   |-- fire/
|   |   |-- 01_detect_fire_image.py
|   |   |-- 02_live_fire_detection_save_by_ip.py
|   |   `-- 03_fire_agent_go_to_fire.py
|   |-- swarm/
|   |   |-- 08_wifi_motor_spin_by_ip.py
|   |   |-- 09_setup_new_drone.py
|   |   |-- 10_swarm_controller.py
|   |   `-- drone_ips.txt
|   `-- yolo/
|       |-- 01_detect_person_image.py
|       |-- 04_live_person_detection_save_by_ip.py
|       `-- 05_human_agent_go_to_person.py
|-- src/
|   |-- chutes_agent.py
|   |-- env_utils.py
|   |-- fire_utils.py
|   |-- media_utils.py
|   |-- tello_stream.py
|   |-- tello_utils.py
|   `-- yolo_utils.py
`-- samples/
    |-- fire/
    `-- yolo/
```

## Safety Checklist

- Run the fire agent in dry-run before flight mode.
- Do not fly near real fire, heat, smoke, candles, people, pets, or fragile objects.
- Use a large clear indoor area with good lighting.
- Use prop guards when possible.
- Keep hands, hair, clothing, and cables away from propellers.
- Confirm battery is above the configured minimum.
- Keep the drone low and within line of sight.
- Keep one hand ready to press `q` or `Ctrl+C`.
- If behavior looks wrong, stop the script and land.

## Troubleshooting

### No drone connection

- Confirm the laptop is on the correct `TELLO-*` Wi-Fi or router Wi-Fi.
- Close the Tello / RoboMaster mobile app.
- Power-cycle the drone.
- On Windows, confirm the Wi-Fi adapter has a valid `192.168.10.x` address for direct mode.

### No video window

- Run from native Windows Python if WSL video or UDP forwarding is unreliable.
- Confirm `opencv-python` installed.
- Close any other app using the Tello video stream.

### Fire is not detected

- Start with `scripts/fire/01_detect_fire_image.py`.
- Try `--profile candle` for small flames.
- Increase flame size in the frame or move the camera closer.
- Try `--center-zoom` if the target is near the center.
- Try `--tiled --every 8` if the target is small and CPU can handle it.

### Chutes command is slow

- The script prints Chutes latency.
- Keep `--decision-every` at a few seconds so motion, camera, and recording stay stable.
- The local safety layer still validates every response.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
