<a id="readme-top"></a>

<div align="center">

<img src="assets/Swarm_2.png" alt="Swarm" width="60%" />

# PoC: Real-Time Fire Detection

Fire detection from a DJI RoboMaster TT / Tello camera using Score's top Detect-fire skill.

Live camera -> fire model -> annotated evidence -> Chutes decision -> validated drone command.

</div>

## Demo

<p align="center">
  <b>PoC: Fire Tracker running on a real Tello drone</b>
</p>

<p align="center">
  <a href="https://youtu.be/isjBdhPj0QY" target="_blank" rel="noopener noreferrer">
    <img src="https://img.youtube.com/vi/isjBdhPj0QY/maxresdefault.jpg" alt="PoC Fire Tracker drone flight - click to watch" width="720" loading="lazy" />
  </a>
</p>

<p align="center">
  <a href="https://youtu.be/isjBdhPj0QY" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Watch%20on-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch on YouTube" />
  </a>
</p>

## What This Is

This repository is a focused proof of concept for one use case: a Tello-class drone observes a controlled fire visual target, detects it locally, records annotated evidence, and uses Chutes to choose the next cautious movement command.

The tested loop is:

```text
Tello camera
  -> OpenCV frame
  -> Score / Manako fire detector
  -> target position and size
  -> Chutes command decision
  -> local safety validator
  -> Tello SDK command
  -> annotated video
```

The Tello setup scripts are support tools. They help validate connectivity, camera streaming, station-mode networking, and bench tests before running the fire PoC.

Want to build your own autonomous drone? Check out [Langostino](https://github.com/swarm-subnet/Langostino), Swarm's open-source autonomous drone platform.

## Fire Model

The fire detector uses Score's current top Detect-fire miner from Manako:

```text
manak0/Detect-fire
https://console.scorevision.io/elements/manak0%2FDetect-fire
```

Thanks to Manako and Score for making the model available.

The compatible ONNX weights are downloaded and cached from:

```text
SuperBitDev/fire1
```

The implementation in [src/fire_utils.py](src/fire_utils.py) prepares each camera frame, runs ONNX Runtime locally, filters weak boxes, and returns clean `fire` detections for drawing and control. Smoke detection exists behind `--include-smoke`, but the default PoC tracks fire only.

## Chutes Integration

Chutes is used as the command-selection layer. It does not directly control the drone.

For each decision cycle, Python sends Chutes a compact JSON summary:

- whether fire is visible
- selected target confidence
- target position in the frame
- horizontal offset from center
- target coverage percentage
- remaining forward movement budget
- current search state
- last executed command

Chutes must return exactly one small Tello SDK command:

```text
stop
cw 1..30
ccw 1..30
forward 20..50
```

Python validates the response before execution. Takeoff, landing, emergency shutdown behavior, movement caps, and lost-target handling remain local in the script.

## Quick Start

### 1. Install

```bash
git clone https://github.com/swarm-subnet/PoC-Real-time-fire-detection.git
cd PoC-Real-time-fire-detection

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv venv-win
.\venv-win\Scripts\Activate.ps1
pip install -r requirements.txt
```

Native Windows Python is recommended for Tello video streaming.

### 2. Configure Chutes

Create a local `.env` file:

```bash
cp .env.example .env
```

Set:

```env
CHUTES_API_KEY=your_chutes_api_key_here
CHUTES_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-TEE
```

`.env` is ignored by git. Do not commit API keys or local network credentials.

### 3. Connect The Drone

For a direct Tello connection:

1. Power on the RoboMaster TT / Tello.
2. Connect the laptop to the `TELLO-*` Wi-Fi network.
3. Close the Tello / RoboMaster mobile app.
4. Check the camera stream:

```bash
python scripts/04_camera_preview.py
```

For station-mode drones, pass the detected drone IP with `--ip`. Station-mode setup utilities are under [scripts/swarm](scripts/swarm), but router credentials are not stored in this repository.

## Run The Fire PoC

### Offline Image Test

This downloads the fire model if needed, downloads a sample fire image if no image is provided, and saves an annotated result.

```bash
python scripts/fire/01_detect_fire_image.py
```

Use your own image:

```bash
python scripts/fire/01_detect_fire_image.py path/to/fire_image.jpg
```

### Live Camera Detection, No Flight

This is the main camera-only test. It opens the drone stream, detects fire, draws boxes, saves frames, and records video.

```bash
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132
```

If `--ip` is omitted, the script tries IPs from [scripts/swarm/drone_ips.txt](scripts/swarm/drone_ips.txt), then falls back to direct mode at `192.168.10.1`.

Useful options:

```bash
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --profile candle
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --center-zoom --center-crop 0.70
python scripts/fire/02_live_fire_detection_save_by_ip.py --ip 192.168.1.132 --tiled --every 8
```

### Agent Dry Run, No Flight

Dry run opens the camera, detects fire, asks Chutes for the command it would run, records video, but does not take off and does not move.

```bash
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132
```

Expected behavior:

- fire centered and far: usually `forward 50`
- fire left: small `ccw`
- fire right: small `cw`
- fire close enough: stop / target reached path
- no fire: at most two forward-search decisions before landing logic triggers

### Agent Flight Mode

Run this only after the dry-run video and printed Chutes commands look correct.

```bash
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --enable-flight
```

Conservative options:

```bash
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --enable-flight --max-forward-cm 200 --forward-step-cm 30
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --enable-flight --max-no-fire-forwards 1
python scripts/fire/03_fire_agent_go_to_fire.py --ip 192.168.1.132 --enable-flight --lost-fire-land-seconds 3
```

Press `q` in the OpenCV window to request shutdown. The script attempts to stop motion, land if airborne, stop the stream, save the recording, and close cleanly.

## Outputs

Generated evidence is written under ignored `captures/` folders:

```text
captures/fire/
captures/fire_live/
captures/fire_videos/
captures/fire_agent/
captures/fire_agent_videos/
```

The cached model is written under ignored `models/fire1/`.

## Support Tools

Camera and drone validation:

```bash
python scripts/01_connection_test.py
python scripts/02_motor_spin_test.py
python scripts/03_first_flight.py
python scripts/04_camera_preview.py
```

Station-mode and multi-drone utilities:

```bash
python scripts/swarm/09_setup_new_drone.py --ssid <wifi-name> --password <wifi-password>
python scripts/swarm/08_wifi_motor_spin_by_ip.py
python scripts/swarm/10_swarm_controller.py
```

Depth experiments are intentionally documented separately in [scripts/depth/README.md](scripts/depth/README.md).

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- requirements-depth.txt
|-- scripts/
|   |-- 01_connection_test.py
|   |-- 02_motor_spin_test.py
|   |-- 03_first_flight.py
|   |-- 04_camera_preview.py
|   |-- fire/
|   |   |-- 01_detect_fire_image.py
|   |   |-- 02_live_fire_detection_save_by_ip.py
|   |   `-- 03_fire_agent_go_to_fire.py
|   |-- depth/
|   |   |-- README.md
|   |   `-- 01_depth_capture.py
|   `-- swarm/
|       |-- 08_wifi_motor_spin_by_ip.py
|       |-- 09_setup_new_drone.py
|       |-- 10_swarm_controller.py
|       `-- drone_ips.txt
|-- src/
|   |-- chutes_agent.py
|   |-- detection_utils.py
|   |-- depth_utils.py
|   |-- env_utils.py
|   |-- fire_utils.py
|   |-- media_utils.py
|   |-- tello_stream.py
|   `-- tello_utils.py
`-- samples/
    `-- fire/
```

## Troubleshooting

### No drone connection

- Confirm the laptop is connected to the correct `TELLO-*` network or the same router as the station-mode drone.
- Close the Tello / RoboMaster mobile app.
- Power-cycle the drone.
- In direct mode, the drone normally responds at `192.168.10.1`.

### No video window

- Use native Windows Python if WSL video or UDP forwarding is unreliable.
- Confirm `opencv-python` installed.
- Close any other app using the Tello video stream.

### Fire is not detected

- Start with `python scripts/fire/01_detect_fire_image.py`.
- Use `--profile candle` for small controlled flames.
- Make the target larger in the frame or move the camera closer.
- Try `--center-zoom` when the target is expected near the center.
- Try `--tiled --every 8` for small targets if your CPU can handle slower inference.

### Chutes command is slow

- The agent script prints Chutes latency.
- Keep `--decision-every` at a few seconds so motion, camera, and recording stay stable.
- The local safety validator still checks every response before execution.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
