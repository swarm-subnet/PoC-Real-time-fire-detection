# Swarm Scripts

Swarm entrypoints are grouped by purpose. Run commands from the repository root.

## Provisioning

Configure drones onto the target router/hotspot Wi-Fi and register station-mode IPs.

```powershell
.\venv-win\Scripts\python.exe scripts\swarm\provisioning\14_auto_setup_tello_wifi.py
```

Manual one-drone setup, when already connected to a `TELLO-*` AP:

```powershell
.\venv-win\Scripts\python.exe scripts\swarm\provisioning\09_setup_new_drone.py --ssid <wifi-name> --password <wifi-password>
```

## Dashboard

Live status, video wall, motor spin, guarded takeoff, and best-effort landing.
By default the dashboard also runs a person-only YOLO detector on the live
camera frames and overlays bounding boxes on each tile.

```powershell
.\venv-win\Scripts\python.exe scripts\swarm\dashboard\13_swarm_dashboard.py
```

Optional detector override:

```powershell
.\venv-win\Scripts\python.exe scripts\swarm\dashboard\13_swarm_dashboard.py --person-imgsz 416
```

Person detection is CUDA-only. If PyTorch cannot use GPU `cuda:0`, the detector
reports an error instead of falling back to CPU. By default the dashboard uses
`yolo11s.pt` at detector size 640, plus the lower original Tello stream profile
for smoother multi-drone display. Use 416 only if you need more speed after
confirming boxes appear reliably.

## Diagnostics

Bench propeller check:

```powershell
.\venv-win\Scripts\python.exe scripts\swarm\diagnostics\08_wifi_motor_spin_by_ip.py
```

Battery drain while props spin:

```powershell
.\venv-win\Scripts\python.exe scripts\swarm\diagnostics\11_battery_motor_drain_by_ip.py --duration 180
```

Hover battery test:

```powershell
.\venv-win\Scripts\python.exe scripts\swarm\diagnostics\12_swarm_hover_battery_test.py --hover-seconds 60
```

GPU person detector test:

```powershell
.\venv-win\Scripts\python.exe scripts\swarm\diagnostics\16_test_gpu_person_detector.py
```

The default benchmark runs 10 warmup batches and 100 measured batches of 5
frames, matching the expected five-drone camera wall.

## Control

Legacy interactive controller:

```powershell
.\venv-win\Scripts\python.exe scripts\swarm\control\10_swarm_controller.py
```

## Config

Registered station-mode drone IPs live in:

```text
scripts/swarm/config/drone_ips.txt
```
