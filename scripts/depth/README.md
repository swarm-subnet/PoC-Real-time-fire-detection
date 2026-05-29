# Depth Experiments

These scripts are not part of the main fire-detection PoC. They are included for sim-to-real experiments: converting real Tello RGB camera frames into depth-like observations that can be compared with Swarm Subnet simulation depth.

```text
Tello RGB camera
  -> monocular depth model
  -> metric depth map
  -> 128x128 normalized depth array
  -> visual depth images and CSV statistics
```

The current implementation uses:

```text
depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
```

The model estimates depth in meters. The saved normalized depth follows the subnet convention:

```text
0.0 = 0.5m or closer
1.0 = 20m or farther
```

## Install CPU Dependencies

```bash
pip install -r requirements-depth.txt
```

For Windows/NVIDIA CUDA 12.9 testing, use the root `requirements-windows-cuda.txt` instead.

## Commands

Run one local image:

```bash
python scripts/depth/01_depth_capture.py image path/to/image.jpg
```

Run over saved Tello RGB frames:

```bash
python scripts/depth/01_depth_capture.py folder captures/depth_live/<run-id> --limit 10
```

Run live depth capture:

```bash
python scripts/depth/01_depth_capture.py live --ip 192.168.1.132
```

Use CUDA when available:

```bash
python scripts/depth/01_depth_capture.py --device cuda live --ip 192.168.1.132
```

Each saved sample includes the source RGB image, colorized depth image, normalized depth PNG, `128x128` `.npy` array, raw meter-depth `.npy` array, side-by-side image, and a `depth_stats.csv` row.
