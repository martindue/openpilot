# Batch Process — Offline Driving Model Inference on External Videos

## Purpose

Process dashboard-mounted webcam videos stored on a NAS through openpilot's driving model **offline** (no car, no comma device needed). This produces per-frame driving perception outputs: planned trajectory, lane lines, road edges, lead vehicles, and driver pose — saved as JSON, CSV, and/or annotated video overlays.

## Data Source

Videos are recorded by SmartEye camera systems and stored on a NAS at:

```
/mnt/nas/dmstagingarea01/DAC160/Recordings/
```

Each recording is a subdirectory containing:

```
DAC160-001-A1-0-3782-12852-20251112T071831/
├── Cam001.avi                              # Video file (2560x1440 @ 30fps, or 1920x1080)
├── *_gva.csv                               # ~100Hz sensor log (speed, IMU, GPS)
├── *.sel / *.sma / *.smb                   # SmartEye gaze/tracking data (not used)
└── recording.xml                           # Recording metadata (not used)
```

The naming convention is `{camera}-{subject}-{route}-{segment}-{ids}-{timestamp}`.

### GVA CSV format

The `*_gva.csv` file contains sensor data at ~100Hz with columns including:
- `FILETIME_100ns_UTC` — Windows FILETIME timestamp (100ns ticks since 1601-01-01)
- `Speed (kph)` — GPS-derived vehicle speed
- `AP can.speed` — CAN bus speed (fallback when GPS unavailable)
- `Roll`, `Pitch`, `Yaw` — orientation in degrees
- `Acceleration X/Y/Z (g)`, `Angular velocity X/Y/Z (degree/s)` — IMU data

## Quick Start

```bash
# From the openpilot repo root, with the .venv activated:

# CPU (default) — ~4 fps
python -m openpilot.tools.batch_process.run \
  --input-dir /mnt/nas/dmstagingarea01/DAC160/Recordings/ \
  --camera-config tools/batch_process/cameras/dac160.json \
  --output-dir /tmp/openpilot_batch_output \
  --formats json,csv,video

# GPU (CUDA) — ~38 fps on MX450
python -m openpilot.tools.batch_process.run \
  --input-dir /mnt/nas/dmstagingarea01/DAC160/Recordings/ \
  --camera-config tools/batch_process/cameras/dac160.json \
  --output-dir /tmp/openpilot_batch_output \
  --formats json,csv,video \
  --device cuda
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | *required* | Path to recordings directory on NAS |
| `--camera-config` | *required* | Path to camera intrinsics JSON file |
| `--output-dir` | *required* | Where to write results |
| `--formats` | `json,csv` | Comma-separated: `json`, `csv`, `video` |
| `--device` | `cpu` | Tinygrad device: `cpu`, `cuda`, `amd`, `nv`, `cl`, `metal` |
| `--glob-pattern` | `*/Cam001.avi` | Glob pattern to find video files |
| `--mount-rpy` | `0,0,0` | Camera mount roll, pitch, yaw in degrees |
| `--traffic-convention` | `lhd` | `lhd` (left-hand drive / right-side traffic) or `rhd` |
| `--default-speed` | `0.0` | Fallback speed in km/h when no GVA CSV found |
| `--max-videos` | `0` (all) | Limit number of videos to process |
| `--max-frames` | `0` (all) | Limit frames per video |

## Output Formats

Per recording, outputs are written to `{output-dir}/{recording-name}/`:

- **`model_outputs.json`** — Full model outputs per frame: plan (position, velocity, acceleration, orientation), lane lines (4 lines × 33 points), road edges (2 edges × 33 points), lead vehicles (3 leads with position/velocity/probability), pose, and metadata.
- **`model_outputs.csv`** — Flattened summary per frame: frame_id, elapsed_time, v_ego, lateral offset at key distances, lead distance/speed, lane width, path curvature.
- **`annotated.mp4`** — Input video with overlaid lane lines, road edges, planned path, lead vehicle markers, and speed readout.

## Architecture

```
run.py                  CLI entry point, video discovery, orchestration
│
├── camera_config.py    Load/scale camera intrinsics from JSON
├── preprocessor.py     Undistort → remap to 1928×1208 → BGR→NV12
├── gva_reader.py       Parse *_gva.csv, interpolate speed per frame time
├── feeder.py           VisionIPC server + cereal message publishing
├── model_runner.py     Feed frames through ModelState, yield per-frame results
├── outputs.py          Collect results → JSON/CSV serialization
├── annotator.py        Overlay model outputs on video frames → MP4
│
└── cameras/
    └── dac160.json     Camera intrinsics for DAC160 at 1080p reference
```

### Processing Pipeline (per frame)

1. **Read** BGR frame from video via OpenCV
2. **Preprocess** — undistort lens distortion, remap to 1928×1208 with centered principal point, convert BGR→NV12
3. **Feed** — send NV12 frame via VisionIPC, publish cereal messages (roadCameraState, carState with speed, liveCalibration)
4. **Infer** — ModelState runs the two-stage tinygrad model (vision encoder → policy decoder)
5. **Collect** — extract plan, lane lines, road edges, leads, pose from raw model output
6. **Save** — serialize to JSON/CSV; optionally re-read video and draw annotations

### Key Technical Details

- **Camera intrinsics**: Specified at a reference resolution (1080p for DAC160). The preprocessor scales them to the actual source video resolution, then computes undistort+remap tables that produce a 1928×1208 output with centered principal point — matching openpilot's internal `CameraConfig` assumption.
- **Speed injection**: Vehicle speed (`vEgo`) is critical for the model's longitudinal predictions. Speed is read from the co-located `*_gva.csv` file (interpolated to frame timestamps). If no CSV exists, falls back to `--default-speed`.
- **Device selection**: The tinygrad model is JIT-compiled to device-specific pickle files. `--device cuda` loads `*_tinygrad_cuda.pkl`, `--device cpu` loads `*_tinygrad_cpu.pkl`. The default `.pkl` files are whichever was last compiled. See "Compiling Model Pickles" below.
- **ZMQ backend**: Offline processing uses ZMQ for cereal messaging (set via `ZMQ=1` env var), since the native shared-memory IPC requires the full openpilot daemon.
- **modeld.py DEV override**: `selfdrive/modeld/modeld.py` line 4 unconditionally sets `DEV='CPU'` on non-TICI hardware. `model_runner.py` saves and restores the user's `DEV` choice around this import.

## Adding a New Camera

1. Calibrate the camera to obtain intrinsics (focal length, principal point, distortion coefficients) at a known reference resolution.
2. Create a JSON file in `cameras/` following the format:

```json
{
  "name": "MyCam",
  "description": "Description, intrinsics at 1080p reference",
  "imageWidth": 1920,
  "imageHeight": 1080,
  "intrinsics": {
    "focalLengthX": 1130.40,
    "focalLengthY": 1130.40,
    "principalPointX": 960.0,
    "principalPointY": 540.0,
    "distortionR2": 0.0,
    "distortionR4": 0.0,
    "distortionR6": 0.0,
    "distortionT1": 0.0,
    "distortionT2": 0.0
  }
}
```

3. Pass `--camera-config cameras/mycam.json` to the CLI. The intrinsics are automatically scaled to match the actual video resolution.

## Compiling Model Pickles for GPU

The pre-compiled pickle files are device-specific. To use a GPU, you need to compile pickles for that device:

```bash
cd selfdrive/modeld/models

# Back up existing CPU pickles
cp driving_vision_tinygrad.pkl driving_vision_tinygrad_cpu.pkl
cp driving_policy_tinygrad.pkl driving_policy_tinygrad_cpu.pkl

# Compile for CUDA
DEV=CUDA .venv/bin/python3 ../../tinygrad_repo/examples/openpilot/compile3.py $(pwd)/driving_vision.onnx
DEV=CUDA .venv/bin/python3 ../../tinygrad_repo/examples/openpilot/compile3.py $(pwd)/driving_policy.onnx

# Save CUDA pickles with device suffix
cp driving_vision_tinygrad.pkl driving_vision_tinygrad_cuda.pkl
cp driving_policy_tinygrad.pkl driving_policy_tinygrad_cuda.pkl
```

The `--device` flag then selects the correct pickles at runtime. Available devices depend on tinygrad support and your hardware: `CPU`, `CUDA`, `NV`, `AMD`, `CL`, `METAL`.

## Performance

Measured on 2560×1440 @ 30fps input video, NVIDIA GeForce MX450 (2GB VRAM):

| | CPU | CUDA (MX450) |
|---|---|---|
| Preprocess | ~13 ms | ~8 ms |
| Model inference | ~215 ms | ~17 ms |
| **Total per frame** | **~228 ms** | **~27 ms** |
| **Throughput** | **4.4 fps** | **37.5 fps** |

GPU mode exceeds the 30 fps source rate, enabling faster-than-realtime batch processing.
