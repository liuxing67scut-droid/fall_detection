# Fall Detection

[中文](README.md) | [English](README_EN.md)

A single-person indoor fall detection prototype based on Raspberry Pi 4B, a USB camera, and OpenCV. The project uses an edge-side capture and visualization pipeline with optional PC-assisted pose inference to improve pose detection responsiveness under the limited computing resources of Raspberry Pi.

## Features

- Developed as the fall detection module for a course-design "smart home robot". The related devices can be mounted on the robot and used to detect and report falls while the robot patrols indoor environments. The current version is limited to single-person scenarios.
- Supports both `YOLO11 + Box` and `MediaPipe + Pose` pipelines, making it possible to compare bounding-box-based detection with pose-keypoint-based detection.
- Provides a `MediaPipe (PC) + Pose (Pi)` collaborative mode: Raspberry Pi handles camera capture and visualization, while the PC runs the heavier pose inference model.
- Fall detection does not rely on a single-frame result. It uses state-machine logic based on body-box geometry, pose angle, hip drop speed, and temporal continuity.
- Provides a reserved fall alert upload interface for sending screenshots and alert metadata to a private website or another server.

## Runtime Modes

After startup, the program shows a main menu with the following modes:

| Menu | Mode | Description |
| --- | --- | --- |
| `1` | `HOG + Box` | Uses OpenCV HOG pedestrian detection as a lightweight fallback, mainly for validating the project pipeline |
| `2` | `YOLO11 + Box` | Uses YOLO11 ONNX with OpenCV DNN to detect person boxes, then applies tracking and box-geometry rules for fall detection |
| `3` | `MediaPipe + Pose` | Runs the local MediaPipe/OpenCV Zoo pose pipeline on Raspberry Pi, draws 33 pose keypoints, and applies pose-rule fall detection |
| `4` | `MediaPipe (PC) + Pose (Pi)` | Raspberry Pi handles capture and display, while a PC-side FastAPI service runs MediaPipe Pose inference; Raspberry Pi continues visualization and fall-state logic |

Keyboard shortcuts:

- `q` / `Esc`: exit
- `s`: save the current snapshot to `data/`
- `i`: show or hide status information
- `j`: show or hide skeleton keypoints in pose modes

## System Architecture

```text
USB Camera
    |
    v
CameraManager
    |
    +--> bbox pipeline
    |       PersonDetector(HOG / YOLO11 ONNX)
    |       -> SinglePersonTracker
    |       -> FallDetector
    |       -> Renderer
    |
    +--> pose pipeline
            local:  MPPersonDetector -> MPPoseEstimator
            remote: PoseRemoteClient -> PC Pose Server
            -> PoseFallDetector
            -> Renderer

FallDetector / PoseFallDetector
    -> FallAlertClient
    -> Platform Server POST /api/events/fall
```

Core modules:

- `camera/CameraManager`: camera opening, parameter setup, background capture thread, and latest-frame delivery
- `detect/PersonDetector`: YOLO11 ONNX inference, HOG fallback, and single-person detection filtering
- `tracking/SinglePersonTracker`: single-target continuity tracking, short-term missing-frame holding, and box smoothing
- `fall/FallDetector`: bbox fall state machine based on aspect ratio, body-height baseline, and temporal continuity
- `pose/MPPersonDetector`, `pose/MPPoseEstimator`: local MediaPipe/OpenCV Zoo pose pipeline
- `remote/PoseRemoteClient`: asynchronous JPEG upload, PC pose-service requests, and keypoint response parsing
- `fall/PoseFallDetector`: pose fall state machine based on trunk angle, vertical span, hip drop speed, knee angle, and temporal continuity
- `alert/FallAlertClient`: asynchronous screenshot and event metadata upload when a fall is raised or cleared
- `ui/Renderer`: preview rendering for frames, boxes, skeletons, and status text

## Project Structure

```text
fall_detect/
├── config/              # Runtime configuration
├── doc/                 # Design and integration documents
├── include/             # C++ headers
├── models/              # Model files
├── pc_pose_server/      # PC-side service
├── src/                 # C++ source code
└── CMakeLists.txt
```

## Dependencies

Main program:

- C++17
- CMake 3.16+
- OpenCV with `core`, `dnn`, `highgui`, `imgcodecs`, `imgproc`, `objdetect`, and `videoio`
- libcurl, optional; required for remote pose mode and fall alert upload

PC pose service:

- Python 3.10+
- FastAPI
- Uvicorn
- MediaPipe
- Pillow
- NumPy

Python dependencies are listed in `pc_pose_server/requirements.txt`.

## Build and Run

### Raspberry Pi / Linux

`~/fall_detect` is an example path. Replace it with the actual project path on your Raspberry Pi.

First build:

```bash
cd ~/fall_detect
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

If `build/` has already been configured, daily use only requires rebuilding and starting the program:

```bash
cd ~/fall_detect
cmake --build build -j4
./build/fall_detect
```

### PC Pose Service

Mode `4) MediaPipe (PC) + Pose (Pi)` requires the PC-side service to be started first.

The Windows path below is an example from the current development machine. Replace it with the actual project path on your own PC.

Initial environment setup:

```powershell
cd "c:\Users\17646\Desktop\fall_detect\pc_pose_server"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the service later:

```powershell
cd "c:\Users\17646\Desktop\fall_detect\pc_pose_server"
.\.venv\Scripts\Activate.ps1
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

The PC service also requires a MediaPipe Pose Landmarker Lite model:

```text
pc_pose_server/models/pose_landmarker_lite.task
```

You can also specify a custom model path with an environment variable:

```powershell
$env:PC_POSE_MODEL_PATH="C:\path\to\pose_landmarker_lite.task"
```

Health check:

```bash
curl http://<pc-ip>:8000/health
```

Configure the remote pose service URL on Raspberry Pi in `config/app.yaml`:

```yaml
remote_pose_server_url: "http://<pc-ip>:8000"
```

## Configuration

The main configuration file is `config/app.yaml`.

Common options:

- `pipeline_mode`: default pipeline, `bbox` or `pose`
- `bbox_backend`: bbox backend, `hog` or `yolo11`
- `pose_backend`: pose backend, `mediapipe` or `pc_mediapipe`
- `camera_source`: camera source, commonly `"0"`
- `frame_width`, `frame_height`, `frame_fps`: requested camera parameters
- `detector_model_path`: YOLO11 ONNX model path
- `detector_interval`: bbox detection interval, used to balance performance and stability
- `mp_persondet_model_path`, `mp_pose_model_path`: local MediaPipe/OpenCV Zoo pose model paths
- `remote_pose_server_url`: PC pose service URL
- `fall_alert_enabled`: whether fall alert upload is enabled
- `fall_alert_base_url`, `fall_alert_event_path`: platform alert endpoint

Note: `pipeline_mode`, `bbox_backend`, and `pose_backend` are overwritten by the startup menu. Fine-grained parameters are still loaded from `app.yaml`.

## Fall Detection Logic

### bbox Pipeline

`YOLO11/HOG -> SinglePersonTracker -> FallDetector`:

- Tracks one main target to reduce detection-box jitter and short-term missing-frame issues.
- Maintains a dynamic body-height baseline from upright frames.
- Uses body-box aspect ratio and current-height-to-baseline ratio to determine suspected falls.
- Enters `fall_detected` only after the suspected state lasts for the configured duration.
- Returns to `normal` after an upright recovery state lasts for the configured duration.

### pose Pipeline

`MediaPipe 33 keypoints -> PoseFallDetector`:

- Counts visible keypoints.
- Computes trunk angle from the shoulder center to the hip center.
- Maintains a vertical body-span baseline from upright frames.
- Computes hip downward speed.
- Uses knee angle to suppress obvious crouching-related false positives.
- Outputs state through a `normal / suspected_fall / fall_detected / no_target` state machine.

## Fall Alert Upload

The main program integrates the Pi-side `FallAlertClient`. When the state first enters `fall_detected` from any non-fall state, it uploads one `raise` event. When the state recovers from `fall_detected` to `normal`, it uploads one `clear` event.

Recommended platform endpoint:

```http
POST /api/events/fall
Content-Type: multipart/form-data
```

Uploaded fields:

- `source_device`
- `frame_id`
- `ts_ms`
- `mode`
- `alert_action`
- `fall_state`
- `message`
- `fps`
- `image`

For more details, see `doc/摔倒报警上传联动方案.md`.

## Documents

- `doc/摔倒检测项目方案草案_v1.md`: project roadmap, technical route, and current conclusions
- `doc/摔倒报警上传联动方案.md`: fall alert upload API and integration workflow
- `pc_pose_server/README.md`: PC pose service usage
