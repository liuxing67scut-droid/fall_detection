# PC Pose Server

This folder contains the PC-side service for mode `4) MediaPipe (PC) + Pose (Pi)`.

## Purpose

- run a real MediaPipe Pose Landmarker model on the PC
- receive one JPEG frame from Raspberry Pi
- return `33` pose landmarks as JSON
- keep Raspberry Pi focused on capture, rendering, and fall logic

## Model file

Download the official MediaPipe Pose Landmarker Lite `.task` model and place it at:

```text
pc_pose_server/models/pose_landmarker_lite.task
```

If you want to use a different location, set:

```powershell
$env:PC_POSE_MODEL_PATH="C:\path\to\your\pose_landmarker_lite.task"
```

## Run on Windows PowerShell

```powershell
cd pc_pose_server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

If the model loads correctly, `/health` should report `model_ready: true`.

## Local PC test

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expected response when the model is ready:

```json
{
  "ok": true,
  "service": "pc-pose-server",
  "mode": "mediapipe",
  "model_ready": true
}
```

## Raspberry Pi test

On Raspberry Pi:

```bash
curl http://100.87.247.58:8000/health
```

If that succeeds, mode `4) MediaPipe (PC) + Pose (Pi)` in the Pi app can continue with `/analyze_pose`.

## Response shape

`POST /analyze_pose` returns:

- `frame_id`
- `ts_ms`
- `latency_ms`
- `pose_score`
- `keypoints`

Each keypoint contains:

- `id`
- `x`
- `y`
- `z`
- `visibility`
- `presence`
- `score`

`x` and `y` are normalized to `0..1`, so the Raspberry Pi can map them back to its local preview size.
