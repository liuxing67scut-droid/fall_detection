from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import time

from fastapi import FastAPI, File, Form, UploadFile

from services.pose_service import PoseService, PoseServiceConfig

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"

pose_service = PoseService(
    PoseServiceConfig(
        model_path=Path(os.getenv("PC_POSE_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
    )
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    pose_service.initialize()
    yield
    pose_service.close()

app = FastAPI(
    title="Fall Detect PC Pose Server",
    version="0.1.0",
    description="MediaPipe pose server for Raspberry Pi to PC inference mode.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return pose_service.health_payload()


@app.post("/analyze_pose")
async def analyze_pose(
    frame_id: int = Form(...),
    ts_ms: int = Form(...),
    image: UploadFile = File(...),
) -> dict:
    image_bytes = await image.read()
    start = time.perf_counter()
    response = pose_service.analyze_pose(image_bytes, frame_id, ts_ms)
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)

    print(
        f"[PCPose] frame_id={frame_id} ts_ms={ts_ms} "
        f"image={response.get('image_width', 0)}x{response.get('image_height', 0)} "
        f"bytes={len(image_bytes)} keypoints={len(response.get('keypoints', []))} "
        f"pose_score={response.get('pose_score', 0.0):.3f} "
        f"latency_ms={response.get('latency_ms', 0.0):.3f} "
        f"handler_ms={elapsed_ms:.3f} status={response.get('status_message', 'unknown')}"
    )

    response.pop("image_width", None)
    response.pop("image_height", None)
    return response
