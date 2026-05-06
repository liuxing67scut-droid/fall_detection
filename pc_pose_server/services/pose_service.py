from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import threading
import time
from typing import Any

import mediapipe as mp
import numpy as np
from PIL import Image


@dataclass
class PoseServiceConfig:
    model_path: Path
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class PoseService:
    def __init__(self, config: PoseServiceConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._landmarker: Any | None = None
        self._ready = False
        self._last_error = ""
        self._last_video_ts_ms = -1

    def initialize(self) -> None:
        self._ready = False
        self._last_error = ""

        if not self._config.model_path.is_file():
            self._last_error = f"model file not found: {self._config.model_path}"
            return

        try:
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self._config.model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=self._config.num_poses,
                min_pose_detection_confidence=self._config.min_pose_detection_confidence,
                min_pose_presence_confidence=self._config.min_pose_presence_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            self._ready = True
            self._last_video_ts_ms = -1
        except Exception as exc:  # pragma: no cover - runtime dependency/environment issue
            self._landmarker = None
            self._ready = False
            self._last_error = str(exc)

    def close(self) -> None:
        with self._lock:
            if self._landmarker is not None:
                try:
                    self._landmarker.close()
                except Exception:
                    pass
            self._landmarker = None
            self._ready = False

    def health_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "pc-pose-server",
            "mode": "mediapipe",
            "model_ready": self._ready,
            "model_path": str(self._config.model_path),
            "error": self._last_error or None,
        }

    def analyze_pose(self, image_bytes: bytes, frame_id: int, ts_ms: int) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            width, height, rgb = self._decode_rgb_image(image_bytes)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            return {
                "ok": False,
                "frame_id": frame_id,
                "ts_ms": ts_ms,
                "latency_ms": latency_ms,
                "pose_score": 0.0,
                "keypoints": [],
                "status_message": "decode_failed",
                "error": str(exc),
                "image_width": 0,
                "image_height": 0,
            }

        if not self._ready or self._landmarker is None:
            latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            return {
                "ok": False,
                "frame_id": frame_id,
                "ts_ms": ts_ms,
                "latency_ms": latency_ms,
                "pose_score": 0.0,
                "keypoints": [],
                "status_message": "model_not_ready",
                "image_width": width,
                "image_height": height,
            }

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            with self._lock:
                video_ts_ms = max(int(ts_ms), self._last_video_ts_ms + 1)
                result = self._landmarker.detect_for_video(mp_image, video_ts_ms)
                self._last_video_ts_ms = video_ts_ms
        except Exception as exc:  # pragma: no cover - runtime dependency/environment issue
            self._last_error = str(exc)
            latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            return {
                "ok": False,
                "frame_id": frame_id,
                "ts_ms": ts_ms,
                "latency_ms": latency_ms,
                "pose_score": 0.0,
                "keypoints": [],
                "status_message": "inference_failed",
                "error": str(exc),
                "image_width": width,
                "image_height": height,
            }

        keypoints: list[dict[str, float | int]] = []
        pose_score = 0.0
        status_message = "no_pose"
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            scores: list[float] = []
            for idx, landmark in enumerate(landmarks):
                visibility = self._safe_float(getattr(landmark, "visibility", 0.0))
                presence = self._safe_float(getattr(landmark, "presence", 0.0))
                score = min(visibility, presence)
                scores.append(score)
                keypoints.append(
                    {
                        "id": idx,
                        "x": self._safe_float(landmark.x),
                        "y": self._safe_float(landmark.y),
                        "z": self._safe_float(landmark.z),
                        "visibility": visibility,
                        "presence": presence,
                        "score": score,
                    }
                )
            pose_score = float(sum(scores) / len(scores)) if scores else 0.0
            status_message = "pose_ok"

        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return {
            "ok": True,
            "frame_id": frame_id,
            "ts_ms": ts_ms,
            "latency_ms": latency_ms,
            "pose_score": pose_score,
            "keypoints": keypoints,
            "status_message": status_message,
            "image_width": width,
            "image_height": height,
        }

    @staticmethod
    def _decode_rgb_image(image_bytes: bytes) -> tuple[int, int, np.ndarray]:
        with Image.open(BytesIO(image_bytes)) as frame:
            rgb_image = frame.convert("RGB")
            width, height = rgb_image.size
            rgb = np.ascontiguousarray(np.asarray(rgb_image))
        return width, height, rgb

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
