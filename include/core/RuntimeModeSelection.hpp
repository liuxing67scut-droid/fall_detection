#pragma once

#include <string>

namespace asdun {

struct RuntimeModeSelection {
  std::string pipeline_mode = "bbox";
  std::string bbox_backend = "yolo11";
  std::string pose_backend = "mediapipe";
};

}  // namespace asdun
