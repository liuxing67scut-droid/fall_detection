#pragma once

#include <memory>
#include <string>

#include "camera/CameraManager.hpp"
#include "detect/PersonDetector.hpp"
#include "fall/FallDetector.hpp"
#include "tracking/SinglePersonTracker.hpp"

namespace asdun {

class Renderer;

struct AppConfig {
  std::string camera_source = "0";
  int frame_width = 640;
  int frame_height = 480;
  int frame_fps = 30;
  std::string window_name = "Fall Detection Preview";
  int frame_timeout_ms = 1000;
  int detector_interval = 2;
  std::string detector_model_path = "./models/yolo11n.onnx";
  int detector_input_width = 640;
  int detector_input_height = 640;
  float detector_confidence_threshold = 0.40F;
  float detector_nms_threshold = 0.45F;
  int detector_person_class_id = 0;
  bool detector_swap_rb = true;
  bool detector_use_hog_fallback = true;
  int detector_max_detections = 1;
  float tracker_new_track_conf_threshold = 0.25F;
  float tracker_hold_track_conf_threshold = 0.10F;
  float tracker_match_iou_threshold = 0.20F;
  float tracker_max_center_distance_ratio = 0.45F;
  int tracker_confirm_frames = 2;
  int tracker_max_missing_frames = 6;
  float tracker_smooth_alpha = 0.35F;
  float fall_upright_aspect_ratio_max = 0.85F;
  float fall_baseline_update_alpha = 0.08F;
  float fall_suspect_aspect_ratio_threshold = 1.10F;
  float fall_suspect_height_ratio_threshold = 0.65F;
  int fall_confirm_hold_ms = 700;
  float fall_recover_aspect_ratio_threshold = 0.85F;
  float fall_recover_height_ratio_threshold = 0.80F;
  int fall_recover_hold_ms = 900;
};

class App {
 public:
  explicit App(std::string config_path);
  ~App();
  int run();

 private:
  bool loadConfig();
  bool initComponents();
  int previewLoop();
  static std::string trim(std::string value);

  std::string config_path_;
  AppConfig config_;
  std::unique_ptr<CameraManager> camera_;
  std::unique_ptr<PersonDetector> detector_;
  std::unique_ptr<SinglePersonTracker> tracker_;
  std::unique_ptr<FallDetector> fall_detector_;
  std::unique_ptr<Renderer> renderer_;
};

}  // namespace asdun
