#pragma once

#include <memory>
#include <optional>
#include <string>

#include "alert/FallAlertClient.hpp"
#include "camera/CameraManager.hpp"
#include "core/RuntimeModeSelection.hpp"
#include "detect/PersonDetector.hpp"
#include "fall/FallDetector.hpp"
#include "fall/PoseFallDetector.hpp"
#include "pose/MPPersonDetector.hpp"
#include "pose/MPPoseEstimator.hpp"
#include "pose/MoveNetPoseEstimator.hpp"
#include "remote/PoseRemoteClient.hpp"
#include "tracking/SinglePersonTracker.hpp"

namespace asdun {

class Renderer;

struct AppConfig {
  std::string pipeline_mode = "bbox";
  std::string bbox_backend = "yolo11";
  std::string pose_backend = "mediapipe";
  std::string camera_source = "0";
  int frame_width = 640;
  int frame_height = 480;
  int frame_fps = 30;
  std::string window_name = "Fall Detection Preview";
  int frame_timeout_ms = 1000;
  std::string pose_model_path = "./models/movenet_lightning_fp16_nchw.onnx";
  int pose_input_width = 192;
  int pose_input_height = 192;
  float pose_keypoint_score_threshold = 0.20F;
  bool pose_show_detector_box = false;
  bool pose_show_detector_aux_points = false;
  bool pose_show_landmark_bbox = true;
  float pose_fall_landmark_score_threshold = 0.25F;
  int pose_fall_min_visible_keypoints = 10;
  float pose_fall_upright_trunk_angle_deg_max = 25.0F;
  float pose_fall_baseline_update_alpha = 0.08F;
  float pose_fall_suspect_trunk_angle_deg_min = 55.0F;
  float pose_fall_suspect_span_ratio_max = 0.62F;
  float pose_fall_suspect_hip_drop_speed_min = 0.35F;
  float pose_fall_knee_angle_crouch_max = 115.0F;
  int pose_fall_confirm_hold_ms = 500;
  float pose_fall_recover_trunk_angle_deg_max = 30.0F;
  float pose_fall_recover_span_ratio_min = 0.78F;
  int pose_fall_recover_hold_ms = 800;
  std::string mp_persondet_model_path = "./models/person_detection_mediapipe_2023mar.onnx";
  std::string mp_pose_model_path = "./models/pose_estimation_mediapipe_2023mar.onnx";
  float mp_persondet_score_threshold = 0.50F;
  float mp_persondet_nms_threshold = 0.30F;
  int mp_persondet_top_k = 1;
  std::string remote_pose_server_url = "http://100.87.247.58:8000";
  std::string remote_pose_health_path = "/health";
  std::string remote_pose_analyze_path = "/analyze_pose";
  int remote_pose_connect_timeout_ms = 300;
  int remote_pose_timeout_ms = 1500;
  int remote_pose_submit_interval_ms = 220;
  int remote_pose_jpeg_quality = 80;
  bool remote_pose_debug = false;
  bool fall_alert_enabled = false;
  std::string fall_alert_base_url = "http://100.87.247.58:9000";
  std::string fall_alert_event_path = "/api/events/fall";
  std::string fall_alert_device_id = "pi-01";
  std::string fall_alert_device_token{};
  std::string fall_alert_message = "Possible fall detected in the scene. Please check immediately.";
  int fall_alert_connect_timeout_ms = 1000;
  int fall_alert_timeout_ms = 3000;
  int fall_alert_cooldown_ms = 30000;
  int fall_alert_jpeg_quality = 85;
  bool fall_alert_debug = false;
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
  explicit App(std::string config_path, std::optional<RuntimeModeSelection> runtime_selection = std::nullopt);
  ~App();
  int run();

 private:
  bool loadConfig();
  void applyRuntimeSelection();
  bool initComponents();
  int previewLoop();
  bool isPosePipeline() const;
  bool isMediaPipePoseBackend() const;
  bool isRemotePoseBackend() const;
  bool isHogBBoxBackend() const;
  void maybeQueueFallAlert(const cv::Mat& frame_bgr,
                           std::uint64_t frame_id,
                           std::uint64_t ts_ms,
                           double fps,
                           FallState current_state,
                           FallState& previous_state);
  std::string currentModeLabel() const;
  static std::string trim(std::string value);

  std::string config_path_;
  std::optional<RuntimeModeSelection> runtime_selection_;
  AppConfig config_;
  std::unique_ptr<CameraManager> camera_;
  std::unique_ptr<PersonDetector> detector_;
  std::unique_ptr<SinglePersonTracker> tracker_;
  std::unique_ptr<FallDetector> fall_detector_;
  std::unique_ptr<MoveNetPoseEstimator> pose_estimator_;
  std::unique_ptr<MPPersonDetector> mp_person_detector_;
  std::unique_ptr<MPPoseEstimator> mp_pose_estimator_;
  std::unique_ptr<PoseRemoteClient> pose_remote_client_;
  std::unique_ptr<FallAlertClient> fall_alert_client_;
  std::unique_ptr<PoseFallDetector> pose_fall_detector_;
  std::unique_ptr<Renderer> renderer_;
  bool show_status_overlay_ = true;
  bool show_pose_joints_ = true;
};

}  // namespace asdun
