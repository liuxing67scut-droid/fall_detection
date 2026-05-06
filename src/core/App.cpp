#include "core/App.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ui/Renderer.hpp"

namespace asdun {

namespace {

constexpr std::array<const char*, 33> kMediaPipePoseNames = {
    "nose",         "left_eye_inner",  "left_eye",       "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear",     "right_ear",      "mouth_left",     "mouth_right",     "left_shoulder",
    "right_shoulder", "left_elbow",    "right_elbow",    "left_wrist",     "right_wrist",     "left_pinky",
    "right_pinky",  "left_index",      "right_index",    "left_thumb",     "right_thumb",     "left_hip",
    "right_hip",    "left_knee",       "right_knee",     "left_ankle",     "right_ankle",     "left_heel",
    "right_heel",   "left_foot_index", "right_foot_index"};

bool parseBoolValue(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string trackerStateLabel(const TrackedPerson& tracked_person) {
  if (!tracked_person.has_target) {
    return "lost";
  }
  if (tracked_person.is_holding) {
    return "holding";
  }
  if (tracked_person.is_confirmed) {
    return "confirmed";
  }
  return "tentative";
}

std::string fallStateLabel(FallState state) {
  switch (state) {
    case FallState::NoTarget:
      return "no_target";
    case FallState::Normal:
      return "normal";
    case FallState::SuspectedFall:
      return "suspected_fall";
    case FallState::FallDetected:
      return "fall_detected";
  }
  return "unknown";
}

std::string normalizePipelineMode(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string normalizeBboxBackend(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string normalizePoseBackend(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool isFallTransitionToDetected(FallState previous_state, FallState current_state) {
  return previous_state != FallState::FallDetected && current_state == FallState::FallDetected;
}

cv::Scalar fallStateColor(FallState state) {
  switch (state) {
    case FallState::Normal:
      return cv::Scalar(60, 220, 60);
    case FallState::SuspectedFall:
      return cv::Scalar(0, 165, 255);
    case FallState::FallDetected:
      return cv::Scalar(0, 70, 255);
    case FallState::NoTarget:
      break;
  }
  return cv::Scalar(210, 210, 210);
}

std::string fallStateShortLabel(FallState state) {
  switch (state) {
    case FallState::Normal:
      return "normal";
    case FallState::SuspectedFall:
      return "suspect";
    case FallState::FallDetected:
      return "fall";
    case FallState::NoTarget:
      break;
  }
  return "track";
}

float clamp01(float value) {
  return std::clamp(value, 0.0F, 1.0F);
}

PoseFrameResult remoteResponseToPoseResult(const RemotePoseResponse& response, const cv::Size& frame_size) {
  PoseFrameResult result;
  if (!response.ok || frame_size.width <= 0 || frame_size.height <= 0) {
    return result;
  }

  result.keypoints.reserve(response.keypoints.size());
  for (const auto& remote_keypoint : response.keypoints) {
    PoseKeypoint keypoint;
    if (remote_keypoint.id >= 0 &&
        remote_keypoint.id < static_cast<int>(kMediaPipePoseNames.size())) {
      keypoint.name = kMediaPipePoseNames[static_cast<std::size_t>(remote_keypoint.id)];
    } else {
      keypoint.name = "kp_" + std::to_string(remote_keypoint.id);
    }
    keypoint.score = remote_keypoint.score;
    keypoint.point = cv::Point2f(clamp01(remote_keypoint.x) * static_cast<float>(frame_size.width),
                                 clamp01(remote_keypoint.y) * static_cast<float>(frame_size.height));
    if (keypoint.score > 0.0F) {
      result.has_pose = true;
    }
    result.keypoints.push_back(std::move(keypoint));
  }
  return result;
}

void drawLabeledBox(cv::Mat& canvas, const cv::Rect& box, const cv::Scalar& color, const std::string& label) {
  if (box.width <= 0 || box.height <= 0) {
    return;
  }

  cv::rectangle(canvas, box, color, 2);
  if (label.empty()) {
    return;
  }

  cv::putText(canvas,
              label,
              cv::Point(box.x, std::max(18, box.y - 8)),
              cv::FONT_HERSHEY_SIMPLEX,
              0.45,
              cv::Scalar(0, 0, 0),
              3,
              cv::LINE_AA);
  cv::putText(canvas,
              label,
              cv::Point(box.x, std::max(18, box.y - 8)),
              cv::FONT_HERSHEY_SIMPLEX,
              0.45,
              color,
              1,
              cv::LINE_AA);
}

void drawTrackedPerson(cv::Mat& canvas, const TrackedPerson& tracked_person, const FallStatus& fall_status) {
  if (!tracked_person.has_target || tracked_person.box.width <= 0 || tracked_person.box.height <= 0) {
    return;
  }

  const FallState visual_state = fall_status.state != FallState::NoTarget ? fall_status.state : FallState::Normal;
  drawLabeledBox(canvas, tracked_person.box, fallStateColor(visual_state), fallStateShortLabel(visual_state));
}

void drawPoseResult(cv::Mat& canvas,
                    const PoseFrameResult& pose_result,
                    float keypoint_score_threshold,
                    bool show_pose_joints) {
  static const std::vector<std::pair<int, int>> kCoco17Edges = {
      {0, 1},   {0, 2},   {1, 3},   {2, 4},   {5, 6},   {5, 7},   {7, 9},   {6, 8},   {8, 10},
      {5, 11},  {6, 12},  {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}};
  static const std::vector<std::pair<int, int>> kMediaPipe33Edges = {
      {0, 1},   {1, 2},   {2, 3},   {3, 7},   {0, 4},   {4, 5},   {5, 6},   {6, 8},   {9, 10},
      {11, 12}, {11, 13}, {13, 15}, {15, 17}, {15, 19}, {15, 21}, {17, 19}, {12, 14}, {14, 16},
      {16, 18}, {16, 20}, {16, 22}, {18, 20}, {11, 23}, {12, 24}, {23, 24}, {23, 25}, {24, 26},
      {25, 27}, {26, 28}, {27, 29}, {28, 30}, {29, 31}, {30, 32}, {27, 31}, {28, 32}};

  if (!pose_result.has_pose) {
    return;
  }
  if (!show_pose_joints) {
    return;
  }

  const auto& skeleton_edges =
      pose_result.keypoints.size() >= 33 ? kMediaPipe33Edges : kCoco17Edges;

  for (const auto& edge : skeleton_edges) {
    if (edge.first >= static_cast<int>(pose_result.keypoints.size()) ||
        edge.second >= static_cast<int>(pose_result.keypoints.size())) {
      continue;
    }
    const PoseKeypoint& a = pose_result.keypoints[static_cast<std::size_t>(edge.first)];
    const PoseKeypoint& b = pose_result.keypoints[static_cast<std::size_t>(edge.second)];
    if (a.score < keypoint_score_threshold || b.score < keypoint_score_threshold) {
      continue;
    }
    cv::line(canvas, a.point, b.point, cv::Scalar(0, 210, 255), 2, cv::LINE_AA);
  }

  for (const auto& keypoint : pose_result.keypoints) {
    if (keypoint.score < keypoint_score_threshold) {
      continue;
    }
    cv::circle(canvas, keypoint.point, 3, cv::Scalar(0, 255, 120), cv::FILLED, cv::LINE_AA);
  }
}

cv::Rect computePoseLandmarkBox(const PoseFrameResult& pose_result, float keypoint_score_threshold) {
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();

  for (const auto& keypoint : pose_result.keypoints) {
    if (keypoint.score < keypoint_score_threshold) {
      continue;
    }
    min_x = std::min(min_x, cvRound(keypoint.point.x));
    min_y = std::min(min_y, cvRound(keypoint.point.y));
    max_x = std::max(max_x, cvRound(keypoint.point.x));
    max_y = std::max(max_y, cvRound(keypoint.point.y));
  }

  if (min_x == std::numeric_limits<int>::max() || min_y == std::numeric_limits<int>::max() ||
      max_x <= min_x || max_y <= min_y) {
    return {};
  }

  const int width = max_x - min_x;
  const int height = max_y - min_y;
  const int pad_x = std::max(8, width / 12);
  const int pad_y = std::max(8, height / 12);
  return cv::Rect(min_x - pad_x, min_y - pad_y, width + pad_x * 2, height + pad_y * 2);
}

void drawPoseFallBox(cv::Mat& canvas, const cv::Rect& landmark_box, const PoseFallStatus& pose_fall_status) {
  if (landmark_box.width <= 0 || landmark_box.height <= 0) {
    return;
  }

  const FallState visual_state =
      pose_fall_status.state != FallState::NoTarget ? pose_fall_status.state : FallState::Normal;
  drawLabeledBox(canvas, landmark_box, fallStateColor(visual_state), fallStateShortLabel(visual_state));
}

void drawMediaPipePerson(cv::Mat& canvas,
                         const MPPersonRegion& person_region,
                         bool show_detector_box,
                         bool show_detector_aux_points) {
  if (!person_region.valid()) {
    return;
  }

  if (show_detector_box) {
    cv::rectangle(canvas, person_region.box, cv::Scalar(255, 220, 0), 2);
  }
  if (show_detector_aux_points) {
    for (const auto& point : person_region.aux_keypoints) {
      cv::circle(canvas, point, 3, cv::Scalar(0, 200, 255), cv::FILLED, cv::LINE_AA);
    }
  }
}

}  // namespace

App::App(std::string config_path, std::optional<RuntimeModeSelection> runtime_selection)
    : config_path_(std::move(config_path)), runtime_selection_(std::move(runtime_selection)) {}

App::~App() = default;

int App::run() {
  if (!loadConfig()) {
    std::cerr << "[App] failed to load config: " << config_path_ << std::endl;
    return 1;
  }
  applyRuntimeSelection();
  if (!initComponents()) {
    std::cerr << "[App] failed to initialize components." << std::endl;
    return 1;
  }
  return previewLoop();
}

bool App::loadConfig() {
  std::ifstream fin(config_path_);
  if (!fin.is_open()) {
    std::cout << "[App] config not found, using defaults: " << config_path_ << std::endl;
    return true;
  }

  std::string line;
  while (std::getline(fin, line)) {
    const auto hash_pos = line.find('#');
    if (hash_pos != std::string::npos) {
      line = line.substr(0, hash_pos);
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }

    std::string key = trim(line.substr(0, colon));
    std::string value = trim(line.substr(colon + 1));
    if (!value.empty() && value.front() == '"' && value.back() == '"' && value.size() >= 2) {
      value = value.substr(1, value.size() - 2);
    }
    if (key.empty() || value.empty()) {
      continue;
    }

    try {
      if (key == "pipeline_mode") {
        config_.pipeline_mode = normalizePipelineMode(value);
      } else if (key == "bbox_backend") {
        config_.bbox_backend = normalizeBboxBackend(value);
      } else if (key == "pose_backend") {
        config_.pose_backend = normalizePoseBackend(value);
      } else if (key == "camera_source") {
        config_.camera_source = value;
      } else if (key == "frame_width") {
        config_.frame_width = std::stoi(value);
      } else if (key == "frame_height") {
        config_.frame_height = std::stoi(value);
      } else if (key == "frame_fps") {
        config_.frame_fps = std::stoi(value);
      } else if (key == "window_name") {
        config_.window_name = value;
      } else if (key == "frame_timeout_ms") {
        config_.frame_timeout_ms = std::stoi(value);
      } else if (key == "pose_model_path") {
        config_.pose_model_path = value;
      } else if (key == "pose_input_width") {
        config_.pose_input_width = std::stoi(value);
      } else if (key == "pose_input_height") {
        config_.pose_input_height = std::stoi(value);
      } else if (key == "pose_keypoint_score_threshold") {
        config_.pose_keypoint_score_threshold = std::stof(value);
      } else if (key == "pose_show_detector_box") {
        config_.pose_show_detector_box = parseBoolValue(value);
      } else if (key == "pose_show_detector_aux_points") {
        config_.pose_show_detector_aux_points = parseBoolValue(value);
      } else if (key == "pose_show_landmark_bbox") {
        config_.pose_show_landmark_bbox = parseBoolValue(value);
      } else if (key == "pose_fall_landmark_score_threshold") {
        config_.pose_fall_landmark_score_threshold = std::stof(value);
      } else if (key == "pose_fall_min_visible_keypoints") {
        config_.pose_fall_min_visible_keypoints = std::stoi(value);
      } else if (key == "pose_fall_upright_trunk_angle_deg_max") {
        config_.pose_fall_upright_trunk_angle_deg_max = std::stof(value);
      } else if (key == "pose_fall_baseline_update_alpha") {
        config_.pose_fall_baseline_update_alpha = std::stof(value);
      } else if (key == "pose_fall_suspect_trunk_angle_deg_min") {
        config_.pose_fall_suspect_trunk_angle_deg_min = std::stof(value);
      } else if (key == "pose_fall_suspect_span_ratio_max") {
        config_.pose_fall_suspect_span_ratio_max = std::stof(value);
      } else if (key == "pose_fall_suspect_hip_drop_speed_min") {
        config_.pose_fall_suspect_hip_drop_speed_min = std::stof(value);
      } else if (key == "pose_fall_knee_angle_crouch_max") {
        config_.pose_fall_knee_angle_crouch_max = std::stof(value);
      } else if (key == "pose_fall_confirm_hold_ms") {
        config_.pose_fall_confirm_hold_ms = std::stoi(value);
      } else if (key == "pose_fall_recover_trunk_angle_deg_max") {
        config_.pose_fall_recover_trunk_angle_deg_max = std::stof(value);
      } else if (key == "pose_fall_recover_span_ratio_min") {
        config_.pose_fall_recover_span_ratio_min = std::stof(value);
      } else if (key == "pose_fall_recover_hold_ms") {
        config_.pose_fall_recover_hold_ms = std::stoi(value);
      } else if (key == "mp_persondet_model_path") {
        config_.mp_persondet_model_path = value;
      } else if (key == "mp_pose_model_path") {
        config_.mp_pose_model_path = value;
      } else if (key == "mp_persondet_score_threshold") {
        config_.mp_persondet_score_threshold = std::stof(value);
      } else if (key == "mp_persondet_nms_threshold") {
        config_.mp_persondet_nms_threshold = std::stof(value);
      } else if (key == "mp_persondet_top_k") {
        config_.mp_persondet_top_k = std::stoi(value);
      } else if (key == "remote_pose_server_url") {
        config_.remote_pose_server_url = value;
      } else if (key == "remote_pose_health_path") {
        config_.remote_pose_health_path = value;
      } else if (key == "remote_pose_analyze_path") {
        config_.remote_pose_analyze_path = value;
      } else if (key == "remote_pose_connect_timeout_ms") {
        config_.remote_pose_connect_timeout_ms = std::stoi(value);
      } else if (key == "remote_pose_timeout_ms") {
        config_.remote_pose_timeout_ms = std::stoi(value);
      } else if (key == "remote_pose_submit_interval_ms") {
        config_.remote_pose_submit_interval_ms = std::stoi(value);
      } else if (key == "remote_pose_jpeg_quality") {
        config_.remote_pose_jpeg_quality = std::stoi(value);
      } else if (key == "remote_pose_debug") {
        config_.remote_pose_debug = parseBoolValue(value);
      } else if (key == "fall_alert_enabled") {
        config_.fall_alert_enabled = parseBoolValue(value);
      } else if (key == "fall_alert_base_url") {
        config_.fall_alert_base_url = value;
      } else if (key == "fall_alert_event_path") {
        config_.fall_alert_event_path = value;
      } else if (key == "fall_alert_device_id") {
        config_.fall_alert_device_id = value;
      } else if (key == "fall_alert_device_token") {
        config_.fall_alert_device_token = value;
      } else if (key == "fall_alert_message") {
        config_.fall_alert_message = value;
      } else if (key == "fall_alert_connect_timeout_ms") {
        config_.fall_alert_connect_timeout_ms = std::stoi(value);
      } else if (key == "fall_alert_timeout_ms") {
        config_.fall_alert_timeout_ms = std::stoi(value);
      } else if (key == "fall_alert_cooldown_ms") {
        config_.fall_alert_cooldown_ms = std::stoi(value);
      } else if (key == "fall_alert_jpeg_quality") {
        config_.fall_alert_jpeg_quality = std::stoi(value);
      } else if (key == "fall_alert_debug") {
        config_.fall_alert_debug = parseBoolValue(value);
      } else if (key == "detector_interval") {
        config_.detector_interval = std::stoi(value);
      } else if (key == "detector_model_path") {
        config_.detector_model_path = value;
      } else if (key == "detector_input_width") {
        config_.detector_input_width = std::stoi(value);
      } else if (key == "detector_input_height") {
        config_.detector_input_height = std::stoi(value);
      } else if (key == "detector_confidence_threshold") {
        config_.detector_confidence_threshold = std::stof(value);
      } else if (key == "detector_nms_threshold") {
        config_.detector_nms_threshold = std::stof(value);
      } else if (key == "detector_person_class_id") {
        config_.detector_person_class_id = std::stoi(value);
      } else if (key == "detector_swap_rb") {
        config_.detector_swap_rb = parseBoolValue(value);
      } else if (key == "detector_use_hog_fallback") {
        config_.detector_use_hog_fallback = parseBoolValue(value);
      } else if (key == "detector_max_detections") {
        config_.detector_max_detections = std::stoi(value);
      } else if (key == "tracker_new_track_conf_threshold") {
        config_.tracker_new_track_conf_threshold = std::stof(value);
      } else if (key == "tracker_hold_track_conf_threshold") {
        config_.tracker_hold_track_conf_threshold = std::stof(value);
      } else if (key == "tracker_match_iou_threshold") {
        config_.tracker_match_iou_threshold = std::stof(value);
      } else if (key == "tracker_max_center_distance_ratio") {
        config_.tracker_max_center_distance_ratio = std::stof(value);
      } else if (key == "tracker_confirm_frames") {
        config_.tracker_confirm_frames = std::stoi(value);
      } else if (key == "tracker_max_missing_frames") {
        config_.tracker_max_missing_frames = std::stoi(value);
      } else if (key == "tracker_smooth_alpha") {
        config_.tracker_smooth_alpha = std::stof(value);
      } else if (key == "fall_upright_aspect_ratio_max") {
        config_.fall_upright_aspect_ratio_max = std::stof(value);
      } else if (key == "fall_baseline_update_alpha") {
        config_.fall_baseline_update_alpha = std::stof(value);
      } else if (key == "fall_suspect_aspect_ratio_threshold") {
        config_.fall_suspect_aspect_ratio_threshold = std::stof(value);
      } else if (key == "fall_suspect_height_ratio_threshold") {
        config_.fall_suspect_height_ratio_threshold = std::stof(value);
      } else if (key == "fall_confirm_hold_ms") {
        config_.fall_confirm_hold_ms = std::stoi(value);
      } else if (key == "fall_recover_aspect_ratio_threshold") {
        config_.fall_recover_aspect_ratio_threshold = std::stof(value);
      } else if (key == "fall_recover_height_ratio_threshold") {
        config_.fall_recover_height_ratio_threshold = std::stof(value);
      } else if (key == "fall_recover_hold_ms") {
        config_.fall_recover_hold_ms = std::stoi(value);
      }
    } catch (const std::exception&) {
      std::cerr << "[App] ignore invalid config entry: " << key << std::endl;
    }
  }
  return true;
}

void App::applyRuntimeSelection() {
  if (!runtime_selection_.has_value()) {
    return;
  }
  config_.pipeline_mode = normalizePipelineMode(runtime_selection_->pipeline_mode);
  config_.bbox_backend = normalizeBboxBackend(runtime_selection_->bbox_backend);
  config_.pose_backend = normalizePoseBackend(runtime_selection_->pose_backend);
}

bool App::initComponents() {
  cv::setNumThreads(1);

  camera_ = std::make_unique<CameraManager>(
      config_.camera_source, config_.frame_width, config_.frame_height, config_.frame_fps);
  renderer_ = std::make_unique<Renderer>(config_.window_name);

  if (!camera_->start()) {
    return false;
  }

  if (config_.fall_alert_enabled) {
    FallAlertClientConfig alert_config;
    alert_config.enabled = config_.fall_alert_enabled;
    alert_config.base_url = config_.fall_alert_base_url;
    alert_config.event_path = config_.fall_alert_event_path;
    alert_config.device_id = config_.fall_alert_device_id;
    alert_config.device_token = config_.fall_alert_device_token;
    alert_config.connect_timeout_ms = config_.fall_alert_connect_timeout_ms;
    alert_config.timeout_ms = config_.fall_alert_timeout_ms;
    alert_config.cooldown_ms = config_.fall_alert_cooldown_ms;
    alert_config.jpeg_quality = config_.fall_alert_jpeg_quality;
    alert_config.debug = config_.fall_alert_debug;
    fall_alert_client_ = std::make_unique<FallAlertClient>(std::move(alert_config));
    fall_alert_client_->start();
  }

  if (isPosePipeline()) {
    pose_fall_detector_ = std::make_unique<PoseFallDetector>(config_.pose_fall_landmark_score_threshold,
                                                             config_.pose_fall_min_visible_keypoints,
                                                             config_.pose_fall_upright_trunk_angle_deg_max,
                                                             config_.pose_fall_baseline_update_alpha,
                                                             config_.pose_fall_suspect_trunk_angle_deg_min,
                                                             config_.pose_fall_suspect_span_ratio_max,
                                                             config_.pose_fall_suspect_hip_drop_speed_min,
                                                             config_.pose_fall_knee_angle_crouch_max,
                                                             config_.pose_fall_confirm_hold_ms,
                                                             config_.pose_fall_recover_trunk_angle_deg_max,
                                                             config_.pose_fall_recover_span_ratio_min,
                                                             config_.pose_fall_recover_hold_ms);
    if (isRemotePoseBackend()) {
      PoseRemoteClientConfig remote_config;
      remote_config.server_url = config_.remote_pose_server_url;
      remote_config.health_path = config_.remote_pose_health_path;
      remote_config.analyze_path = config_.remote_pose_analyze_path;
      remote_config.connect_timeout_ms = config_.remote_pose_connect_timeout_ms;
      remote_config.timeout_ms = config_.remote_pose_timeout_ms;
      remote_config.submit_interval_ms = config_.remote_pose_submit_interval_ms;
      remote_config.jpeg_quality = config_.remote_pose_jpeg_quality;
      remote_config.debug = config_.remote_pose_debug;
      pose_remote_client_ = std::make_unique<PoseRemoteClient>(std::move(remote_config));
      if (!pose_remote_client_->start()) {
        std::cerr << "[App] remote pose client is unavailable." << std::endl;
        return false;
      }
    } else if (isMediaPipePoseBackend()) {
      mp_person_detector_ = std::make_unique<MPPersonDetector>(config_.mp_persondet_model_path,
                                                               config_.mp_persondet_score_threshold,
                                                               config_.mp_persondet_nms_threshold,
                                                               config_.mp_persondet_top_k);
      mp_pose_estimator_ =
          std::make_unique<MPPoseEstimator>(config_.mp_pose_model_path, config_.pose_keypoint_score_threshold);
      mp_person_detector_->init();
      mp_pose_estimator_->init();
    } else {
      pose_estimator_ = std::make_unique<MoveNetPoseEstimator>(config_.pose_model_path,
                                                               config_.pose_input_width,
                                                               config_.pose_input_height,
                                                               config_.pose_keypoint_score_threshold);
      pose_estimator_->init();
    }
  } else {
    const bool use_hog_backend = isHogBBoxBackend();
    const std::string detector_model_path = use_hog_backend ? "" : config_.detector_model_path;
    const bool detector_use_hog_fallback = use_hog_backend;
    detector_ = std::make_unique<PersonDetector>(detector_model_path,
                                                 config_.detector_input_width,
                                                 config_.detector_input_height,
                                                 config_.detector_confidence_threshold,
                                                 config_.detector_nms_threshold,
                                                 config_.detector_person_class_id,
                                                 config_.detector_swap_rb,
                                                 detector_use_hog_fallback,
                                                 config_.detector_max_detections);
    tracker_ = std::make_unique<SinglePersonTracker>(config_.tracker_new_track_conf_threshold,
                                                     config_.tracker_hold_track_conf_threshold,
                                                     config_.tracker_match_iou_threshold,
                                                     config_.tracker_max_center_distance_ratio,
                                                     config_.tracker_confirm_frames,
                                                     config_.tracker_max_missing_frames,
                                                     config_.tracker_smooth_alpha);
    fall_detector_ = std::make_unique<FallDetector>(config_.fall_upright_aspect_ratio_max,
                                                    config_.fall_baseline_update_alpha,
                                                    config_.fall_suspect_aspect_ratio_threshold,
                                                    config_.fall_suspect_height_ratio_threshold,
                                                    config_.fall_confirm_hold_ms,
                                                    config_.fall_recover_aspect_ratio_threshold,
                                                    config_.fall_recover_height_ratio_threshold,
                                                    config_.fall_recover_hold_ms);
    detector_->init();
  }

  std::cout << "[App] camera preview started." << std::endl;
  std::cout << "[App] press q or Esc to exit, press s to save a snapshot." << std::endl;
  std::cout << "[App] pipeline=" << config_.pipeline_mode << std::endl;
  if (isPosePipeline()) {
    std::cout << "[App] pose_backend=" << config_.pose_backend << std::endl;
    if (isRemotePoseBackend()) {
      std::cout << "[App] " << pose_remote_client_->statusMessage() << std::endl;
    } else if (isMediaPipePoseBackend()) {
      std::cout << "[App] " << mp_person_detector_->statusMessage() << std::endl;
      std::cout << "[App] " << mp_pose_estimator_->statusMessage() << std::endl;
    } else {
      std::cout << "[App] " << pose_estimator_->statusMessage() << std::endl;
    }
  } else {
    std::cout << "[App] bbox_backend=" << config_.bbox_backend << std::endl;
    std::cout << "[App] " << detector_->statusMessage() << std::endl;
  }
  if (fall_alert_client_) {
    std::cout << "[App] " << fall_alert_client_->statusMessage() << std::endl;
  }
  return true;
}

int App::previewLoop() {
  int saved_frame_count = 0;
  std::vector<PersonDetection> cached_detections;
  TrackedPerson tracked_person;
  FallStatus fall_status;
  PoseFrameResult pose_result;
  PoseFallStatus pose_fall_status;
  MPPersonRegion mp_person_region;
  RemotePoseResponse remote_pose_response;
  double smoothed_fps = 0.0;
  std::uint64_t last_frame_ts_ms = 0;
  double last_detect_ms = 0.0;
  FallState last_box_alert_state = FallState::NoTarget;
  FallState last_pose_alert_state = FallState::NoTarget;
  while (true) {
    FramePacket frame_packet;
    if (!camera_->getLatestFrame(frame_packet, static_cast<std::uint32_t>(config_.frame_timeout_ms))) {
      std::cerr << "[App] timeout waiting for camera frame." << std::endl;
      break;
    }

    if (last_frame_ts_ms > 0 && frame_packet.ts_ms > last_frame_ts_ms) {
      const double instant_fps = 1000.0 / static_cast<double>(frame_packet.ts_ms - last_frame_ts_ms);
      smoothed_fps = (smoothed_fps <= 0.0) ? instant_fps : (0.82 * smoothed_fps + 0.18 * instant_fps);
    }
    last_frame_ts_ms = frame_packet.ts_ms;

    const int safe_interval = std::max(config_.detector_interval, 1);
    if (isPosePipeline()) {
      if (isRemotePoseBackend()) {
        if (pose_remote_client_) {
          pose_remote_client_->submit(frame_packet.bgr, frame_packet.frame_id, frame_packet.ts_ms);
          if (const auto latest = pose_remote_client_->consumeLatest(); latest.has_value()) {
            remote_pose_response = *latest;
            pose_result = remoteResponseToPoseResult(remote_pose_response, frame_packet.bgr.size());
            if (pose_fall_detector_) {
              pose_fall_status = pose_fall_detector_->update(pose_result, frame_packet.ts_ms);
            }
            last_detect_ms = remote_pose_response.latency_ms;
          }
        }
      } else {
        const auto pose_begin = std::chrono::steady_clock::now();
        if (isMediaPipePoseBackend()) {
          const std::vector<MPPersonRegion> regions = mp_person_detector_->detect(frame_packet.bgr);
          mp_person_region = regions.empty() ? MPPersonRegion{} : regions.front();
          pose_result = mp_pose_estimator_->estimate(frame_packet.bgr, mp_person_region);
        } else {
          pose_result = pose_estimator_->estimate(frame_packet.bgr);
        }
        if (pose_fall_detector_) {
          pose_fall_status = pose_fall_detector_->update(pose_result, frame_packet.ts_ms);
        }
        const auto pose_end = std::chrono::steady_clock::now();
        last_detect_ms = std::chrono::duration<double, std::milli>(pose_end - pose_begin).count();
      }
      maybeQueueFallAlert(frame_packet.bgr,
                          frame_packet.frame_id,
                          frame_packet.ts_ms,
                          smoothed_fps,
                          pose_fall_status.state,
                          last_pose_alert_state);
    } else {
      if (frame_packet.frame_id == 1 || (frame_packet.frame_id % static_cast<std::uint64_t>(safe_interval)) == 0) {
        const auto detect_begin = std::chrono::steady_clock::now();
        cached_detections = detector_->detect(frame_packet.bgr);
        const auto detect_end = std::chrono::steady_clock::now();
        last_detect_ms =
            std::chrono::duration<double, std::milli>(detect_end - detect_begin).count();
        tracked_person = tracker_->update(cached_detections, frame_packet.frame_id);
      } else {
        tracked_person = tracker_->current();
      }
      fall_status = fall_detector_->update(tracked_person, frame_packet.ts_ms);
      maybeQueueFallAlert(frame_packet.bgr,
                          frame_packet.frame_id,
                          frame_packet.ts_ms,
                          smoothed_fps,
                          fall_status.state,
                          last_box_alert_state);
    }

    cv::Mat canvas = frame_packet.bgr.clone();
    if (isPosePipeline()) {
      if (isMediaPipePoseBackend()) {
        drawMediaPipePerson(canvas,
                            mp_person_region,
                            config_.pose_show_detector_box,
                            config_.pose_show_detector_aux_points);
      }
      drawPoseResult(canvas, pose_result, config_.pose_keypoint_score_threshold, show_pose_joints_);
      if (config_.pose_show_landmark_bbox) {
        cv::Rect landmark_box = computePoseLandmarkBox(pose_result, config_.pose_keypoint_score_threshold);
        landmark_box &= cv::Rect(0, 0, canvas.cols, canvas.rows);
        if (landmark_box.width > 0 && landmark_box.height > 0) {
          drawPoseFallBox(canvas, landmark_box, pose_fall_status);
        }
      }
    } else {
      drawTrackedPerson(canvas, tracked_person, fall_status);
    }

    std::vector<std::string> status_lines;
    if (show_status_overlay_) {
      {
        std::ostringstream oss;
        oss << "source=" << config_.camera_source << " size=" << frame_packet.bgr.cols << "x"
            << frame_packet.bgr.rows;
        status_lines.push_back(oss.str());
      }
      {
        std::ostringstream oss;
        oss << "pipeline=" << config_.pipeline_mode;
        status_lines.push_back(oss.str());
      }
      if (isPosePipeline()) {
        {
          std::ostringstream oss;
          oss << "pose_backend=" << config_.pose_backend;
          status_lines.push_back(oss.str());
        }
        if (isRemotePoseBackend()) {
          std::ostringstream oss;
          oss << pose_remote_client_->statusMessage()
              << " frame_id=" << remote_pose_response.frame_id
              << " keypoints=" << pose_result.keypoints.size();
          status_lines.push_back(oss.str());
        } else if (isMediaPipePoseBackend()) {
          std::ostringstream oss;
          oss << mp_person_detector_->statusMessage()
              << " person=" << (mp_person_region.valid() ? "yes" : "no");
          status_lines.push_back(oss.str());
          std::ostringstream pose_oss;
          pose_oss << mp_pose_estimator_->statusMessage()
                   << " keypoints=" << pose_result.keypoints.size();
          status_lines.push_back(pose_oss.str());
        } else {
          std::ostringstream oss;
          oss << pose_estimator_->statusMessage()
              << " keypoints=" << pose_result.keypoints.size();
          status_lines.push_back(oss.str());
        }
        {
          std::ostringstream oss;
          oss << "pose_fall=" << fallStateLabel(pose_fall_status.state)
              << " angle=" << std::fixed << std::setprecision(1) << pose_fall_status.trunk_angle_deg
              << " span=" << std::fixed << std::setprecision(2) << pose_fall_status.vertical_span_ratio
              << " drop=" << std::fixed << std::setprecision(2) << pose_fall_status.hip_drop_speed
              << " vis=" << pose_fall_status.visible_keypoints;
          status_lines.push_back(oss.str());
        }
      } else {
        {
          std::ostringstream oss;
          oss << "bbox_backend=" << config_.bbox_backend;
          status_lines.push_back(oss.str());
        }
        {
          std::ostringstream oss;
          oss << "persons=" << cached_detections.size() << " " << detector_->statusMessage();
          status_lines.push_back(oss.str());
        }
        {
          std::ostringstream oss;
          oss << "track=" << trackerStateLabel(tracked_person)
              << " conf=" << std::fixed << std::setprecision(2) << tracked_person.confidence
              << " miss=" << tracked_person.missing_frames;
          status_lines.push_back(oss.str());
        }
        {
          std::ostringstream oss;
          oss << "fall=" << fallStateLabel(fall_status.state)
              << " ar=" << std::fixed << std::setprecision(2) << fall_status.aspect_ratio
              << " hr=" << std::fixed << std::setprecision(2) << fall_status.height_ratio;
          status_lines.push_back(oss.str());
        }
      }
      {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1)
            << "req_fps=" << config_.frame_fps
            << " actual_fps=" << smoothed_fps
            << " detect_ms=" << last_detect_ms;
        if (isRemotePoseBackend()) {
          oss << " remote_every_ms=" << config_.remote_pose_submit_interval_ms;
        } else {
          oss << " det_every=" << safe_interval;
        }
        status_lines.push_back(oss.str());
      }
      if (fall_alert_client_) {
        status_lines.push_back(fall_alert_client_->statusMessage());
      }
    }
    {
      std::ostringstream oss;
      oss << "keys: q/Esc exit | s save snapshot | i info";
      if (isPosePipeline()) {
        oss << " | j joints";
      }
      status_lines.push_back(oss.str());
    }

    renderer_->drawPreview(canvas, status_lines);

    const int key = renderer_->waitKey(1);
    if (key == 'q' || key == 'Q' || key == 27) {
      break;
    }
    if (key == 's' || key == 'S') {
      const std::string output_path = "data/snapshot_" + std::to_string(saved_frame_count++) + ".jpg";
      if (cv::imwrite(output_path, frame_packet.bgr)) {
        std::cout << "[App] snapshot saved: " << output_path << std::endl;
      } else {
        std::cerr << "[App] failed to save snapshot: " << output_path << std::endl;
      }
    }
    if (key == 'i' || key == 'I') {
      show_status_overlay_ = !show_status_overlay_;
    }
    if ((key == 'j' || key == 'J') && isPosePipeline()) {
      show_pose_joints_ = !show_pose_joints_;
    }
  }

  camera_->stop();
  renderer_->closeWindow();
  return 0;
}

bool App::isPosePipeline() const {
  return normalizePipelineMode(config_.pipeline_mode) == "pose";
}

bool App::isMediaPipePoseBackend() const {
  return normalizePoseBackend(config_.pose_backend) == "mediapipe";
}

bool App::isRemotePoseBackend() const {
  const std::string backend = normalizePoseBackend(config_.pose_backend);
  return backend == "pc_mediapipe" || backend == "remote";
}

bool App::isHogBBoxBackend() const {
  return normalizeBboxBackend(config_.bbox_backend) == "hog";
}

void App::maybeQueueFallAlert(const cv::Mat& frame_bgr,
                              std::uint64_t frame_id,
                              std::uint64_t ts_ms,
                              double fps,
                              FallState current_state,
                              FallState& previous_state) {
  if (!fall_alert_client_) {
    previous_state = current_state;
    return;
  }

  if (isFallTransitionToDetected(previous_state, current_state)) {
    FallAlertEvent event;
    event.source_device = config_.fall_alert_device_id;
    event.frame_id = frame_id;
    event.ts_ms = ts_ms;
    event.mode = currentModeLabel();
    event.fall_state = fallStateLabel(current_state);
    event.message = config_.fall_alert_message;
    event.fps = fps;
    fall_alert_client_->enqueue(frame_bgr, event);
  }

  previous_state = current_state;
}

std::string App::currentModeLabel() const {
  if (isPosePipeline()) {
    return "pose:" + config_.pose_backend;
  }
  return "bbox:" + config_.bbox_backend;
}

std::string App::trim(std::string value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

}  // namespace asdun
