#include "core/App.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
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

cv::Scalar trackerColor(const TrackedPerson& tracked_person) {
  if (!tracked_person.has_target) {
    return cv::Scalar(0, 160, 255);
  }
  if (tracked_person.is_holding) {
    return cv::Scalar(0, 200, 255);
  }
  if (tracked_person.is_confirmed) {
    return cv::Scalar(0, 220, 0);
  }
  return cv::Scalar(255, 210, 0);
}

void drawTrackedPerson(cv::Mat& canvas, const TrackedPerson& tracked_person) {
  if (!tracked_person.has_target || tracked_person.box.width <= 0 || tracked_person.box.height <= 0) {
    return;
  }

  const cv::Scalar color = trackerColor(tracked_person);
  cv::rectangle(canvas, tracked_person.box, color, 2);

  std::ostringstream oss;
  oss << "track " << std::fixed << std::setprecision(2) << tracked_person.confidence
      << " " << trackerStateLabel(tracked_person);
  if (tracked_person.missing_frames > 0) {
    oss << " miss=" << tracked_person.missing_frames;
  }

  cv::putText(canvas,
              oss.str(),
              cv::Point(tracked_person.box.x, std::max(12, tracked_person.box.y - 8)),
              cv::FONT_HERSHEY_SIMPLEX,
              0.45,
              color,
              1,
              cv::LINE_AA);
}

void drawFallState(cv::Mat& canvas, const FallStatus& fall_status) {
  const std::string label = fallStateLabel(fall_status.state);
  cv::Scalar color(180, 220, 180);
  if (fall_status.state == FallState::SuspectedFall) {
    color = cv::Scalar(0, 210, 255);
  } else if (fall_status.state == FallState::FallDetected) {
    color = cv::Scalar(0, 70, 255);
  }

  const cv::Point anchor(18, std::max(28, canvas.rows - 22));
  cv::putText(canvas,
              label,
              anchor,
              cv::FONT_HERSHEY_SIMPLEX,
              0.7,
              cv::Scalar(0, 0, 0),
              4,
              cv::LINE_AA);
  cv::putText(canvas,
              label,
              anchor,
              cv::FONT_HERSHEY_SIMPLEX,
              0.7,
              color,
              2,
              cv::LINE_AA);
}

}  // namespace

App::App(std::string config_path) : config_path_(std::move(config_path)) {}

App::~App() = default;

int App::run() {
  if (!loadConfig()) {
    std::cerr << "[App] failed to load config: " << config_path_ << std::endl;
    return 1;
  }
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
      if (key == "camera_source") {
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

bool App::initComponents() {
  cv::setNumThreads(1);

  camera_ = std::make_unique<CameraManager>(
      config_.camera_source, config_.frame_width, config_.frame_height, config_.frame_fps);
  detector_ = std::make_unique<PersonDetector>(config_.detector_model_path,
                                               config_.detector_input_width,
                                               config_.detector_input_height,
                                               config_.detector_confidence_threshold,
                                               config_.detector_nms_threshold,
                                               config_.detector_person_class_id,
                                               config_.detector_swap_rb,
                                               config_.detector_use_hog_fallback,
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
  renderer_ = std::make_unique<Renderer>(config_.window_name);

  if (!camera_->start()) {
    return false;
  }

  detector_->init();

  std::cout << "[App] camera preview started." << std::endl;
  std::cout << "[App] press q or Esc to exit, press s to save a snapshot." << std::endl;
  std::cout << "[App] " << detector_->statusMessage() << std::endl;
  return true;
}

int App::previewLoop() {
  int saved_frame_count = 0;
  std::vector<PersonDetection> cached_detections;
  TrackedPerson tracked_person;
  FallStatus fall_status;
  double smoothed_fps = 0.0;
  std::uint64_t last_frame_ts_ms = 0;
  double last_detect_ms = 0.0;
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

    cv::Mat canvas = frame_packet.bgr.clone();
    drawTrackedPerson(canvas, tracked_person);
    drawFallState(canvas, fall_status);

    std::vector<std::string> status_lines;
    {
      std::ostringstream oss;
      oss << "source=" << config_.camera_source << " size=" << frame_packet.bgr.cols << "x" << frame_packet.bgr.rows;
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
    {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(1)
          << "req_fps=" << config_.frame_fps
          << " actual_fps=" << smoothed_fps
          << " detect_ms=" << last_detect_ms
          << " det_every=" << safe_interval;
      status_lines.push_back(oss.str());
    }
    status_lines.push_back("keys: q/Esc exit | s save snapshot");

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
  }

  camera_->stop();
  renderer_->closeWindow();
  return 0;
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
