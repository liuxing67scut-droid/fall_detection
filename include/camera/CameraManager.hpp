#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

namespace asdun {

struct FramePacket {
  cv::Mat bgr;
  std::uint64_t ts_ms = 0;
  std::uint64_t frame_id = 0;
};

class CameraManager {
 public:
  CameraManager(std::string source, int width, int height, int fps);
  ~CameraManager();

  bool start();
  void stop();

  bool getLatestFrame(FramePacket& out, std::uint32_t timeout_ms);
  int width() const { return width_; }
  int height() const { return height_; }
  int fps() const { return fps_; }

 private:
  bool openCapture();
  void captureLoop();

  std::string source_;
  int width_ = 640;
  int height_ = 480;
  int fps_ = 30;

  cv::VideoCapture cap_;
  std::atomic<bool> running_{false};
  std::thread capture_thread_;

  std::mutex frame_mutex_;
  std::condition_variable frame_cv_;
  FramePacket latest_frame_;
  bool has_frame_ = false;
  std::uint64_t frame_counter_ = 0;
  std::uint64_t last_delivered_frame_id_ = 0;
};

}  // namespace asdun
