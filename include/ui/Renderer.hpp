#pragma once

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace asdun {

class Renderer {
 public:
  explicit Renderer(std::string window_name);
  ~Renderer();

  void drawPreview(const cv::Mat& frame_bgr, const std::vector<std::string>& status_lines) const;
  int waitKey(int delay_ms) const;
  void closeWindow() const;

 private:
  void ensureWindow() const;

  std::string window_name_;
  mutable bool window_created_ = false;
};

}  // namespace asdun
