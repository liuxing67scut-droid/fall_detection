#include "ui/Renderer.hpp"

#include <algorithm>
#include <utility>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

void drawStatusLine(cv::Mat& canvas, const std::string& text, const cv::Point& origin) {
  constexpr double kFontScale = 0.45;
  constexpr int kTextThickness = 1;

  // A thin shadow keeps the overlay readable without blocking the preview.
  cv::putText(canvas,
              text,
              origin + cv::Point(1, 1),
              cv::FONT_HERSHEY_SIMPLEX,
              kFontScale,
              cv::Scalar(20, 20, 20),
              2,
              cv::LINE_AA);
  cv::putText(canvas,
              text,
              origin,
              cv::FONT_HERSHEY_SIMPLEX,
              kFontScale,
              cv::Scalar(80, 255, 80),
              kTextThickness,
              cv::LINE_AA);
}

}  // namespace

Renderer::Renderer(std::string window_name) : window_name_(std::move(window_name)) { ensureWindow(); }

Renderer::~Renderer() { closeWindow(); }

void Renderer::ensureWindow() const {
  if (!window_created_) {
    cv::namedWindow(window_name_, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    window_created_ = true;
  }
}

void Renderer::drawPreview(const cv::Mat& frame_bgr, const std::vector<std::string>& status_lines) const {
  ensureWindow();

  cv::Mat canvas = frame_bgr.clone();
  int y = 20;
  for (const auto& line : status_lines) {
    drawStatusLine(canvas, line, cv::Point(12, y));
    y += 20;
  }

  cv::imshow(window_name_, canvas);
}

int Renderer::waitKey(int delay_ms) const { return cv::waitKey(delay_ms); }

void Renderer::closeWindow() const {
  if (window_created_) {
    cv::destroyWindow(window_name_);
    cv::waitKey(1);
    window_created_ = false;
  }
}

}  // namespace asdun
