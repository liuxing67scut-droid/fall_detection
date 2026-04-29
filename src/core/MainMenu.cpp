#include "core/MainMenu.hpp"

#include <iostream>
#include <optional>
#include <string>

namespace asdun {

std::optional<RuntimeModeSelection> showMainMenu() {
  while (true) {
    std::cout << "========================" << std::endl;
    std::cout << "Fall Detect Main Menu" << std::endl;
    std::cout << "1) HOG + Box" << std::endl;
    std::cout << "2) YOLO11 + Box" << std::endl;
    std::cout << "3) MediaPipe + Pose" << std::endl;
    std::cout << "4) PC Mode (Later)" << std::endl;
    std::cout << "0) Exit" << std::endl;
    std::cout << "Enter command: ";

    std::string input;
    if (!std::getline(std::cin, input)) {
      return std::nullopt;
    }

    if (input == "0") {
      return std::nullopt;
    }
    if (input == "1") {
      return RuntimeModeSelection{"bbox", "hog", "mediapipe"};
    }
    if (input == "2") {
      return RuntimeModeSelection{"bbox", "yolo11", "mediapipe"};
    }
    if (input == "3") {
      return RuntimeModeSelection{"pose", "yolo11", "mediapipe"};
    }
    if (input == "4") {
      std::cout << "[Menu] PC Mode is not implemented yet." << std::endl;
      continue;
    }

    std::cout << "[Menu] Invalid command. Please try again." << std::endl;
  }
}

}  // namespace asdun
