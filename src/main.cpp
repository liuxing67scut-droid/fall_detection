#include <exception>
#include <iostream>

#include "core/App.hpp"
#include "core/MainMenu.hpp"

int main() {
  try {
    const auto selection = asdun::showMainMenu();
    if (!selection.has_value()) {
      return 0;
    }
    asdun::App app("./config/app.yaml", selection);
    return app.run();
  } catch (const std::exception& ex) {
    std::cerr << "[FATAL] " << ex.what() << std::endl;
    return 1;
  }
}
