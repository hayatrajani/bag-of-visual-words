#include <array>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "homework_3.h"

using ImageRow = std::array<std::tuple<std::string, float>, 3>;

int main() {
  const float score{0.5};
  ImageRow row1{std::make_tuple("../web_app/data/000000.png", score),
                std::make_tuple("../web_app/data/000100.png", score),
                std::make_tuple("../web_app/data/000200.png", score)};
  ImageRow row2{std::make_tuple("../web_app/data/000300.png", score),
                std::make_tuple("../web_app/data/000400.png", score),
                std::make_tuple("../web_app/data/000500.png", score)};
  ImageRow row3{std::make_tuple("../web_app/data/000600.png", score),
                std::make_tuple("../web_app/data/000700.png", score),
                std::make_tuple("../web_app/data/000800.png", score)};
  std::vector<ImageRow> rows{row1, row2, row3};
  image_browser::CreateImageBrowser("Test Program!", "../web_app/style.css",
                                    rows);
  return EXIT_SUCCESS;
}