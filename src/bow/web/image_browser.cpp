#include "bow/web/image_browser.hpp"

#include <filesystem>
#include <string>
#include <vector>

#include "bow/web/html_writer.hpp"

namespace fs = std::filesystem;

namespace bow::web::image_browser {

void createImageBrowser(
    const std::string& query_image_path,
    const std::vector<std::pair<std::string, float>>& similarities,
    const std::string& output_dir, const std::string& css_path) {
  if (!fs::exists(output_dir)) {
    fs::create_directories(output_dir);
  }
  fs::path query_image{query_image_path};
  HTML_Writer html_writer((output_dir / query_image.stem()).string() + ".html");
  html_writer.openDocument();
  html_writer.addTitle("Comparison Results for " +
                       query_image.filename().string());
  if (fs::is_regular_file(css_path)) {
    html_writer.addCSS(css_path);
  } else {
    std::cerr << "[INFO] No CSS found!\n";
  }
  html_writer.openBody();
  html_writer.openRow();
  html_writer.addImage(query_image_path, 0, true);
  html_writer.closeRow();
  int image_count{};
  for (const auto& similarity : similarities) {
    if (image_count == 0) {
      html_writer.openRow();
    }
    html_writer.addImage(similarity.first, similarity.second);
    ++image_count;
    if (image_count == 3) {
      html_writer.closeRow();
      image_count = 0;
    }
  }
  if (image_count != 0) {
    html_writer.closeRow();
  }
  html_writer.closeBody();
  html_writer.closeDocument();
}

}  // namespace bow::web::image_browser