// @file    html_writer.cpp
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]

#include "bow/web/html_writer.hpp"

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace bow::web {

HTML_Writer::HTML_Writer(const std::string& html_file_path)
    : html_file_path_{html_file_path} {
  fout_.open(html_file_path, std::ios_base::out);
  if (!fout_) {
    throw std::runtime_error("Cannot open file " + html_file_path);
  }
  fout_.precision(4);
}

void HTML_Writer::openDocument() { fout_ << "<!DOCTYPE html>\n<html>\n"; }

void HTML_Writer::closeDocument() { fout_ << "</html>\n"; }

void HTML_Writer::openBody() { fout_ << "<body>\n"; }

void HTML_Writer::closeBody() { fout_ << "</body>\n"; }

void HTML_Writer::openRow() { fout_ << "<div class=\"row\">\n"; }

void HTML_Writer::closeRow() { fout_ << "</div>\n"; }

void HTML_Writer::addCSS(const std::string& stylesheet) {
  fout_ << "<head>\n<link rel=\"stylesheet\" type=\"text/css\" href=\""
        << fs::relative(stylesheet, fs::path(html_file_path_).parent_path())
               .string()
        << "\" />\n</head>\n";
}

void HTML_Writer::addTitle(const std::string& title) {
  fout_ << "<title>" << title << "</title>\n";
}

void HTML_Writer::addImage(const std::string& image_path, float distance,
                           bool query_image) {
  fs::path image_path_(image_path);
  if (!fs::exists(image_path)) {
    std::cerr << "[ERROR] Image does not exist!\n";
  } else if (image_path_.extension() != ".png") {
    std::cerr << "[ERROR] Invalid image!\n";
    return;
  }
  if (query_image) {
    fout_ << "<div class=\"column\" style=\"border: 5px solid green;\">\n";
    fout_ << "<h3>" << image_path_.filename().string() << "</h3>\n";
    fout_ << "<img src=\""
          << fs::relative(image_path, fs::path(html_file_path_).parent_path())
                 .string()
          << "\" />\n";
    fout_ << "<p><b>Query Image</b></p>\n</div>\n";
  } else {
    fout_ << "<div class=\"column\">\n";
    fout_ << "<h3>" << image_path_.filename().string() << "</h3>\n";
    fout_ << "<img src=\""
          << fs::relative(image_path, fs::path(html_file_path_).parent_path())
                 .string()
          << "\" />\n";
    fout_ << "<p><b>Distance Measure: " << distance << "</b></p>\n</div>\n";
  }
}

}  // namespace bow::web