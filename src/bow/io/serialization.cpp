#include "bow/io/serialization.hpp"

#include <fstream>
#include <iostream>

#include <opencv2/core/mat.hpp>

namespace bow::io::serialization {

void serialize(const cv::Mat& descriptors, const std::string& image_path,
               const std::string& filename) {
  std::ofstream out_file(filename, std::ios_base::out | std::ios_base::binary);
  if (!out_file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  int type{descriptors.type()};
  std::size_t size{sizeof(int)};
  out_file.write(reinterpret_cast<const char*>(&descriptors.rows), size);
  out_file.write(reinterpret_cast<const char*>(&descriptors.cols), size);
  out_file.write(reinterpret_cast<char*>(&type), size);
  out_file.write(reinterpret_cast<char*>(descriptors.data),
                 descriptors.elemSize() * descriptors.rows * descriptors.cols);
  int image_path_size = image_path.size();
  out_file.write(reinterpret_cast<char*>(&image_path_size), size);
  out_file.write(image_path.data(), image_path_size);
}

std::tuple<cv::Mat, std::string> deserialize(const std::string& filename) {
  std::ifstream in_file(filename, std::ios_base::in | std::ios_base::binary);
  if (!in_file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  std::string image_path;
  int rows{};
  int cols{};
  int type{};
  std::size_t size{sizeof(int)};
  in_file.read(reinterpret_cast<char*>(&rows), size);
  in_file.read(reinterpret_cast<char*>(&cols), size);
  in_file.read(reinterpret_cast<char*>(&type), size);
  cv::Mat descriptors = cv::Mat::zeros(rows, cols, type);
  in_file.read(reinterpret_cast<char*>(descriptors.data),
               descriptors.elemSize() * descriptors.rows * descriptors.cols);
  int image_path_size{};
  in_file.read(reinterpret_cast<char*>(&image_path_size), size);
  image_path.resize(image_path_size);
  in_file.read(image_path.data(), image_path_size);
  return std::make_tuple(descriptors, image_path);
}

}  // namespace bow::io::serialization