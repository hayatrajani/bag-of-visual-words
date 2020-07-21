#include "bow/core/descriptor.hpp"

#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace bow {

FeatureDescriptor::FeatureDescriptor(const std::string& image_path)
    : image_path_{image_path} {
  const cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  std::vector<cv::KeyPoint> keypoints;
  auto detector = cv::xfeatures2d::SIFT::create();
  detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors_);
}

FeatureDescriptor FeatureDescriptor::deserialize(const std::string& filename) {
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
  return {image_path, descriptors};
}

void FeatureDescriptor::serialize(const std::string& filename) {
  std::ofstream out_file(filename, std::ios_base::out | std::ios_base::binary);
  if (!out_file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  int type{descriptors_.type()};
  std::size_t size{sizeof(int)};
  out_file.write(reinterpret_cast<const char*>(&descriptors_.rows), size);
  out_file.write(reinterpret_cast<const char*>(&descriptors_.cols), size);
  out_file.write(reinterpret_cast<char*>(&type), size);
  out_file.write(
      reinterpret_cast<char*>(descriptors_.data),
      descriptors_.elemSize() * descriptors_.rows * descriptors_.cols);
  int image_path_size = image_path_.size();
  out_file.write(reinterpret_cast<char*>(&image_path_size), size);
  out_file.write(image_path_.data(), image_path_size);
}

}  // namespace bow