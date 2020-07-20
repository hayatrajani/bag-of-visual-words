#ifndef BOW_IO_SERIALIZATION_HPP_
#define BOW_IO_SERIALIZATION_HPP_

#include <string>

#include <opencv2/core/mat.hpp>

namespace bow::io::serialization {

void serialize(const cv::Mat& descriptor, const std::string& image_path,
               const std::string& filename);

std::tuple<cv::Mat, std::string> deserialize(const std::string& filename);

}  // namespace bow::io::serialization

#endif