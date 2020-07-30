#ifndef BOW_FEATURE_DESCRIPTOR_HPP_
#define BOW_FEATURE_DESCRIPTOR_HPP_

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace bow {

class FeatureDescriptor {
 private:
  const std::string image_path_;
  cv::Mat descriptors_;

 public:
  FeatureDescriptor(const std::string& image_path, const cv::Mat& descriptors)
      : image_path_{image_path}, descriptors_{descriptors.clone()} {}
  explicit FeatureDescriptor(const std::string& image_path);

  static FeatureDescriptor deserialize(const std::string& filename);
  void serialize(const std::string& filename);

  std::string getImagePath() const { return image_path_; }
  cv::Mat getDescriptors() const { return descriptors_; }

  int size() const { return descriptors_.rows; }
  bool empty() const { return descriptors_.empty(); }
};

}  // namespace bow

#endif