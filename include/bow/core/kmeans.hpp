#ifndef BOW_KMEANS_HPP_
#define BOW_KMEANS_HPP_

#include <vector>

#include <opencv2/core/mat.hpp>

#include "bow/core/descriptor.hpp"

namespace bow {

cv::Mat kMeans(const std::vector<FeatureDescriptor>& descriptor_dataset, int k,
               int max_iter, bool use_opencv_kmeans = true,
               double epsilon = 0.001);

}  // namespace bow

#endif