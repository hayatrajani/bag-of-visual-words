#ifndef BOW_KMEANS_HPP_
#define BOW_KMEANS_HPP_

#include <vector>

#include <opencv2/core/mat.hpp>

namespace bow {

cv::Mat kMeans(const std::vector<cv::Mat>& descriptors, int k, int max_iter,
               bool use_opencv_kmeans = true, double epsilon = 0.001);

}  // namespace bow

#endif