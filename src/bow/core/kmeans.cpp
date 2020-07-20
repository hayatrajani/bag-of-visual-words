#include "bow/core/kmeans.hpp"

#include <vector>

#include <opencv2/core.hpp>

namespace bow {

cv::Mat kMeans(const std::vector<cv::Mat>& descriptors, int k, int max_iter,
               bool use_opencv_kmeans, double epsilon) {
  if (use_opencv_kmeans) {
    const double eps{epsilon};
    const int attempts{1};
    cv::Mat stacked_descriptors;
    for (const auto& descriptor : descriptors) {
      stacked_descriptors.push_back(descriptor);
    }
    cv::Mat centers;
    cv::Mat labels;
    cv::kmeans(stacked_descriptors, k, labels,
               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                max_iter, eps),
               attempts, cv::KMEANS_RANDOM_CENTERS, centers);
    return centers;
  }
  return {};
}

}  // namespace bow