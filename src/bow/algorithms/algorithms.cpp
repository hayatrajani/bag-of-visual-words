#include "bow/algorithms/algorithms.hpp"

#include <limits>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>

#include "bow/core/descriptor.hpp"

namespace bow::algorithms {

int nearestNeighbour(const cv::Mat& descriptor, const cv::Mat& codebook,
                     cv::flann::GenericIndex<cvflann::L2<float>>* kdtree) {
  if (descriptor.empty() || codebook.empty()) {
    throw std::runtime_error("Empty input(s)!");
  }
  if (descriptor.rows > 1) {
    throw std::runtime_error("Descriptor must be a row vector not a matrix!");
  }
  if (codebook.rows == 1) {
    return 0;
  }
  if (kdtree) {
    int k{1};
    std::vector<int> indices(k);
    std::vector<float> distances(k);
    kdtree->knnSearch(descriptor, indices, distances, k,
                      cvflann::SearchParams());
    return indices[0];
  }
  int nearest_cluster_idx{};
  float min_dist{std::numeric_limits<float>::max()};
  for (int r = 0; r < codebook.rows; ++r) {
    auto diff = codebook.row(r) - descriptor;
    float dist = static_cast<float>(std::sqrt(diff.dot(diff)));
    if (dist < min_dist) {
      min_dist = dist;
      nearest_cluster_idx = r;
    }
  }
  return nearest_cluster_idx;
}

cv::Mat kMeans(const std::vector<FeatureDescriptor>& descriptor_dataset,
               int num_clusters, int max_iter, double epsilon,
               bool use_opencv_kmeans, bool use_flann) {
  if (descriptor_dataset.empty()) {
    throw std::runtime_error("Empty dataset!");
  }
  cv::Mat stacked_descriptors;
  for (const auto& descriptor : descriptor_dataset) {
    stacked_descriptors.push_back(descriptor.getDescriptors());
  }
  if (num_clusters > stacked_descriptors.rows) {
    throw std::runtime_error(
        "Number of clusters greater than the total number of data points!");
  }
  if (num_clusters == stacked_descriptors.rows) {
    return stacked_descriptors;
  }
  cv::Mat centers;
  cv::Mat labels;
  if (use_opencv_kmeans) {
    cv::kmeans(stacked_descriptors, num_clusters, labels,
               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                max_iter, epsilon),
               1, cv::KMEANS_RANDOM_CENTERS, centers);
    return centers;
  }
  return {};
}

}  // namespace bow::algorithms