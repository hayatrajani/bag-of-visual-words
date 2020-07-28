#include "bow/algorithms/algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>

#include "bow/core/descriptor.hpp"

using flannL2index = cv::flann::GenericIndex<cvflann::L2<float>>;

namespace bow::algorithms {

namespace {

void initClusterCenters(const cv::Mat& dataset, cv::Mat& centers,
                        int num_clusters) {
  std::vector<int> range(dataset.rows);
  std::iota(range.begin(), range.end(), 0);
  std::shuffle(range.begin(), range.end(), std::mt19937{42});
  for (int k{}; k < num_clusters; ++k) {
    centers.push_back(dataset.row(range[k]));
  }
}

void kmeans_(const cv::Mat& stacked_descriptors, cv::Mat& labels,
             cv::Mat& centers, int num_clusters, int max_iter, double epsilon,
             bool use_flann) {
  auto type = stacked_descriptors.type();
  std::unique_ptr<flannL2index> kdtree{};
  // initialize cluster centers
  initClusterCenters(stacked_descriptors, centers, num_clusters);
  // repeat for max_iter iterations
  for (int i{}; i < max_iter; ++i) {
    // assign data points to their nearest cluster
    if (use_flann) {
      kdtree = std::make_unique<flannL2index>(centers,
                                              cvflann::AutotunedIndexParams());
    }
    labels = cv::Mat::zeros(num_clusters, stacked_descriptors.rows, CV_8U);
    for (int m{}; m < stacked_descriptors.rows; ++m) {
      int k =
          nearestNeighbour(stacked_descriptors.row(m), centers, kdtree.get());
      *labels.ptr<uchar>(k, m) = 1;
    }
    // re-compute cluster centers
    std::vector<float> delta(num_clusters);
    for (int k{}; k < num_clusters; ++k) {
      auto old_center = centers.row(k).clone();
      cv::Mat descriptors_sum =
          cv::Mat::zeros(1, stacked_descriptors.cols, type);
      for (int m{}; m < stacked_descriptors.rows; ++m) {
        descriptors_sum +=
            stacked_descriptors.row(m) * *labels.ptr<uchar>(k, m);
      }
      auto num_descriptors = cv::sum(labels.row(k))[0];
      if (num_descriptors != 0) {
        auto new_center = descriptors_sum / num_descriptors;
        centers.row(k) = new_center;
        delta[k] = cv::norm(old_center, new_center);
      }
    }
    // stop if the average change in centroids is smaller than epsilon
    auto avg_delta =
        (std::accumulate(delta.begin(), delta.end(), 0.0) / delta.size());
    if (avg_delta <= epsilon) {
      break;
    }
  }
}

}  // anonymous namespace

int nearestNeighbour(const cv::Mat& descriptor, const cv::Mat& codebook,
                     flannL2index* kdtree) {
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
    auto dist = cv::norm(codebook.row(r), descriptor);
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
  } else {
    kmeans_(stacked_descriptors, labels, centers, num_clusters, max_iter,
            epsilon, use_flann);
  }
  return centers;
}

}  // namespace bow::algorithms