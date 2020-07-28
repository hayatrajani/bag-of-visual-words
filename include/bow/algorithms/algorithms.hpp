#ifndef BOW_ALGORITHMS_HPP_
#define BOW_ALGORITHMS_HPP_

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/flann.hpp>

#include "bow/core/descriptor.hpp"

namespace bow::algorithms {

int nearestNeighbour(
    const cv::Mat& descriptor, const cv::Mat& codebook,
    cv::flann::GenericIndex<cvflann::L2<float>>* kdtree = nullptr);

cv::Mat kMeans(const std::vector<FeatureDescriptor>& descriptor_dataset,
               int num_clusters, int max_iter, double epsilon = 1e-6,
               bool use_opencv_kmeans = true, bool use_flann = true);

}  // namespace bow::algorithms

#endif