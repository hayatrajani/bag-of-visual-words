// @file    algorithms.hpp
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]

#ifndef BOW_ALGORITHMS_HPP_
#define BOW_ALGORITHMS_HPP_

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/flann.hpp>

#include "bow/core/descriptor.hpp"

namespace bow::algorithms {

/**
 * @brief This function searches for a data point in the search space that is
 * closest to the query point by comparing Euclidean distances. Alternatively,
 * a FLANN-based search can be performed by setting the kdtree parameter.
 *
 * @param descriptor A row vector representing the data point for which the
 *                   nearest neighbor is being queried.
 * @param codebook   A set of data points forming the search space.
 * @param kdtree     An optional parameter pointing to the kdtree to use for
 *                   performing a FLANN-based search.
 *
 * @return The row index of the codebook matrix representing the data point
 * closest to the query point.
 */
int nearestNeighbour(
    const cv::Mat& descriptor, const cv::Mat& codebook,
    cv::flann::GenericIndex<cvflann::L2<float>>* kdtree = nullptr);

/**
 * @brief This function preforms kMeans clustering to partition the input
 * dataset into a set of k clusters, each represented by a cluster center
 * - a row vector of the same dimensionality as that of the input dataset.
 *
 * @param descriptor_dataset The dataset to be clustered.
 * @param num_clusters       The number of clusters to partition the dataset in.
 * @param max_iter           The maximum number of iterations before termination
 * @param epsilon            The desired accuracy to be reached for an early
 *                           termination; default 1e-6.
 * @param use_opencv_kmeans  Set this to true to use the OpenCV implementation
 *                           of the kMeans clustering algorithm; default false.
 * @param use_flann          Set this to true to use a FLANN-based search for
 *                           grouping the dataset into clusters; default false.
 *
 * @return A matrix of row vectors representing the cluster centers.
 */
cv::Mat kMeans(const std::vector<FeatureDescriptor>& descriptor_dataset,
               int num_clusters, int max_iter, double epsilon = 1e-6,
               bool use_opencv_kmeans = false, bool use_flann = false);

}  // namespace bow::algorithms

#endif