// @file    test_algorithms.cpp
// @author  Ignacio Vizzo   [ivizzo@uni-bonn.de]
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]
//
// Original Copyright (c) 2020 Ignacio Vizzo, all rights reserved

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "bow/algorithms/algorithms.hpp"
#include "test_data.hpp"
#include "test_utils.hpp"

static void TestKMeans(const cv::Mat& gt_cluster, bool use_cv_kmeans = true,
                       bool use_flann = false) {
  const auto& data = getDummyData();
  const double epsilon = 1e-6;
  const int dict_size = gt_cluster.rows;
  const int iterations = 10;
  auto centroids = bow::algorithms::kMeans(data, dict_size, iterations, epsilon,
                                           use_cv_kmeans, use_flann);

  EXPECT_EQ(centroids.rows, dict_size);
  EXPECT_EQ(centroids.size, gt_cluster.size);

  // Need to sort the output, otherwise the comparison will fail
  cv::sort(centroids, centroids, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  EXPECT_TRUE(mat_are_equal<float>(centroids, gt_cluster))
      << "gt_centroids:\n"
      << gt_cluster << "\ncomputed centroids:\n"
      << centroids;
}

TEST(NearestNeighbour, EmptyDescriptor) {
  EXPECT_THROW(bow::algorithms::nearestNeighbour({}, get5Kmeans()),
               std::runtime_error);
}

TEST(NearestNeighbour, DescriptorMatrix) {
  EXPECT_THROW(
      bow::algorithms::nearestNeighbour(getAllFeatures(), get5Kmeans()),
      std::runtime_error);
}

TEST(NearestNeighbour, EmptyCodebook) {
  EXPECT_THROW(bow::algorithms::nearestNeighbour(
                   cv::Mat_<float>(1, getNumColumns(), 7.0F), {}),
               std::runtime_error);
}

TEST(NearestNeighbour, UnitCodebook) {
  auto index = bow::algorithms::nearestNeighbour(
      cv::Mat_<float>(1, getNumColumns(), 7.0F),
      cv::Mat_<float>(1, getNumColumns(), 70.0F));
  EXPECT_EQ(index, 0);
}

TEST(NearestNeighbour, TrivialExample) {
  const auto gt = cv::Mat_<float>(1, getNumColumns(), 20.0F);
  const auto codebook = get5Kmeans();
  auto index = bow::algorithms::nearestNeighbour(
      cv::Mat_<float>(1, getNumColumns(), 15.0F), codebook);
  EXPECT_TRUE(mat_are_equal<float>(gt, codebook.row(index)))
      << "expected:\n"
      << gt << "\ncomputed:\n"
      << codebook.row(index);
}

TEST(KMeansClustering, EmptyData) {
  const int dict_size = 1;
  const int iterations = 10;

  EXPECT_THROW(bow::algorithms::kMeans({}, dict_size, iterations),
               std::runtime_error);
}

TEST(KMeansClustering, NegativeClusters) {
  const auto& data = getDummyData();
  const int dict_size = -1;
  const int iterations = 10;

  EXPECT_THROW(bow::algorithms::kMeans(data, dict_size, iterations),
               std::runtime_error);
}

TEST(KMeansClustering, NullClusters) {
  const auto& data = getDummyData();
  const int dict_size = 0;
  const int iterations = 10;

  EXPECT_THROW(bow::algorithms::kMeans(data, dict_size, iterations),
               std::runtime_error);
}

TEST(KMeansClustering, MoreLabelsThanFeatures) {
  const auto& data = getDummyData();
  const int dict_size = getMaxFeatures() + 1;
  const int iterations = 10;

  EXPECT_THROW(bow::algorithms::kMeans(data, dict_size, iterations),
               std::runtime_error);
}

TEST(KMeansClustering, SelectAllFeatures) { TestKMeans(getAllFeatures()); }

TEST(KMeansClustering, MinimumSignificantCluster_CV) {
  TestKMeans(get5Kmeans());
}

TEST(KMeansClustering, Use3Words_CV) { TestKMeans(get3Kmeans()); }

TEST(KMeansClustering, MinimumSignificantCluster_Custom_NN) {
  TestKMeans(get5Kmeans(), false);
}

TEST(KMeansClustering, Use3Words_Custom_NN) { TestKMeans(get3Kmeans(), false); }

TEST(KMeansClustering, MinimumSignificantCluster_Custom_FLANN) {
  TestKMeans(get5Kmeans(), false, true);
}

TEST(KMeansClustering, Use3Words_Custom_FLAN) {
  TestKMeans(get3Kmeans(), false, true);
}
