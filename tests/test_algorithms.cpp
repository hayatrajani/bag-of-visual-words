// @file    test_algorithms.hpp
// @author  Ignacio Vizzo (creator) [ivizzo@uni-bonn.de]
// @author  Hayat Rajani (editor)   [hayat.rajani@uni-bonn.de]
//
// Original Copyright (c) 2020 Ignacio Vizzo, all rights reserved

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "bow/algorithms/algorithms.hpp"
#include "test_data.hpp"
#include "test_utils.hpp"

static void TestKMeans(const cv::Mat& gt_cluster, bool use_cv_kmeans = true,
                       bool use_flann = false) {
  auto& data = getDummyData();
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

TEST(KMeansClustering, EmptyData) {
  const int dict_size = 1;
  const int iterations = 10;

  EXPECT_THROW(bow::algorithms::kMeans({}, dict_size, iterations),
               std::runtime_error);
}

TEST(KMeansClustering, NegativeClusters) {
  auto& data = getDummyData();
  const int dict_size = -1;
  const int iterations = 10;

  EXPECT_THROW(bow::algorithms::kMeans(data, dict_size, iterations),
               std::runtime_error);
}

TEST(KMeansClustering, NullClusters) {
  auto& data = getDummyData();
  const int dict_size = 0;
  const int iterations = 10;

  EXPECT_THROW(bow::algorithms::kMeans(data, dict_size, iterations),
               std::runtime_error);
}

TEST(KMeansClustering, MoreLabelsThanFeatures) {
  auto& data = getDummyData();
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
TEST(KMeansClustering, Use2Words_CV) { TestKMeans(get2Kmeans()); }

TEST(KMeansClustering, MinimumSignificantCluster_CustomNN) {
  TestKMeans(get5Kmeans(), false);
}
TEST(KMeansClustering, Use3Words_CustomNN) { TestKMeans(get3Kmeans(), false); }
TEST(KMeansClustering, Use2Words_CustomNN) { TestKMeans(get2Kmeans(), false); }

TEST(KMeansClustering, MinimumSignificantCluster_CustomFLAN) {
  TestKMeans(get5Kmeans(), false, true);
}
TEST(KMeansClustering, Use3Words_CustomFLAN) {
  TestKMeans(get3Kmeans(), false, true);
}
TEST(KMeansClustering, Use2Words_CustomFLAN) {
  TestKMeans(get2Kmeans(), false, true);
}