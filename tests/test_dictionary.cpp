// @file    test_dictionary.cpp
// @author  Ignacio Vizzo   [ivizzo@uni-bonn.de]
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]
//
// Original Copyright (c) 2020 Ignacio Vizzo, all rights reserved

#include <gtest/gtest.h>

#include <filesystem>

#include <opencv2/opencv.hpp>

#include "bow/core/dictionary.hpp"
#include "test_data.hpp"
#include "test_utils.hpp"

namespace fs = std::filesystem;

namespace {

const int max_iter = 10;
const int dict_size = 5;
auto& dictionary = bow::Dictionary::getInstance();

}  // anonymous namespace

TEST(Dictionary, BuildEmptyDictionary) {
  dictionary.build({}, dict_size, max_iter);
  ASSERT_TRUE(dictionary.empty());
  ASSERT_TRUE(!dictionary.getIndex());
}

TEST(Dictionary, BuildEmptyDictionaryFromData) {
  dictionary.setVocabulary({});
  ASSERT_TRUE(dictionary.empty());
  ASSERT_TRUE(!dictionary.getIndex());
}

TEST(Dictionary, BuildDictionary) {
  const auto& descriptors = getDummyData();
  dictionary.build(descriptors, dict_size, max_iter, 1e-6, true, false);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_TRUE(!dictionary.getIndex());
  ASSERT_EQ(dictionary.size(), dict_size);

  const auto& gt_cluster = get5Kmeans();
  const auto& centroids = dictionary.getVocabulary();

  // Need to sort the output, otherwise the comparison will fail
  cv::sort(centroids, centroids, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  EXPECT_TRUE(mat_are_equal<float>(centroids, gt_cluster))
      << "gt_centroids:\n"
      << gt_cluster << "\ncomputed centroids:\n"
      << centroids;
}

TEST(Dictionary, BuildDictionaryWithFLANN) {
  const auto& descriptors = getDummyData();
  dictionary.build(descriptors, dict_size, max_iter);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_TRUE(dictionary.getIndex());
  ASSERT_EQ(dictionary.size(), dict_size);

  const auto& gt_cluster = get5Kmeans();
  const auto& centroids = dictionary.getVocabulary();

  // Need to sort the output, otherwise the comparison will fail
  cv::sort(centroids, centroids, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  EXPECT_TRUE(mat_are_equal<float>(centroids, gt_cluster))
      << "gt_centroids:\n"
      << gt_cluster << "\ncomputed centroids:\n"
      << centroids;
}

TEST(Dictionary, BuildDictionaryFromData) {
  const auto& gt_cluster = get5Kmeans();

  dictionary.setVocabulary(gt_cluster);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_TRUE(!dictionary.getIndex());
  ASSERT_EQ(dictionary.size(), dict_size);

  const auto& centroids = dictionary.getVocabulary();

  // Need to sort the output, otherwise the comparison will fail
  cv::sort(centroids, centroids, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  EXPECT_TRUE(mat_are_equal<float>(centroids, gt_cluster))
      << "gt_centroids:\n"
      << gt_cluster << "\ncomputed centroids:\n"
      << centroids;
}

TEST(Dictionary, BuildDictionaryFromDataWithFLANN) {
  const auto& gt_cluster = get5Kmeans();

  dictionary.setVocabulary(gt_cluster, true);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_TRUE(dictionary.getIndex());
  ASSERT_EQ(dictionary.size(), dict_size);

  const auto& centroids = dictionary.getVocabulary();

  // Need to sort the output, otherwise the comparison will fail
  cv::sort(centroids, centroids, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  EXPECT_TRUE(mat_are_equal<float>(centroids, gt_cluster))
      << "gt_centroids:\n"
      << gt_cluster << "\ncomputed centroids:\n"
      << centroids;
}

TEST(Dictionary, SerializationFakeFile) {
  EXPECT_THROW(dictionary.serialize({}), std::runtime_error);
  EXPECT_THROW(dictionary.deserialize({}), std::runtime_error);
}

TEST(Dictionary, SerializationTrivial) {
  const std::string file_name = "temp.bin";
  const auto& gt_cluster = get5Kmeans();

  dictionary.setVocabulary(gt_cluster);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_TRUE(!dictionary.getIndex());
  ASSERT_EQ(dictionary.size(), dict_size);

  dictionary.serialize(file_name);
  ASSERT_TRUE(fs::exists(file_name));

  dictionary.setVocabulary({});
  dictionary.deserialize(file_name);

  const auto& centroids = dictionary.getVocabulary();

  // Need to sort the output, otherwise the comparison will fail
  cv::sort(centroids, centroids, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  EXPECT_TRUE(mat_are_equal<float>(centroids, gt_cluster))
      << "gt_centroids:\n"
      << gt_cluster << "\ncomputed centroids:\n"
      << centroids;

  fs::remove(file_name);
}

TEST(Dictionary, SerializationFlann) {
  const std::string file_name = "temp.bin";
  const std::string flann_file_name = "temp_flann.bin";
  const auto& gt_cluster = get5Kmeans();

  dictionary.setVocabulary(gt_cluster, true);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_TRUE(dictionary.getIndex());
  ASSERT_EQ(dictionary.size(), dict_size);

  dictionary.serialize(file_name, flann_file_name);
  ASSERT_TRUE(fs::exists(file_name));
  ASSERT_TRUE(fs::exists(flann_file_name));

  dictionary.setVocabulary({});
  dictionary.deserialize(file_name, true, flann_file_name);

  ASSERT_TRUE(dictionary.getIndex());
  const auto& centroids = dictionary.getVocabulary();

  // Need to sort the output, otherwise the comparison will fail
  cv::sort(centroids, centroids, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  EXPECT_TRUE(mat_are_equal<float>(centroids, gt_cluster))
      << "gt_centroids:\n"
      << gt_cluster << "\ncomputed centroids:\n"
      << centroids;

  fs::remove(file_name);
  fs::remove(flann_file_name);
}

TEST(Dictionary, SerializationFlannNoFile) {
  const std::string file_name = "temp.bin";
  const std::string default_flann_file = "bow_index_params.flann";
  const auto& gt_cluster = get5Kmeans();

  dictionary.setVocabulary(gt_cluster, true);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_TRUE(dictionary.getIndex());
  ASSERT_EQ(dictionary.size(), dict_size);

  dictionary.serialize(file_name);
  ASSERT_TRUE(fs::exists(file_name));
  ASSERT_TRUE(fs::exists(default_flann_file));

  dictionary.setVocabulary({});
  dictionary.deserialize(file_name, true);

  ASSERT_TRUE(dictionary.getIndex());
  const auto& centroids = dictionary.getVocabulary();

  // Need to sort the output, otherwise the comparison will fail
  cv::sort(centroids, centroids, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  EXPECT_TRUE(mat_are_equal<float>(centroids, gt_cluster))
      << "gt_centroids:\n"
      << gt_cluster << "\ncomputed centroids:\n"
      << centroids;

  fs::remove(file_name);
  fs::remove(default_flann_file);
}

TEST(Dictionary, SerNoFlannDeserWithFlann) {
  const std::string file_name = "temp.bin";
  const auto& gt_cluster = get5Kmeans();

  dictionary.setVocabulary(gt_cluster);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_TRUE(!dictionary.getIndex());
  ASSERT_EQ(dictionary.size(), dict_size);

  dictionary.serialize(file_name);
  ASSERT_TRUE(fs::exists(file_name));

  dictionary.setVocabulary({});
  dictionary.deserialize(file_name, true);

  ASSERT_TRUE(dictionary.getIndex());
  const auto& centroids = dictionary.getVocabulary();

  // Need to sort the output, otherwise the comparison will fail
  cv::sort(centroids, centroids, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  EXPECT_TRUE(mat_are_equal<float>(centroids, gt_cluster))
      << "gt_centroids:\n"
      << gt_cluster << "\ncomputed centroids:\n"
      << centroids;

  fs::remove(file_name);
}

TEST(Dictionary, SerNoFlannDeserWithFlannNoFile) {
  const std::string file_name = "temp.bin";
  const std::string fake_file = "lorem_ipsum.bin";
  const auto& gt_cluster = get5Kmeans();

  dictionary.setVocabulary(gt_cluster);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_TRUE(!dictionary.getIndex());
  ASSERT_EQ(dictionary.size(), dict_size);

  dictionary.serialize(file_name);
  ASSERT_TRUE(fs::exists(file_name));
  ASSERT_TRUE(!fs::exists(fake_file));

  dictionary.setVocabulary({});
  dictionary.deserialize(file_name, true, fake_file);

  ASSERT_TRUE(dictionary.getIndex());
  const auto& centroids = dictionary.getVocabulary();

  // Need to sort the output, otherwise the comparison will fail
  cv::sort(centroids, centroids, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  EXPECT_TRUE(mat_are_equal<float>(centroids, gt_cluster))
      << "gt_centroids:\n"
      << gt_cluster << "\ncomputed centroids:\n"
      << centroids;

  fs::remove(file_name);
}