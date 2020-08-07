// @file    test_histogram.cpp
// @author  Ignacio Vizzo   [ivizzo@uni-bonn.de]
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]
//
// Original Copyright (c) 2020 Ignacio Vizzo, all rights reserved

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <numeric>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "bow/core/dictionary.hpp"
#include "bow/core/histogram.hpp"
#include "test_data.hpp"
#include "test_utils.hpp"

namespace fs = std::filesystem;

namespace {

const std::string dummy_image_file{"dummy.png"};
auto& dictionary = bow::Dictionary::getInstance();
std::vector<float> gt_histogram_data{5, 5, 5, 5, 5};

// Test data source:
// https://github.com/ovysotska/in_simple_english/blob/master/bag_of_visual_words.ipynb
size_t dataset_size{4};
std::vector<bow::Histogram> histogram_dataset{
    bow::Histogram(dummy_image_file, {5, 2, 1, 0, 0}),
    bow::Histogram(dummy_image_file, {4, 0, 1, 1, 0}),
    bow::Histogram(dummy_image_file, {3, 1, 1, 0, 2}),
    bow::Histogram(dummy_image_file, {1, 2, 1, 0, 0})};
std::vector<bow::Histogram> gt_reweighted_dataset{
    bow::Histogram(dummy_image_file, {0, 0.07, 0, 0, 0}),
    bow::Histogram(dummy_image_file, {0, 0, 0, 0.23, 0}),
    bow::Histogram(dummy_image_file, {0, 0.04, 0, 0, 0.4}),
    bow::Histogram(dummy_image_file, {0, 0.14, 0, 0, 0})};
std::vector<std::vector<float>> gt_similarities{
    {0, 0, 0.9, 1}, {0, 1, 1, 1}, {0, 0.9, 0.9, 1}, {0, 0, 0.9, 1}};
std::vector<float> gt_idf{0, 0.2876, 0, 1.3862, 1.3862};

}  // anonymous namespace

TEST(Histogram, EmptyDictionary) {
  dictionary.setVocabulary({});
  ASSERT_THROW(bow::Histogram("", getAllFeatures(), dictionary),
               std::runtime_error);
}

TEST(Histogram, EmptyDescriptors) {
  dictionary.setVocabulary(get5Kmeans());
  cv::Mat empty;
  auto histogram = bow::Histogram("", empty, dictionary);
  ASSERT_TRUE(histogram.empty());
  ASSERT_EQ(histogram.size(), 0);
}

TEST(Histogram, CreateFromDictionary) {
  dictionary.setVocabulary(get5Kmeans());
  auto histogram =
      bow::Histogram(dummy_image_file, getAllFeatures(), dictionary);
  ASSERT_FALSE(histogram.getImagePath().empty());
  ASSERT_FALSE(histogram.empty());
  ASSERT_GT(histogram.size(), 0);
  ASSERT_EQ(histogram.size(), dictionary.size())
      << "The numbers of bins in the histogram must match the number of words "
         "in the dictionary";
  ASSERT_EQ(gt_histogram_data, histogram.data());
}

TEST(Histogram, NonTrivialExample) {
  dictionary.setVocabulary(get5Kmeans());
  const auto& descriptors = get3Features();
  auto histogram = bow::Histogram(dummy_image_file, descriptors, dictionary);
  ASSERT_FALSE(histogram.empty());
  ASSERT_GT(histogram.size(), 0);
  ASSERT_EQ(histogram.size(), dictionary.size())
      << "The numbers of bins in the histogram must match the number of words "
         "in the dictionary";

  //  05,  05,  05,  05,  05,  05,  05,  05,  05 <--- closest cluster == 0 (00)
  //  15,  15,  15,  15,  15,  15,  15,  15,  15 <--- closest cluster == 1 (20)
  // 115, 115, 115, 115, 115, 115, 115, 115, 115 <--- closest cluster == 4 (80)
  // Therefore, the histogram must look like: bins [1, 1, 0, 0, 1]
  std::vector<float> res{1, 1, 0, 0, 1};
  ASSERT_EQ(res, histogram.data());
}

TEST(Histogram, PrintToStdout) {
  dictionary.setVocabulary(get5Kmeans());
  auto histogram =
      bow::Histogram(dummy_image_file, getAllFeatures(), dictionary);
  ASSERT_FALSE(histogram.empty());
  ASSERT_GT(histogram.size(), 0);

  testing::internal::CaptureStdout();
  std::cout << histogram;
  std::string cout = testing::internal::GetCapturedStdout();
  ASSERT_THAT(cout, testing::HasSubstr("5, 5, 5, 5, 5"));
}

TEST(Histogram, ReadWriteFakeFile) {
  auto histogram = bow::Histogram(dummy_image_file, gt_histogram_data);
  EXPECT_THROW(histogram.writeToCSV({}), std::runtime_error);
  EXPECT_THROW(bow::Histogram::readFromCSV({}), std::runtime_error);
}

TEST(Histogram, ReadWriteEmptyData) {
  auto histogram = bow::Histogram("", {});

  const std::string file_name{"temp.csv"};
  histogram.writeToCSV(file_name);
  ASSERT_TRUE(fs::exists(file_name));

  auto csv_histogram = bow::Histogram::readFromCSV(file_name);
  EXPECT_TRUE(csv_histogram.empty());
  EXPECT_TRUE(csv_histogram.getImagePath().empty());

  fs::remove(file_name);
}

TEST(Histogram, ReadWriteCSV) {
  auto histogram = bow::Histogram(dummy_image_file, gt_histogram_data);
  ASSERT_FALSE(histogram.empty());
  ASSERT_GT(histogram.size(), 0);

  const std::string file_name{"temp.csv"};
  histogram.writeToCSV(file_name);
  ASSERT_TRUE(fs::exists(file_name));

  auto csv_histogram = bow::Histogram::readFromCSV(file_name);
  EXPECT_EQ(histogram.data(), csv_histogram.data());
  EXPECT_EQ(histogram.getImagePath(), csv_histogram.getImagePath());

  fs::remove(file_name);
}

TEST(Histogram, Iterators) {
  dictionary.setVocabulary(get5Kmeans());
  const auto& descriptors = getAllFeatures();
  auto histogram = bow::Histogram(dummy_image_file, descriptors, dictionary);
  ASSERT_FALSE(histogram.empty());
  ASSERT_GT(histogram.size(), 0);

  for (const auto& bin : histogram) {
    // Do nothing,if this works, then the iterators also work!
    (void)bin;
  }

  // Accumulate all bins in histogram
  auto sum = std::accumulate(histogram.begin(), histogram.end(), 0.0F);
  ASSERT_EQ(sum, descriptors.rows)
      << "The number of the input descriptors must match the sum of all the "
         "bins in the histogram, check the homework diagram";
}

TEST(Histogram, ConstIterators) {
  dictionary.setVocabulary(get5Kmeans());
  const auto& descriptors = getAllFeatures();
  const auto histogram =
      bow::Histogram(dummy_image_file, descriptors, dictionary);
  ASSERT_FALSE(histogram.empty());
  ASSERT_GT(histogram.size(), 0);

  for (const auto& bin : histogram) {
    // Do nothing,if this works, then the iterators also work!
    (void)bin;
  }

  std::string gt_hist{"5, 5, 5, 5, 5"};
  std::stringstream ss;
  ss << "bins = [";
  std::copy(histogram.cbegin(), histogram.cend(),
            std::ostream_iterator<int>(ss, ", "));
  ss << "\b\b]";
  testing::internal::CaptureStdout();
  std::cout << ss.str();
  std::string cout = testing::internal::GetCapturedStdout();
  ASSERT_THAT(cout, testing::HasSubstr(gt_hist)) << ss.str();
}

TEST(Histogram, AccessOperators) {
  dictionary.setVocabulary(get5Kmeans());
  const auto& descriptors = getAllFeatures();
  auto histogram = bow::Histogram(dummy_image_file, descriptors, dictionary);
  ASSERT_FALSE(histogram.empty());
  ASSERT_GT(histogram.size(), 0);

  for (size_t i = 0; i < histogram.size(); i++) {
    histogram[i]++;
    gt_histogram_data[i]++;
  }
  ASSERT_EQ(gt_histogram_data, histogram.data());

  const auto& const_hist = histogram;
  for (size_t i = 0; i < histogram.size(); i++) {
    (void)const_hist[i];
  }
}

TEST(Histogram, ComputeIDF) {
  bow::Histogram::computeIDF(histogram_dataset);
  ASSERT_TRUE(bow::Histogram::hasIDF());
  ASSERT_TRUE(vec_are_equal(gt_idf, bow::Histogram::getIDF()));
}

TEST(Histogram, ComputeIDF_EmptyDataset) {
  bow::Histogram::computeIDF({});
  ASSERT_FALSE(bow::Histogram::hasIDF());
  ASSERT_TRUE(bow::Histogram::getIDF().empty());
}

TEST(Histogram, SaveLoadIDF) {
  bow::Histogram::computeIDF(histogram_dataset);
  ASSERT_TRUE(bow::Histogram::hasIDF());
  auto idf = bow::Histogram::getIDF();

  const std::string file_name{"temp.bin"};
  bow::Histogram::saveIDF(file_name);
  ASSERT_TRUE(fs::exists(file_name));

  bow::Histogram::computeIDF({});
  ASSERT_FALSE(bow::Histogram::hasIDF());

  bow::Histogram::loadIDF(file_name);
  ASSERT_TRUE(bow::Histogram::hasIDF());
  ASSERT_TRUE(vec_are_equal(idf, bow::Histogram::getIDF()));

  fs::remove(file_name);
}

TEST(Histogram, SaveLoadIDF_EmptyData) {
  bow::Histogram::computeIDF({});
  ASSERT_FALSE(bow::Histogram::hasIDF());

  const std::string file_name{"temp.bin"};
  ASSERT_NO_THROW(bow::Histogram::saveIDF(file_name));
  ASSERT_TRUE(fs::exists(file_name));

  ASSERT_NO_THROW(bow::Histogram::loadIDF(file_name));
  ASSERT_FALSE(bow::Histogram::hasIDF());

  fs::remove(file_name);
}

TEST(Histogram, SaveLoadIDF_FakeFile) {
  ASSERT_THROW(bow::Histogram::saveIDF(""), std::runtime_error);
  ASSERT_THROW(bow::Histogram::loadIDF(""), std::runtime_error);
}

TEST(Histogram, Reweight) {
  bow::Histogram::computeIDF(histogram_dataset);
  ASSERT_TRUE(bow::Histogram::hasIDF());
  for (size_t i = 0; i < dataset_size; ++i) {
    histogram_dataset[i].reweight();
    EXPECT_TRUE(vec_are_equal(histogram_dataset[i].data(),
                              gt_reweighted_dataset[i].data(), 1e-2));
  }
}

TEST(Histogram, CompareWithEmpty) {
  auto hist = bow::Histogram(dummy_image_file, gt_histogram_data);
  ASSERT_FALSE(hist.empty());
  ASSERT_GT(hist.size(), 0);

  auto empty_hist = bow::Histogram("", {});
  ASSERT_TRUE(empty_hist.empty());

  ASSERT_EQ(hist.compare(empty_hist), 1);
}

TEST(Histogram, CompareBothEmpty) {
  auto h1 = bow::Histogram("", {});
  ASSERT_TRUE(h1.empty());

  auto h2 = bow::Histogram("", {});
  ASSERT_TRUE(h2.empty());

  ASSERT_EQ(h1.compare(h2), 0);
}

TEST(Histogram, CompareList) {
  for (size_t i = 0; i < dataset_size; ++i) {
    auto results = gt_reweighted_dataset[i].compare(gt_reweighted_dataset);
    std::vector<float> similarities{};
    similarities.reserve(results.size());
    for (auto& result : results) {
      similarities.emplace_back(result.second);
    }
    EXPECT_TRUE(vec_are_equal(similarities, gt_similarities[i], 1e-2));
  }
}

TEST(Histogram, CompareTopK) {
  auto similarities =
      gt_reweighted_dataset[0].compare(gt_reweighted_dataset, 2);
  for (size_t i = 0; i < similarities.size(); ++i) {
    EXPECT_FLOAT_EQ(similarities[i].second, gt_similarities[0][i]);
  }
}

TEST(Histogram, CompareBotK) {
  auto similarities =
      gt_reweighted_dataset[1].compare(gt_reweighted_dataset, -2);
  for (size_t i = 0; i < similarities.size(); ++i) {
    EXPECT_FLOAT_EQ(similarities[i].second,
                    gt_similarities[1][dataset_size - (i + 1)]);
  }
}