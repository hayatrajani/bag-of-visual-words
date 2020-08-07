// @file    test_dataset.cpp
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include <opencv2/core/mat.hpp>

#include "bow/io/dataset.hpp"
#include "test_data.hpp"
#include "test_utils.hpp"

namespace fs = std::filesystem;
namespace ds = bow::io::dataset;

namespace {

const std::string temp_dir{"temp"};
const std::string dummy_image{"dummy.png"};
const std::string lenna{"test_data/lenna.png"};
const std::string dataset_parent_path{"test_data/dummy_dataset/"};
const std::string histogram_dataset_path{dataset_parent_path + "histograms/"};
const std::string descriptor_dataset_path{dataset_parent_path + "descriptors/"};
const std::string image_dataset_path{dataset_parent_path + "images/"};
const std::string invalid_image{image_dataset_path + "invalid.jpg"};

const int max_iter = 10;
const int num_clusters = 5;
auto& dictionary = bow::Dictionary::getInstance();

std::vector<float> gt_histogram_data{5, 5, 5, 5, 5};
auto dummy_descriptors = bow::FeatureDescriptor(dummy_image, getAllFeatures());
auto dummy_descriptor_dataset = getDummyData(histogram_dataset_path);

const int dataset_size{10};
const int dummy_dataset_size{5};

}  // anonymous namespace

TEST(Dataset, DatasetSize) {
  ASSERT_EQ(ds::datasetSize(image_dataset_path), 11);
}

TEST(Dataset, ExtractDescriptors) {
  auto descriptors = ds::extractDescriptors(lenna);

  ASSERT_EQ(descriptors.getImagePath(), lenna);
  ASSERT_FALSE(descriptors.empty());
}

TEST(Dataset, ExtractDescriptorsVerbose) {
  testing::internal::CaptureStdout();
  auto descriptors = ds::extractDescriptors(lenna, true);

  ASSERT_EQ(descriptors.getImagePath(), lenna);
  ASSERT_FALSE(descriptors.empty());

  std::string cout = testing::internal::GetCapturedStdout();
  ASSERT_FALSE(cout.empty());
  ASSERT_THAT(cout, testing::HasSubstr("Done"));
}

TEST(Dataset, ExtractDescriptorsFakeFile) {
  ASSERT_THROW(ds::extractDescriptors(dummy_image), std::runtime_error);
}

TEST(Dataset, ExtractDescriptorsInvalidImage) {
  ASSERT_THROW(ds::extractDescriptors(invalid_image), std::runtime_error);
}

TEST(Dataset, BuildDescriptorDataset) {
  auto descriptor_dataset = ds::buildDescriptorDataset(image_dataset_path);

  ASSERT_FALSE(descriptor_dataset.empty());
  ASSERT_EQ(descriptor_dataset.size(), dataset_size);
}

TEST(Dataset, BuildDescriptorDatasetEmpty) {
  fs::create_directory(temp_dir);
  EXPECT_THROW(ds::buildDescriptorDataset(temp_dir, true), std::runtime_error);
  fs::remove_all(temp_dir);
}

TEST(Dataset, BuildDescriptorDatasetToDiskVerbose) {
  testing::internal::CaptureStdout();

  auto descriptor_dataset =
      ds::buildDescriptorDataset(image_dataset_path, true, true);

  ASSERT_FALSE(descriptor_dataset.empty());
  ASSERT_EQ(descriptor_dataset.size(), dataset_size);

  ASSERT_TRUE(fs::exists(descriptor_dataset_path));
  ASSERT_FALSE(fs::is_empty(descriptor_dataset_path));
  ASSERT_EQ(ds::datasetSize(descriptor_dataset_path, ".bin"), dataset_size);

  std::string cout = testing::internal::GetCapturedStdout();
  ASSERT_FALSE(cout.empty());
  ASSERT_THAT(cout, testing::HasSubstr("Done"));
}

TEST(Dataset, LoadDescriptorDataset) {
  auto descriptor_dataset = ds::loadDescriptorDataset(descriptor_dataset_path);

  ASSERT_FALSE(descriptor_dataset.empty());
  ASSERT_EQ(descriptor_dataset.size(), dataset_size);
}

TEST(Dataset, LoadDescriptorDatasetVerbose) {
  testing::internal::CaptureStdout();
  auto descriptor_dataset =
      ds::loadDescriptorDataset(descriptor_dataset_path, true);

  ASSERT_FALSE(descriptor_dataset.empty());
  ASSERT_EQ(descriptor_dataset.size(), dataset_size);

  std::string cout = testing::internal::GetCapturedStdout();
  ASSERT_FALSE(cout.empty());
  ASSERT_THAT(cout, testing::HasSubstr("Done"));

  fs::remove_all(descriptor_dataset_path);
}

TEST(Dataset, LoadDescriptorDatasetEmpty) {
  fs::create_directory(temp_dir);
  EXPECT_THROW(ds::loadDescriptorDataset(temp_dir), std::runtime_error);
  fs::remove_all(temp_dir);
}

TEST(Dataset, ComputeHistogramNoDict) {
  dictionary.setVocabulary({});
  ASSERT_THROW(ds::computeHistogram(dummy_descriptors), std::runtime_error);
}

TEST(Dataset, ComputeHistogram) {
  dictionary.setVocabulary(get5Kmeans());
  auto histogram = ds::computeHistogram(dummy_descriptors);

  ASSERT_FALSE(histogram.empty());
  ASSERT_EQ(histogram.size(), num_clusters);
}

TEST(Dataset, ComputeHistogramReweightVerbose) {
  testing::internal::CaptureStdout();

  dictionary.setVocabulary(get5Kmeans());
  auto histogram = ds::computeHistogram(dummy_descriptors, true, true);

  ASSERT_FALSE(histogram.empty());
  ASSERT_EQ(histogram.size(), num_clusters);

  std::string cout = testing::internal::GetCapturedStdout();
  ASSERT_FALSE(cout.empty());
  ASSERT_THAT(cout, testing::HasSubstr("Done"));
}

TEST(Dataset, ComputeHistogramReweightNoIDF) {
  testing::internal::CaptureStderr();

  bow::Histogram::computeIDF({});
  dictionary.setVocabulary(get5Kmeans());
  auto histogram = ds::computeHistogram(dummy_descriptors, true);

  ASSERT_FALSE(histogram.empty());
  ASSERT_EQ(histogram.size(), num_clusters);

  std::string cerr = testing::internal::GetCapturedStderr();
  ASSERT_FALSE(cerr.empty());
  ASSERT_THAT(cerr, testing::HasSubstr("[ERROR]"));
}

TEST(Dataset, BuildHistogramDataset) {
  auto histogram_dataset = ds::buildHistogramDataset(dummy_descriptor_dataset,
                                                     num_clusters, max_iter);
  ASSERT_FALSE(histogram_dataset.empty());
  ASSERT_EQ(histogram_dataset.size(), dummy_dataset_size);
}

TEST(Dataset, BuildHistogramDatasetToDiskVerbose) {
  testing::internal::CaptureStdout();

  auto histogram_dataset = ds::buildHistogramDataset(
      dummy_descriptor_dataset, num_clusters, max_iter, 1e-6, false, false,
      false, true, true);
  ASSERT_FALSE(histogram_dataset.empty());
  ASSERT_EQ(histogram_dataset.size(), dummy_dataset_size);
  std::cout << "DONE!!\n";

  ASSERT_TRUE(fs::exists(histogram_dataset_path));
  ASSERT_FALSE(fs::is_empty(histogram_dataset_path));
  ASSERT_EQ(ds::datasetSize(histogram_dataset_path, ".csv"),
            dummy_dataset_size);

  std::string cout = testing::internal::GetCapturedStdout();
  ASSERT_FALSE(cout.empty());
  ASSERT_THAT(cout, testing::HasSubstr("Done"));
}

TEST(Dataset, BuildHistogramDatasetReweightToDiskVerbose) {
  testing::internal::CaptureStdout();

  auto histogram_dataset =
      ds::buildHistogramDataset(dummy_descriptor_dataset, num_clusters,
                                max_iter, 1e-6, false, false, true, true, true);
  ASSERT_FALSE(histogram_dataset.empty());
  ASSERT_EQ(histogram_dataset.size(), dummy_dataset_size);

  std::string cout = testing::internal::GetCapturedStdout();
  ASSERT_FALSE(cout.empty());
  ASSERT_THAT(cout, testing::HasSubstr("Done"));
}

TEST(Dataset, LoadHistogramDataset) {
  auto histogram_dataset = ds::loadHistogramDataset(histogram_dataset_path);

  ASSERT_FALSE(histogram_dataset.empty());
  ASSERT_EQ(histogram_dataset.size(), dummy_dataset_size);
}

TEST(Dataset, LoadHistogramDatasetVerbose) {
  testing::internal::CaptureStdout();
  auto histogram_dataset =
      ds::loadHistogramDataset(histogram_dataset_path, true);

  ASSERT_FALSE(histogram_dataset.empty());
  ASSERT_EQ(histogram_dataset.size(), dummy_dataset_size);

  std::string cout = testing::internal::GetCapturedStdout();
  ASSERT_FALSE(cout.empty());
  ASSERT_THAT(cout, testing::HasSubstr("Done"));

  fs::remove_all(histogram_dataset_path);
}

TEST(Dataset, LoadHistogramDatasetEmpty) {
  fs::create_directory(temp_dir);
  EXPECT_THROW(ds::loadHistogramDataset(temp_dir), std::runtime_error);
  fs::remove_all(temp_dir);
}