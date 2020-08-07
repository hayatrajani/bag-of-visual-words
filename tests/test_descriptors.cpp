// @file    test_descriptors.cpp
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]

#include <gtest/gtest.h>

#include <filesystem>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "bow/core/descriptor.hpp"
#include "test_utils.hpp"

namespace fs = std::filesystem;

namespace {

const std::string dummy_image{"dummy.png"};
const std::string lenna{"test_data/lenna.png"};
const std::string featureless_image{"test_data/featureless.png"};

cv::Mat computeSifts(const std::string& file_name) {
  const cv::Mat image = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  auto detector = cv::xfeatures2d::SIFT::create();
  detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
  return descriptors;
}

}  // anonymous namespace

TEST(Descriptor, BuildFromData) {
  auto data = computeSifts(lenna);
  auto descriptor = bow::FeatureDescriptor(lenna, data);
  ASSERT_FALSE(descriptor.getImagePath().empty());
  ASSERT_FALSE(descriptor.empty());
  ASSERT_EQ(descriptor.size(), data.rows);
}

TEST(Descriptor, BuildFromImage) {
  auto gt_data = computeSifts(lenna);
  auto descriptor = bow::FeatureDescriptor(lenna);

  ASSERT_FALSE(descriptor.getImagePath().empty());
  EXPECT_EQ(descriptor.getImagePath(), lenna);

  ASSERT_FALSE(descriptor.empty());
  ASSERT_EQ(descriptor.size(), gt_data.rows);
  EXPECT_TRUE(mat_are_equal<float>(descriptor.getDescriptors(), gt_data));
}

TEST(Descriptor, BuildFromFeaturelessImage) {
  auto descriptor = bow::FeatureDescriptor(featureless_image);

  ASSERT_FALSE(descriptor.getImagePath().empty());
  EXPECT_EQ(descriptor.getImagePath(), featureless_image);

  ASSERT_TRUE(descriptor.empty());
}

TEST(Descriptor, Serialization) {
  const std::string file_name = "temp.bin";

  auto descriptor = bow::FeatureDescriptor(lenna);
  ASSERT_FALSE(descriptor.getImagePath().empty());
  ASSERT_FALSE(descriptor.empty());

  descriptor.serialize(file_name);
  ASSERT_TRUE(fs::exists(file_name));

  auto bin_descriptors = bow::FeatureDescriptor::deserialize(file_name);
  ASSERT_FALSE(bin_descriptors.getImagePath().empty());
  ASSERT_FALSE(bin_descriptors.empty());

  EXPECT_EQ(descriptor.getImagePath(), bin_descriptors.getImagePath());
  EXPECT_TRUE(mat_are_equal<float>(descriptor.getDescriptors(),
                                   bin_descriptors.getDescriptors()));

  fs::remove(file_name);
}

TEST(Descriptor, SerializationEmptyData) {
  const std::string file_name = "temp.bin";

  auto descriptor = bow::FeatureDescriptor(featureless_image);
  ASSERT_FALSE(descriptor.getImagePath().empty());
  ASSERT_TRUE(descriptor.empty());

  descriptor.serialize(file_name);
  ASSERT_TRUE(fs::exists(file_name));

  auto bin_descriptors = bow::FeatureDescriptor::deserialize(file_name);
  ASSERT_FALSE(bin_descriptors.getImagePath().empty());
  ASSERT_TRUE(bin_descriptors.empty());

  EXPECT_EQ(descriptor.getImagePath(), bin_descriptors.getImagePath());
  EXPECT_TRUE(mat_are_equal<float>(descriptor.getDescriptors(),
                                   bin_descriptors.getDescriptors()));

  fs::remove(file_name);
}

TEST(Descriptor, SerializationFakeFile) {
  auto descriptor = bow::FeatureDescriptor(featureless_image);
  EXPECT_THROW(descriptor.serialize({}), std::runtime_error);
  EXPECT_THROW(bow::FeatureDescriptor::deserialize({}), std::runtime_error);
}