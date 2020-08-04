// @file    test_data.cpp
// @author  Ignacio Vizzo   [ivizzo@uni-bonn.de]
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]
//
// Original Copyright (c) 2020 Ignacio Vizzo, all rights reserved

#include "test_data.hpp"

#include <vector>

#include <opencv2/opencv.hpp>

#include "bow/core/descriptor.hpp"

namespace {

// init some parameters
const int rows_num = 1;
const int cols_num = 10;
const int max_features = 25;

cv::Mat generateMat(int min, int max, int steps, int repeat) {
  cv::Mat data;
  for (int i = min; i < max; i += steps) {
    for (int j = 0; j < repeat; j++) {
      cv::Mat_<float> vec(rows_num, cols_num, static_cast<float>(i));
      data.push_back(vec);
    }
  }
  return data;
}

cv::Mat generateRows(std::vector<float> values) {
  cv::Mat data;
  for (float& value : values) {
    data.push_back(cv::Mat_<float>(rows_num, cols_num, value));
  }
  return data;
}

}  // anonymous namespace

int getMaxFeatures() { return max_features; }

int getNumColumns() { return cols_num; }

cv::Mat get3Features() { return generateRows({5.0F, 15.0F, 115.0F}); }

cv::Mat get3Kmeans() { return generateRows({0.0F, 30.0F, 70.0F}); }

cv::Mat get5Kmeans() { return generateMat(0, 100, 20, 1); }

cv::Mat getAllFeatures() { return generateMat(0, 100, 20, 5); }

std::vector<bow::FeatureDescriptor> getDummyData() {
  std::vector<bow::FeatureDescriptor> data;
  for (size_t i = 0; i < 100; i += 20) {
    auto row = generateMat(i, i + 1, i + 1, 5);
    data.emplace_back(bow::FeatureDescriptor("dummy.png", row));
  }
  return data;
}