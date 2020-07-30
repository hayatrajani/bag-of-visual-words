// @file    test_data.cpp
// @author  Ignacio Vizzo (creator) [ivizzo@uni-bonn.de]
// @author  Hayat Rajani (editor)   [hayat.rajani@uni-bonn.de]
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

}  // namespace

int getMaxFeatures() { return max_features; }
int getNumColumns() { return cols_num; }

std::vector<bow::FeatureDescriptor>& getDummyData() {
  static std::vector<bow::FeatureDescriptor> data;
  for (int i = 0; i < 100; i += 20) {
    for (size_t j = 0; j < 5; j++) {
      data.emplace_back(bow::FeatureDescriptor(
          "test.png", cv::Mat_<float>(rows_num, cols_num, i)));
    }
  }
  return data;
}

cv::Mat getAllFeatures() {
  cv::Mat data;
  for (int i = 0; i < 100; i += 20) {
    for (size_t j = 0; j < 5; j++) {
      data.push_back(cv::Mat_<float>(rows_num, cols_num, i));
    }
  }
  return data;
}

cv::Mat get3Features() {
  cv::Mat data;
  data.push_back(cv::Mat_<float>(rows_num, cols_num, 5.0F));
  data.push_back(cv::Mat_<float>(rows_num, cols_num, 15.0F));
  data.push_back(cv::Mat_<float>(rows_num, cols_num, 115.0F));
  return data;
}

cv::Mat get5Kmeans() {
  cv::Mat data;
  for (int i = 0; i < 100; i += 20) {
    cv::Mat_<float> vec(rows_num, cols_num, i);
    data.push_back(vec);
  }
  return data;
}

cv::Mat get3Kmeans() {
  cv::Mat data;
  data.push_back(cv::Mat_<float>(rows_num, cols_num, 0.0F));
  data.push_back(cv::Mat_<float>(rows_num, cols_num, 30.0F));
  data.push_back(cv::Mat_<float>(rows_num, cols_num, 70.0F));
  return data;
}

cv::Mat get2Kmeans() {
  cv::Mat data;
  data.push_back(cv::Mat_<float>(rows_num, cols_num, 20.000002F));
  data.push_back(cv::Mat_<float>(rows_num, cols_num, 70.0F));
  return data;
}