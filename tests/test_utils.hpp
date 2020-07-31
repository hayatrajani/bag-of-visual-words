// @file    test_utils.hpp
// @author  Ignacio Vizzo   [ivizzo@uni-bonn.de]
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]
//
// Original Copyright (c) 2020 Ignacio Vizzo, all rights reserved

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_

#include <algorithm>
#include <vector>

#include <opencv2/opencv.hpp>

template <typename T>
bool inline mat_are_equal(const cv::Mat& m1, const cv::Mat& m2) {
  return std::equal(m1.begin<T>(), m1.end<T>(), m2.begin<T>(),
                    [](T a, T b) { return abs(a - b) <= 1e-4; });
}

template <typename T>
bool inline vec_are_equal(const std::vector<T>& v1, const std::vector<T>& v2) {
  return std::equal(v1.begin(), v1.end(), v2.begin(),
                    [](T a, T b) { return abs(a - b) <= 1e-4; });
}

#endif