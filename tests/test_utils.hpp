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
bool inline mat_are_equal(const cv::Mat& m1, const cv::Mat& m2,
                          float epsilon = 1e-4) {
  return std::equal(m1.begin<T>(), m1.end<T>(), m2.begin<T>(),
                    [&epsilon](T a, T b) { return abs(a - b) <= epsilon; });
}

template <typename T>
bool inline vec_are_equal(const std::vector<T>& v1, const std::vector<T>& v2,
                          float epsilon = 1e-4) {
  return std::equal(v1.begin(), v1.end(), v2.begin(),
                    [&epsilon](T a, T b) { return abs(a - b) <= epsilon; });
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  if (!v.empty()) {
    out << v.front();
    for (auto i{v.begin() + 1}; i != v.end(); ++i) {
      out << ", " << *i;
    }
  }
  return out;
}

// trim from start (in place)
inline void ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
            return std::isspace(ch) == 0;
          }));
}

// trim from end (in place)
inline void rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return std::isspace(ch) == 0; })
              .base(),
          s.end());
}

// trim from both ends (in place)
inline std::string trim(const std::string& s_input) {
  std::string s{s_input};
  ltrim(s);
  rtrim(s);
  return s;
}

#endif