// @file    histogram.hpp
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]

#ifndef BOW_HISTOGRAM_HPP_
#define BOW_HISTOGRAM_HPP_

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "bow/core/dictionary.hpp"

namespace bow {

class Histogram {
 private:
  static std::vector<float> idf_;
  const std::string image_path_;
  std::vector<float> data_;

 public:
  Histogram(const std::string& image_path, const std::vector<float>& data)
      : image_path_{image_path}, data_{data} {}
  Histogram(const std::string& image_path, const cv::Mat& descriptors,
            const Dictionary& dictionary);

  static Histogram readFromCSV(const std::string& filename);
  void writeToCSV(const std::string& filename) const;
  friend std::ostream& operator<<(std::ostream& out,
                                  const Histogram& histogram);

  float operator[](int index) const { return data_[index]; }
  float& operator[](int index) { return data_[index]; }
  std::vector<float> data() const { return data_; }
  std::string getImagePath() const { return image_path_; }

  std::size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  std::vector<float>::iterator begin() { return data_.begin(); }
  std::vector<float>::const_iterator begin() const { return data_.cbegin(); }
  std::vector<float>::const_iterator cbegin() const { return data_.cbegin(); }

  std::vector<float>::iterator end() { return data_.end(); }
  std::vector<float>::const_iterator end() const { return data_.cend(); }
  std::vector<float>::const_iterator cend() const { return data_.cend(); }

  static void computeIDF(const std::vector<Histogram>& histogram_dataset);
  static void saveIDF(const std::string& filename);
  static void loadIDF(const std::string& filename);
  static std::vector<float> getIDF() { return idf_; }
  static bool hasIDF() { return !idf_.empty(); }
  void reweight();

  float compare(const Histogram& other) const;
  std::vector<std::pair<std::string, float>> compare(
      const std::vector<Histogram>& histograms, int top_k = 0) const;
};

}  // namespace bow

#endif