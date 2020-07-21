#include "bow/core/histogram.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/flann.hpp>

#include "bow/core/dictionary.hpp"

namespace bow {

std::vector<float> Histogram::idf_{};

int nearestNeighbour(const cv::Mat& descriptor, const cv::Mat& codebook,
                     flannL2index* kdtree = nullptr) {
  if (kdtree) {
    int k{1};
    std::vector<int> indices(k);
    std::vector<float> distances(k);
    kdtree->knnSearch(descriptor, indices, distances, k,
                      cvflann::SearchParams());
    return indices[0];
  }
  int nearest_cluster_idx{};
  float min_dist{std::numeric_limits<float>::max()};
  for (int r = 0; r < codebook.rows; ++r) {
    auto diff = codebook.row(r) - descriptor;
    float dist = static_cast<float>(std::sqrt(diff.dot(diff)));
    if (dist < min_dist) {
      min_dist = dist;
      nearest_cluster_idx = r;
    }
  }
  return nearest_cluster_idx;
}

Histogram::Histogram(const std::string& image_path, const cv::Mat& descriptors,
                     const Dictionary& dictionary)
    : image_path_{image_path} {
  if (!descriptors.empty()) {
    if (!dictionary.empty()) {
      const cv::Mat& codebook = dictionary.getVocabulary();
      flannL2index* kdtree = dictionary.getIndex();
      data_.resize(dictionary.size());
      for (int r = 0; r < descriptors.rows; ++r) {
        data_[nearestNeighbour(descriptors.row(r), codebook, kdtree)]++;
      }
    } else {
      throw std::runtime_error("Empty codebook!");
    }
  }
}

Histogram Histogram::readFromCSV(const std::string& filename) {
  std::ifstream in(filename, std::ios_base::in);
  if (!in) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  std::string image_path;
  std::string line;
  // read image_path
  std::getline(in, image_path);
  image_path.erase(0, 2);
  // ignore header
  std::getline(in, line);
  // read bin count
  std::getline(in, line, ',');
  int bin_count{std::stoi(line)};
  // read data
  std::vector<float> data;
  if (bin_count != 0) {
    data.reserve(bin_count);
    while (std::getline(in, line, ',')) {
      data.emplace_back(std::stoi(line));
    }
  }
  return {image_path, data};
}

void Histogram::writeToCSV(const std::string& filename) const {
  std::ofstream out{filename};
  if (!out) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  out << "# " << image_path_ << '\n';
  out << "# Format: number of bins followed by bin frequencies\n";
  out << data_.size() << ", ";
  out << *this << '\n';
}

std::ostream& operator<<(std::ostream& out, const Histogram& histogram) {
  if (!histogram.data_.empty()) {
    out << histogram.data_.front();
    for (auto i{histogram.data_.begin() + 1}; i != histogram.data_.end(); ++i) {
      out << ", " << *i;
    }
  }
  return out;
}

void Histogram::computeIDF(const std::vector<Histogram>& histogram_dataset) {
  float dataset_size = histogram_dataset.size();
  int codebook_size = histogram_dataset[0].size();
  if (!idf_.empty()) {
    idf_.clear();
  }
  idf_.resize(codebook_size);
  for (const auto& histogram : histogram_dataset) {
    if (!histogram.empty()) {
      for (int c = 0; c < codebook_size; ++c) {
        if (histogram[c] > 0) {
          idf_[c]++;
        }
      }
    }
  }
  for (int c = 0; c < codebook_size; ++c) {
    idf_[c] = std::log(dataset_size / idf_[c]);
  }
}

void Histogram::saveIDF(const std::string& filename) {
  std::ofstream out_file(filename, std::ios_base::out | std::ios_base::binary);
  if (!out_file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  int data_size = idf_.size();
  out_file.write(reinterpret_cast<char*>(&data_size), sizeof(data_size));
  out_file.write(reinterpret_cast<char*>(&idf_[0]), data_size * sizeof(float));
}

void Histogram::loadIDF(const std::string& filename) {
  std::ifstream in_file(filename, std::ios_base::in | std::ios_base::binary);
  if (!in_file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  int data_size{};
  in_file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
  idf_.resize(data_size);
  in_file.read(reinterpret_cast<char*>(&idf_[0]), data_size * sizeof(float));
}

void Histogram::reweight() {
  if (!(data_.empty() || idf_.empty())) {
    int codebook_size = idf_.size();
    float num_words = std::accumulate(data_.begin(), data_.end(), 0);
    for (int c = 0; c < codebook_size; ++c) {
      data_[c] *= idf_[c] / num_words;
    }
  }
}

float Histogram::compare(const Histogram& other) const {
  if (data_.empty() || other.empty()) {
    return 1.0F;
  }
  if (data_.empty() && other.empty()) {
    return 0.0F;
  }
  return 1.0F -
         static_cast<float>(
             std::inner_product(data_.begin(), data_.end(), other.begin(), 0) /
             (std::sqrt(std::inner_product(data_.begin(), data_.end(),
                                           data_.begin(), 0)) *
              std::sqrt(std::inner_product(other.begin(), other.end(),
                                           other.begin(), 0))));
}

std::vector<std::pair<std::string, float>> Histogram::compare(
    const std::vector<Histogram>& histograms, int top_k) const {
  std::vector<std::pair<std::string, float>> similarities;
  int size = histograms.size();
  similarities.reserve(size);
  for (const Histogram& histogram : histograms) {
    similarities.emplace_back(
        std::make_pair(histogram.getImagePath(), compare(histogram)));
  }
  std::sort(
      similarities.begin(), similarities.end(),
      [](const auto& p1, const auto& p2) { return p1.second < p2.second; });
  if (top_k == 0 || abs(top_k) >= size) {
    return similarities;
  }
  if (top_k > 0) {
    return std::vector<std::pair<std::string, float>>(
        similarities.begin(), similarities.begin() + top_k);
  }
  return std::vector<std::pair<std::string, float>>(
      similarities.rbegin(), similarities.rbegin() - top_k);
}

}  // namespace bow