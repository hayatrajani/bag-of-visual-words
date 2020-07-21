#include "bow/core/dictionary.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/flann.hpp>

#include "bow/core/descriptor.hpp"
#include "bow/core/kmeans.hpp"

namespace fs = std::filesystem;

namespace bow {

void Dictionary::build(int max_iter, int dict_size,
                       const std::vector<FeatureDescriptor>& descriptor_dataset,
                       bool build_flann_index, bool use_opencv_kmeans,
                       double epsilon) {
  if (!descriptor_dataset.empty()) {
    codebook_ = kMeans(descriptor_dataset, dict_size, max_iter,
                       use_opencv_kmeans, epsilon);
    if (build_flann_index) {
      buildIndex();
    }
  }
}

void Dictionary::serialize(const std::string& dict_filename,
                           const std::string& flann_params_filename) const {
  std::ofstream out_file(dict_filename,
                         std::ios_base::out | std::ios_base::binary);
  if (!out_file) {
    throw std::runtime_error("Cannot open file: " + dict_filename);
  }
  int type{codebook_.type()};
  std::size_t size{sizeof(int)};
  out_file.write(reinterpret_cast<const char*>(&codebook_.rows), size);
  out_file.write(reinterpret_cast<const char*>(&codebook_.cols), size);
  out_file.write(reinterpret_cast<char*>(&type), size);
  out_file.write(reinterpret_cast<char*>(codebook_.data),
                 codebook_.elemSize() * codebook_.rows * codebook_.cols);
  if (kdtree_) {
    if (!flann_params_filename.empty()) {
      kdtree_->save(flann_params_filename);
    } else {
      fs::path flann_params_path{dict_filename};
      flann_params_path = flann_params_path.parent_path();
      flann_params_path /= "bow_index_params.flann";
      kdtree_->save(flann_params_path);
    }
  }
}

void Dictionary::deserialize(const std::string& dict_filename,
                             bool build_flann_index,
                             const std::string& flann_params_filename) {
  std::ifstream in_file(dict_filename,
                        std::ios_base::in | std::ios_base::binary);
  if (!in_file) {
    throw std::runtime_error("Cannot open file: " + dict_filename);
  }
  int rows{};
  int cols{};
  int type{};
  std::size_t size{sizeof(int)};
  in_file.read(reinterpret_cast<char*>(&rows), size);
  in_file.read(reinterpret_cast<char*>(&cols), size);
  in_file.read(reinterpret_cast<char*>(&type), size);
  codebook_ = cv::Mat::zeros(rows, cols, type);
  in_file.read(reinterpret_cast<char*>(codebook_.data),
               codebook_.elemSize() * codebook_.rows * codebook_.cols);
  if (build_flann_index) {
    if (!flann_params_filename.empty()) {
      if (fs::exists(flann_params_filename)) {
        buildIndex(cvflann::SavedIndexParams(flann_params_filename));
      } else {
        buildIndex();
      }
    } else {
      fs::path flann_params_path{dict_filename};
      flann_params_path = flann_params_path.parent_path();
      flann_params_path /= "bow_index_params.flann";
      if (fs::exists(flann_params_path)) {
        buildIndex(cvflann::SavedIndexParams(flann_params_path));
      } else {
        buildIndex();
      }
    }
  }
}

}  // namespace bow