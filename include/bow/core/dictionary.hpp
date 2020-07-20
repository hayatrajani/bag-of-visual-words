#ifndef BOW_DICTIONARY_HPP_
#define BOW_DICTIONARY_HPP_

#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/flann.hpp>

#include "bow/core/kmeans.hpp"

namespace bow {

using flannL2index = cv::flann::GenericIndex<cvflann::L2<float>>;

class Dictionary {
 private:
  cv::Mat codebook_;
  std::unique_ptr<flannL2index> kdtree_{};

  Dictionary() = default;
  ~Dictionary() = default;

  void buildIndex(const cvflann::IndexParams& index_params =
                      cvflann::AutotunedIndexParams()) {
    kdtree_ = std::make_unique<flannL2index>(codebook_, index_params);
  }

 public:
  Dictionary(const Dictionary&) = delete;
  Dictionary& operator=(const Dictionary&) = delete;
  Dictionary(Dictionary&&) = delete;
  Dictionary& operator=(Dictionary&&) = delete;

  static Dictionary& getInstance() {
    static Dictionary instance;
    return instance;
  }

  void build(int max_iter, int dict_size,
             const std::vector<cv::Mat>& descriptors,
             bool build_flann_index = false, bool use_opencv_kmeans = true,
             double epsilon = 0.001);

  void serialize(const std::string& dict_filename,
                 const std::string& flann_params_filename = "") const;
  void deserialize(const std::string& dict_filename,
                   bool build_flann_index = false,
                   const std::string& flann_params_filename = "");

  const cv::Mat& getVocabulary() const { return codebook_; }
  flannL2index* getIndex() const { return kdtree_.get(); }
  /*std::optional<flannL2index> get_index() const {
    if (kdtree_) {
      return *kdtree_;
    }
    return {};
  }*/

  int size() const { return codebook_.rows; }
  bool empty() const { return codebook_.empty(); }
};

}  // namespace bow

#endif