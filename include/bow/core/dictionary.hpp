#ifndef BOW_DICTIONARY_HPP_
#define BOW_DICTIONARY_HPP_

#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/flann.hpp>

#include "bow/core/descriptor.hpp"

namespace bow {

using flannL2index = cv::flann::GenericIndex<cvflann::L2<float>>;

class Dictionary {
 private:
  cv::Mat codebook_;
  std::unique_ptr<flannL2index> kdtree_{};

  Dictionary() = default;
  ~Dictionary() = default;

  void buildIndex(const cvflann::IndexParams& index_params =
                      cvflann::AutotunedIndexParams());

 public:
  Dictionary(const Dictionary&) = delete;
  Dictionary& operator=(const Dictionary&) = delete;
  Dictionary(Dictionary&&) = delete;
  Dictionary& operator=(Dictionary&&) = delete;

  static Dictionary& getInstance() {
    static Dictionary instance;
    return instance;
  }

  void build(const std::vector<FeatureDescriptor>& descriptor_dataset,
             int dict_size, int max_iter, double epsilon = 1e-6,
             bool use_opencv_kmeans = true, bool use_flann = true);
  void setVocabulary(const cv::Mat& codebook) { codebook_ = codebook.clone(); }

  void serialize(const std::string& dict_filename,
                 const std::string& flann_params_filename = "") const;
  void deserialize(const std::string& dict_filename,
                   bool build_flann_index = false,
                   const std::string& flann_params_filename = "");

  const cv::Mat& getVocabulary() const { return codebook_; }
  flannL2index* getIndex() const { return kdtree_.get(); }

  int size() const { return codebook_.rows; }
  bool empty() const { return codebook_.empty(); }
};

}  // namespace bow

#endif