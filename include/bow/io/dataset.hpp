#ifndef BOW_IO_DATASET_HPP_
#define BOW_IO_DATASET_HPP_

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "bow/core/descriptor.hpp"
#include "bow/core/histogram.hpp"

namespace fs = std::filesystem;

namespace bow::io::dataset {

int datasetSize(const fs::path& path, const std::string& extension = "");

FeatureDescriptor extractDescriptors(const std::string& image_path,
                                     bool verbose = false);

std::vector<FeatureDescriptor> buildDescriptorDataset(
    const fs::path& dataset_path, bool save_to_disk = false,
    bool verbose = false);

std::vector<FeatureDescriptor> loadDescriptorDataset(
    const fs::path& dataset_path, bool verbose = false);

Histogram computeHistogram(const FeatureDescriptor& descriptor,
                           bool reweight = true, bool verbose = false);

std::vector<Histogram> buildHistogramDataset(
    const std::vector<FeatureDescriptor>& descriptor_dataset, int max_iter,
    int num_clusters, bool use_flann, bool use_opencv_kmeans, float epsilon,
    bool reweight = true, bool save_to_disk = true, bool verbose = false);

std::vector<Histogram> loadHistogramDataset(const fs::path& dataset_path,
                                            bool verbose = false);

}  // namespace bow::io::dataset

#endif