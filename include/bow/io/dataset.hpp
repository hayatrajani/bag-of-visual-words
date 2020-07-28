#ifndef BOW_IO_DATASET_HPP_
#define BOW_IO_DATASET_HPP_

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "bow/core/descriptor.hpp"
#include "bow/core/histogram.hpp"

namespace bow::io::dataset {

int datasetSize(const std::filesystem::path& path,
                const std::string& extension = "");

FeatureDescriptor extractDescriptors(const std::string& image_path,
                                     bool verbose = false);

std::vector<FeatureDescriptor> buildDescriptorDataset(
    const std::filesystem::path& dataset_path, bool save_to_disk = false,
    bool verbose = false);

std::vector<FeatureDescriptor> loadDescriptorDataset(
    const std::filesystem::path& dataset_path, bool verbose = false);

Histogram computeHistogram(const FeatureDescriptor& descriptor,
                           bool reweight = true, bool verbose = false);

std::vector<Histogram> buildHistogramDataset(
    const std::vector<FeatureDescriptor>& descriptor_dataset, int num_clusters,
    int max_iter, float epsilon = 1e-6, bool use_opencv_kmeans = true,
    bool use_flann = true, bool reweight = true, bool save_to_disk = true,
    bool verbose = false);

std::vector<Histogram> loadHistogramDataset(
    const std::filesystem::path& dataset_path, bool verbose = false);

}  // namespace bow::io::dataset

#endif