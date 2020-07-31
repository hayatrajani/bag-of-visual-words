// @file    dataset.hpp
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]

#ifndef BOW_IO_DATASET_HPP_
#define BOW_IO_DATASET_HPP_

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "bow/core/descriptor.hpp"
#include "bow/core/histogram.hpp"

namespace bow::io::dataset {

/**
 * @brief This function counts the number of files in the specified directory.
 * The extension parameter optionally enables counting files of the given type.
 *
 * @param dir_path  The path to the directory in question.
 * @param extension The type of files to look for; optional.
 *
 * @return The number of files in the directory.
 */
int datasetSize(const std::filesystem::path& path,
                const std::string& extension = "");

/**
 * @brief A convenience function to extract SIFT feature descriptors from the
 * given image.
 *
 * @param image_path The path to the (png) image file.
 * @param verbose    Set this to true to enable verbose outputs; default false.
 *
 * @return An instance of type bow::FeatureDescriptor representing the SIFT
 * feature descriptors.
 */
FeatureDescriptor extractDescriptors(const std::string& image_path,
                                     bool verbose = false);

/**
 * @brief A convenience function to extract SIFT feature descriptors from the
 * images in a dataset. The extracted descriptors can optionally be stored in a
 * directory called "descriptors" under the dataset path. Note that any
 * pre-existing descriptors will be overwritten, if present.
 *
 * @param dataset_path The path to the (png) image dataset.
 * @param save_to_disk Set this to true to store the extracted feature
 *                     descriptors; default false.
 * @param verbose      Set this to true to enable verbose outputs; default
 *                     false.
 *
 * @return A vector of instances of type bow::FeatureDescriptor representing the
 * SIFT feature descriptors of the images in the dataset.
 */
std::vector<FeatureDescriptor> buildDescriptorDataset(
    const std::filesystem::path& dataset_path, bool save_to_disk = false,
    bool verbose = false);

/**
 * @brief A convenience function to read in a previously computed feature
 * descriptor dataset and load the data into a vector.
 *
 * @param dataset_path The path to the descriptor dataset.
 * @param verbose      Set this to true to enable verbose outputs; default
 *                     false.
 *
 * @return A vector of instances of type bow::FeatureDescriptor representing the
 * SIFT feature descriptors in the dataset.
 */
std::vector<FeatureDescriptor> loadDescriptorDataset(
    const std::filesystem::path& dataset_path, bool verbose = false);

/**
 * @brief A convenience function to compute a histogram from an image's
 * feature descriptors. This function should only be called after a call to
 * buildHistogramDataset(). This function internally queries bow::Dictionary to
 * fetch the codebook vector and throws an error if it does not already exist.
 * Further, it assumes that the inverse document frequencies of the dataset have
 * already been computed if the reweight parameter is set to true.
 *
 * @param descriptor The feature descriptor of the image in question.
 * @param reweight   Set this to true to perform TF-IDF reweighting of the
 *                   computed histogram; default false.
 * @param verbose    Set this to true to enable verbose outputs; default false.
 *
 * @return A vector of instances of type bow::FeatureDescriptor representing the
 * SIFT feature descriptors of the images in the dataset.
 */
Histogram computeHistogram(const FeatureDescriptor& descriptor,
                           bool reweight = false, bool verbose = false);

/**
 * @brief A convenience function to compute histograms from a dataset of feature
 * descriptor. This function internally performs kMeans clustering on the
 * dataset to generate a codebook vector of type bow::Dictionary for the
 * histograms. It further computes and saves the inverse document frequencies of
 * the computed histogram dataset if the reweight parameter is set to true. The
 * computed histograms also can optionally be stored in a directory called
 * "histograms" under the path of the original image dataset. Note that any
 * pre-existing histograms will be overwritten, if present.
 *
 * @param descriptor_dataset The dataset of feature descriptors.
 * @param num_clusters       The number of clusters to partition the dataset in.
 * @param max_iter           The maximum number of iterations before termination
 * @param epsilon            The desired accuracy to be reached for an early
 *                           termination; default 1e-6.
 * @param use_opencv_kmeans  Set this to true to use the OpenCV implementation
 *                           of the kMeans clustering algorithm; default false.
 * @param use_flann          Set this to true to use a FLANN-based search for
 *                           grouping the dataset into clusters; default false.
 * @param reweight           Set this to true to perform TF-IDF reweighting of
 *                           the computed histogram; default false.
 * @param save_to_disk       Set this to true to store the computed histograms;
 *                           default false.
 * @param verbose            Set this to true to enable verbose outputs; default
 *                           false.
 *
 * @return A vector of instances of type bow::Histogram representing the
 * histograms of the images in the dataset.
 */
std::vector<Histogram> buildHistogramDataset(
    const std::vector<FeatureDescriptor>& descriptor_dataset, int num_clusters,
    int max_iter, float epsilon = 1e-6, bool use_opencv_kmeans = false,
    bool use_flann = false, bool reweight = false, bool save_to_disk = false,
    bool verbose = false);

/**
 * @brief A convenience function to read in a previously computed histogram
 * dataset and load the data into a vector.
 *
 * @param dataset_path The path to the histogram dataset.
 * @param verbose      Set this to true to enable verbose outputs; default
 *                     false.
 *
 * @return A vector of instances of type bow::Histogram representing the
 * histograms of the images in the dataset.
 */
std::vector<Histogram> loadHistogramDataset(
    const std::filesystem::path& dataset_path, bool verbose = false);

}  // namespace bow::io::dataset

#endif