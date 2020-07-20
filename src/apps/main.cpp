#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "bow/io/dataset.hpp"

namespace po = boost::program_options;
namespace ds = bow::io::dataset;
namespace fs = std::filesystem;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, " "));
  return os;
}

int main(int argc, char** argv) {
  // clang-format off
  po::options_description general_options_description("General Options");
  general_options_description.add_options()
    ("help,h", "display help message")
    ("verbose,v", "print verbose output")
    ("config-file,c", po::value<std::string>()->default_value(""),
      "path to the configuration file")
    ("image-path,I", po::value<std::string>(),
      "path to image dataset")
    ("descriptor-path,D", po::value<std::string>(),
      "path to precomputed feature descriptors")
    ("histogram-path,H", po::value<std::string>(),
      "path to precomputed image histograms")
  ;
  po::options_description config_options_description("Configuration Options");
  config_options_description.add_options()
    ("use-flann", po::value<bool>()->default_value(true),
      "use FLANN for histogram computations")
    ("use-opencv-kmeans", po::value<bool>()->default_value(true),
      "use opencv kmeans implementation")
    ("num-clusters,k", po::value<int>()->default_value(100),
      "number of clusters")
    ("max-iter,m", po::value<int>()->default_value(25),
      "maximum number of iterations")
    ("epsilon,e", po::value<float>()->default_value(0.001),
      "stop iterations if specified accuracy, epsilon, is reached "
      "(only for opencv kmeans)")
    ("num-similar,n", po::value<int>()->default_value(1),
      "number of similar images to find")
    ("reweight", po::value<bool>()->default_value(true),
      "perform TF-IDF reweighting for histograms")
    ("save-histograms", po::value<bool>()->default_value(true),
      "save histogram dataset to disk")
    ("save-descriptors", po::value<bool>()->default_value(false),
      "save descriptors dataset to disk")
  ;
  po::options_description shared_options_description;
  shared_options_description.add_options()
    ("query-path,Q", po::value<std::vector<std::string>>()->multitoken(),
      "path to query image(s)")
  ;
  // clang-format on

  po::options_description cmdline_options;
  cmdline_options.add(general_options_description)
      .add(config_options_description)
      .add(shared_options_description);

  po::options_description config_options;
  config_options.add(config_options_description)
      .add(shared_options_description);

  po::options_description display_options;
  display_options.add(general_options_description)
      .add(shared_options_description)
      .add(config_options_description)
      .add(shared_options_description);

  po::variables_map var_map;
  try {
    po::store(po::parse_command_line(argc, argv, cmdline_options), var_map);
    const auto config_file_path{var_map["config-file"].as<std::string>()};
    if (!config_file_path.empty()) {
      std::ifstream config_file{config_file_path};
      if (!config_file) {
        std::cerr << "[ERROR] Cannot open config file: " << config_file_path
                  << '\n';
        return EXIT_FAILURE;
      }
      po::store(po::parse_config_file(config_file, config_options), var_map);
    }
  } catch (const po::error& e) {
    std::cerr << "[ERROR] Invalid Option\n" << e.what() << '\n';
    return EXIT_FAILURE;
  }

  if (var_map.count("help")) {
    std::cout << display_options << '\n';
    return EXIT_SUCCESS;
  }

  const auto verbose{var_map.count("verbose") != 0};
  const auto use_flann{var_map["use-flann"].as<bool>()};
  const auto use_opencv_kmeans{var_map["use-opencv-kmeans"].as<bool>()};
  const auto num_clusters{var_map["num-clusters"].as<int>()};
  const auto max_iter{var_map["max-iter"].as<int>()};
  const auto epsilon{var_map["epsilon"].as<float>()};
  const auto num_similar{var_map["num-similar"].as<int>()};
  const auto reweight{var_map["reweight"].as<bool>()};
  const auto hist_to_disk{var_map["save-histograms"].as<bool>()};
  const auto desc_to_disk{var_map["save-descriptors"].as<bool>()};

  std::vector<bow::Histogram> histogram_dataset;

  try {
    if (var_map.count("image-path")) {
      const fs::path dataset_path{var_map["image-path"].as<std::string>()};
      const auto [descriptors, image_paths] =
          ds::buildDescriptorDataset(dataset_path, desc_to_disk, verbose);
      histogram_dataset = ds::buildHistogramDataset(
          descriptors, image_paths, max_iter, num_clusters, use_flann,
          use_opencv_kmeans, epsilon, reweight, hist_to_disk, verbose);
    } else if (var_map.count("descriptor-path")) {
      const fs::path dataset_path{var_map["descriptor-path"].as<std::string>()};
      const auto [descriptors, image_paths] =
          ds::loadDescriptorDataset(dataset_path, verbose);
      histogram_dataset = ds::buildHistogramDataset(
          descriptors, image_paths, max_iter, num_clusters, use_flann,
          use_opencv_kmeans, epsilon, reweight, hist_to_disk, verbose);
    } else if (var_map.count("histogram-path")) {
      const fs::path dataset_path{var_map["histogram-path"].as<std::string>()};
      histogram_dataset = ds::loadHistogramDataset(dataset_path, verbose);
    } else {
      std::cerr << "[ERROR] Path to dataset not specified\n";
      return EXIT_FAILURE;
    }

    if (var_map.count("query-path")) {
      const auto& query_paths{
          var_map["query-path"].as<std::vector<std::string>>()};
      for (const std::string& query_path : query_paths) {
        if (fs::exists(query_path)) {
          if (query_path.compare(query_path.length() - 4, 4, ".png") == 0) {
            auto histogram = ds::computeHistogram(
                ds::extractDescriptors(query_path, verbose), query_path,
                reweight, verbose);
            auto similarities =
                histogram.compare(histogram_dataset, num_similar);
            std::cout << "Images similar to: " << histogram.getImagePath()
                      << '\n';
            for (const auto& similarity : similarities) {
              std::cout << '\t' << similarity.first << '\n';
            }
            std::cout << '\n';
          } else {
            std::cerr << "Invalid image! Skipping " << query_path << '\n';
          }
        } else {
          std::cerr << "Image does not exist! Skipping " << query_path << '\n';
        }
      }
    } else {
      std::cout << "[WARNING] No query image(s) found! Histograms were "
                   "nonetheless computed and stored within the specified "
                   "dataset directory\n";
    }
  } catch (const std::runtime_error& e) {
    std::cerr << "[ERROR] " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
