#include "bow/io/dataset.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "bow/core/dictionary.hpp"
#include "bow/core/histogram.hpp"
#include "bow/io/serialization.hpp"

using cv::xfeatures2d::SiftDescriptorExtractor;
using cv::xfeatures2d::SiftFeatureDetector;

namespace fs = std::filesystem;

namespace bow::io::dataset {

static void histToDisk_(bool save_to_disk, bool verbose,
                        const fs::path& hist_dataset_path,
                        const fs::path& image_path, const Histogram& histogram);

int datasetSize(const fs::path& path, const std::string& extension) {
  if (!extension.empty()) {
    return std::count_if(fs::directory_iterator(path), {},
                         [&extension](auto file) {
                           return file.path().extension() == extension;
                         });
  }
  return std::distance(fs::directory_iterator(path), {});
}

cv::Mat extractDescriptors(const std::string& image_path, bool verbose) {
  if (verbose) {
    std::cout << "Extracting descriptors from " << image_path << '\n';
  }
  // read image
  const cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  // detect key points
  auto detector = SiftFeatureDetector::create();
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(image, keypoints);
  // extract the SIFT descriptors
  cv::Mat descriptors;
  auto extractor = SiftDescriptorExtractor::create();
  extractor->compute(image, keypoints, descriptors);
  if (verbose) {
    std::cout << "Done\n";
  }
  return descriptors;
}

std::tuple<std::vector<cv::Mat>, std::vector<std::string>>
buildDescriptorDataset(const fs::path& dataset_path, bool save_to_disk,
                       bool verbose) {
  if (verbose) {
    std::cout << "Building descriptor dataset...\n";
  }
  int file_count{datasetSize(dataset_path, ".png")};
  if (file_count == 0) {
    throw std::runtime_error("No valid image files found!");
  }
  fs::path desc_dataset_path;
  if (save_to_disk) {
    desc_dataset_path =
        (dataset_path.string().back() == fs::path::preferred_separator)
            ? dataset_path.parent_path().parent_path() / "descriptors"
            : dataset_path.parent_path() / "descriptors";
    if (verbose) {
      std::cout << "Creating a directory to save the descriptor dataset:\n\t"
                << desc_dataset_path
                << "\nNote that any pre-existing files will be overwritten!\n";
    }
    if (fs::exists(desc_dataset_path)) {
      fs::remove_all(desc_dataset_path);
    }
    fs::create_directory(desc_dataset_path);
  }
  std::vector<cv::Mat> sift_dataset;
  sift_dataset.reserve(file_count);
  std::vector<std::string> image_paths;
  image_paths.reserve(file_count);
  for (const auto& image_file : fs::directory_iterator(dataset_path)) {
    const fs::path image_path{image_file.path()};
    const std::string image_path_str{image_path.string()};
    if (verbose) {
      std::cout << "Processing " << image_path.filename() << '\n';
    }
    if (image_path.extension() == ".png") {
      auto descriptors = extractDescriptors(image_path_str);
      sift_dataset.emplace_back(descriptors);
      image_paths.emplace_back(image_path_str);
      if (save_to_disk) {
        const std::string desc_file_path{
            (desc_dataset_path / image_path.stem()).string() + ".bin"};
        try {
          if (verbose) {
            std::cout << "Writing to disk\n";
          }
          serialization::serialize(descriptors, image_path_str, desc_file_path);
        } catch (const std::runtime_error& e) {
          std::cerr << "[ERROR] Descriptors for image " << image_path_str
                    << " not saved to disk! " << e.what() << '\n';
        }
      }
    } else {
      if (verbose) {
        std::cout << "Skipping...\n";
      }
    }
  }
  if (verbose) {
    std::cout << "Done\n\n";
  }
  return std::make_tuple(sift_dataset, image_paths);
}

std::tuple<std::vector<cv::Mat>, std::vector<std::string>>
loadDescriptorDataset(const fs::path& dataset_path, bool verbose) {
  if (verbose) {
    std::cout << "Loading descriptor dataset...\n";
  }
  int file_count{datasetSize(dataset_path, ".bin")};
  if (file_count == 0) {
    throw std::runtime_error("No valid descriptors found!");
  }
  std::vector<cv::Mat> sift_dataset;
  sift_dataset.reserve(file_count);
  std::vector<std::string> image_paths;
  image_paths.reserve(file_count);
  for (const auto& desc_file : fs::directory_iterator(dataset_path)) {
    const fs::path desc_file_path{desc_file.path()};
    if (verbose) {
      std::cout << "Processing " << desc_file_path.filename() << '\n';
    }
    if (desc_file_path.extension() == ".bin") {
      try {
        auto [descriptors, image_path] =
            serialization::deserialize(desc_file_path.string());
        sift_dataset.emplace_back(descriptors);
        image_paths.emplace_back(image_path);
      } catch (const std::runtime_error& e) {
        std::cerr << "[ERROR] Descriptors not loaded! " << e.what() << '\n';
      }
    } else {
      if (verbose) {
        std::cout << "Skipping...\n";
      }
    }
  }
  if (verbose) {
    std::cout << "Done\n\n";
  }
  return std::make_tuple(sift_dataset, image_paths);
}

Histogram computeHistogram(const cv::Mat& descriptors,
                           const std::string& image_path, bool reweight,
                           bool verbose) {
  if (verbose) {
    std::cout << "Fetching codebook\n";
  }
  Dictionary& dictionary = Dictionary::getInstance();
  try {
    if (verbose) {
      std::cout << "Computing histogram for " << image_path << '\n';
    }
    Histogram histogram(image_path, descriptors, dictionary);
    if (reweight) {
      if (verbose) {
        std::cout << "Reweighting histogram\n";
      }
      if (!Histogram::hasIDF()) {
        histogram.reweight();
      } else {
        std::cerr
            << "[ERROR] IDFs not computed! Call Histogram::computeIDF() on the "
               "histogram dataset and manually reweight() the histogram\n";
      }
    }
    if (verbose) {
      std::cout << "Done\n";
    }
    return histogram;
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(
        std::string(e.what()) +
        " Check if the histogram dataset was computed without errors.");
  }
}

std::vector<Histogram> buildHistogramDataset(
    const std::vector<cv::Mat>& descriptors,
    const std::vector<std::string>& image_paths, int max_iter, int num_clusters,
    bool use_flann, bool use_opencv_kmeans, float epsilon, bool reweight,
    bool save_to_disk, bool verbose) {
  if (verbose) {
    std::cout << "Building histogram dataset...\n";
    std::cout << "Building codebook\n";
  }
  Dictionary& dictionary = Dictionary::getInstance();
  dictionary.build(max_iter, num_clusters, descriptors, use_flann,
                   use_opencv_kmeans, epsilon);
  fs::path hist_dataset_path;
  if (save_to_disk) {
    fs::path image_path{image_paths[0]};
    hist_dataset_path = image_path.parent_path().parent_path() / "histograms";
    if (verbose) {
      std::cout << "Creating a directory to save the histogram dataset:\n\t"
                << hist_dataset_path
                << "\nNote that any pre-existing files will be overwritten!\n";
    }
    if (fs::exists(hist_dataset_path)) {
      fs::remove_all(hist_dataset_path);
    }
    fs::create_directory(hist_dataset_path);
    if (verbose) {
      std::cout << "Writing codebook to disk\n";
    }
    try {
      dictionary.serialize((hist_dataset_path / "bow_codebook.dict").string());
    } catch (const std::runtime_error& e) {
      std::cerr << "[ERROR] Codebook not saved to disk! " << e.what() << '\n';
    }
  }
  std::size_t file_count{descriptors.size()};
  std::vector<Histogram> histogram_dataset;
  histogram_dataset.reserve(file_count);
  try {
    for (std::size_t i = 0; i < file_count; ++i) {
      if (verbose) {
        std::cout << "Computing histogram for image "
                  << fs::path(image_paths[i]).filename() << '\n';
      }
      histogram_dataset.emplace_back(
          Histogram(image_paths[i], descriptors[i], dictionary));
      if (!reweight) {
        histToDisk_(save_to_disk, verbose, hist_dataset_path, image_paths[i],
                    histogram_dataset[i]);
      }
    }
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(
        std::string(e.what()) +
        " Check if the descriptors were generated without errors.");
  }
  if (reweight) {
    if (verbose) {
      std::cout << "Computing histogram dataset's IDFs for reweighting\n";
    }
    Histogram::computeIDF(histogram_dataset);
    if (save_to_disk) {
      try {
        if (verbose) {
          std::cout << "Writing IDFs to disk\n";
        }
        std::cout << hist_dataset_path << '\n';
        Histogram::saveIDF(
            (hist_dataset_path / "histogram_dataset.idf").string());
      } catch (const std::runtime_error& e) {
        std::cerr << "[ERROR] Histogram dataset's IDFs not saved to disk! "
                  << e.what() << '\n';
      }
    }
    for (std::size_t i = 0; i < file_count; ++i) {
      if (verbose) {
        std::cout << "Reweighting histogram for image "
                  << fs::path(image_paths[i]).filename() << '\n';
      }
      histogram_dataset[i].reweight();
      histToDisk_(save_to_disk, verbose, hist_dataset_path, image_paths[i],
                  histogram_dataset[i]);
    }
  }
  if (verbose) {
    std::cout << "Done\n\n";
  }
  return histogram_dataset;
}

std::vector<Histogram> loadHistogramDataset(const fs::path& dataset_path,
                                            bool verbose) {
  if (verbose) {
    std::cout << "Loading histogram dataset...\n";
  }
  int file_count{datasetSize(dataset_path, ".csv")};
  if (file_count == 0) {
    throw std::runtime_error("No valid histogram files found!");
  }
  if (verbose) {
    std::cout << "Loading codebook\n";
  }
  Dictionary& dictionary = Dictionary::getInstance();
  try {
    dictionary.deserialize((dataset_path / "bow_codebook.dict").string());
  } catch (const std::runtime_error& e) {
    throw std::runtime_error("Codebook not loaded! " + std::string(e.what()));
  }
  std::vector<Histogram> histogram_dataset;
  histogram_dataset.reserve(file_count);
  for (const auto& hist_file : fs::directory_iterator(dataset_path)) {
    const fs::path hist_file_path{hist_file.path()};
    if (verbose) {
      std::cout << "Processing " << hist_file_path.filename() << '\n';
    }
    if (hist_file_path.extension() == ".csv") {
      try {
        histogram_dataset.emplace_back(
            Histogram::readFromCSV(hist_file_path.string()));
      } catch (const std::runtime_error& e) {
        std::cerr << "[ERROR] Histogram not loaded! " << e.what() << '\n';
      }
    } else {
      if (verbose) {
        std::cout << "Skipping...\n";
      }
    }
  }
  if (verbose) {
    std::cout << "Loading histogram dataset's IDFs\n";
  }
  try {
    Histogram::loadIDF((dataset_path / "histogram_dataset.idf").string());
  } catch (const std::runtime_error& e) {
    std::cerr << "[WARNING] Histogram dataset's IDFs not loaded! " << e.what()
              << " Manually call Histogram::computeIDF() on the histogram "
                 "dataset and reweight() all histograms if necessary!\n";
  }
  if (verbose) {
    std::cout << "Done\n\n";
  }
  return histogram_dataset;
}

static void histToDisk_(bool save_to_disk, bool verbose,
                        const fs::path& hist_dataset_path,
                        const fs::path& image_path,
                        const Histogram& histogram) {
  if (save_to_disk) {
    const std::string hist_file_path{
        (hist_dataset_path / image_path.stem()).string() + ".csv"};
    try {
      if (verbose) {
        std::cout << "Writing to disk\n";
      }
      histogram.writeToCSV(hist_file_path);
    } catch (const std::runtime_error& e) {
      std::cerr << "[ERROR] Histogram for image " << image_path
                << " not saved to disk! " << e.what() << '\n';
    }
  }
}

}  // namespace bow::io::dataset