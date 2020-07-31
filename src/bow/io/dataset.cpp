// @file    dataset.cpp
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]

#include "bow/io/dataset.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "bow/core/descriptor.hpp"
#include "bow/core/dictionary.hpp"
#include "bow/core/histogram.hpp"

namespace fs = std::filesystem;

namespace bow::io::dataset {

static void histToDisk_(bool save_to_disk, bool verbose,
                        const fs::path& hist_dataset_path,
                        const fs::path& image_path,
                        const Histogram& histogram) {
  if (save_to_disk) {
    const std::string hist_file_path{
        (hist_dataset_path / image_path.stem()).string() + ".csv"};
    try {
      if (verbose) {
        std::cout << "\tWriting to disk\n";
      }
      histogram.writeToCSV(hist_file_path);
    } catch (const std::runtime_error& e) {
      std::cerr << "\t[ERROR] Histogram for image " << image_path
                << " not saved to disk! " << e.what() << '\n';
    }
  }
}

int datasetSize(const fs::path& dir_path, const std::string& extension) {
  if (!extension.empty()) {
    return std::count_if(fs::directory_iterator(dir_path), {},
                         [&extension](auto file) {
                           return file.path().extension() == extension;
                         });
  }
  return std::distance(fs::directory_iterator(dir_path), {});
}

FeatureDescriptor extractDescriptors(const std::string& image_path,
                                     bool verbose) {
  if (verbose) {
    std::cout << "Extracting descriptors from " << image_path << '\n';
  }
  if (!fs::exists(image_path)) {
    throw std::runtime_error("Image does not exist!");
  }
  if (image_path.compare(image_path.length() - 4, 4, ".png") != 0) {
    throw std::runtime_error("Invalid image!");
  }
  FeatureDescriptor descriptor(image_path);
  if (verbose) {
    std::cout << "Done\n\n";
  }
  return descriptor;
}

std::vector<FeatureDescriptor> buildDescriptorDataset(
    const fs::path& dataset_path, bool save_to_disk, bool verbose) {
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
      std::cout
          << "\tCreating a directory to save the dataset:\n\t"
          << desc_dataset_path
          << "\n\tNote that any pre-existing files will be overwritten!\n";
    }
    if (fs::exists(desc_dataset_path)) {
      fs::remove_all(desc_dataset_path);
    }
    fs::create_directory(desc_dataset_path);
  }
  std::vector<FeatureDescriptor> descriptor_dataset;
  descriptor_dataset.reserve(file_count);
  for (const auto& image_file : fs::directory_iterator(dataset_path)) {
    const fs::path image_path{image_file.path()};
    const std::string image_path_str{image_path.string()};
    if (verbose) {
      std::cout << "\tProcessing " << image_path.filename() << '\n';
    }
    if (image_path.extension() == ".png") {
      descriptor_dataset.emplace_back(FeatureDescriptor(image_path_str));
      if (save_to_disk) {
        const std::string desc_file_path{
            (desc_dataset_path / image_path.stem()).string() + ".bin"};
        try {
          if (verbose) {
            std::cout << "\tWriting to disk\n";
          }
          descriptor_dataset.back().serialize(desc_file_path);
        } catch (const std::runtime_error& e) {
          std::cerr << "\t[ERROR] Descriptors for image " << image_path_str
                    << " not saved to disk! " << e.what() << '\n';
        }
      }
    } else {
      if (verbose) {
        std::cout << "\tSkipping...\n";
      }
    }
  }
  if (verbose) {
    std::cout << "Done\n\n";
  }
  return descriptor_dataset;
}

std::vector<FeatureDescriptor> loadDescriptorDataset(
    const fs::path& dataset_path, bool verbose) {
  if (verbose) {
    std::cout << "Loading descriptor dataset...\n";
  }
  int file_count{datasetSize(dataset_path, ".bin")};
  if (file_count == 0) {
    throw std::runtime_error("No valid descriptors found!");
  }
  std::vector<FeatureDescriptor> descriptor_dataset;
  descriptor_dataset.reserve(file_count);
  for (const auto& desc_file : fs::directory_iterator(dataset_path)) {
    const fs::path desc_file_path{desc_file.path()};
    if (verbose) {
      std::cout << "\tProcessing " << desc_file_path.filename() << '\n';
    }
    if (desc_file_path.extension() == ".bin") {
      try {
        descriptor_dataset.emplace_back(
            FeatureDescriptor::deserialize(desc_file_path.string()));
      } catch (const std::runtime_error& e) {
        std::cerr << "\t[ERROR] Descriptors not loaded! " << e.what() << '\n';
      }
    } else {
      if (verbose) {
        std::cout << "\tSkipping...\n";
      }
    }
  }
  if (verbose) {
    std::cout << "Done\n\n";
  }
  return descriptor_dataset;
}

Histogram computeHistogram(const FeatureDescriptor& descriptor, bool reweight,
                           bool verbose) {
  if (verbose) {
    std::cout << "Fetching codebook\n";
  }
  Dictionary& dictionary = Dictionary::getInstance();
  try {
    if (verbose) {
      std::cout << "Computing histogram for " << descriptor.getImagePath()
                << '\n';
    }
    Histogram histogram(descriptor.getImagePath(), descriptor.getDescriptors(),
                        dictionary);
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
      std::cout << "Done\n\n";
    }
    return histogram;
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(
        std::string(e.what()) +
        " Check if the histogram dataset was computed without errors.");
  }
}

std::vector<Histogram> buildHistogramDataset(
    const std::vector<FeatureDescriptor>& descriptor_dataset, int num_clusters,
    int max_iter, float epsilon, bool use_opencv_kmeans, bool use_flann,
    bool reweight, bool save_to_disk, bool verbose) {
  if (verbose) {
    std::cout << "Building histogram dataset...\n";
    std::cout << "\tBuilding codebook\n";
  }
  Dictionary& dictionary = Dictionary::getInstance();
  dictionary.build(descriptor_dataset, num_clusters, max_iter, epsilon,
                   use_opencv_kmeans, use_flann);
  fs::path hist_dataset_path;
  if (save_to_disk) {
    fs::path image_path{descriptor_dataset[0].getImagePath()};
    hist_dataset_path = image_path.parent_path().parent_path() / "histograms";
    if (verbose) {
      std::cout
          << "\tCreating a directory to save the histogram dataset:\n\t"
          << hist_dataset_path
          << "\n\tNote that any pre-existing files will be overwritten!\n";
    }
    if (fs::exists(hist_dataset_path)) {
      fs::remove_all(hist_dataset_path);
    }
    fs::create_directory(hist_dataset_path);
    if (verbose) {
      std::cout << "\tWriting codebook to disk\n";
    }
    try {
      dictionary.serialize((hist_dataset_path / "bow_codebook.dict").string());
    } catch (const std::runtime_error& e) {
      std::cerr << "\t[ERROR] Codebook not saved to disk! " << e.what() << '\n';
    }
  }
  std::vector<Histogram> histogram_dataset;
  histogram_dataset.reserve(descriptor_dataset.size());
  try {
    for (const auto& descriptor : descriptor_dataset) {
      const std::string image_path{descriptor.getImagePath()};
      if (verbose) {
        std::cout << "\tComputing histogram for image "
                  << fs::path(image_path).filename() << '\n';
      }
      histogram_dataset.emplace_back(
          Histogram(image_path, descriptor.getDescriptors(), dictionary));
      if (!reweight) {
        histToDisk_(save_to_disk, verbose, hist_dataset_path, image_path,
                    histogram_dataset.back());
      }
    }
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(
        std::string(e.what()) +
        " Check if the descriptors were generated without errors.");
  }
  if (reweight) {
    if (verbose) {
      std::cout << "\tComputing histogram dataset's IDFs for reweighting\n";
    }
    Histogram::computeIDF(histogram_dataset);
    if (save_to_disk) {
      try {
        if (verbose) {
          std::cout << "\tWriting IDFs to disk\n";
        }
        Histogram::saveIDF(
            (hist_dataset_path / "histogram_dataset.idf").string());
      } catch (const std::runtime_error& e) {
        std::cerr << "\t[ERROR] Histogram dataset's IDFs not saved to disk! "
                  << e.what() << '\n';
      }
    }
    for (auto& histogram : histogram_dataset) {
      if (verbose) {
        std::cout << "\tReweighting histogram for image "
                  << fs::path(histogram.getImagePath()).filename() << '\n';
      }
      histogram.reweight();
      histToDisk_(save_to_disk, verbose, hist_dataset_path,
                  histogram.getImagePath(), histogram);
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
    std::cout << "\tLoading codebook\n";
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
      std::cout << "\tProcessing " << hist_file_path.filename() << '\n';
    }
    if (hist_file_path.extension() == ".csv") {
      try {
        histogram_dataset.emplace_back(
            Histogram::readFromCSV(hist_file_path.string()));
      } catch (const std::runtime_error& e) {
        std::cerr << "\t[ERROR] Histogram not loaded! " << e.what() << '\n';
      }
    } else {
      if (verbose) {
        std::cout << "\tSkipping...\n";
      }
    }
  }
  if (verbose) {
    std::cout << "\tLoading histogram dataset's IDFs\n";
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

}  // namespace bow::io::dataset