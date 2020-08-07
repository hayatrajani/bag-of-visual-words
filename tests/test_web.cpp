// @file    test_web.cpp
// @author  Ignacio Vizzo   [ivizzo@uni-bonn.de]
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]
//
// Original Copyright (c) 2020 Ignacio Vizzo, all rights reserved

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "bow/web/image_browser.hpp"
#include "test_utils.hpp"

namespace fs = std::filesystem;
namespace ib = bow::web::image_browser;

namespace {

const int dataset_size{10 + 1};

const std::string dataset_path{"test_data/dummy_dataset/images/"};
const std::string query_path{dataset_path + "door_3.png"};
const std::string css_path{"test_data/default_style.css"};
const std::string out_dir{"test_data/web_output/"};
const std::string html_path{out_dir + "door_3.html"};

std::vector<std::pair<std::string, float>> getDummySimilarities(
    const std::string& dataset_path, int dataset_size) {
  std::vector<std::pair<std::string, float>> dummy_similarities;
  dummy_similarities.reserve(dataset_size);
  for (const auto& image_file : fs::directory_iterator(dataset_path)) {
    dummy_similarities.emplace_back(
        std::make_pair(image_file.path().string(), 0.1F));
  }
  return dummy_similarities;
}

auto dummy_similarities = getDummySimilarities(dataset_path, dataset_size);

}  // anonymous namespace

TEST(Web, CreateImageBrowser) {
  testing::internal::CaptureStderr();

  ib::createImageBrowser(query_path, dummy_similarities, out_dir, css_path);
  ASSERT_TRUE(fs::exists(html_path));

  auto cerr = testing::internal::GetCapturedStderr();
  ASSERT_FALSE(cerr.empty());
  ASSERT_THAT(cerr, testing::HasSubstr("[ERROR]"));

  std::ifstream in_file(html_path, std::ios_base::in);
  std::string fin((std::istreambuf_iterator<char>(in_file)),
                  std::istreambuf_iterator<char>());
  std::ifstream html_example("../tests/example.html");
  std::string line;
  while (std::getline(html_example, line)) {
    ASSERT_THAT(fin, testing::HasSubstr(trim(line)));
  }
  html_example.close();
  in_file.close();

  fs::remove_all(out_dir);
}

TEST(Web, InvalidHTML) {
  ASSERT_THROW(ib::createImageBrowser(query_path, dummy_similarities, ""),
               std::runtime_error);
}

TEST(Web, NoCSS) {
  testing::internal::CaptureStderr();

  ib::createImageBrowser(query_path, dummy_similarities, out_dir);
  ASSERT_TRUE(fs::exists(html_path));

  auto cerr = testing::internal::GetCapturedStderr();
  ASSERT_FALSE(cerr.empty());
  ASSERT_THAT(cerr, testing::HasSubstr("[INFO]"));

  fs::remove_all(out_dir);
}

TEST(Web, FakeQueryImage) {
  testing::internal::CaptureStderr();

  ib::createImageBrowser("fake_file.png", dummy_similarities, out_dir,
                         css_path);
  ASSERT_TRUE(fs::exists(out_dir + "fake_file.html"));

  auto cerr = testing::internal::GetCapturedStderr();
  ASSERT_FALSE(cerr.empty());
  ASSERT_THAT(cerr, testing::HasSubstr("[ERROR]"));

  fs::remove_all(out_dir);
}

TEST(Web, EmptySimilarities) {
  testing::internal::CaptureStderr();

  ib::createImageBrowser(query_path, {}, out_dir, css_path);
  ASSERT_FALSE(fs::exists(html_path));

  auto cerr = testing::internal::GetCapturedStderr();
  ASSERT_FALSE(cerr.empty());
  ASSERT_THAT(cerr, testing::HasSubstr("[ERROR]"));
}