// @file    image_browser.hpp
// @author  Ignacio Vizzo   [ivizzo@uni-bonn.de]
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]
//
// Original Copyright (c) 2020 Ignacio Vizzo, all rights reserved

#ifndef BOW_WEB_IMAGE_BROWSER_HPP_
#define BOW_WEB_IMAGE_BROWSER_HPP_

#include <string>
#include <vector>

namespace bow::web::image_browser {

/**
 * @brief This function generates an HTMl file displaying the results of the
 * similarity comparison.
 *
 * @param query_image_path Path to query image.
 * @param similarities     A vector of pairs of image paths with their
 *                         respective distances to the query image.
 * @param output_dir       Path to output directory, if any.
 * @param css_path         Path to CSS file, if any.
 */
void createImageBrowser(
    const std::string& query_image_path,
    const std::vector<std::pair<std::string, float>>& similarities,
    const std::string& output_dir = "results",
    const std::string& css_path = "default_style.css");

}  // namespace bow::web::image_browser

#endif
