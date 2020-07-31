// @file    html_writer.h
// @author  Ignacio Vizzo   [ivizzo@uni-bonn.de]
// @author  Hayat Rajani    [hayat.rajani@uni-bonn.de]
//
// Original Copyright (c) 2020 Ignacio Vizzo, all rights reserved

#ifndef BOW_WEB_HTML_WRITER_H_
#define BOW_WEB_HTML_WRITER_H_

#include <fstream>
#include <iostream>
#include <string>

namespace bow::web {

class HTML_Writer {
 private:
  const std::string html_file_path_;
  std::ofstream fout_;

 public:
  explicit HTML_Writer(const std::string& html_file_path);

  /**
   * @brief This function will print to the output stream the begining of an
   * HTML5 file. It should be called only once at the begining of your program.
   */
  void openDocument();

  /**
   * @brief This function will close the HTML5 file. It should be called only
   * once at the end of your program.
   */
  void closeDocument();

  /**
   * @brief This function prints a <body> clause to the output stream.
   */
  void openBody();

  /**
   * @brief This function prints a </body> clause to the output stream.
   */
  void closeBody();

  /**
   * @brief This function adds a new row division to your application by
   * printing a <div class="row"> tag to the output stream.
   */
  void openRow();

  /**
   * @brief This function closes a row division in your application. Make sure
   * you only call this function after a call to openRow()
   */
  void closeRow();

  /**
   * @brief This function adds a stylesheet to your web application.
   *
   * @param stylesheet The path where the CSS file is located, for example,
   * "<path>/style.css"
   */
  void addCSS(const std::string& stylesheet);

  /**
   * @brief This function adds a title to your web application by printing a
   * <title>""</title> tag to the output stream.
   *
   * @param title The string containing the title.
   */
  void addTitle(const std::string& title);

  /**
   * @brief This function adds a new image to your web application using the
   * <img src=""> tag. It also displays the score and the name of the image.
   *
   * @param img_path    The path to the image
   * @param distance    The distance measure of the given image
   * @param query_image If or not the image being added is the query image.
   */
  void addImage(const std::string& image_path, float distance,
                bool query_image = false);
};

}  // namespace bow::web

#endif
