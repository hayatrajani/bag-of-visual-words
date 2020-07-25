// @file    html_writer.h
// @author  Ignacio Vizzo (creator) [ivizzo@uni-bonn.de]
// @author  Hayat Rajani (modifier) [hayat.rajani@uni-bonn.de]
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
   * @brief openDocument() will print to the output stream the begining of a
   * HTML5 file. This function should be called only once at the begining of
   * your program.
   */
  void openDocument();

  /**
   * @brief closeDocument() will close the HTML5 file. This function should be
   * called only once at the end of your program.
   */
  void closeDocument();

  /**
   * @brief openBody() prints a <body> clause to the output stream.
   */
  void openBody();

  /**
   * @brief closeBody() prints a </body> clause to the output stream.
   */
  void closeBody();

  /**
   * @brief openRow() adds a new row division to your application.
   */
  void openRow();

  /**
   * @brief closeRow() closes a row division of your application. Make sure you
   * only call this function after a call to openRow()
   */
  void closeRow();

  /**
   * @brief addCSS() adds a stylesheet to your web application.
   *
   * @param stylesheet The path where the CSS file is located, for example,
   * "<path>/style.css"
   */
  void addCSS(const std::string& stylesheet);

  /**
   * @brief addTitle() adds a Title to your web application.
   *
   * @param title The string containing the title.
   */
  void addTitle(const std::string& title);

  /**
   * @brief addImage() adds a new image to your web application using the
   * <img src=""> clause. It also prints the score and the name of the image.
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
