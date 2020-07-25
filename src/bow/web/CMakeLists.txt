add_library(html_writer html_writer.cpp)
set_target_properties(html_writer PROPERTIES PREFIX "")

add_library(image_browser image_browser.cpp)
set_target_properties(image_browser PROPERTIES PREFIX "")
target_link_libraries(image_browser PRIVATE html_writer)

install(TARGETS html_writer image_browser DESTINATION lib)
