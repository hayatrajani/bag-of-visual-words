add_executable(main main.cpp)
target_link_libraries(main PRIVATE dataset image_browser Boost::program_options)
install(TARGETS main DESTINATION bin)
install(FILES bow_params.cfg default_style.css DESTINATION bin)
