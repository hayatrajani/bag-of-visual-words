add_executable(main main.cpp)
target_link_libraries(main PRIVATE Boost::program_options dataset)
install(TARGETS main DESTINATION bin)
install(FILES bow_params.cfg DESTINATION bin)
