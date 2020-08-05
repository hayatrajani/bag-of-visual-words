set(TEST_BINARY ${PROJECT_NAME}_test)

add_executable(${TEST_BINARY}
               test_data.cpp
               test_descriptors.cpp
               test_algorithms.cpp
               test_dictionary.cpp
               test_histograms.cpp
               test_dataset.cpp)

target_link_libraries(${TEST_BINARY} descriptor algorithms dictionary histogram dataset GTest::Main)

gtest_discover_tests(${TEST_BINARY} WORKING_DIRECTORY
                     ${CMAKE_CURRENT_SOURCE_DIR})
