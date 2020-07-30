set(TEST_BINARY ${PROJECT_NAME}_test)

add_executable(${TEST_BINARY}
               test_data.cpp
               test_algorithms.cpp)

target_link_libraries(${TEST_BINARY} algorithms GTest::Main)

gtest_discover_tests(${TEST_BINARY} WORKING_DIRECTORY
                     ${CMAKE_CURRENT_SOURCE_DIR})
