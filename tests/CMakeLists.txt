set(TEST_BINARY ${PROJECT_NAME}_test)
add_executable(${TEST_BINARY} test_bow.cpp)
target_link_libraries(${TEST_BINARY} bow GTest::Main)
gtest_discover_tests(${TEST_BINARY})
