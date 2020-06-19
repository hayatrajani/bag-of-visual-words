# Code Coverage Configuration
if(ENABLE_COVERAGE)
  # NOTE: Coverage only works/makes sense with debug builds
  set(CMAKE_BUILD_TYPE "Debug")
  set(CXX_COVERAGE_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COVERAGE_FLAGS}")
  message(STATUS "Enabling coverage instrumentation: ${CXX_COVERAGE_FLAGS}")

  add_custom_target(test-coverage
                    COMMENT "Run test coverage for all sources expect for tests"
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND ${CMAKE_SOURCE_DIR}/.ci/test_coverage.sh)
endif(ENABLE_COVERAGE)
