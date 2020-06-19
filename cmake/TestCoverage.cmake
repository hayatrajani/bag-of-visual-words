# Code Coverage Configuration
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
if(ENABLE_COVERAGE)
    # NOTE: Coverage only works/makes sense with debug builds
    set(CMAKE_BUILD_TYPE "Debug")
    set(CXX_COVERAGE_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COVERAGE_FLAGS}")
    message(STATUS "Enabling coverage instrumentation: ${CXX_COVERAGE_FLAGS}")
endif(ENABLE_COVERAGE)
