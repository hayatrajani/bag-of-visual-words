# ~~~
# @file      CMakeLists.txt
# @author    Ignacio Vizzo (creator)    [ivizzo@uni-bonn.de]
# @author    Hayat Rajani (modifier)    [hayat.rajani@uni-bonn.de]
#
# Original Copyright (c) 2020 Ignacio Vizzo, all rights reserved
cmake_minimum_required(VERSION 3.1)
project(bow_final_project)

# Set build type if not set.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Build options
option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" ON)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" ON)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

# Attempt to find OpenCV4 on your system, for more details please read
# /usr/share/OpenCV/OpenCVConfig.cmake
find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
  message(STATUS "OpenCV directories: ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "OpenCV not found, please read the README.md")
endif(OpenCV_FOUND)

# Attempt to find Boost on your system, for more details please read
# /usr/share/cmake-version/Modules/FindBoost.cmake
find_package(Boost COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
  message(STATUS "Found Boost version ${Boost_VERSION}")
  message(STATUS "Boost directories: ${Boost_INCLUDE_DIRS}")
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Boost not found, please read the README.md")
endif(Boost_FOUND)

# Enable testing
enable_testing()
find_package(GTest REQUIRED)

# Set additional flags.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Code Coverage, clang-format, clang-tidy, cppcheck, Configuration:
include(TestCoverage)
include(ClangFormat)
include(StaticAnalyzers)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Set build output directories
set(BUILD_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/results)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BUILD_OUTPUT_DIRECTORY}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BUILD_OUTPUT_DIRECTORY}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_OUTPUT_DIRECTORY}/bin)

# Use this directory to test your installation script
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/results)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

# After all setup is done, we can go to our src/ directory to build our files
add_subdirectory(src)
add_subdirectory(tests)
