cmake_minimum_required(VERSION 3.18)
project(ml_from_scratch CUDA CXX)

# Set CMake policies and options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Option to build with or without CUDA
option(USE_CUDA "Build with CUDA support" ON)

# Find necessary packages
find_package(pybind11 REQUIRED)
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
endif()

# Add subdirectories
add_subdirectory(common)
add_subdirectory(tasks)