cmake_minimum_required(VERSION 3.18)
project(cuda_utils LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Set output directories for this subdirectory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/infinite_ml/common/cuda)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/infinite_ml/common/cuda)

# Build common CUDA utilities
if(USE_CUDA)
    # Source files
    set(COMMON_CUDA_SOURCES
        cuda_device_info.cu
        utils.cu
    )
    
    # Build CUDA library
    cuda_add_library(common_cuda SHARED ${COMMON_CUDA_SOURCES})
    
    target_include_directories(common_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Create pybind11 module
    pybind11_add_module(cuda_utils cuda_bindings.cpp)
    
    # Set RPATH to look in the same directory as the module
    set_target_properties(cuda_utils PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    # Link pybind module with our common_cuda library
    target_link_libraries(cuda_utils PRIVATE 
        common_cuda
        ${CUDA_LIBRARIES}
    )
    
    # Install the libraries
    install(
        TARGETS common_cuda cuda_utils
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/common/cuda
        RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/common/cuda
    )
    
    # Create Python package structure
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py "# Python package marker\ntry:\n    from .cuda_utils import *\nexcept ImportError:\n    print('Failed to import CUDA utilities')\n")
    
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
            DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/common/cuda)
    
    # Install header files
    install(FILES utils.cuh
            DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/common/cuda/include)
    
    # Install any additional header files if they exist
    file(GLOB CUDA_HEADERS "*.h" "*.cuh")
    if(CUDA_HEADERS)
        install(FILES ${CUDA_HEADERS}
                DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/common/cuda/include)
    endif()
endif()