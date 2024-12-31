cmake_minimum_required(VERSION 3.10)
project(infinite_ml LANGUAGES CXX)

# Enable CUDA language if CUDA is available
option(USE_CUDA "Build with CUDA support" ON)
if(USE_CUDA)
    enable_language(CUDA)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set RPATH settings for all targets
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# Add module path for custom Find* modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Set default Python site-packages if not provided
if(NOT DEFINED PYTHON_SITE_PACKAGES)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0], end='')"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    )
    message(STATUS "Using Python site-packages: ${PYTHON_SITE_PACKAGES}")
endif()

# Define output directories for libraries and binaries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Handle CUDA
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    
    # Set CUDA settings
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Handle CUDA architectures based on CMake version
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        # Modern way
        set(CMAKE_CUDA_ARCHITECTURES 35 50 60 70 75 80 CACHE STRING "CUDA architectures")
    else()
        # Legacy way for older CMake
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_35 -arch=sm_50 -arch=sm_60 -arch=sm_70 -arch=sm_75)
    endif()
    
    # Include CUDA architecture detection
    include(DetectCudaArch)
    detect_cuda_arch(DETECTED_CUDA_ARCH)
    message(STATUS "CUDA architecture detection result: ${DETECTED_CUDA_ARCH}")
    
    # Load pybind11 setup helper
    include(SetupPybind)
endif()

# Add common libraries
add_subdirectory(infinite_ml/common)

# Add task directories
add_subdirectory(infinite_ml/tasks)

# Create Python package structure
set(INIT_CONTENT "# Python package marker\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py ${INIT_CONTENT})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml)