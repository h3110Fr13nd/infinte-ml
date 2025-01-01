cmake_minimum_required(VERSION 3.10)
project(vector_addition CUDA CXX)

set(TASK_NAME addition)
set(TASK_PATH task_001_addition)

# Set output directories for this task
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/infinite_ml/tasks/${TASK_PATH}/cuda)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/infinite_ml/tasks/${TASK_PATH}/cuda)

# Force position independent code for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if(USE_CUDA)
    # Find CUDA
    find_package(CUDA REQUIRED)Building infinite-ml @ file:///home/h3110fr13nd/Desktop/dev/aiml/infinite-ml
    
    # Include common headers
    include_directories(
        ${CMAKE_SOURCE_DIR}/infinite_ml/common/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Set CUDA options
    # Find CUDA architecture
    detect_cuda_arch(CUDA_ARCH)
    message(STATUS "Detected CUDA arch: ${CUDA_ARCH}")
    
    # Process the semicolon-separated list
    # Take the highest architecture in the list for best compatibility
    set(CUDA_NVCC_FLAGS "-std=c++11")
    
    # Convert the semicolon-separated list to a CMake list
    string(REPLACE ";" " " CUDA_ARCH_STR "${CUDA_ARCH}")
    message(STATUS "Architecture string: ${CUDA_ARCH_STR}")
    
    # Extract the highest architecture from the list
    list(LENGTH CUDA_ARCH ARCH_COUNT)
    if(ARCH_COUNT GREATER 0)
        list(GET CUDA_ARCH -1 HIGHEST_ARCH)
        
        # Extract major and minor parts (if format is like "75" for 7.5)
        string(REGEX MATCH "^([0-9])([0-9])$" ARCH_MATCH ${HIGHEST_ARCH})
        if(ARCH_MATCH)
            set(MAJOR_ARCH ${CMAKE_MATCH_1})
            set(MINOR_ARCH ${CMAKE_MATCH_2})
            set(FORMATTED_ARCH "${MAJOR_ARCH}.${MINOR_ARCH}")
            message(STATUS "Using highest CUDA arch: ${FORMATTED_ARCH}")
            
            # Set CUDA flags for the highest architecture
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode=arch=compute_${MAJOR_ARCH}${MINOR_ARCH},code=sm_${MAJOR_ARCH}${MINOR_ARCH} -Xcompiler -fPIC)
        else()
            # Try alternate format like "7.5"
            string(REGEX MATCH "^([0-9]+)\.([0-9]+)$" ARCH_MATCH ${HIGHEST_ARCH})
            if(ARCH_MATCH)
                set(MAJOR_ARCH ${CMAKE_MATCH_1})
                set(MINOR_ARCH ${CMAKE_MATCH_2})
                message(STATUS "Using highest CUDA arch: ${MAJOR_ARCH}.${MINOR_ARCH}")
                
                # Set CUDA flags for the highest architecture
                set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode=arch=compute_${MAJOR_ARCH}${MINOR_ARCH},code=sm_${MAJOR_ARCH}${MINOR_ARCH} -Xcompiler -fPIC)
            else()
                message(WARNING "Could not parse CUDA architecture '${HIGHEST_ARCH}', falling back to sm_70")
                set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode=arch=compute_70,code=sm_70 -Xcompiler -fPIC)
            endif()
        endif()
    else()
        # Fallback to safe architecture values if no architectures found
        message(WARNING "No CUDA architectures detected, falling back to sm_70")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode=arch=compute_70,code=sm_70 -Xcompiler -fPIC)
    endif()
    
    # Set increased heap size to help with memory allocation
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lcudadevrt --default-stream per-thread)
    
    # Add compilation flags
    add_compile_definitions(USE_CUDA)

    # Add logger library
    add_library(logger STATIC logger.cpp)
    
    # Ensure logger is compiled with position independent code
    set_property(TARGET logger PROPERTY POSITION_INDEPENDENT_CODE ON)

    # Source files
    set(SRC_FILES
        addition.cu
    )

    # Define shared library target
    cuda_add_library(${TASK_NAME}_cuda SHARED ${SRC_FILES})
    
    # Make sure we have the right include directories
    target_include_directories(${TASK_NAME}_cuda PUBLIC
        ${CMAKE_SOURCE_DIR}/infinite_ml/common/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Setup pybind11 module
    pybind11_add_module(addition bindings.cpp)

    # Set RPATH to look in the same directory as the module
    set_target_properties(addition PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    
    # Link with our CUDA library
    target_link_libraries(addition PRIVATE
        ${TASK_NAME}_cuda
        ${CUDA_LIBRARIES}
        logger
    )

    # Install the libraries to the correct location
    install(
        TARGETS ${TASK_NAME}_cuda addition logger
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/tasks/${TASK_PATH}/cuda
        RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/tasks/${TASK_PATH}/cuda
        ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/tasks/${TASK_PATH}/cuda
    )

    # Create Python package structure
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py "# Python package marker\ntry:\n    from .addition import *\nexcept ImportError as e:\n    import os\n    print(f'Error importing CUDA module: {e}')\n    print(f'Current directory: {os.getcwd()}')\n    print(f'Files in directory: {os.listdir(\".\")}')\n    def addition():\n        return {'success': False, 'message': '', 'error_message': 'CUDA module not compiled properly.'}\n")

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
            DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/tasks/${TASK_PATH}/cuda)
            
    # Copy shared libraries to the build directory for development use
    add_custom_command(TARGET ${TASK_NAME}_cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/infinite_ml/tasks/${TASK_PATH}/cuda/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TASK_NAME}_cuda> ${CMAKE_SOURCE_DIR}/infinite_ml/tasks/${TASK_PATH}/cuda/
        COMMENT "Copying shared library to source directory for development"
    )
    
    add_custom_command(TARGET addition POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/infinite_ml/tasks/${TASK_PATH}/cuda/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:addition> ${CMAKE_SOURCE_DIR}/infinite_ml/tasks/${TASK_PATH}/cuda/
        COMMENT "Copying Python module to source directory for development"
    )
else()
    # Create a no-op Python package when CUDA is not available
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py "# Python package marker\ndef addition():\n    return {'success': False, 'message': '', 'error_message': 'CUDA support not enabled.'}\n")
    
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
            DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/tasks/${TASK_PATH}/cuda)
endif()
