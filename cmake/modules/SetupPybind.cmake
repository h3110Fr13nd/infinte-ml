# Function to easily create pybind11 modules for tasks
function(setup_task_binding)
    # Parse arguments
    set(options)
    set(oneValueArgs TARGET_NAME SOURCE TASK_NAME TASK_PATH)
    set(multiValueArgs LINK_LIBS INCLUDE_DIRS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Error checking
    if(NOT ARG_TARGET_NAME)
        message(FATAL_ERROR "setup_task_binding: TARGET_NAME not specified")
        return()
    endif()
    
    if(NOT ARG_SOURCE)
        message(FATAL_ERROR "setup_task_binding: SOURCE not specified")
        return()
    endif()
    
    if(NOT ARG_TASK_PATH)
        message(FATAL_ERROR "setup_task_binding: TASK_PATH not specified")
        return()
    endif()

    # Create Python binding module
    pybind11_add_module(${ARG_TARGET_NAME} ${ARG_SOURCE})
    
    # Set include directories
    target_include_directories(${ARG_TARGET_NAME} PRIVATE 
        ${CMAKE_SOURCE_DIR}/infinite_ml/tasks/${ARG_TASK_PATH}
        ${CMAKE_SOURCE_DIR}/infinite_ml/common/cuda
    )
    
    # If additional include directories are provided
    if(ARG_INCLUDE_DIRS)
        target_include_directories(${ARG_TARGET_NAME} PRIVATE ${ARG_INCLUDE_DIRS})
    endif()
    
    # Link with required libraries
    if(ARG_LINK_LIBS)
        target_link_libraries(${ARG_TARGET_NAME} PRIVATE ${ARG_LINK_LIBS})
    endif()
    
    # Always link with CUDA libraries if CUDA is enabled
    if(USE_CUDA)
        target_link_libraries(${ARG_TARGET_NAME} PRIVATE ${CUDA_LIBRARIES})
    endif()
    
    # Set installation path
    install(TARGETS ${ARG_TARGET_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/tasks/${ARG_TASK_PATH}
        RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/tasks/${ARG_TASK_PATH})
        
    # Add debug information
    message(STATUS "Created binding for task: ${ARG_TARGET_NAME}")
endfunction()