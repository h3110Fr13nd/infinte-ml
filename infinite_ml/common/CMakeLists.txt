# Check if CUDA directory exists
if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
    if(USE_CUDA)
        add_subdirectory(cuda)
    endif()
endif()

# Create Python package structure for common modules
set(COMMON_INIT_CONTENT "# Python package marker\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py ${COMMON_INIT_CONTENT})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/common)
