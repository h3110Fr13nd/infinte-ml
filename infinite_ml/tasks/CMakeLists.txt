# Find all task directories and add them as subdirectories
file(GLOB TASK_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*")
foreach(TASK_DIR ${TASK_DIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${TASK_DIR})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TASK_DIR}/cuda/CMakeLists.txt)
            # Configure the task-specific cuda directory
            add_subdirectory(${TASK_DIR}/cuda)
            
            # Create Python package structure for this task
            set(TASK_INIT_CONTENT "# Python package marker\n")
            file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${TASK_DIR}/__init__.py ${TASK_INIT_CONTENT})
            install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${TASK_DIR}/__init__.py
                  DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/tasks/${TASK_DIR})
        endif()
    endif()
endforeach()

# Create Python package structure
set(TASKS_INIT_CONTENT "# Python package marker\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py ${TASKS_INIT_CONTENT})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        DESTINATION ${PYTHON_SITE_PACKAGES}/infinite_ml/tasks)

