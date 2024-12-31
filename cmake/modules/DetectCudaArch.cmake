# Function to detect CUDA architecture
function(detect_cuda_arch out_var)
    # Skip detection if CUDA architectures are already defined
    if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED CUDA_ARCHITECTURES)
        if(DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(${out_var} ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
        else()
            set(${out_var} ${CUDA_ARCHITECTURES} PARENT_SCOPE)
        endif()
        return()
    endif()

    # Create a simple cuda program to detect architecture
    set(_cuda_detect_code "
        #include <stdio.h>
        #include <cuda_runtime.h>
        int main() {
            int count = 0;
            if (cudaSuccess != cudaGetDeviceCount(&count)) {
                printf(\"75\"); // Default if can't get device count
                return 1;
            }
            if (count == 0) {
                printf(\"75\"); // Default if no CUDA devices
                return 0;
            }
            
            cudaDeviceProp prop;
            if (cudaSuccess != cudaGetDeviceProperties(&prop, 0)) {
                printf(\"75\"); // Default if can't get properties
                return 1;
            }
            
            int major = prop.major;
            int minor = prop.minor;
            printf(\"%d%d\", major, minor);
            return 0;
        }
    ")
    
    # Write the detection program to a file
    set(_cuda_detect_file ${CMAKE_BINARY_DIR}/detect_cuda_arch.cu)
    file(WRITE ${_cuda_detect_file} "${_cuda_detect_code}")
    
    # Attempt to compile and run the program
    try_run(
        _cuda_run_result _cuda_compile_result
        ${CMAKE_BINARY_DIR} ${_cuda_detect_file}
        CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}
        LINK_LIBRARIES ${CUDA_LIBRARIES}
        RUN_OUTPUT_VARIABLE _cuda_run_output
    )
    
    # Set the detected architecture or default if detection fails
    if(_cuda_compile_result AND _cuda_run_result EQUAL 0 AND _cuda_run_output)
        message(STATUS "Detected CUDA architecture: ${_cuda_run_output}")
        set(${out_var} ${_cuda_run_output} PARENT_SCOPE)
    else()
        # Default to a common architecture if detection fails
        set(${out_var} "75" PARENT_SCOPE)
        message(STATUS "Failed to detect CUDA architecture, using default architecture 75")
    endif()
endfunction()