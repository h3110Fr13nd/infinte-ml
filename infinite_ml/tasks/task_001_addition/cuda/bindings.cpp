#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "addition.cuh"
#include "logger.h"
#include <vector>
#include <string>

namespace py = pybind11;

// Wrapper function for CUDA vector addition that accepts numpy arrays
PerformanceResult vectorAddCudaPython(
    py::array_t<float> pyA, 
    py::array_t<float> pyB, 
    py::array_t<float> pyC, 
    int threadsPerBlock) 
{
    // Get buffers for the numpy arrays
    py::buffer_info bufA = pyA.request();
    py::buffer_info bufB = pyB.request();
    py::buffer_info bufC = pyC.request();
    
    // Check dimensions
    if (bufA.ndim != 1 || bufB.ndim != 1 || bufC.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    
    // Check sizes match
    size_t numElements = bufA.shape[0];
    if (bufB.shape[0] != numElements || bufC.shape[0] != numElements) {
        throw std::runtime_error("Input shapes must match");
    }
    
    // Get pointers to the data
    float* ptrA = static_cast<float*>(bufA.ptr);
    float* ptrB = static_cast<float*>(bufB.ptr);
    float* ptrC = static_cast<float*>(bufC.ptr);
    
    // Call the CUDA function
    return vectorAddCuda(ptrA, ptrB, ptrC, numElements, threadsPerBlock);
}

// Wrapper function for CPU vector addition that accepts numpy arrays
PerformanceResult vectorAddCPUPython(
    py::array_t<float> pyA, 
    py::array_t<float> pyB, 
    py::array_t<float> pyC)
{
    // Get buffers for the numpy arrays
    py::buffer_info bufA = pyA.request();
    py::buffer_info bufB = pyB.request();
    py::buffer_info bufC = pyC.request();
    
    // Check dimensions
    if (bufA.ndim != 1 || bufB.ndim != 1 || bufC.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    
    // Check sizes match
    size_t numElements = bufA.shape[0];
    if (bufB.shape[0] != numElements || bufC.shape[0] != numElements) {
        throw std::runtime_error("Input shapes must match");
    }
    
    // Get pointers to the data
    float* ptrA = static_cast<float*>(bufA.ptr);
    float* ptrB = static_cast<float*>(bufB.ptr);
    float* ptrC = static_cast<float*>(bufC.ptr);
    
    // Call the CPU function
    return vectorAddCPU(ptrA, ptrB, ptrC, numElements);
}

// Function to verify result from Python
void verifyResultPython(
    py::array_t<float> pyA, 
    py::array_t<float> pyB, 
    py::array_t<float> pyC)
{
    // Get buffers for the numpy arrays
    py::buffer_info bufA = pyA.request();
    py::buffer_info bufB = pyB.request();
    py::buffer_info bufC = pyC.request();
    
    // Check dimensions
    if (bufA.ndim != 1 || bufB.ndim != 1 || bufC.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    
    // Check sizes match
    size_t numElements = bufA.shape[0];
    if (bufB.shape[0] != numElements || bufC.shape[0] != numElements) {
        throw std::runtime_error("Input shapes must match");
    }
    
    // Get pointers to the data
    float* ptrA = static_cast<float*>(bufA.ptr);
    float* ptrB = static_cast<float*>(bufB.ptr);
    float* ptrC = static_cast<float*>(bufC.ptr);
    
    // Call the verification function
    verifyResult(ptrA, ptrB, ptrC, numElements);
}

// Python module
PYBIND11_MODULE(addition, m) {
    m.doc() = "CUDA Vector Addition Module"; // Optional module docstring
    
    // Bind the PerformanceResult struct to Python
    py::class_<PerformanceResult>(m, "PerformanceResult")
        .def_readwrite("name", &PerformanceResult::name)
        .def_readwrite("blocks", &PerformanceResult::blocks)
        .def_readwrite("threads_per_block", &PerformanceResult::threadsPerBlock)
        .def_readwrite("time_ms", &PerformanceResult::timeMs)
        .def_readwrite("num_additions_performed", &PerformanceResult::numAdditionsPerformed)
        .def_readwrite("device_name", &PerformanceResult::deviceName)
        .def("__repr__", [](const PerformanceResult &pr) {
            return "<PerformanceResult: " + pr.name + 
                   ", Blocks=" + std::to_string(pr.blocks) + 
                   ", Threads=" + std::to_string(pr.threadsPerBlock) + 
                   ", Time=" + std::to_string(pr.timeMs) + "ms>";
        });
    
    // Bind LogLevel enum to Python
    py::enum_<LogLevel>(m, "LogLevel")
        .value("DEBUG", DEBUG)
        .value("INFO", INFO)
        .value("WARNING", WARNING)
        .value("ERROR", ERROR)
        .value("CRITICAL", CRITICAL)
        .export_values();
    
    // Bind Logger class to Python
    py::class_<Logger>(m, "Logger")
        .def(py::init<const std::string &>())
        .def("log", &Logger::log)
        .def("debug", &Logger::debug)
        .def("info", &Logger::info)
        .def("warning", &Logger::warning)
        .def("error", &Logger::error)
        .def("critical", &Logger::critical);
    
    // Expose the vector addition functions to Python
    m.def("vector_add_cuda", &vectorAddCudaPython, 
          "Perform vector addition on the GPU", 
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("threads_per_block"));
    
    m.def("vector_add_cpu", &vectorAddCPUPython, 
          "Perform vector addition on the CPU", 
          py::arg("a"), py::arg("b"), py::arg("c"));
    
    m.def("verify_result", &verifyResultPython, 
          "Verify the result of vector addition", 
          py::arg("a"), py::arg("b"), py::arg("c"));
    
    m.def("initialize_logger", &initializeLogger, 
          "Initialize the global logger", 
          py::arg("log_filename"));
    
    m.def("cleanup_logger", &cleanupLogger, 
          "Clean up the global logger");
    
    m.def("log_performance_results_cuda", &logPerformanceResultsCuda, 
          "Log CUDA performance results", 
          py::arg("result"));
    
    m.def("log_performance_results_cpu", &logPerformanceResultsCPU, 
          "Log CPU performance results", 
          py::arg("result"));
}
