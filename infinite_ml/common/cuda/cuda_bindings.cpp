#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

// Function declarations from cuda_device_info.cu
std::string getCudaDeviceInfo();
int getDeviceCount();

// Function declarations from utils.cu
bool isCudaAvailable();
size_t getDeviceMemory(int deviceId);
void syncDevice();

PYBIND11_MODULE(cuda_utils, m) {
    m.doc() = "CUDA utility functions for infinite_ml";
    
    // Expose functions from cuda_device_info.cu
    m.def("get_device_info", &getCudaDeviceInfo, "Get CUDA device information");
    m.def("get_device_count", &getDeviceCount, "Get number of CUDA devices");
    
    // Expose functions from utils.cu
    m.def("is_cuda_available", &isCudaAvailable, "Check if CUDA is available");
    m.def("get_device_memory", &getDeviceMemory, "Get total memory for a CUDA device",
          py::arg("device_id") = 0);
    m.def("sync_device", &syncDevice, "Synchronize CUDA device");
}
