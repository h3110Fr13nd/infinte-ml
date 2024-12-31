#include <pybind11/pybind11.h>
#include <string>
#include "hello_world.cuh"

namespace py = pybind11;

PYBIND11_MODULE(hello_world, m) {
    m.doc() = "Hello World CUDA example for infinite-ml";
    
    // Expose the HelloWorldResult struct to Python
    // py::class_<HelloWorldResult>(m, "HelloWorldResult")
    //     .def(py::init<>())
    //     .def_readwrite("success", &HelloWorldResult::success)
    //     .def_readwrite("message", &HelloWorldResult::message)
    //     .def_readwrite("error_message", &HelloWorldResult::errorMessage);
    
    // Expose the sayHelloFromCuda function to Python
    m.def("hello_world", &helloWorldCuda,
          "A simple function that demonstrates CUDA functionality");
}
