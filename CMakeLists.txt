cmake_minimum_required(VERSION 3.15)
project(keisler_cpp_backend)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Hardcoded paths based on manual check
set(PYBIND11_INCLUDES 
    "/usr/include/python3.11"
    "/root/keisler_engine/venv/lib/python3.11/site-packages/pybind11/include"
)
message(STATUS "PYBIND11_INCLUDES: ${PYBIND11_INCLUDES}")

# ONNX Runtime paths
set(ONNXRUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-sdk)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_DIR}/lib)

include_directories(${PYBIND11_INCLUDES} ${ONNXRUNTIME_INCLUDE_DIR})

# Python includes
execute_process(COMMAND keisler_engine/venv/bin/python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
                OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories(${PYTHON_INCLUDE_DIR})

add_library(keisler_cpp_backend MODULE src/cpp/main.cpp)

# Link ONNX Runtime
target_link_libraries(keisler_cpp_backend PRIVATE ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)

set_target_properties(keisler_cpp_backend PROPERTIES 
    PREFIX ""
    SUFFIX ".so"
)
