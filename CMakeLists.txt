cmake_minimum_required(VERSION 3.15)
project(weathergraph_backend)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pybind11 is resolved from the active Python environment.
# Pass the cmake config dir at configure time:
#   cmake -S . -B build -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
# pybind11_add_module handles Python include paths, PREFIX, SUFFIX, and
# -fvisibility=hidden automatically — no hardcoded /root/ paths required.
find_package(pybind11 CONFIG REQUIRED)

set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-sdk")

pybind11_add_module(weathergraph_backend src/cpp/main.cpp)

target_include_directories(weathergraph_backend PRIVATE
    "${ONNXRUNTIME_DIR}/include"
)

target_link_libraries(weathergraph_backend PRIVATE
    "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
)

# Optional: Link against mimalloc if available to reduce memory fragmentation
find_package(mimalloc QUIET)
if(mimalloc_FOUND)
    target_link_libraries(weathergraph_backend PRIVATE mimalloc)
    message(STATUS "Found mimalloc, linking to weathergraph_backend")
else()
    message(STATUS "mimalloc not found. Consider using LD_PRELOAD=/usr/lib/libmimalloc.so at runtime.")
endif()
