# Building from Source

This developer guide describes how to configure, compile, and package WeatherGraph's C++ core backend (`weathergraph_backend.so`) from source code.

---

## 1. Compiler Toolchain Requirements

To compile the C++ shared libraries, your host development machine must have the following system dependencies installed:

*   **Compiler**: A toolchain supporting C++17 or newer.
    *   `gcc` (version 9.0 or newer)
    *   `clang` (version 10.0 or newer)
*   **Build System**: `cmake` (version 3.18 or newer) and `make` (or `ninja`).
*   **Python Headers**: `python3-dev` or `python3-devel` matching the version of Python you are using for runtime.
*   **Shared Library Utilities**: `patchelf` (on Linux) is used to embed and locate relative paths (`rpath`) for `libonnxruntime.so` within the compiled Python wrapper.

### Installing Dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake patchelf python3-dev
```

---

## 2. Compilation Mechanism: scikit-build-core

WeatherGraph uses **`scikit-build-core`** as its PEP 517 build backend. When you install the package, `scikit-build-core` parses `pyproject.toml`, invokes `cmake` under the hood to compile the C++ source files in `src/cpp/` using `pybind11`, and places the resulting shared library directly inside the python package.

### Build Layout
During compilation, CMake targets are built and placed as follows:

```text
build/
  └── (CMake compilation artifacts)
weathergraph/
  └── core/
        ├── weathergraph_backend.so      # Compiled pybind11 module
        └── libonnxruntime.so.1.18.0     # Vendored ONNX Runtime library
```

---

## 3. Build & Install Commands

### Development Install (Editable Mode)
For development, install the package in editable mode (`-e`). This compiles the C++ code once and allows you to modify Python files without recompiling.

```bash
pip install -e .
```

### Building Distribution Wheels
To build static wheel binaries (`.whl`) and source archives (`.tar.gz`) for packaging or distribution:

```bash
pip install build
python -m build
```
The compiled wheel file will be saved in the `dist/` directory.

---

## 4. Custom CMake Configurations

You can pass configuration flags to the underlying CMake execution using the `CMAKE_ARGS` environment variable:

### CUDA Support
Enable CUDA GPU execution provider bindings:

```bash
CMAKE_ARGS="-DONNXRUNTIME_CUDA=ON" pip install -e .
```

### Custom ONNX Runtime SDK Path
By default, the CMake build script automatically downloads the ONNX Runtime library archive matching your system architecture. If you are developing on an offline cluster, you can point CMake to a pre-extracted local ONNX Runtime SDK directory:

```bash
CMAKE_ARGS="-DONNXRUNTIME_ROOT=/opt/onnxruntime-sdk-1.18.0" pip install -e .
```

### Debug Build
Compile the C++ code with debug symbols and without optimizations:

```bash
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug" pip install -e .
```
This is essential for profiling C++ sessions or attaching GDB debuggers.