#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

class WeatherGraphModel {
private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info;
    std::string input_name;
    std::string output_name;
    std::vector<int64_t> output_shape;
    bool cpu_mem_arena_enabled;
    bool mem_pattern_enabled;

public:
    WeatherGraphModel(const std::string& model_path,
                      int intra_op_threads = 1,
                      bool disable_cpu_mem_arena = false,
                      bool disable_mem_pattern = false)
        : env(ORT_LOGGING_LEVEL_WARNING, "WeatherGraphEnv"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          cpu_mem_arena_enabled(!disable_cpu_mem_arena),
          mem_pattern_enabled(!disable_mem_pattern) {
        
        Ort::SessionOptions session_options;
        if (intra_op_threads > 0) {
            session_options.SetIntraOpNumThreads(intra_op_threads);
        }
        if (disable_cpu_mem_arena) {
            session_options.DisableCpuMemArena();
        }
        if (disable_mem_pattern) {
            session_options.DisableMemPattern();
        }
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = session->GetInputNameAllocated(0, allocator);
        input_name = input_name_ptr.get();
        auto output_name_ptr = session->GetOutputNameAllocated(0, allocator);
        output_name = output_name_ptr.get();

        auto output_info = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        output_shape = output_info.GetShape();
        for (auto dim : output_shape) {
            if (dim <= 0) {
                throw std::runtime_error("Model output must have a fully defined static shape.");
            }
        }
        
        std::cout << "Model loaded. Input: " << input_name << ", Output: " << output_name << std::endl;
    }

    std::vector<int64_t> get_output_shape() const {
        return output_shape;
    }

    bool is_cpu_mem_arena_enabled() const {
        return cpu_mem_arena_enabled;
    }

    bool is_mem_pattern_enabled() const {
        return mem_pattern_enabled;
    }

    py::array_t<float> predict(py::array_t<float> input_array) {
        py::buffer_info buf = input_array.request();
        
        // Safety Check: Ensure the input array is C-contiguous.
        // For a 3D array [D0, D1, D2], C-contiguous means:
        // strides[2] = sizeof(T)
        // strides[1] = D2 * strides[2]
        // strides[0] = D1 * strides[1]
        
        bool is_contiguous = true;
        ssize_t expected_stride = sizeof(float);
        for (int i = buf.ndim - 1; i >= 0; --i) {
            if (buf.strides[i] != expected_stride) {
                is_contiguous = false;
                break;
            }
            expected_stride *= buf.shape[i];
        }

        if (!is_contiguous) {
            throw std::runtime_error("Input array must be C-contiguous (no strides). Call .copy() in Python if necessary.");
        }

        float* input_ptr = static_cast<float*>(buf.ptr);

        std::vector<int64_t> input_shape;
        for (auto s : buf.shape) input_shape.push_back(s);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_ptr, buf.size, input_shape.data(), input_shape.size());

        std::vector<ssize_t> py_output_shape(output_shape.begin(), output_shape.end());
        py::array_t<float> output_array(py_output_shape);
        py::buffer_info out_buf = output_array.request();
        float* output_ptr = static_cast<float*>(out_buf.ptr);

        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            memory_info, output_ptr, out_buf.size, output_shape.data(), output_shape.size());

        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};

        session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);

        return output_array;
    }
};

PYBIND11_MODULE(weathergraph_backend, m) {
    py::class_<WeatherGraphModel>(m, "WeatherGraphEngine")
    .def(py::init<const std::string&, int, bool, bool>(),
         py::arg("model_path"),
         py::arg("intra_op_threads") = 1,
         py::arg("disable_cpu_mem_arena") = false,
         py::arg("disable_mem_pattern") = false)
    .def("output_shape", &WeatherGraphModel::get_output_shape)
    .def("cpu_mem_arena_enabled", &WeatherGraphModel::is_cpu_mem_arena_enabled)
    .def("mem_pattern_enabled", &WeatherGraphModel::is_mem_pattern_enabled)
        .def("predict", &WeatherGraphModel::predict);
}
