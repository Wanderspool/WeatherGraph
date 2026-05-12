#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

namespace {

using ProviderOptions = std::unordered_map<std::string, std::string>;

std::string normalize_execution_provider(std::string execution_provider) {
    std::transform(
        execution_provider.begin(),
        execution_provider.end(),
        execution_provider.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        }
    );

    if (execution_provider == "nvidia") {
        return "cuda";
    }
    if (execution_provider == "trt") {
        return "tensorrt";
    }
    if (execution_provider == "amd") {
        return "rocm";
    }
    if (execution_provider == "intel") {
        return "openvino";
    }

    return execution_provider;
}

bool is_accelerator_provider(const std::string& execution_provider) {
    return execution_provider == "cuda"
        || execution_provider == "tensorrt"
        || execution_provider == "rocm"
        || execution_provider == "openvino";
}

ProviderOptions merge_provider_options(const std::string& execution_provider,
                                       int execution_device_id,
                                       std::uint64_t execution_memory_limit,
                                       const ProviderOptions& execution_provider_options) {
    ProviderOptions merged_options = execution_provider_options;

    if (is_accelerator_provider(execution_provider) && merged_options.find("device_id") == merged_options.end()) {
        merged_options["device_id"] = std::to_string(execution_device_id);
    }

    if (execution_memory_limit > 0) {
        if ((execution_provider == "cuda" || execution_provider == "rocm")
            && merged_options.find("gpu_mem_limit") == merged_options.end()) {
            merged_options["gpu_mem_limit"] = std::to_string(execution_memory_limit);
        }
        if (execution_provider == "tensorrt"
            && merged_options.find("trt_max_workspace_size") == merged_options.end()) {
            merged_options["trt_max_workspace_size"] = std::to_string(execution_memory_limit);
        }
    }

    return merged_options;
}

struct ProviderOptionArrays {
    std::vector<std::string> keys;
    std::vector<std::string> values;
    std::vector<const char*> key_ptrs;
    std::vector<const char*> value_ptrs;
};

ProviderOptionArrays build_provider_option_arrays(const ProviderOptions& execution_provider_options) {
    ProviderOptionArrays option_arrays;
    option_arrays.keys.reserve(execution_provider_options.size());
    option_arrays.values.reserve(execution_provider_options.size());
    option_arrays.key_ptrs.reserve(execution_provider_options.size());
    option_arrays.value_ptrs.reserve(execution_provider_options.size());

    for (const auto& [key, value] : execution_provider_options) {
        option_arrays.keys.push_back(key);
        option_arrays.values.push_back(value);
    }

    for (size_t index = 0; index < option_arrays.keys.size(); ++index) {
        option_arrays.key_ptrs.push_back(option_arrays.keys[index].c_str());
        option_arrays.value_ptrs.push_back(option_arrays.values[index].c_str());
    }

    return option_arrays;
}

void append_cuda_provider(Ort::SessionOptions& session_options,
                          const ProviderOptions& execution_provider_options) {
    auto& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));

    try {
        if (!execution_provider_options.empty()) {
            auto option_arrays = build_provider_option_arrays(execution_provider_options);
            Ort::ThrowOnError(
                api.UpdateCUDAProviderOptions(
                    cuda_options,
                    option_arrays.key_ptrs.data(),
                    option_arrays.value_ptrs.data(),
                    option_arrays.key_ptrs.size()
                )
            );
        }
        session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    } catch (...) {
        api.ReleaseCUDAProviderOptions(cuda_options);
        throw;
    }

    api.ReleaseCUDAProviderOptions(cuda_options);
}

void append_tensorrt_provider(Ort::SessionOptions& session_options,
                              const ProviderOptions& execution_provider_options) {
    auto& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));

    try {
        if (!execution_provider_options.empty()) {
            auto option_arrays = build_provider_option_arrays(execution_provider_options);
            Ort::ThrowOnError(
                api.UpdateTensorRTProviderOptions(
                    tensorrt_options,
                    option_arrays.key_ptrs.data(),
                    option_arrays.value_ptrs.data(),
                    option_arrays.key_ptrs.size()
                )
            );
        }
        session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
    } catch (...) {
        api.ReleaseTensorRTProviderOptions(tensorrt_options);
        throw;
    }

    api.ReleaseTensorRTProviderOptions(tensorrt_options);
}

void append_rocm_provider(Ort::SessionOptions& session_options,
                          const ProviderOptions& execution_provider_options) {
    auto& api = Ort::GetApi();
    OrtROCMProviderOptions* rocm_options = nullptr;
    Ort::ThrowOnError(api.CreateROCMProviderOptions(&rocm_options));

    try {
        if (!execution_provider_options.empty()) {
            auto option_arrays = build_provider_option_arrays(execution_provider_options);
            Ort::ThrowOnError(
                api.UpdateROCMProviderOptions(
                    rocm_options,
                    option_arrays.key_ptrs.data(),
                    option_arrays.value_ptrs.data(),
                    option_arrays.key_ptrs.size()
                )
            );
        }
        session_options.AppendExecutionProvider_ROCM(*rocm_options);
    } catch (...) {
        api.ReleaseROCMProviderOptions(rocm_options);
        throw;
    }

    api.ReleaseROCMProviderOptions(rocm_options);
}

void append_openvino_provider(Ort::SessionOptions& session_options,
                              const ProviderOptions& execution_provider_options) {
    session_options.AppendExecutionProvider_OpenVINO_V2(execution_provider_options);
}

void append_execution_provider(Ort::SessionOptions& session_options,
                               const std::string& execution_provider,
                               int execution_device_id,
                               std::uint64_t execution_memory_limit,
                               const ProviderOptions& execution_provider_options) {
    if (execution_provider == "cpu") {
        if (!execution_provider_options.empty()) {
            throw std::invalid_argument(
                "execution_provider_options are only valid for accelerator execution providers."
            );
        }
        return;
    }

    const ProviderOptions merged_options = merge_provider_options(
        execution_provider,
        execution_device_id,
        execution_memory_limit,
        execution_provider_options
    );

    if (execution_provider == "cuda") {
        append_cuda_provider(session_options, merged_options);
        return;
    }
    if (execution_provider == "tensorrt") {
        append_tensorrt_provider(session_options, merged_options);
        return;
    }
    if (execution_provider == "rocm") {
        append_rocm_provider(session_options, merged_options);
        return;
    }
    if (execution_provider == "openvino") {
        append_openvino_provider(session_options, merged_options);
        return;
    }

    throw std::invalid_argument(
        "execution_provider must be one of 'cpu', 'cuda', 'tensorrt', 'rocm', or 'openvino'."
    );
}

}  // namespace

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
    std::string execution_provider_name;
    int execution_device_id;
    std::uint64_t execution_memory_limit;
    bool cpu_ep_fallback_enabled;

public:
    WeatherGraphModel(const std::string& model_path,
                      int intra_op_threads = 1,
                      bool disable_cpu_mem_arena = false,
                      bool disable_mem_pattern = false,
                      const std::string& execution_provider = "cpu",
                      int execution_device_id = 0,
                      std::uint64_t execution_memory_limit = 0,
                      bool disable_cpu_ep_fallback = false,
                      const ProviderOptions& execution_provider_options = {})
        : env(ORT_LOGGING_LEVEL_WARNING, "WeatherGraphEnv"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          cpu_mem_arena_enabled(!disable_cpu_mem_arena),
          mem_pattern_enabled(!disable_mem_pattern),
          execution_provider_name(normalize_execution_provider(execution_provider)),
          execution_device_id(execution_device_id),
          execution_memory_limit(execution_memory_limit),
          cpu_ep_fallback_enabled(!disable_cpu_ep_fallback) {
        if (execution_device_id < 0) {
            throw std::invalid_argument("execution_device_id must be >= 0.");
        }
        if (disable_cpu_ep_fallback && execution_provider_name == "cpu") {
            throw std::invalid_argument("disable_cpu_ep_fallback requires a non-CPU execution_provider.");
        }

        
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

        if (disable_cpu_ep_fallback) {
            session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
        }

        try {
            append_execution_provider(
                session_options,
                execution_provider_name,
                execution_device_id,
                execution_memory_limit,
                execution_provider_options
            );
        } catch (const Ort::Exception& ex) {
            throw std::runtime_error(
                std::string("Failed to enable execution provider '") + execution_provider_name +
                "': " + ex.what() +
                ". Ensure the ONNX Runtime build and provider shared libraries for this accelerator are available."
            );
        }

        try {
            session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        } catch (const Ort::Exception& ex) {
            if (execution_provider_name != "cpu") {
                throw std::runtime_error(
                    std::string("Failed to create ") + execution_provider_name +
                    "-backed ONNX Runtime session: " + ex.what() +
                    ". Verify that the matching execution-provider shared libraries and device runtime stack are available."
                );
            }
            throw;
        }

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = session->GetInputNameAllocated(0, allocator);
        input_name = input_name_ptr.get();
        auto output_name_ptr = session->GetOutputNameAllocated(0, allocator);
        output_name = output_name_ptr.get();

        auto output_type_info = session->GetOutputTypeInfo(0);
        auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
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

    std::string get_execution_provider() const {
        return execution_provider_name;
    }

    bool is_cpu_ep_fallback_enabled() const {
        return cpu_ep_fallback_enabled;
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
    .def(py::init<const std::string&, int, bool, bool, const std::string&, int, std::uint64_t, bool, const ProviderOptions&>(),
         py::arg("model_path"),
         py::arg("intra_op_threads") = 1,
         py::arg("disable_cpu_mem_arena") = false,
         py::arg("disable_mem_pattern") = false,
         py::arg("execution_provider") = "cpu",
        py::arg("execution_device_id") = 0,
        py::arg("execution_memory_limit") = 0,
        py::arg("disable_cpu_ep_fallback") = false,
        py::arg("execution_provider_options") = ProviderOptions{})
    .def("output_shape", &WeatherGraphModel::get_output_shape)
    .def("cpu_mem_arena_enabled", &WeatherGraphModel::is_cpu_mem_arena_enabled)
    .def("mem_pattern_enabled", &WeatherGraphModel::is_mem_pattern_enabled)
    .def("execution_provider", &WeatherGraphModel::get_execution_provider)
    .def("cpu_ep_fallback_enabled", &WeatherGraphModel::is_cpu_ep_fallback_enabled)
        .def("predict", &WeatherGraphModel::predict);
}
