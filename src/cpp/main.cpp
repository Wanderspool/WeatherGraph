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
#include <cmath>
#include <cstring>
#include <optional>
#include <random>
#include <utility>

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

// ── Perturbation Engine ──────────────────────────────────────────────────────
// High-performance Gaussian noise generator with per-channel scale control.
// Uses std::mt19937_64 for reproducible results when a fixed seed is supplied.

class PerturbationEngine {
    std::mt19937_64 rng_;
    std::normal_distribution<float> normal_{0.0f, 1.0f};

public:
    /// Construct with an explicit seed.  seed==0 uses std::random_device.
    explicit PerturbationEngine(uint64_t seed)
        : rng_(seed == 0 ? std::random_device{}() : seed) {}

    /// Apply additive Gaussian noise in-place.
    /// channel_scales[channels]: per-channel σ.  0.0 → skip (static channel).
    void perturb(float* buffer, size_t nodes, size_t channels,
                 const float* channel_scales) {
        for (size_t n = 0; n < nodes; ++n) {
            for (size_t c = 0; c < channels; ++c) {
                float sigma = channel_scales[c];
                if (sigma > 0.0f) {
                    buffer[n * channels + c] += sigma * normal_(rng_);
                }
            }
        }
    }
};

// ── Ensemble Result ──────────────────────────────────────────────────────────
// Returned to Python via zero-copy pybind11 array views.

struct EnsembleResult {
    py::array_t<float> mean;        // [agg_steps, nodes, 78]
    py::array_t<float> std_dev;     // [agg_steps, nodes, 78]
    py::dict probabilities;         // rule_name → py::array_t<float>[agg_steps, nodes]
    int total_members;
    int total_steps;
    std::vector<int> aggregated_step_indices;
};

// ── Threshold Rule (internal) ────────────────────────────────────────────────

struct ThresholdRule {
    std::string name;
    int channel_idx;
    float value;
    bool is_greater;  // true: ">", false: "<"
};

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
    std::unique_ptr<Ort::Session> constraints_session;
    std::string constraints_input_name;
    std::string constraints_output_name;

public:
    WeatherGraphModel(const std::string& model_path,
                      int intra_op_threads = 1,
                      bool disable_cpu_mem_arena = false,
                      bool disable_mem_pattern = false,
                      const std::string& execution_provider = "cpu",
                      int execution_device_id = 0,
                      std::uint64_t execution_memory_limit = 0,
                      bool disable_cpu_ep_fallback = false,
                      const ProviderOptions& execution_provider_options = {},
                      const std::string& constraints_model_path = "")
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

        if (!constraints_model_path.empty()) {
            try {
                constraints_session = std::make_unique<Ort::Session>(env, constraints_model_path.c_str(), session_options);
                auto constraints_input_name_ptr = constraints_session->GetInputNameAllocated(0, allocator);
                constraints_input_name = constraints_input_name_ptr.get();
                auto constraints_output_name_ptr = constraints_session->GetOutputNameAllocated(0, allocator);
                constraints_output_name = constraints_output_name_ptr.get();
                std::cout << "Constraints Model loaded. Input: " << constraints_input_name << ", Output: " << constraints_output_name << std::endl;
            } catch (const Ort::Exception& ex) {
                throw std::runtime_error(std::string("Failed to load constraints model: ") + ex.what());
            }
        }
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

        // Data Sanitization: Replace NaN and Inf with 0.0f
        for (ssize_t i = 0; i < buf.size; ++i) {
            if (std::isnan(input_ptr[i]) || std::isinf(input_ptr[i])) {
                input_ptr[i] = 0.0f;
            }
        }

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

        if (constraints_session) {
            // Apply hard constraints inplace using the output tensor as input
            const char* c_input_names[] = {constraints_input_name.c_str()};
            const char* c_output_names[] = {constraints_output_name.c_str()};
            
            Ort::Value constraints_output_tensor = Ort::Value::CreateTensor<float>(
                memory_info, output_ptr, out_buf.size, output_shape.data(), output_shape.size());

            constraints_session->Run(Ort::RunOptions{nullptr}, c_input_names, &output_tensor, 1, c_output_names, &constraints_output_tensor, 1);
        }

        return output_array;
    }

    // ── Ensemble Inference ────────────────────────────────────────────────────
    // O(1)-memory ensemble prediction with Welford online aggregation,
    // per-step Gaussian perturbation, and threshold probability maps.
    //
    // Memory: mean[agg×N×78] + M2[agg×N×78] + buf_in[N×78] + buf_out[N×78]
    //         + prob_counters[rules×agg×N].
    // Does NOT depend on `members`.

    EnsembleResult predict_ensemble(
            py::array_t<float> initial_state,
            int steps,
            int members,
            py::array_t<float> channel_scales,
            py::list threshold_channels,
            py::list threshold_values,
            py::list threshold_ops,
            py::list threshold_names,
            py::list aggregate_steps_list,
            uint64_t seed) {

        // ── 1. Input validation ──────────────────────────────────────────────
        py::buffer_info state_buf = initial_state.request();
        if (state_buf.ndim != 3 || state_buf.shape[0] != 1 ||
            state_buf.shape[2] != output_shape[2]) {
            throw std::invalid_argument(
                "initial_state must have shape [1, nodes, " +
                std::to_string(output_shape[2]) + "].");
        }

        // C-contiguity check
        {
            ssize_t expected = sizeof(float);
            for (int i = state_buf.ndim - 1; i >= 0; --i) {
                if (state_buf.strides[i] != expected) {
                    throw std::runtime_error(
                        "initial_state must be C-contiguous.");
                }
                expected *= state_buf.shape[i];
            }
        }

        if (members < 1) {
            throw std::invalid_argument("members must be >= 1.");
        }
        if (steps < 1) {
            throw std::invalid_argument("steps must be >= 1.");
        }

        py::buffer_info scales_buf = channel_scales.request();
        if (scales_buf.ndim != 1 ||
            scales_buf.shape[0] != output_shape[2]) {
            throw std::invalid_argument(
                "channel_scales must have shape [" +
                std::to_string(output_shape[2]) + "].");
        }
        const float* scales_ptr =
            static_cast<const float*>(scales_buf.ptr);

        const size_t num_nodes =
            static_cast<size_t>(state_buf.shape[1]);
        const size_t num_channels =
            static_cast<size_t>(output_shape[2]);
        const size_t state_size = num_nodes * num_channels;

        // ── 2. Aggregate step set ────────────────────────────────────────────
        std::vector<int> agg_steps;
        if (aggregate_steps_list.size() == 0) {
            agg_steps.resize(steps);
            for (int s = 0; s < steps; ++s) agg_steps[s] = s;
        } else {
            for (auto item : aggregate_steps_list) {
                int s = item.cast<int>();
                if (s < 0 || s >= steps) {
                    throw std::invalid_argument(
                        "aggregate_steps values must be in [0, steps).");
                }
                agg_steps.push_back(s);
            }
            std::sort(agg_steps.begin(), agg_steps.end());
        }
        const size_t agg_count = agg_steps.size();

        // Build fast lookup: step → agg_idx (-1 if not aggregated)
        std::vector<int> step_to_agg(steps, -1);
        for (size_t a = 0; a < agg_count; ++a) {
            step_to_agg[static_cast<size_t>(agg_steps[a])] =
                static_cast<int>(a);
        }

        // ── 3. Parse threshold rules ─────────────────────────────────────────
        std::vector<ThresholdRule> rules;
        {
            size_t n_rules = threshold_channels.size();
            if (threshold_values.size() != n_rules ||
                threshold_ops.size() != n_rules ||
                threshold_names.size() != n_rules) {
                throw std::invalid_argument(
                    "threshold_channels, threshold_values, threshold_ops, "
                    "and threshold_names must have the same length.");
            }
            for (size_t r = 0; r < n_rules; ++r) {
                ThresholdRule rule;
                rule.name = threshold_names[r].cast<std::string>();
                rule.channel_idx = threshold_channels[r].cast<int>();
                rule.value = threshold_values[r].cast<float>();
                std::string op = threshold_ops[r].cast<std::string>();
                if (op == ">") {
                    rule.is_greater = true;
                } else if (op == "<") {
                    rule.is_greater = false;
                } else {
                    throw std::invalid_argument(
                        "threshold op must be '>' or '<', got '" + op + "'.");
                }
                if (rule.channel_idx < 0 ||
                    static_cast<size_t>(rule.channel_idx) >= num_channels) {
                    throw std::invalid_argument(
                        "threshold channel_idx out of range.");
                }
                rules.push_back(std::move(rule));
            }
        }
        const size_t n_rules = rules.size();

        // ── 4. One-time allocation ───────────────────────────────────────────
        // mean_accum and std_dev are returned to Python as numpy arrays.
        py::array_t<float> mean_array(
            {static_cast<ssize_t>(agg_count),
             static_cast<ssize_t>(num_nodes),
             static_cast<ssize_t>(num_channels)});
        py::array_t<float> std_dev_array(
            {static_cast<ssize_t>(agg_count),
             static_cast<ssize_t>(num_nodes),
             static_cast<ssize_t>(num_channels)});

        float* mean_ptr =
            static_cast<float*>(mean_array.request().ptr);
        float* std_dev_ptr =
            static_cast<float*>(std_dev_array.request().ptr);

        const size_t agg_state_size = agg_count * state_size;
        std::memset(mean_ptr, 0, agg_state_size * sizeof(float));

        // M2 accumulator (temporary, freed after finalization)
        std::vector<float> M2(agg_state_size, 0.0f);

        // Probability counters: [n_rules][agg_count * num_nodes]
        std::vector<std::vector<int>> prob_counters(
            n_rules,
            std::vector<int>(agg_count * num_nodes, 0));

        // Work buffers (reused across all ensemble members)
        std::vector<float> buf_in(state_size);
        std::vector<float> buf_out_storage(state_size);

        const float* initial_ptr =
            static_cast<const float*>(state_buf.ptr);

        // ONNX tensor shapes (batch dim = 1)
        std::vector<int64_t> onnx_input_shape = {
            1,
            static_cast<int64_t>(num_nodes),
            static_cast<int64_t>(num_channels)};

        const char* in_names[] = {input_name.c_str()};
        const char* out_names[] = {output_name.c_str()};

        // ── 5. PRNG ──────────────────────────────────────────────────────────
        PerturbationEngine prng(seed);

        // ── 6. Ensemble loop ─────────────────────────────────────────────────
        for (int m = 1; m <= members; ++m) {
            // 6a. Copy initial state and apply initial perturbation
            std::memcpy(buf_in.data(), initial_ptr,
                        state_size * sizeof(float));
            prng.perturb(buf_in.data(), num_nodes, num_channels,
                         scales_ptr);

            // NaN/Inf sanitize after perturbation
            for (size_t i = 0; i < state_size; ++i) {
                if (std::isnan(buf_in[i]) || std::isinf(buf_in[i])) {
                    buf_in[i] = 0.0f;
                }
            }

            for (int step = 0; step < steps; ++step) {
                // 6b. ONNX inference: buf_in → buf_out
                Ort::Value onnx_input = Ort::Value::CreateTensor<float>(
                    memory_info,
                    buf_in.data(),
                    state_size,
                    onnx_input_shape.data(),
                    onnx_input_shape.size());

                Ort::Value onnx_output = Ort::Value::CreateTensor<float>(
                    memory_info,
                    buf_out_storage.data(),
                    state_size,
                    output_shape.data(),
                    output_shape.size());

                session->Run(
                    Ort::RunOptions{nullptr},
                    in_names, &onnx_input, 1,
                    out_names, &onnx_output, 1);

                // Apply constraints model if present
                if (constraints_session) {
                    const char* c_in[] =
                        {constraints_input_name.c_str()};
                    const char* c_out[] =
                        {constraints_output_name.c_str()};

                    Ort::Value c_input =
                        Ort::Value::CreateTensor<float>(
                            memory_info,
                            buf_out_storage.data(),
                            state_size,
                            output_shape.data(),
                            output_shape.size());

                    Ort::Value c_output =
                        Ort::Value::CreateTensor<float>(
                            memory_info,
                            buf_out_storage.data(),
                            state_size,
                            output_shape.data(),
                            output_shape.size());

                    constraints_session->Run(
                        Ort::RunOptions{nullptr},
                        c_in, &c_input, 1,
                        c_out, &c_output, 1);
                }

                float* out_ptr = buf_out_storage.data();

                // 6c. Welford update + threshold counting
                //     (only for aggregate steps)
                int agg_idx = step_to_agg[static_cast<size_t>(step)];
                if (agg_idx >= 0) {
                    size_t agg_offset =
                        static_cast<size_t>(agg_idx) * state_size;

                    for (size_t i = 0; i < state_size; ++i) {
                        float val = out_ptr[i];
                        float delta =
                            val - mean_ptr[agg_offset + i];
                        mean_ptr[agg_offset + i] +=
                            delta / static_cast<float>(m);
                        float delta2 =
                            val - mean_ptr[agg_offset + i];
                        M2[agg_offset + i] += delta * delta2;
                    }

                    // Threshold counting
                    size_t prob_offset =
                        static_cast<size_t>(agg_idx) * num_nodes;
                    for (size_t r = 0; r < n_rules; ++r) {
                        int ch = rules[r].channel_idx;
                        float thresh = rules[r].value;
                        bool is_gt = rules[r].is_greater;

                        for (size_t node = 0; node < num_nodes;
                             ++node) {
                            float v =
                                out_ptr[node * num_channels +
                                        static_cast<size_t>(ch)];
                            bool hit = is_gt ? (v > thresh)
                                             : (v < thresh);
                            if (hit) {
                                prob_counters[r]
                                    [prob_offset + node] += 1;
                            }
                        }
                    }
                }

                // 6d. Prepare for next step: copy output → input
                //     and apply per-step perturbation
                std::memcpy(buf_in.data(), out_ptr,
                            state_size * sizeof(float));

                if (step < steps - 1) {
                    prng.perturb(buf_in.data(), num_nodes,
                                 num_channels, scales_ptr);

                    // NaN/Inf sanitize
                    for (size_t i = 0; i < state_size; ++i) {
                        if (std::isnan(buf_in[i]) ||
                            std::isinf(buf_in[i])) {
                            buf_in[i] = 0.0f;
                        }
                    }
                }
            }  // step loop
        }  // member loop

        // ── 7. Finalization ──────────────────────────────────────────────────
        // std_dev = sqrt(M2 / members)
        float inv_members = 1.0f / static_cast<float>(members);
        for (size_t i = 0; i < agg_state_size; ++i) {
            float variance = M2[i] * inv_members;
            std_dev_ptr[i] = std::sqrt(
                std::max(variance, 0.0f));
        }

        // Probability maps: counters / members
        py::dict prob_dict;
        for (size_t r = 0; r < n_rules; ++r) {
            py::array_t<float> prob_array(
                {static_cast<ssize_t>(agg_count),
                 static_cast<ssize_t>(num_nodes)});
            float* prob_ptr =
                static_cast<float*>(prob_array.request().ptr);

            for (size_t i = 0; i < agg_count * num_nodes; ++i) {
                prob_ptr[i] =
                    static_cast<float>(prob_counters[r][i]) *
                    inv_members;
            }
            prob_dict[py::cast(rules[r].name)] = prob_array;
        }

        // ── 8. Build result ──────────────────────────────────────────────────
        EnsembleResult result;
        result.mean = std::move(mean_array);
        result.std_dev = std::move(std_dev_array);
        result.probabilities = std::move(prob_dict);
        result.total_members = members;
        result.total_steps = steps;
        result.aggregated_step_indices = std::move(agg_steps);

        return result;
    }
};

PYBIND11_MODULE(weathergraph_backend, m) {
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error(std::string("ONNX Runtime Error: ") + e.what());
        } catch (const std::bad_alloc& e) {
            throw std::runtime_error(std::string("Memory Allocation Error (OOM): ") + e.what());
        } catch (const std::out_of_range& e) {
            throw std::out_of_range(std::string("Out of Range Error: ") + e.what());
        }
    });

    py::class_<EnsembleResult>(m, "EnsembleResult")
        .def_readonly("mean", &EnsembleResult::mean)
        .def_readonly("std_dev", &EnsembleResult::std_dev)
        .def_readonly("probabilities", &EnsembleResult::probabilities)
        .def_readonly("total_members", &EnsembleResult::total_members)
        .def_readonly("total_steps", &EnsembleResult::total_steps)
        .def_readonly("aggregated_step_indices",
                       &EnsembleResult::aggregated_step_indices);

    py::class_<WeatherGraphModel>(m, "WeatherGraphEngine")
    .def(py::init<const std::string&, int, bool, bool, const std::string&, int, std::uint64_t, bool, const ProviderOptions&, const std::string&>(),
         py::arg("model_path"),
         py::arg("intra_op_threads") = 1,
         py::arg("disable_cpu_mem_arena") = false,
         py::arg("disable_mem_pattern") = false,
         py::arg("execution_provider") = "cpu",
        py::arg("execution_device_id") = 0,
        py::arg("execution_memory_limit") = 0,
        py::arg("disable_cpu_ep_fallback") = false,
        py::arg("execution_provider_options") = ProviderOptions{},
        py::arg("constraints_model_path") = "")
    .def("output_shape", &WeatherGraphModel::get_output_shape)
    .def("cpu_mem_arena_enabled", &WeatherGraphModel::is_cpu_mem_arena_enabled)
    .def("mem_pattern_enabled", &WeatherGraphModel::is_mem_pattern_enabled)
    .def("execution_provider", &WeatherGraphModel::get_execution_provider)
    .def("cpu_ep_fallback_enabled", &WeatherGraphModel::is_cpu_ep_fallback_enabled)
    .def("predict", &WeatherGraphModel::predict)
    .def("predict_ensemble", &WeatherGraphModel::predict_ensemble,
         py::arg("initial_state"),
         py::arg("steps") = 40,
         py::arg("members") = 50,
         py::arg("channel_scales"),
         py::arg("threshold_channels") = py::list{},
         py::arg("threshold_values") = py::list{},
         py::arg("threshold_ops") = py::list{},
         py::arg("threshold_names") = py::list{},
         py::arg("aggregate_steps") = py::list{},
         py::arg("seed") = static_cast<uint64_t>(0));
}
