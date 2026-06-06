# Testing & Validation

WeatherGraph enforces strict quality control through a comprehensive testing framework. All code additions, optimizations, and API adjustments must pass the test suite before being merged.

This guide describes how to run and extend the tests.

---

## 1. Test Architecture

The `tests/` directory is organized into thematic files targeting different components:

```text
tests/
  ├── test_cpp_backend.py          # Verifies ONNX Runtime configurations in C++
  ├── test_accessor.py             # Validates the xarray accessor registration
  ├── test_cf_compliance.py       # Checks CF-1.11 compliance on exported files
  ├── test_memory_leak.py          # Monitors memory leaks during repeated rollouts
  ├── test_historical_validation.py# Compares forecasts against historical targets
  └── doc_examples/                # Sourced code snippets compiled in documentation
        ├── test_adapters.py
        ├── test_ensembles.py
        └── ...
```

---

## 2. Running the Test Suite

To run the full test suite, install the `[test]` package extras and execute `pytest` with the root directory in your python path:

```bash
# 1. Install test dependencies
pip install 'weathergraph[test]'

# 2. Run all tests
PYTHONPATH=. pytest
```

### Running Specific Test Domains
If you are developing a specific feature, you can target individual test files:

```bash
# Test the C++ backend and option forwarding
PYTHONPATH=. pytest tests/test_cpp_backend.py

# Run only the documentation example tests
PYTHONPATH=. pytest tests/doc_examples/
```

---

## 3. Advanced Test Domains

### Memory Leak Testing (`test_memory_leak.py`)
WeatherGraph's C++ bindings pass raw arrays between Python and C++ namespaces. If pointers or memory arenas are not handled correctly, they can trigger Resident Set Size (RSS) memory bloat.

The memory leak tests run repeated multi-step autoregressive rollouts (often hundreds of steps) while monitoring the process RSS via `psutil` and `memory-profiler`. The test asserts that the memory delta after initial steps remains flat within strict tolerances ($\le 100\text{ KB}$ deviation).

```bash
PYTHONPATH=. pytest tests/test_memory_leak.py -v
```

### Historical Validation (`test_historical_validation.py`)
To ensure that optimizations (such as FP16 conversions, spatial tiling, or CUDA execution provider settings) do not introduce numerical drift or scientific regressions, the historical validation suite:
1.  Loads a target reanalysis slice.
2.  Runs a multi-step rollout prediction.
3.  Calculates the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) against historical benchmarks.
4.  Asserts that values stay within strict error tolerances (typically $\le 10^{-5}$ deviation).

---

## 4. Writing Documented Examples

All code blocks displayed in the tutorials and guides are extracted dynamically from python files inside `tests/doc_examples/` using MkDocs snippets:

```markdown
--8<-- "tests/doc_examples/test_adapters.py:imports"
```

When adding new python examples to the documentation:
1.  Create a test inside `tests/doc_examples/`.
2.  Wrap the snippet using comment blocks:
    ```python
    # --8<-- [start:my_example]
    # code goes here
    # --8<-- [end:my_example]
    ```
3.  Write test assertions using mocks (`unittest.mock.patch`) to ensure the code executes successfully.
4.  Reference the block in the markdown file.