import os
import tempfile
import numpy as np
import pytest
import psutil
import onnx
import onnx.helper as helper
from onnx import TensorProto
from keisler_engine import GraphWeatherModel
import keisler_cpp_backend

def create_dummy_onnx(path):
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 71042, 78])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 71042, 78])
    node_def = helper.make_node('Identity', ['input'], ['output'])
    op = helper.make_opsetid("ai.onnx", 14)
    graph_def = helper.make_graph([node_def], 'dummy', [input_tensor], [output_tensor])
    model_def = helper.make_model(graph_def, producer_name='dummy', opset_imports=[op])
    model_def.ir_version = 8
    onnx.save(model_def, path)

@pytest.fixture
def mock_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path)
        yield keisler_cpp_backend.KeislerEngine(model_path)

def test_engine_inference_zero_copy(mock_engine):
    """Verifies that the engine performs inference correctly and returns expected shapes."""
    input_data = np.random.randn(1, 71042, 78).astype(np.float32)
    output_data = mock_engine.predict(input_data)
    
    assert output_data.shape == (1, 71042, 78)
    np.testing.assert_allclose(input_data, output_data)

def test_memory_safety_non_contiguous(mock_engine):
    """Verifies the engine rejects non-contiguous arrays, preventing C++ segfaults."""
    input_data = np.random.randn(1, 71042, 78).astype(np.float32)
    non_contiguous = input_data[:, ::-1, :]
    
    assert not non_contiguous.flags.c_contiguous
    
    with pytest.raises(RuntimeError, match="C-contiguous"):
        mock_engine.predict(non_contiguous)

def test_robustness_nan_inf(mock_engine):
    """Verifies the engine processes NaN/Inf without throwing C++ Floating Point Exceptions."""
    input_data = np.random.randn(1, 71042, 78).astype(np.float32)
    input_data[0, 0, 0] = np.nan
    input_data[0, 1, 0] = np.inf
    
    output_data = mock_engine.predict(input_data)
    
    assert np.isnan(output_data[0, 0, 0])
    assert np.isinf(output_data[0, 1, 0])
