import os
import tempfile
import numpy as np
import pytest
import psutil
import onnx
import onnx.helper as helper
from onnx import TensorProto
import weathergraph_backend

def create_dummy_onnx(path):
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 71042, 78])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 71042, 78])
    node_def = helper.make_node('Identity', ['input'], ['output'])
    op = helper.make_opsetid("ai.onnx", 14)
    graph_def = helper.make_graph([node_def], 'dummy', [input_tensor], [output_tensor])
    model_def = helper.make_model(graph_def, producer_name='dummy', opset_imports=[op])
    model_def.ir_version = 8
    onnx.save(model_def, path)

def test_memory_leak_infinite_rollout():
    """
    Stress test the C++ backend for unbounded RSS growth during repeated inference.

    This guards the runtime contract itself, not a specific deployment claim like
    "all workloads fit in 1 GB RAM".
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path)
        
        process = psutil.Process(os.getpid())
        
        # Initialize engine and allocate buffer
        engine = weathergraph_backend.WeatherGraphEngine(model_path)
        input_data = np.random.randn(1, 71042, 78).astype(np.float32)
        
        ITERATIONS = 1000
        mem_at_100 = 0
        
        for i in range(ITERATIONS):
            _ = engine.predict(input_data)
            
            # Record baseline memory after initial caching and warm-up
            if i == 100:
                mem_at_100 = process.memory_info().rss
                
        mem_final = process.memory_info().rss
        
        # Calculate growth after warm-up
        diff_mb = (mem_final - mem_at_100) / 1024 / 1024
        
        # Assert no leak larger than 1MB (allowing for minor OS jitter)
        assert diff_mb < 1.0, f"Memory leak detected! Growth: {diff_mb:.2f} MB"
