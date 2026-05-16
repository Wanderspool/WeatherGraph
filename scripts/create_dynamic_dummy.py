import onnx
import onnx.helper as helper
from onnx import TensorProto
import sys

def create_dummy_model(node_count, output_path):
    # Input: [1, node_count, 78] (batch, nodes, channels)
    # Output: [1, node_count, 78]
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, node_count, 78])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, node_count, 78])
    
    # Just an Identity node
    node_def = helper.make_node(
        'Identity',
        ['input'],
        ['output']
    )

    op = helper.make_opsetid("ai.onnx", 14)

    graph_def = helper.make_graph(
        [node_def],
        'dummy-weather-model',
        [input_tensor],
        [output_tensor]
    )

    model_def = helper.make_model(graph_def, producer_name='dummy', opset_imports=[op])
    model_def.ir_version = 8
    onnx.save(model_def, output_path)
    print(f"Dummy model saved as {output_path} with {node_count} nodes")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_dynamic_dummy.py <node_count> <output_path>")
    else:
        create_dummy_model(int(sys.argv[1]), sys.argv[2])
