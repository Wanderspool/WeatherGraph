import onnx
import onnx.helper as helper
from onnx import TensorProto

def create_dummy_model():
    # Input: [1, 71042, 78] (batch, nodes, channels)
    # Output: [1, 71042, 78]
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 71042, 78])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 71042, 78])
    
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
 # Try IR version 8 (ONNX 1.11+)
    onnx.save(model_def, 'dummy_model.onnx')
    print("Dummy model saved as dummy_model.onnx")

if __name__ == "__main__":
    create_dummy_model()
