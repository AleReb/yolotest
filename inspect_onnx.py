import onnxruntime as ort
import numpy as np

model_path = "yolo11n.onnx"

try:
    session = ort.InferenceSession(model_path)
    
    print("Input inputs:")
    for i in session.get_inputs():
        print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

    print("\nOutput outputs:")
    for i in session.get_outputs():
        print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

    # Run a dummy inference to see actual output structure
    input_name = session.get_inputs()[0].name
    # Assuming standard [1, 3, 640, 640] input
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
    
    outputs = session.run(None, {input_name: dummy_input})
    print(f"\nActual output shape: {outputs[0].shape}")
    
except Exception as e:
    print(f"Error: {e}")
