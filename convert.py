import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load your existing pickle model
with open("model (4).pkl", "rb") as f:
    model = pickle.load(f)

# Convert it to ONNX (9 features input)
initial_type = [('float_input', FloatTensorType([None, 9]))]
onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

# Save the lightweight ONNX file
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Successfully created model.onnx!")
