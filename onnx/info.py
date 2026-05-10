from pathlib import Path
import onnxruntime as ort

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

######################

run="nevo"

######################

run_dir = root_dir / "runs" / run
onnx_dir = run_dir / "onnx"

encoder = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])
decoder = ort.InferenceSession(str(onnx_dir / "decoder.onnx"), providers=["CPUExecutionProvider"])

print("Enc Inputs:")
for i in encoder.get_inputs():
    print(i.name, i.shape, i.type)
print("\nEnc Outputs:")
for o in encoder.get_outputs():
    print(o.name, o.shape, o.type)
print("\nDec Inputs:")
for i in decoder.get_inputs():
    print(i.name, i.shape, i.type)
print("\nDec Outputs:")
for o in decoder.get_outputs():
    print(o.name, o.shape, o.type)
