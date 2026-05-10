import onnxruntime as ort

ENCODER_LOC = r"" #encoder onnx file
DECODER_LOC = r"" #decoder onnx file

encoder = ort.InferenceSession(ENCODER_LOC, providers=["CPUExecutionProvider"])
decoder = ort.InferenceSession(DECODER_LOC, providers=["CPUExecutionProvider"])

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
