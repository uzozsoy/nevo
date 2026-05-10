from encode_decode import ram_mb,ResidualVectorQuantizer,model_dir_from_nevo
import os, psutil, time
import numpy as np
from pathlib import Path

######################
n_codebooks_used = 4
fs          = 8000
######################
model_loc = None #location of .nevo zip file
if not model_loc:
    #########################
    run="stabilize/nevo_wo_mel_mask"
    #########################
    cur_dir = Path(__file__).resolve().parent
    onnx_dir = cur_dir.parent
    root_dir = onnx_dir.parent

    run_dir = root_dir / "runs"
    import re
    parts = re.split(r"[\\/]", run)
    name = parts[-1]
    sub_dir = Path(*parts[:-1])  # empty Path(".") if no folders

    model_loc = run_dir / sub_dir / name / "onnx" / (name + ".nevo")
    model_loc = str(model_loc.resolve())

print("Model location:", model_loc)
model_dir = model_dir_from_nevo(model_loc)

p = psutil.Process(os.getpid())
p.cpu_affinity([1])

code_inputs=np.load("test_codes.npy")
n_frames=code_inputs.shape[0]

initial_ram = ram_mb()

import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

decoder = ort.InferenceSession(str(model_dir + r"\decoder.onnx"), providers=["CPUExecutionProvider"],sess_options=so)

# -------------------------------------------------------------------------------

dec_input_names = [i.name for i in decoder.get_inputs()]
dec_output_names = [o.name for o in decoder.get_outputs()]

# -------------------------------------------------------------------------------

frame_size = decoder.get_outputs()[0].shape[2]
latent_dim = decoder.get_inputs()[0].shape[1]

# -------------------------------------------------------------------------------

audio_outputs = np.zeros((n_frames, 1, 1, frame_size), dtype=np.float32)
dummy_embedding = np.zeros((1, latent_dim, 1), dtype=np.float32)

dec_feed = {}
for inp in decoder.get_inputs():
    if inp.name == dec_input_names[0]:
        dec_feed[inp.name] = dummy_embedding.astype(np.float32)
    else:
        dec_feed[inp.name] = np.zeros(inp.shape, dtype=np.float32)

# -------------------------------------------------------------------------------

dec_state_input_names = dec_input_names[1:]
dec_state_output_names = dec_output_names[1:]

# -------------------------------------------------------------------------------

codebooks = np.load(str(model_dir + r"\codebook.npy")).astype(np.float32)
quantizer = ResidualVectorQuantizer(codebooks)

# -------------------------------------------------------------------------------

peak_ram = initial_ram
frame_times = []

t1 = time.perf_counter()

for i in range(n_frames):
    t_f1 = time.perf_counter()

    quantized_embedding = quantizer.dequantize(code_inputs[i]).reshape(1, latent_dim, 1)

    # ---- DECODER ----
    dec_feed[dec_input_names[0]] = quantized_embedding
    dec_outputs = decoder.run(dec_output_names, dec_feed)

    dec_out = dec_outputs[0]
    audio_outputs[i] = dec_out

    # ---- STATE FEEDBACK ----

    # decoder states
    for in_name, out_name in zip(dec_state_input_names, dec_state_output_names):
        out_idx = dec_output_names.index(out_name)
        dec_feed[in_name] = dec_outputs[out_idx]

    # ---- Stats ----
    current_ram = ram_mb()
    if current_ram > peak_ram:
        peak_ram = current_ram

    frame_times.append(time.perf_counter() - t_f1)

t2 = time.perf_counter()

# frame_rate = fs / frame_size (e.g. 8000 / 320 = 25 fps)
frame_rate = fs / frame_size
rtf = (t2 - t1) / (n_frames / frame_rate)
print("Real Time Factor:", rtf)
print("Extra RAM used:", peak_ram - initial_ram, "MB")
print("Max Frame Time:", round(max(frame_times) * 1000, 1))
# print("Frame Times (ms):", [round(ft * 1000, 1) for ft in frame_times])

# --------- Playback ---------
import sounddevice as sd
sd.play(audio_outputs.flatten(), fs, blocking=True)