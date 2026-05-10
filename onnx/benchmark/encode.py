from encode_decode import ram_mb,ResidualVectorQuantizer,model_dir_from_nevo
import os, psutil, time
import numpy as np
from pathlib import Path
import soundfile as sf

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

audio_raw , sr = sf.read("test_clip.wav",always_2d=True)
if audio_raw.shape[1] != 1: audio_raw = audio_raw.mean(axis=1, keepdims=True)
audio_raw = audio_raw.flatten().astype(np.float32)

initial_ram = ram_mb()

import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

encoder = ort.InferenceSession(str(model_dir + r"\encoder.onnx"), providers=["CPUExecutionProvider"],sess_options=so)

# -------------------------------------------------------------------------------

enc_input_names = [i.name for i in encoder.get_inputs()]
enc_output_names = [o.name for o in encoder.get_outputs()]

# -------------------------------------------------------------------------------

frame_size = encoder.get_inputs()[0].shape[2]
latent_dim = encoder.get_outputs()[0].shape[1]

# -------------------------------------------------------------------------------

pad = (-len(audio_raw)) % frame_size
audio_padded = np.pad(audio_raw, (0, pad), mode="constant")
n_frames = len(audio_padded) // frame_size

audio_inputs = audio_padded.reshape(n_frames, 1, 1, frame_size).astype(np.float32)
audio_outputs = np.zeros((n_frames, 1, 1, frame_size), dtype=np.float32)

enc_feed = {}
for inp in encoder.get_inputs():
    if inp.name == enc_input_names[0]:
        enc_feed[inp.name] = audio_inputs[0].astype(np.float32)
    else:
        enc_feed[inp.name] = np.zeros(inp.shape, dtype=np.float32)

# -------------------------------------------------------------------------------

enc_state_input_names = enc_input_names[1:]
enc_state_output_names = enc_output_names[1:]

# -------------------------------------------------------------------------------

codebooks = np.load(str(model_dir + r"\codebook.npy")).astype(np.float32)
code_outputs = np.zeros((n_frames, n_codebooks_used), dtype=np.int64)
quantizer = ResidualVectorQuantizer(codebooks)

# -------------------------------------------------------------------------------

peak_ram = initial_ram
frame_times = []

t1 = time.perf_counter()

for i in range(n_frames):
    t_f1 = time.perf_counter()

    # ---- ENCODER ----
    enc_feed[enc_input_names[0]] = audio_inputs[i]

    enc_outputs = encoder.run(enc_output_names, enc_feed)

    # main encoder output (latent representation)
    latent_out = enc_outputs[0]

    # RVQ: quantize & dequantize
    codes = quantizer.quantize(latent_out, n_codebooks_used)
    code_outputs[i] = codes

    # ---- STATE FEEDBACK ----
    # encoder states
    for in_name, out_name in zip(enc_state_input_names, enc_state_output_names):
        out_idx = enc_output_names.index(out_name)
        enc_feed[in_name] = enc_outputs[out_idx]

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

# -------- Save Codes --------
out_path = Path("test_codes.npy")
np.save(out_path, code_outputs)
