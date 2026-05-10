import os, psutil, time
from pathlib import Path
import numpy as np
import soundfile as sf

import tempfile
import zipfile

######################
n_codebooks_used = 8
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

p = psutil.Process(os.getpid())
p.cpu_affinity([1])

def ram_mb() -> float:
    """Return resident memory for the current process in megabytes."""
    return p.memory_info().rss / (1024**2)  # resident set size (actual RAM)

class ResidualVectorQuantizer:
    """NumPy residual vector quantizer for exported `.nevo` codebooks."""

    def __init__(self, codebooks: np.ndarray) -> None:
        self.C = np.asarray(codebooks, dtype=np.float32)  # (Q,K,D)
        self.Q, self.K, self.D = self.C.shape
        self.cn = (self.C * self.C).sum(axis=2)           # (Q,K)

    def quantize(self, x: np.ndarray, n_levels: int | None = None) -> np.ndarray:
        """Return RVQ code indices for one latent vector."""
        r = np.asarray(x, dtype=np.float32).reshape(-1)   # (D,)
        assert r.shape[0] == self.D

        if n_levels is None:
            L = self.Q
        else:
            L = int(n_levels)
            assert 1 <= L <= self.Q

        idx = np.empty((L,), dtype=np.int64)

        for i in range(L):
            Ci = self.C[i]                                # (K,D)
            r_norm = float(np.dot(r, r))                  # scalar
            dot = Ci @ r                                  # (K,)
            dist = self.cn[i] + r_norm - 2.0 * dot        # (K,)
            k = int(np.argmin(dist))
            idx[i] = k
            r -= Ci[k]                                    # residual update

        return idx

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        """Return the summed codebook vector for RVQ code indices."""
        codes = np.asarray(codes, dtype=np.int64).reshape(-1)
        L = codes.shape[0]
        assert L <= self.Q

        x_hat = np.zeros((self.D,), dtype=np.float32)
        for i, k in enumerate(codes):
            x_hat += self.C[i, k]

        return x_hat

def model_dir_from_nevo(nevo_path: str | Path) -> str:
    """
    Extracts a .nevo (zip) bundle into a fresh temp directory and returns the temp
    directory path as a string (model_dir), containing:
      encoder.onnx, decoder.onnx, codebook.npy

    Usage:
        model_dir = model_dir_from_nevo(model_loc)
        encoder = ort.InferenceSession(model_dir + r"\\encoder.onnx", ...)
    """
    nevo_path = Path(nevo_path)

    if not nevo_path.exists():
        raise FileNotFoundError(f".nevo not found: {nevo_path}")

    temp_dir = Path(tempfile.mkdtemp(prefix="nevo_"))

    with zipfile.ZipFile(nevo_path, "r") as z:
        z.extractall(temp_dir)

    # expected filenames inside the zip
    required = ["encoder.onnx", "decoder.onnx", "codebook.npy"]
    missing = [f for f in required if not (temp_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{nevo_path} is missing {missing}. "
            f"Expected files at root of archive: {required}"
        )

    return str(temp_dir.resolve())

if __name__ == "__main__":

    print("Model location:", model_loc)
    # print("Model directory:", model_dir)
    model_dir = model_dir_from_nevo(model_loc)


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
    decoder = ort.InferenceSession(str(model_dir + r"\decoder.onnx"), providers=["CPUExecutionProvider"],sess_options=so)

    # -------------------------------------------------------------------------------

    enc_input_names = [i.name for i in encoder.get_inputs()]
    enc_output_names = [o.name for o in encoder.get_outputs()]

    dec_input_names = [i.name for i in decoder.get_inputs()]
    dec_output_names = [o.name for o in decoder.get_outputs()]

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

    dummy_embedding = np.zeros((1, latent_dim, 1), dtype=np.float32)

    dec_feed = {}
    for inp in decoder.get_inputs():
        if inp.name == dec_input_names[0]:
            dec_feed[inp.name] = dummy_embedding.astype(np.float32)
        else:
            dec_feed[inp.name] = np.zeros(inp.shape, dtype=np.float32)

    # -------------------------------------------------------------------------------

    enc_state_input_names = enc_input_names[1:]
    enc_state_output_names = enc_output_names[1:]

    dec_state_input_names = dec_input_names[1:]
    dec_state_output_names = dec_output_names[1:]

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
        quantized_embedding = quantizer.dequantize(codes).reshape(1, latent_dim, 1)

        # ---- DECODER ----
        dec_feed[dec_input_names[0]] = quantized_embedding
        dec_outputs = decoder.run(dec_output_names, dec_feed)

        dec_out = dec_outputs[0]
        audio_outputs[i] = dec_out

        # ---- STATE FEEDBACK ----
        # encoder states
        for in_name, out_name in zip(enc_state_input_names, enc_state_output_names):
            out_idx = enc_output_names.index(out_name)
            enc_feed[in_name] = enc_outputs[out_idx]

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
    #print("Frame Times (ms):", [round(ft * 1000, 1) for ft in frame_times])

    # -------- Save Codes --------
    out_path = Path("test_codes.npy")
    np.save(out_path, code_outputs)

    # ---------- Save wav ---------
    import soundfile as sf
    wav_out_path = "output.wav"
    sf.write(wav_out_path, audio_outputs.flatten(), fs)

    # --------- Playback ---------
    import sounddevice as sd
    sd.play(audio_outputs.flatten(), fs, blocking=True)
