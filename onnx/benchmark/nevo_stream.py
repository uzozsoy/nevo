import os
import queue
import psutil
import numpy as np
import multiprocessing as mp
import sounddevice as sd
import onnxruntime as ort
import argparse
from pathlib import Path
import tempfile
import zipfile

# =========================================================
# USER SETTINGS (defaults; CLI can override)
# =========================================================
MODEL_LOC = r"/home/umut/PycharmProjects/NevoBenchmark/models/latentd/release_latentd_128.nevo"
N_CODEBOOKS_USED = 4

FS = 8000
CHANNELS = 1
DTYPE = "float32"

INPUT_DEVICE = 5
OUTPUT_DEVICE = 5

ENCODER_CORE = 0
DECODER_CORE = 1

QUEUE_SIZE = 8
LATENCY = "low"

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
        encoder = ort.InferenceSession(model_dir + r"//encoder.onnx", ...)
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

def parse_optional_args() -> tuple[str, int, int, int, int, int | None, int | None]:
    """Parse optional live-streaming CLI overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="Path to .nevo model")
    parser.add_argument("--nc", type=int, default=None, help="Number of codebooks used")
    parser.add_argument("--enc-core", type=int, default=None, help="CPU core for encoder process")
    parser.add_argument("--dec-core", type=int, default=None, help="CPU core for decoder process")
    parser.add_argument("--q-size", type=int, default=None, help="Queue size")
    parser.add_argument("--in-dev", type=int, default=None, help="Input device index")
    parser.add_argument("--out-dev", type=int, default=None, help="Output device index")

    args = parser.parse_args()

    model_loc = args.path if args.path is not None else MODEL_LOC
    n_codebooks_used = args.nc if args.nc is not None else N_CODEBOOKS_USED
    encoder_core = args.enc_core if args.enc_core is not None else ENCODER_CORE
    decoder_core = args.dec_core if args.dec_core is not None else DECODER_CORE
    queue_size = args.q_size if args.q_size is not None else QUEUE_SIZE
    input_device = args.in_dev if args.in_dev is not None else INPUT_DEVICE
    output_device = args.out_dev if args.out_dev is not None else OUTPUT_DEVICE

    return model_loc, n_codebooks_used, encoder_core, decoder_core, queue_size, input_device, output_device


def choose_device(kind: str = "input") -> int:
    """Interactively select an input or output sounddevice index."""
    devices = sd.query_devices()
    valid = []

    print(f"\nAvailable {kind} devices:\n")

    for idx, dev in enumerate(devices):
        if kind == "input" and dev["max_input_channels"] > 0:
            valid.append(idx)
            print(f"[{idx}] {dev['name']} | in={dev['max_input_channels']} out={dev['max_output_channels']}")
        elif kind == "output" and dev["max_output_channels"] > 0:
            valid.append(idx)
            print(f"[{idx}] {dev['name']} | in={dev['max_input_channels']} out={dev['max_output_channels']}")

    if not valid:
        raise RuntimeError(f"No valid {kind} devices found.")

    while True:
        s = input(f"\nSelect {kind} device index: ").strip()
        try:
            idx = int(s)
            if idx in valid:
                return idx
            print("Invalid selection.")
        except ValueError:
            print("Enter a valid integer index.")


def resolve_devices(input_device: int | None, output_device: int | None) -> tuple[int, int]:
    """Resolve missing audio device ids through interactive selection."""
    if input_device is None:
        input_device = choose_device("input")
    if output_device is None:
        output_device = choose_device("output")
    return input_device, output_device

# =========================================================
# ENCODER PROCESS
# =========================================================
def encoder_worker(model_dir: str, n_codebooks_used: int, raw_q: object, code_q: object, status_q: object, core_id: int) -> None:
    """Worker process that encodes audio frames and emits RVQ codes."""
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([core_id])

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        encoder = ort.InferenceSession(
            str(model_dir + "/encoder.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )

        enc_input_names = [i.name for i in encoder.get_inputs()]
        enc_output_names = [o.name for o in encoder.get_outputs()]

        frame_size = encoder.get_inputs()[0].shape[2]

        enc_feed = {}
        for inp in encoder.get_inputs():
            if inp.name == enc_input_names[0]:
                enc_feed[inp.name] = np.zeros((1, 1, frame_size), dtype=np.float32)
            else:
                enc_feed[inp.name] = np.zeros(inp.shape, dtype=np.float32)

        enc_state_input_names = enc_input_names[1:]
        enc_state_output_names = enc_output_names[1:]

        enc_state_pairs = [
            (in_name, enc_output_names.index(out_name))
            for in_name, out_name in zip(enc_state_input_names, enc_state_output_names)
        ]

        codebooks = np.load(str(model_dir + "/codebook.npy")).astype(np.float32)
        quantizer = ResidualVectorQuantizer(codebooks)

        # Tell main process frame size once ready
        status_q.put(("encoder_ready", frame_size))

        while True:
            item = raw_q.get()
            if item is None:
                break

            frame_id, x = item  # x shape: (frame_size,)

            x = x.reshape(1, 1, frame_size).astype(np.float32)
            enc_feed[enc_input_names[0]] = x

            enc_outputs = encoder.run(enc_output_names, enc_feed)
            latent_out = enc_outputs[0]

            codes = quantizer.quantize(latent_out, n_codebooks_used)
            code_q.put((frame_id, codes), block=True)

            for in_name, out_idx in enc_state_pairs:
                enc_feed[in_name] = enc_outputs[out_idx]

    except Exception as e:
        status_q.put(("encoder_error", repr(e)))


# =========================================================
# DECODER PROCESS
# =========================================================
def decoder_worker(model_dir: str, code_q: object, out_q: object, status_q: object, core_id: int) -> None:
    """Worker process that decodes RVQ codes into audio frames."""
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([core_id])

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        decoder = ort.InferenceSession(
            str(model_dir + "/decoder.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )

        dec_input_names = [i.name for i in decoder.get_inputs()]
        dec_output_names = [o.name for o in decoder.get_outputs()]

        frame_size = decoder.get_outputs()[0].shape[2]
        latent_dim = decoder.get_inputs()[0].shape[1]

        dec_feed = {}
        for inp in decoder.get_inputs():
            if inp.name == dec_input_names[0]:
                dec_feed[inp.name] = np.zeros((1, latent_dim, 1), dtype=np.float32)
            else:
                dec_feed[inp.name] = np.zeros(inp.shape, dtype=np.float32)

        dec_state_input_names = dec_input_names[1:]
        dec_state_output_names = dec_output_names[1:]

        dec_state_pairs = [
            (in_name, dec_output_names.index(out_name))
            for in_name, out_name in zip(dec_state_input_names, dec_state_output_names)
        ]

        codebooks = np.load(str(model_dir + "/codebook.npy")).astype(np.float32)
        quantizer = ResidualVectorQuantizer(codebooks)

        status_q.put(("decoder_ready", frame_size))

        while True:
            item = code_q.get()
            if item is None:
                break

            frame_id, codes = item

            quantized_embedding = quantizer.dequantize(codes).reshape(1, latent_dim, 1).astype(np.float32)
            dec_feed[dec_input_names[0]] = quantized_embedding

            dec_outputs = decoder.run(dec_output_names, dec_feed)
            dec_out = dec_outputs[0]  # expected shape (1,1,frame_size)

            for in_name, out_idx in dec_state_pairs:
                dec_feed[in_name] = dec_outputs[out_idx]

            y = dec_out.reshape(frame_size).astype(np.float32)
            out_q.put((frame_id, y), block=True)

    except Exception as e:
        status_q.put(("decoder_error", repr(e)))


# =========================================================
# MAIN LIVE LOOPBACK
# =========================================================
def main() -> None:
    """Run the live codec loopback demo."""
    mp.set_start_method("spawn", force=True)

    model_loc, n_codebooks_used, encoder_core, decoder_core, queue_size, input_device, output_device = parse_optional_args()
    input_device, output_device = resolve_devices(input_device, output_device)

    model_loc = os.path.abspath(model_loc)
    print("Model location:", model_loc)
    print("Input device:", input_device)
    print("Output device:", output_device)

    model_dir = model_dir_from_nevo(model_loc)

    raw_q = mp.Queue(maxsize=queue_size)
    code_q = mp.Queue(maxsize=queue_size)
    out_q = mp.Queue(maxsize=queue_size)
    status_q = mp.Queue()

    enc_p = mp.Process(
        target=encoder_worker,
        args=(model_dir, n_codebooks_used, raw_q, code_q, status_q, encoder_core),
        daemon=True,
    )
    dec_p = mp.Process(
        target=decoder_worker,
        args=(model_dir, code_q, out_q, status_q, decoder_core),
        daemon=True,
    )

    enc_p.start()
    dec_p.start()

    enc_frame_size = None
    dec_frame_size = None

    while enc_frame_size is None or dec_frame_size is None:
        tag, value = status_q.get()

        if tag == "encoder_ready":
            enc_frame_size = value
        elif tag == "decoder_ready":
            dec_frame_size = value
        elif tag == "encoder_error":
            raise RuntimeError(f"Encoder process failed: {value}")
        elif tag == "decoder_error":
            raise RuntimeError(f"Decoder process failed: {value}")

    if enc_frame_size != dec_frame_size:
        raise RuntimeError(f"Encoder frame size {enc_frame_size} != decoder frame size {dec_frame_size}")

    frame_size = enc_frame_size
    print("Frame size:", frame_size)

    next_frame_id = 0
    play_frame_id = 0
    decoded_buffer = {}

    dropped_input = 0
    missing_output = 0
    startup_blocks = 2
    played_blocks = 0

    def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        nonlocal next_frame_id, play_frame_id, dropped_input, missing_output, played_blocks

        if status:
            print(status)

        x = indata[:, 0].copy().astype(np.float32)

        if len(x) != frame_size:
            if len(x) < frame_size:
                x = np.pad(x, (0, frame_size - len(x)))
            else:
                x = x[:frame_size]

        try:
            raw_q.put_nowait((next_frame_id, x))
            next_frame_id += 1
        except queue.Full:
            dropped_input += 1

        while True:
            try:
                fid, y = out_q.get_nowait()
                decoded_buffer[fid] = y
            except queue.Empty:
                break

        if played_blocks < startup_blocks:
            outdata.fill(0)
            played_blocks += 1
            return

        if play_frame_id in decoded_buffer:
            y = decoded_buffer.pop(play_frame_id)
            outdata[:, 0] = y
            play_frame_id += 1
        else:
            outdata.fill(0)
            missing_output += 1

    try:
        with sd.Stream(
                samplerate=FS,
                blocksize=frame_size,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=callback,
                latency=LATENCY,
                device=(input_device, output_device),
        ):
            print("Live loopback started. Speak into the mic. Ctrl+C to stop.")
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            raw_q.put(None, timeout=0.2)
        except Exception:
            pass

        try:
            code_q.put(None, timeout=0.2)
        except Exception:
            pass

        enc_p.join(timeout=1.0)
        dec_p.join(timeout=1.0)

        if enc_p.is_alive():
            enc_p.terminate()
        if dec_p.is_alive():
            dec_p.terminate()

        print("Dropped input frames:", dropped_input)
        print("Missing output frames:", missing_output)


if __name__ == "__main__":
    main()
