import os
import queue
import psutil
import zipfile
import tempfile
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import sounddevice as sd
import onnxruntime as ort
from scipy.signal import lfilter, firwin


# =========================================================
# USER SETTINGS (defaults; CLI can override)
# =========================================================
MODEL_LOC = r"/home/umut/PycharmProjects/NevoBenchmark/models/latentd/release_latentd_128.nevo"
DENOISER_MODEL = r"/home/umut/PycharmProjects/NevoBenchmark/models/gtcrn_simple.onnx"
# GTCRN ONNX model attribution:
# Source: https://github.com/Xiaobin-Rong/gtcrn/tree/main/stream/onnx_models
# License: MIT, Copyright (c) 2024 Rong Xiaobin. See THIRD_PARTY_NOTICES.md.

N_CODEBOOKS_USED = 4

DEVICE_FS = 16000     # GTCRN input/output rate
CODEC_FS = 8000       # codec internal rate

CHANNELS = 1
DTYPE = "float32"

INPUT_DEVICE = 5
OUTPUT_DEVICE = 5

DENOISER_CORE = 2
ENCODER_CORE = 0
DECODER_CORE = 1

QUEUE_SIZE = 0
LATENCY = "low"

class StreamingDownsample2x:
    """Stateful FIR downsampler from 16 kHz to 8 kHz."""

    def __init__(self, num_taps: int = 63, cutoff: float = 0.45) -> None:
        self.taps = firwin(num_taps, cutoff).astype(np.float32)
        self.zi = np.zeros(num_taps - 1, dtype=np.float32)
        self.phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y, self.zi = lfilter(self.taps, [1.0], x, zi=self.zi)

        out = y[self.phase::2].astype(np.float32)
        self.phase = (self.phase + len(x)) & 1
        return out


class StreamingUpsample2x:
    """Stateful FIR upsampler from 8 kHz to 16 kHz."""

    def __init__(self, num_taps: int = 63, cutoff: float = 0.45) -> None:
        self.taps = (2.0 * firwin(num_taps, cutoff)).astype(np.float32)
        self.zi = np.zeros(num_taps - 1, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)

        z = np.zeros(len(x) * 2, dtype=np.float32)
        z[::2] = x

        y, self.zi = lfilter(self.taps, [1.0], z, zi=self.zi)
        return y.astype(np.float32)

class ResidualVectorQuantizer:
    """NumPy residual vector quantizer for exported `.nevo` codebooks."""

    def __init__(self, codebooks: np.ndarray) -> None:
        self.C = np.asarray(codebooks, dtype=np.float32)  # (Q,K,D)
        self.Q, self.K, self.D = self.C.shape
        self.cn = (self.C * self.C).sum(axis=2)           # (Q,K)

    def quantize(self, x: np.ndarray, n_levels: int | None = None) -> np.ndarray:
        """Return RVQ code indices for one latent vector."""
        r = np.asarray(x, dtype=np.float32).reshape(-1)
        assert r.shape[0] == self.D

        if n_levels is None:
            L = self.Q
        else:
            L = int(n_levels)
            assert 1 <= L <= self.Q

        idx = np.empty((L,), dtype=np.int64)

        for i in range(L):
            Ci = self.C[i]
            r_norm = float(np.dot(r, r))
            dot = Ci @ r
            dist = self.cn[i] + r_norm - 2.0 * dot
            k = int(np.argmin(dist))
            idx[i] = k
            r -= Ci[k]

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
    nevo_path = Path(nevo_path)

    if not nevo_path.exists():
        raise FileNotFoundError(f".nevo not found: {nevo_path}")

    temp_dir = Path(tempfile.mkdtemp(prefix="nevo_"))

    with zipfile.ZipFile(nevo_path, "r") as z:
        z.extractall(temp_dir)

    required = ["encoder.onnx", "decoder.onnx", "codebook.npy"]
    missing = [f for f in required if not (temp_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{nevo_path} is missing {missing}. "
            f"Expected files at root of archive: {required}"
        )

    return str(temp_dir.resolve())


def parse_optional_args() -> tuple[str, str, int, int, int, int, int, int | None, int | None]:
    """Parse optional live-streaming CLI overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="Path to .nevo model")
    parser.add_argument("--den-path", type=str, default=None, help="Path to GTCRN ONNX")
    parser.add_argument("--nc", type=int, default=None, help="Number of codebooks used")
    parser.add_argument("--den-core", type=int, default=None, help="CPU core for denoiser process")
    parser.add_argument("--enc-core", type=int, default=None, help="CPU core for encoder process")
    parser.add_argument("--dec-core", type=int, default=None, help="CPU core for decoder process")
    parser.add_argument("--q-size", type=int, default=None, help="Queue size")
    parser.add_argument("--in-dev", type=int, default=None, help="Input device index")
    parser.add_argument("--out-dev", type=int, default=None, help="Output device index")

    args = parser.parse_args()

    model_loc = args.path if args.path is not None else MODEL_LOC
    denoiser_model = args.den_path if args.den_path is not None else DENOISER_MODEL
    n_codebooks_used = args.nc if args.nc is not None else N_CODEBOOKS_USED
    denoiser_core = args.den_core if args.den_core is not None else DENOISER_CORE
    encoder_core = args.enc_core if args.enc_core is not None else ENCODER_CORE
    decoder_core = args.dec_core if args.dec_core is not None else DECODER_CORE
    queue_size = args.q_size if args.q_size is not None else QUEUE_SIZE
    input_device = args.in_dev if args.in_dev is not None else INPUT_DEVICE
    output_device = args.out_dev if args.out_dev is not None else OUTPUT_DEVICE

    return (
        model_loc,
        denoiser_model,
        n_codebooks_used,
        denoiser_core,
        encoder_core,
        decoder_core,
        queue_size,
        input_device,
        output_device,
    )


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
# DENOISER PROCESS (16 kHz GTCRN ONNX)
# =========================================================
def denoiser_worker(denoiser_model: str, raw16_q: object, den16_q: object, status_q: object, core_id: int) -> None:
    """Worker process that streams audio through the GTCRN ONNX denoiser."""
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([core_id])

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            denoiser_model,
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )

        conv_cache = np.zeros((2, 1, 16, 16, 33), dtype=np.float32)
        tra_cache = np.zeros((2, 3, 1, 1, 16), dtype=np.float32)
        inter_cache = np.zeros((2, 1, 33, 16), dtype=np.float32)

        n_fft = 512
        hop = 256
        win = (np.hanning(n_fft) ** 0.5).astype(np.float32)

        in_tail = np.zeros(n_fft - hop, dtype=np.float32)   # 256
        out_tail = np.zeros(n_fft - hop, dtype=np.float32)  # 256

        status_q.put(("denoiser_ready", hop))

        while True:
            item = raw16_q.get()
            if item is None:
                break

            frame_id, x = item  # x shape: (256,), 16 kHz

            x = np.asarray(x, dtype=np.float32).reshape(-1)
            if len(x) != hop:
                if len(x) < hop:
                    x = np.pad(x, (0, hop - len(x)))
                else:
                    x = x[:hop]

            frame = np.concatenate([in_tail, x]).astype(np.float32)  # (512,)
            in_tail = frame[hop:].copy()

            spec = np.fft.rfft(frame * win, n=n_fft)
            mix = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)  # (257,2)
            mix = mix[:, None, :]    # (257,1,2)
            mix = mix[None, ...]     # (1,257,1,2)

            enh, conv_cache, tra_cache, inter_cache = session.run(
                [],
                {
                    "mix": mix,
                    "conv_cache": conv_cache,
                    "tra_cache": tra_cache,
                    "inter_cache": inter_cache,
                },
            )

            enh_complex = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
            y_frame = np.fft.irfft(enh_complex, n=n_fft).astype(np.float32) * win

            y = y_frame[:hop] + out_tail[:hop]
            out_tail = y_frame[hop:].copy()

            den16_q.put((frame_id, y.astype(np.float32)), block=True)

    except Exception as e:
        status_q.put(("denoiser_error", repr(e)))


# =========================================================
# ENCODER PROCESS (expects denoised 16 kHz chunks)
# =========================================================
def encoder_worker(model_dir: str, n_codebooks_used: int, den16_q: object, code_q: object, status_q: object, core_id: int) -> None:
    """Worker process that downsamples denoised audio, encodes it, and emits RVQ codes."""
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
        frame_size = encoder.get_inputs()[0].shape[2]   # codec frame size at 8 kHz

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

        # Stateful 16k -> 8k resampler
        resampler_16_to_8 = StreamingDownsample2x()

        status_q.put(("encoder_ready", frame_size))

        sample_buf_8k = np.zeros(0, dtype=np.float32)
        codec_frame_id = 0

        while True:
            item = den16_q.get()
            if item is None:
                break

            _, y16 = item
            y16 = np.asarray(y16, dtype=np.float32).reshape(-1)

            y8 = resampler_16_to_8.process(y16)
            sample_buf_8k = np.concatenate([sample_buf_8k, y8])

            while len(sample_buf_8k) >= frame_size:
                x8 = sample_buf_8k[:frame_size]
                sample_buf_8k = sample_buf_8k[frame_size:]

                enc_feed[enc_input_names[0]] = x8.reshape(1, 1, frame_size).astype(np.float32)
                enc_outputs = encoder.run(enc_output_names, enc_feed)
                latent_out = enc_outputs[0]

                codes = quantizer.quantize(latent_out, n_codebooks_used)
                code_q.put((codec_frame_id, codes), block=True)

                for in_name, out_idx in enc_state_pairs:
                    enc_feed[in_name] = enc_outputs[out_idx]

                codec_frame_id += 1

    except Exception as e:
        status_q.put(("encoder_error", repr(e)))


# =========================================================
# DECODER PROCESS (outputs 16 kHz playback chunks)
# =========================================================
def decoder_worker(model_dir: str, code_q: object, play16_q: object, status_q: object, core_id: int) -> None:
    """Worker process that decodes RVQ codes and upsamples audio for playback."""
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

        frame_size = decoder.get_outputs()[0].shape[2]   # codec output size at 8 kHz
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

        # Stateful 8k -> 16k resampler
        resampler_8_to_16 = StreamingUpsample2x()

        status_q.put(("decoder_ready", frame_size))

        while True:
            item = code_q.get()
            if item is None:
                break

            _, codes = item

            quantized_embedding = quantizer.dequantize(codes).reshape(1, latent_dim, 1).astype(np.float32)
            dec_feed[dec_input_names[0]] = quantized_embedding

            dec_outputs = decoder.run(dec_output_names, dec_feed)
            dec_out = dec_outputs[0]

            for in_name, out_idx in dec_state_pairs:
                dec_feed[in_name] = dec_outputs[out_idx]

            y8 = dec_out.reshape(frame_size).astype(np.float32)
            y16 = resampler_8_to_16.process(y8)

            play16_q.put(y16.astype(np.float32), block=True)

    except Exception as e:
        status_q.put(("decoder_error", repr(e)))

# =========================================================
# MAIN LIVE LOOPBACK
# =========================================================
def main() -> None:
    """Run live GTCRN-denoise plus codec loopback."""
    mp.set_start_method("spawn", force=True)

    (
        model_loc,
        denoiser_model,
        n_codebooks_used,
        denoiser_core,
        encoder_core,
        decoder_core,
        queue_size,
        input_device,
        output_device,
    ) = parse_optional_args()

    input_device, output_device = resolve_devices(input_device, output_device)

    model_loc = os.path.abspath(model_loc)
    denoiser_model = os.path.abspath(denoiser_model)

    print("Model location:", model_loc)
    print("Denoiser model:", denoiser_model)
    print("Input device:", input_device)
    print("Output device:", output_device)

    model_dir = model_dir_from_nevo(model_loc)

    raw16_q = mp.Queue(maxsize=queue_size)
    den16_q = mp.Queue(maxsize=queue_size)
    code_q = mp.Queue(maxsize=queue_size)
    play16_q = mp.Queue(maxsize=queue_size * 4)
    status_q = mp.Queue()

    den_p = mp.Process(
        target=denoiser_worker,
        args=(denoiser_model, raw16_q, den16_q, status_q, denoiser_core),
        daemon=True,
    )
    enc_p = mp.Process(
        target=encoder_worker,
        args=(model_dir, n_codebooks_used, den16_q, code_q, status_q, encoder_core),
        daemon=True,
    )
    dec_p = mp.Process(
        target=decoder_worker,
        args=(model_dir, code_q, play16_q, status_q, decoder_core),
        daemon=True,
    )

    den_p.start()
    enc_p.start()
    dec_p.start()

    den_ready = False
    enc_frame_size = None
    dec_frame_size = None

    while not den_ready or enc_frame_size is None or dec_frame_size is None:
        tag, value = status_q.get()

        if tag == "denoiser_ready":
            den_ready = True
            print("Denoiser hop:", value)
        elif tag == "encoder_ready":
            enc_frame_size = value
        elif tag == "decoder_ready":
            dec_frame_size = value
        elif tag == "denoiser_error":
            raise RuntimeError(f"Denoiser process failed: {value}")
        elif tag == "encoder_error":
            raise RuntimeError(f"Encoder process failed: {value}")
        elif tag == "decoder_error":
            raise RuntimeError(f"Decoder process failed: {value}")

    if enc_frame_size != dec_frame_size:
        raise RuntimeError(f"Encoder frame size {enc_frame_size} != decoder frame size {dec_frame_size}")

    print("Codec frame size:", enc_frame_size)

    next_input_chunk_id = 0
    play_fifo = np.zeros(0, dtype=np.float32)

    dropped_input = 0
    missing_output = 0
    startup_blocks = 8
    played_blocks = 0

    def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        nonlocal next_input_chunk_id, dropped_input, missing_output, played_blocks, play_fifo

        if status:
            print(status)

        x = indata[:, 0].copy().astype(np.float32)

        if len(x) != 256:
            if len(x) < 256:
                x = np.pad(x, (0, 256 - len(x)))
            else:
                x = x[:256]

        try:
            raw16_q.put_nowait((next_input_chunk_id, x))
            next_input_chunk_id += 1
        except queue.Full:
            dropped_input += 1

        while True:
            try:
                y = play16_q.get_nowait()
                play_fifo = np.concatenate([play_fifo, y])
            except queue.Empty:
                break

        if played_blocks < startup_blocks:
            outdata.fill(0)
            played_blocks += 1
            return

        if len(play_fifo) >= frames:
            outdata[:, 0] = play_fifo[:frames]
            play_fifo = play_fifo[frames:]
        else:
            outdata.fill(0)
            missing_output += 1
    try:
        with sd.Stream(
            samplerate=DEVICE_FS,
            blocksize=256,
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
        for q in (raw16_q, den16_q, code_q):
            try:
                q.put(None, timeout=0.2)
            except Exception:
                pass

        den_p.join(timeout=1.0)
        enc_p.join(timeout=1.0)
        dec_p.join(timeout=1.0)

        for p in (den_p, enc_p, dec_p):
            if p.is_alive():
                p.terminate()

        print("Dropped input chunks:", dropped_input)
        print("Missing output chunks:", missing_output)


if __name__ == "__main__":
    main()
