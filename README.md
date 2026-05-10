# NEVO

NEVO is a neural audio codec research codebase for low-bitrate, frame-based audio compression. The core model encodes mono waveform frames into latent features, quantizes them with residual vector quantization (RVQ), and decodes the quantized latents back to audio.

This repository currently contains the model code, dataset utilities, losses, ONNX export helpers, benchmark scripts, and evaluation/plotting scripts. Some release details are still placeholders because the public training entrypoint, exact environment, and checkpoint locations are not included in this snapshot.

## Repository Layout

```text
.
|-- dataset.py                  # AudioDataset with clean/noise/RIR augmentation
|-- model.py                    # NEVO encoder, decoder, RVQ model, discriminator
|-- requirements.txt            # Core non-PyTorch dependencies
|-- requirements-onnx.txt       # Optional ONNX/export/benchmark dependencies
|-- requirements-plot.txt       # Optional plotting dependencies
|-- requirements-stream.txt     # Optional live audio dependencies
|-- encodec/                    # EnCodec-derived training utilities
|-- helpers/
|   |-- info.py                 # Logging, metrics, codebook usage tracking
|   |-- losses.py               # Mel, PESQ proxy, adversarial, feature losses
|   |-- modules.py              # Convolution, LSTM, mel spectrogram modules
|   `-- utils.py                # Checkpoint, seeding, STOI alignment helpers
|-- onnx/
|   |-- convert.py              # Exports encoder/decoder/codebooks into .nevo bundle
|   |-- io_helpers.py           # ONNX wrapper/state flattening helpers
|   `-- benchmark/              # Offline and live ONNX benchmark scripts
|-- eval/                       # Clip comparison and metrics plotting tools
|-- runs/
|   `-- nevo/
|       `-- nevo_train.py       # Default training experiment
|-- vq/                         # vector-quantize-pytorch-derived RVQ implementation
`-- THIRD_PARTY_NOTICES.md      # Third-party attribution and license notices
```

## Core Concepts

The main model classes live in `model.py`.

- `NevoEncoder`: Converts waveform input shaped `[batch, 1, samples]` into latent frames.
- `ResidualVQ`: Quantizes latent frames using multiple residual codebooks.
- `NevoDecoder`: Reconstructs waveform frames from quantized latents.
- `NevoModel`: End-to-end encoder, RVQ, decoder model.
- `NevoConcatModel`: Variant that groups adjacent latent frames before quantization.
- `MSSTFTDiscriminator`: Multi-scale STFT discriminator used for adversarial training.

The `quantizer_limit` / `use_levels` path controls how many RVQ levels are active. This is used for custom bandwidth operation.

## Data

`AudioDataset` samples fixed-length chunks from one or more audio directories.

It can optionally:

- resample audio to the configured sample rate,
- mix noise at a random magnitude or SNR,
- apply room impulse responses,
- cache discovered audio paths for faster startup.

Dataset output is a tensor shaped `[2, chunk_len]`, where index `0` is the altered/noisy signal and index `1` is the clean target.

Example placeholder:

```python
from dataset import AudioDataset

dataset = AudioDataset(
    dataset_directories=["path/to/clean_audio"],
    sample_rate=8000,
    frames_in_chunk=25,
    frame_length=320,
    noise_prob=0.5,
    noise_directory="path/to/noise",
    rir_prob=0.0,
)
```

## Installation

Install PyTorch and torchaudio separately first. Their wheels depend on your OS, Python version, CUDA/ROCm/CPU target, and driver setup, so they are intentionally not pinned in `requirements.txt`.

Use the official PyTorch install selector:

```text
https://pytorch.org/get-started/locally/
```

This project was last developed with:

```text
torch==2.11.0+cu130
torchaudio==2.11.0+cu130
```

After installing a matching PyTorch/torchaudio pair, install the core non-PyTorch dependencies:

```bash
pip install -r requirements.txt
```

Optional dependency groups:

```bash
pip install -r requirements-plot.txt    # eval/plot_metrics.py
pip install -r requirements-onnx.txt    # ONNX export and offline benchmarks
pip install -r requirements-stream.txt  # live microphone/speaker demos
```

`torchcodec` is not imported directly by this repository. Install it only if your torchaudio build/workflow requires it.

## Training

The default training script is:

```text
runs/nevo/nevo_train.py
```

It defines:

- `hyper`: training, loss, bandwidth, dataset, and miscellaneous run settings,
- `config`: model architecture and discriminator settings,
- dataset roots for clean audio, noise, RIRs, and test clips,
- optimizer/scheduler setup,
- checkpoint save/resume logic,
- PESQ/ESTOI validation and best-checkpoint selection.

The script currently contains local Windows dataset paths. Edit these before training:

```python
hyper["misc"]["test_clips_directory"]
dataset_directories
noise_directory
rir_directory
```

Recommended invocation from the run directory:

```powershell
cd runs/nevo
$env:PYTHONPATH = (Resolve-Path ../..).Path
python nevo_train.py
```

Equivalent shell form:

```bash
cd runs/nevo
PYTHONPATH=../.. python nevo_train.py
```

The run writes metrics under `runs/nevo/metrics/` and, when run from `runs/nevo`, writes checkpoints under `runs/nevo/models/`.

Resume behavior is controlled by `resumefromcheckpoint` in `nevo_train.py`:

- `0`: start from scratch.
- `-1`: resume from `./models/temp/nevo.pt`.
- positive integer: resume from `./models/checkpoints/nevo_<epoch>.pt`.

TODO: replace in-script constants with CLI/config files before release.

## ONNX Export

`onnx/convert.py` exports a trained checkpoint into:

- `encoder.onnx`
- `decoder.onnx`
- `codebook.npy`
- `<run_name>.nevo`

The `.nevo` file is a zip bundle containing the encoder, decoder, and codebook files.

Current export configuration is set inside `onnx/convert.py`:

```python
run = "stabilize/nevo_wo_mel_mask"
use_ema = True
use_best = True
```

TODO: replace these constants with command-line arguments before release.

For the default training run, update `run` to:

```python
run = "nevo"
```

Then run:

```bash
python onnx/convert.py
```

## Benchmarking

The consolidated benchmark CLI is:

```bash
python onnx/benchmark/nevo_bench.py -o encdec -m path/to/model.nevo --nc 4
```

Operations:

- `enc`: encode `onnx/benchmark/test_clip.wav` into `test_codes.npy`.
- `dec`: decode `test_codes.npy` into `output.wav`.
- `encdec`: run encode, quantize, decode, save codes, and save audio.

The benchmark prints real-time factor, peak extra RAM, and max frame time.

## Live Streaming Demos

`onnx/benchmark/nevo_stream.py` runs a live microphone-to-speaker codec loopback using an exported `.nevo` bundle.

```bash
python onnx/benchmark/nevo_stream.py --path path/to/model.nevo --nc 4 --in-dev 0 --out-dev 0
```

`onnx/benchmark/nevo_gtcrn_stream.py` adds a GTCRN ONNX denoiser before the codec.

```bash
python onnx/benchmark/nevo_gtcrn_stream.py --path path/to/model.nevo --den-path path/to/gtcrn_simple.onnx --nc 4
```

TODO: remove or replace hard-coded local default paths and device ids in the streaming scripts before release.

## Evaluation

`eval/clip_comparison.py` compares reconstructed clips against references and writes per-bitrate ESTOI/PESQ results.

`eval/plot_metrics.py` plots metrics saved under a `runs/<run>/metrics/history.csv` layout. The training script writes this file through `helpers.info.EpochMetrics`.

## Checkpoint Format

Several scripts expect checkpoint dictionaries with keys like:

```text
config
hyper
gen_model
dis_models
gen_optimizer
dis_optimizers
gen_scheduler
dis_schedulers
balancer
ema_gen_model
last_epoch_info
best_score
last_epoch_idx
rng_state
```

These are written by `helpers.utils.save_allstates()`.

## Third-Party Attribution

This repository includes or references work derived from:

- Meta EnCodec
- lucidrains/vector-quantize-pytorch
- Xiaobin Rong's GTCRN model

See `THIRD_PARTY_NOTICES.md` for copyright, license, source links, and citations.

## Release TODOs

- Add a root project license file.
- Move training configuration out of `runs/nevo/nevo_train.py` into CLI/config files.
- Replace hard-coded dataset roots, run paths, local model paths, and audio device ids with CLI/config options.
- Decide whether benchmark audio, generated WAVs, `.npy` files, and `__pycache__` files should be distributed.
- Add a small smoke test for model construction, ONNX export, and `.nevo` benchmark loading.
- Add model cards or checkpoint metadata for any released weights.
