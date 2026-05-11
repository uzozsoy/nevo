# NEVO

NEVO is a neural audio codec research codebase for low-bitrate audio compression. The core model encodes mono waveform frames into latent features, quantizes them with residual vector quantization (RVQ), and decodes the quantized latents back to audio.

This repository currently contains the model code, dataset utilities, losses, ONNX export helpers, benchmark scripts, and evaluation/plotting scripts.

## Repository Layout

```text
.
|-- dataset.py                  # AudioDataset with clean/noise/RIR augmentation
|-- model.py                    # NEVO encoder, decoder, RVQ model, discriminator
|-- requirements.txt            # Core non-PyTorch dependencies
|-- requirements-onnx.txt       # Optional ONNX/export/benchmark dependencies
|-- requirements-plot.txt       # Optional plotting dependencies
|-- requirements-stream.txt     # Optional live audio dependencies
|-- encodec/                    # EnCodec's loss balancer training utility
|-- helpers/
|   |-- info.py                 # Logging, metrics, codebook usage tracking
|   |-- losses.py               # Multi scale mel spectrogram, adversarial, feature losses
|   |-- modules.py              # Streaming capable convolution and LSTM modules, and Mel Spectrogram
|   `-- utils.py                # Checkpoint, seeding, STOI alignment helpers
|-- onnx/
|   |-- convert.py              # Exports encoder/decoder/codebooks into .nevo bundle
|   |-- io_helpers.py           # ONNX wrapper/state flattening helpers
|   `-- benchmark/              # Offline and live ONNX benchmark scripts
|-- eval/
|   |-- clip_comparison.py      # Processes wavs in a directory to compute metrics and save converted files    
|   |-- plot_metrics.py         # Allows to plot losses, and metric scores to evaluate different runs
|-- runs/
|   `-- main/
|       `-- nevo/
|           `-- nevo_train.py   # Default training experiment
|-- vq/                         # vector-quantize-pytorch-derived RVQ implementation
`-- THIRD_PARTY_NOTICES.md      # Third-party attribution and license notices
```

## Core Concepts

The main model classes live in `model.py`.

- `NevoEncoder`: Converts waveform input shaped `[batch, 1, samples]` into latent frames.
- `ResidualVQ`: Quantizes latent frames using multiple residual codebooks.
- `NevoDecoder`: Reconstructs waveform frames from quantized latents.
- `NevoModel`: End-to-end encoder, RVQ, decoder model.
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

Example:

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
runs/main/nevo/nevo_train.py
```

It defines:

- `hyper`: training, loss, bandwidth, dataset, and miscellaneous run settings,
- `config`: model architecture and discriminator settings,
- dataset roots for clean audio, noise, RIRs, and test clips,
- optimizer/scheduler setup,
- checkpoint save/resume logic,
- PESQ/ESTOI validation and best-checkpoint selection.

The run writes metrics under `runs/main/nevo/metrics/` and, when run from `runs/main/nevo`, writes checkpoints under `runs/main/nevo/models/`.

You can create as many subdirectories as you want. Create you run as: `runs/**/run-name/run-name_train.py` and use the tools as 

```python
run = "**//run-name"
```

Resume behavior is controlled by `resumefromcheckpoint` in `nevo_train.py`:

- `0`: start from scratch.
- `-1`: resume from `./models/temp/nevo.pt`.
- positive integer: resume from `./models/checkpoints/nevo_<epoch>.pt`.

## ONNX Export

`onnx/convert.py` exports a trained checkpoint into:

- `encoder.onnx`
- `decoder.onnx`
- `codebook.npy`
- `<run_name>.nevo`

The `.nevo` file is a zip bundle containing the encoder, decoder, and codebook files.

Current export configuration is set inside `onnx/convert.py`:

```python
run = "main/nevo"
use_ema = True
use_best = True
```

## Benchmarking

The benchmark CLI for RTF calculation is:

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

`onnx/benchmark/nevo_gtcrn_stream.py` adds a GTCRN ONNX denoiser before the codec. This is the preferred run method.

```bash
python onnx/benchmark/nevo_gtcrn_stream.py --path path/to/model.nevo --den-path path/to/gtcrn_simple.onnx --nc 8
```

## Evaluation

`eval/clip_comparison.py` compares reconstructed clips against references and writes per-bitrate ESTOI/PESQ results.

`eval/plot_metrics.py` plots metrics saved under a `runs/<run>/metrics/history.csv` layout.

## Third-Party Attribution

This repository includes or references work derived from:

- Meta EnCodec
- lucidrains/vector-quantize-pytorch
- Xiaobin Rong's GTCRN model

See `THIRD_PARTY_NOTICES.md` for copyright, license, source links, and citations.

