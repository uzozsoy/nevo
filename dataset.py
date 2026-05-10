import random
from collections.abc import Sequence
from pathlib import Path

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio.*deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*load_with_torchcodec.*",
    category=UserWarning,
)

import torch
import torchaudio
from torch.utils.data import Dataset


AUDIO_EXTS = {
    ".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".mp4"
}


def power(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return mean squared signal power with a numeric floor."""
    return torch.mean(x ** 2) + eps


def snr_linear(snr_db: float) -> float:
    """Convert an SNR value from decibels to linear scale."""
    return 10 ** (snr_db / 10.0)


class AudioDataset(Dataset):
    """Sample clean/altered audio chunks for codec training.

    Args:
        dataset_directories: Audio roots to sample clean speech/audio from.
        dataset_weights: Optional sampling weights for each dataset root.
        frames_in_chunk: Number of model frames returned per sample.
        sample_rate: Target sample rate after loading/resampling.
        frame_length: Samples per model frame.
        samples_per_epoch: Virtual dataset length exposed to DataLoader.
        noise_prob: Probability of adding a sampled noise chunk.
        noise_magnitude: Scalar or range used when `snr_range` is not set.
        snr_range: Optional `(min_db, max_db)` range for SNR-based mixing.
        noise_directory: Audio root used for noise sampling.
        preload_noise: If true, load all noise files during initialization.
        rir_prob: Probability of applying a room impulse response.
        rir_directory: Audio root used for RIR sampling.
        debug: Print cache/loading diagnostics.
    """

    def __init__(
        self,
        dataset_directories: str | Path | Sequence[str | Path],
        dataset_weights: Sequence[float] | None = None,
        frames_in_chunk: int = 25,
        sample_rate: int = 8000,
        frame_length: int = 320,
        samples_per_epoch: int = 100_000,
        noise_prob: float = 0,
        noise_magnitude: float | Sequence[float] = 1,
        snr_range: Sequence[float] | None = None,
        noise_directory: str | Path | None = None,
        preload_noise: bool = False,
        rir_prob: float = 0,
        rir_directory: str | Path | None = None,
        debug: bool = False,
    ) -> None:
        if isinstance(dataset_directories, (str, Path)):
            dataset_directories = [dataset_directories]

        self.directories = [Path(d) for d in dataset_directories]
        self.frames_in_chunk = frames_in_chunk
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.chunk_len = frames_in_chunk * frame_length
        self.samples_per_epoch = samples_per_epoch

        if dataset_weights is None:
            dataset_weights = [1.0] * len(self.directories)

        assert len(dataset_weights) == len(self.directories)

        self.dataset_weights = torch.tensor(dataset_weights, dtype=torch.float32)

        self.dataset_paths = []
        self.file_weights = []

        for directory in self.directories:

            cache_path = directory / f"_audio_dataset_cache_sr{self.sample_rate}_chunk{self.chunk_len}.pt"
            #this cached is used to initialize the dataset faster. if you change the dataset, don't forget to delete this cache

            if cache_path.exists():
                cache = torch.load(cache_path, weights_only=False)
                valid_paths = [Path(p) for p in cache["paths"]]
                weights = cache["weights"]

                if debug:
                    print(f"Loaded cache: {cache_path} | files={len(valid_paths)}")

            else:
                paths = sorted(
                    p for p in directory.rglob("*")
                    if p.is_file() and p.suffix.lower() in AUDIO_EXTS
                )

                valid_paths = []
                weights = []

                for path in paths:
                    try:
                        info = torchaudio.info(str(path))
                        length = int(info.num_frames)

                        if info.sample_rate != self.sample_rate:
                            length = int(length * self.sample_rate / info.sample_rate)

                        if length < self.chunk_len:
                            if debug:
                                print(f"Skipped short file: {path} | samples={length}")
                            continue

                        usable_crop_positions = length - self.chunk_len + 1

                        valid_paths.append(path)
                        weights.append(float(usable_crop_positions))

                    except Exception as e:
                        if debug:
                            print(f"Skipped unreadable file: {path} | {e}")

                torch.save(
                    {
                        "paths": [str(p) for p in valid_paths],
                        "weights": weights,
                        "sample_rate": self.sample_rate,
                        "chunk_len": self.chunk_len,
                    },
                    cache_path,
                )

                if debug:
                    print(f"Saved cache: {cache_path} | files={len(valid_paths)}")

            if len(valid_paths) == 0:
                raise RuntimeError(f"No valid audio files found in {directory}")

            self.dataset_paths.append(valid_paths)
            self.file_weights.append(torch.tensor(weights, dtype=torch.float32))

            if debug:
                print(
                    f"Dataset: {directory} | "
                    f"files={len(valid_paths)} | "
                    f"total_weight={sum(weights):.0f}"
                )

        self.noise_prob = noise_prob
        self.noise_magnitude = noise_magnitude
        self.snr_range = snr_range
        self.use_snr_range = snr_range is not None

        self.preload_noise = preload_noise
        self.noise_paths = []
        self.noises = []
        if noise_prob:
            noise_root = Path(noise_directory)
            self.noise_paths = sorted(
                p for p in noise_root.rglob("*")
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS
            )
            if len(self.noise_paths) == 0:
                raise RuntimeError(f"No noise files found in {noise_root}")

            if self.preload_noise:
                self.noises = [self._load_audio(p) for p in self.noise_paths]

                if debug:
                    total_samples = sum(n.numel() for n in self.noises)
                    total_mb = total_samples * 4 / 1024 / 1024
                    print(
                        f"Preloaded noise files: {len(self.noises)} | "
                        f"approx {total_mb:.1f} MB"
                    )
            else:
                if debug:
                    print(f"Using on-demand noise loading: {len(self.noise_paths)} files")

        self.rir_prob = rir_prob
        self.rir_paths = []
        if rir_prob:
            rir_root = Path(rir_directory)
            self.rir_paths = sorted(
                p for p in rir_root.rglob("*")
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS
            )
            if len(self.rir_paths) == 0:
                raise RuntimeError(f"No RIR files found in {rir_root}")

        if rir_prob:
            rir_root = Path(rir_directory)
            self.rir_paths = sorted(
                p for p in rir_root.rglob("*")
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS
            )

            if len(self.rir_paths) == 0:
                raise RuntimeError(f"No RIR files found in {rir_root}")

            self.rirs = [self._load_rir(p) for p in self.rir_paths]

            if debug:
                print(f"Preloaded RIRs: {len(self.rirs)}")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_audio(self, path: str | Path) -> torch.Tensor:
        """Load mono audio at the dataset sample rate as a flat float tensor."""
        audio, sr = torchaudio.load(str(path))

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        return audio.flatten().float()

    def _load_rir(self, path: str | Path, eps: float = 1e-8) -> torch.Tensor:
        """Load and normalize a mono room impulse response."""
        rir, sr = torchaudio.load(str(path))

        # mono
        rir = rir.mean(dim=0).float()

        # resample
        if sr != self.sample_rate:
            rir = torchaudio.functional.resample(rir, sr, self.sample_rate)

        # old preprocessing logic
        rir = rir - rir.mean()
        rir = rir / torch.linalg.vector_norm(rir, ord=2).clamp_min(eps)

        peak = torch.argmax(rir.abs()).item()
        rir = rir[peak:]

        return rir

    @staticmethod
    def _normalize_if_clipping(audio: torch.Tensor) -> torch.Tensor:
        """Scale audio down only when its peak exceeds the expected range."""
        peak = audio.abs().max()
        if peak > 1.0:
            audio = audio / peak
        return audio

    def _random_crop(self, audio: torch.Tensor) -> torch.Tensor:
        """Return one random `chunk_len` slice from a longer audio tensor."""
        start = random.randint(0, audio.numel() - self.chunk_len)
        return audio[start:start + self.chunk_len]

    def _sample_dataset_idx(self) -> int:
        """Sample a dataset root index according to configured weights."""
        return torch.multinomial(
            self.dataset_weights,
            num_samples=1,
            replacement=True,
        ).item()

    def _sample_file_idx(self, dataset_idx: int) -> int:
        """Sample an audio file index from one dataset root."""
        return torch.multinomial(
            self.file_weights[dataset_idx],
            num_samples=1,
            replacement=True,
        ).item()

    def _sample_clean_chunk(self) -> torch.Tensor:
        """Load one clean file and return a random training chunk."""
        dataset_idx = self._sample_dataset_idx()
        file_idx = self._sample_file_idx(dataset_idx)

        path = self.dataset_paths[dataset_idx][file_idx]
        audio = self._load_audio(path)

        return self._random_crop(audio)

    def _sample_noise_chunk(self, length: int) -> torch.Tensor:
        """Return a noise chunk of exactly `length` samples."""
        if self.preload_noise:
            noise = random.choice(self.noises)
        else:
            path = random.choice(self.noise_paths)
            noise = self._load_audio(path)

        if noise.numel() < length:
            repeats = (length + noise.numel() - 1) // noise.numel()
            noise = noise.repeat(repeats)

        start = random.randint(0, noise.numel() - length)
        return noise[start:start + length]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a tensor shaped `[2, chunk_len]` as `[altered, clean]`."""
        clean = self._sample_clean_chunk()
        clean = self._normalize_if_clipping(clean)

        altered = clean.clone()

        if self.rir_prob and random.random() < self.rir_prob:
            rir = random.choice(self.rirs)

            altered = torchaudio.functional.fftconvolve(altered, rir)
            altered = altered[:self.chunk_len]

        if self.noise_prob and random.random() < self.noise_prob:
            noise = self._sample_noise_chunk(self.chunk_len)

            if self.use_snr_range:
                snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
                multiplier = torch.sqrt(
                    power(altered) / (power(noise) * snr_linear(snr_db))
                )
            else:
                if isinstance(self.noise_magnitude, (tuple, list)):
                    multiplier = random.uniform(
                        self.noise_magnitude[0],
                        self.noise_magnitude[1],
                    )
                else:
                    multiplier = random.uniform(0, self.noise_magnitude)

            altered = altered + multiplier * noise

        return torch.stack([altered, clean], dim=0).float()
