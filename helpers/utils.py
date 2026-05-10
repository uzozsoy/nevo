import torch,time,math,wave,os,tempfile,random
import numpy as np
from pystoi import stoi
from scipy.signal import correlate, correlation_lags
from typing import Any

class TrainingTimer:
    """Track epoch durations and estimate remaining training time."""

    def __init__(self, epochs: int = 0) -> None:
        self.timestamps = np.zeros(epochs+1)
        self.elapsedtimes = np.zeros(epochs)
        self.epochs = epochs
        self.epochcounter = 0

    def convert(self, seconds: float, tostring: bool = True) -> str | tuple[int, int, int, int]:
        """Convert seconds to either a display string or `(d, h, m, s)` tuple."""
        seconds = int(round(seconds))
        days = seconds // (60*60*24)
        seconds = seconds % (60*60*24)

        hours = seconds // (60*60)
        seconds = seconds % (60*60)

        minutes = seconds // 60
        seconds = seconds % 60

        if tostring:
            return str(days) + " d, " + str(hours) + " h, " + str(minutes) + " m, " + str(seconds) + " s "
        else:
            return days, hours, minutes, seconds

    def start(self) -> None:
        """Record the start timestamp."""
        self.timestamps[0]=time.time()

    def step(self, returnremaining: bool = False) -> float | None:
        """Record one completed epoch and optionally estimate seconds remaining."""
        self.epochcounter+=1
        currenttime = time.time()
        self.timestamps[self.epochcounter] = currenttime
        self.elapsedtimes[self.epochcounter-1]=currenttime - self.timestamps[self.epochcounter-1]
        if returnremaining:
            weightedsum = np.sum(self.elapsedtimes * np.arange(1, self.epochs+1))
            return (weightedsum/np.sum(np.arange(1, self.epochcounter+1)))*(self.epochs-self.epochcounter)
    def stop(self) -> None:
        """Placeholder for future timer finalization."""
        pass

class WarmupCosineLR_Scheduler:
    """Callable learning-rate schedule helper for warmup plus cosine decay."""

    def __init__(
        self,
        num_epochs: int,
        num_warmup: int | None = None,
        min_lr_coeff: float = 0.1,
        linear: bool = False,
    ) -> None:
        self.num_epochs = num_epochs
        self.min_lr_coeff = min_lr_coeff
        self.linear = linear
        if num_warmup is None:
            self.num_warmup = num_epochs//10
        else:
            self.num_warmup = num_warmup
    def lrlambda(self, step: int) -> float:
        """Return LR multiplier for a zero-based scheduler step."""
        epoch=step+1
        lrdiff=1-self.min_lr_coeff
        if epoch<self.num_warmup:
            if self.linear is False: return math.sin((math.pi/2)*(epoch/self.num_warmup))
            else: return epoch/self.num_warmup
        else:
            return ((lrdiff/2)*math.cos(  (math.pi/(self.num_epochs-self.num_warmup)) * (epoch-self.num_warmup) ))  +  self.min_lr_coeff+(lrdiff/2)

class Ramp():
    """Linear ramp from an initial value to 1 over a step interval."""

    def __init__(self, init: float = 0.0, start: int = 0, end: int = 0) -> None:
        self.init = init
        self.start = start
        self.end = end

    def val(self, x: int | float) -> float:
        """Return the ramp value at step `x`."""
        if x < self.start: return self.init
        if x >= self.start and x <= self.end: return ((x-self.start)/(self.end-self.start))*(1-self.init)+self.init #line from known 2 points formula
        if x > self.end: return 1

def save_wav(
    gen_model: torch.nn.Module,
    audiodata: torch.Tensor,
    name: str,
    legacy_mode: bool = False,
) -> None:
    """Decode one audio tensor and write it as `outputwavs/<name>.wav`."""
    if legacy_mode:
        gen_model.eval()
        gen_model.vqeval()
        decoded_wch = gen_model(audiodata.unsqueeze(0).unsqueeze(0))
    else:
        gen_model.eval()
        decoded_wch = gen_model.evalforward(audiodata.unsqueeze(0).unsqueeze(0))
    audiotosave = (decoded_wch.cpu().detach().flatten().numpy() * (2 ** 15 - 1)).astype("<h")
    with wave.open("./outputwavs/" + name + ".wav", "w") as f:
        # 2 Channels.
        f.setnchannels(1)
        # 2 bytes per sample.
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(audiotosave.tobytes())

def atomic_torch_save(obj: Any, path: str) -> None:
    """Atomically save a PyTorch object by replacing a temp file in-place."""
    # ensure target dir exists
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

    # create temp file in SAME dir so replace is atomic
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic swap
        # (optional) also fsync the directory for extra crash safety:
        # dir_fd = os.open(d, os.O_DIRECTORY)
        # try: os.fsync(dir_fd)
        # finally: os.close(dir_fd)
    finally:
        # clean up stray temp if something failed before replace
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass

def save_allstates(
    hyper: dict[str, Any],
    config: dict[str, Any],
    epoch: int,
    gen_model: torch.nn.Module,
    dis_model: torch.nn.Module | list[torch.nn.Module] | tuple[torch.nn.Module, ...],
    gen_optimizer: Any,
    dis_optimizer: Any,
    gen_scheduler: Any,
    dis_scheduler: Any,
    balancer: Any,
    savelocation: str,
    ema_gen_model: torch.nn.Module | None = None,
    last_epoch_info: Any = None,
    best_score: float | None = None,
    save_rng_state: bool = False,
) -> None:
    """Save models, optimizers, schedulers, metadata, and optional RNG state."""
    allstates_dict = {
        "hyper": hyper,
        "config": config,
        "last_epoch_idx": int(epoch),
        "gen_model": gen_model.state_dict(),
        "dis_models": [m.state_dict() for m in dis_model] if isinstance(dis_model, (list, tuple)) else dis_model.state_dict(),
        "gen_optimizer": gen_optimizer.state_dict(),
        "dis_optimizers": [o.state_dict() for o in dis_optimizer] if isinstance(dis_optimizer, (list, tuple)) else dis_optimizer.state_dict(),
        "gen_scheduler": gen_scheduler.state_dict() if gen_scheduler is not None else None,
        "dis_schedulers": [s.state_dict() for s in dis_scheduler] if isinstance(dis_scheduler, (list, tuple)) else (dis_scheduler.state_dict() if dis_scheduler is not None else None),
        "balancer": balancer.state_dict() if (balancer is not None and hasattr(balancer, "state_dict")) else None,
        "ema_gen_model": ema_gen_model.state_dict() if ema_gen_model is not None else None,
        "last_epoch_info": last_epoch_info if last_epoch_info is not None else None,
        "best_score": best_score if best_score is not None else None,
    }
    if save_rng_state:
        allstates_dict["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,}
    atomic_torch_save(allstates_dict, savelocation)

def set_rng_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def stoi_aligned(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    fs_sig: int,
    extended: bool = False,
    max_delay_ms: float = 200,
    min_overlap: float = 0.5,
    remove_dc: bool = True,
    debug: bool = False,
) -> float:
    """Align two signals by cross-correlation before computing STOI/ESTOI."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    x_corr = x - np.mean(x) if remove_dc else x
    y_corr = y - np.mean(y) if remove_dc else y

    max_lag = int(round(fs_sig * max_delay_ms / 1000.0))
    max_lag = min(max_lag, len(x) - 1, len(y) - 1)

    corr = correlate(y_corr, x_corr, mode="full", method="fft")
    lags = correlation_lags(len(y_corr), len(x_corr), mode="full")

    valid = (lags >= -max_lag) & (lags <= max_lag)
    best_lag = int(lags[valid][np.argmax(corr[valid])])

    if best_lag > 0:
        x_aligned = x[: len(y) - best_lag]
        y_aligned = y[best_lag:]
    elif best_lag < 0:
        lag = -best_lag
        x_aligned = x[lag:]
        y_aligned = y[: len(x) - lag]
    else:
        n0 = min(len(x), len(y))
        x_aligned = x[:n0]
        y_aligned = y[:n0]

    n = min(len(x_aligned), len(y_aligned))
    x_aligned = x_aligned[:n]
    y_aligned = y_aligned[:n]

    if n < min(len(x), len(y)) * min_overlap:
        raise ValueError(f"Aligned overlap too short: {n} samples, lag={best_lag}")

    score = stoi(x_aligned, y_aligned, fs_sig, extended=extended)

    if debug:
        print(
            f"STOI alignment: lag={best_lag} samples "
            f"({1000.0 * best_lag / fs_sig:.2f} ms), "
            f"overlap={n} samples"
        )

    return score
