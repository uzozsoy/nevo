import torch,re,sys
import csv, json, math, sys, re
from collections.abc import Sequence
from pathlib import Path
from datetime import datetime
from prettytable import PrettyTable
from pprint import pprint
from typing import Any, TextIO

colored = True
RED = "\x1b[31m" if colored else ""
GREEN = "\x1b[32m" if colored else ""
RESET = "\x1b[0m" if colored else ""

class EpochMetrics:
    """Collect, print, and persist per-bitrate epoch metrics.

    Args:
        bitrates: Bitrate labels tracked in each row.
        run_dir: Root run directory where `metrics/` will be written.
        metrics: Optional metric spec dictionaries passed to `add_metric`.
    """

    def __init__(
        self,
        bitrates: Sequence[int | float | str],
        run_dir: str | Path = ".",
        metrics: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.bitrates = list(bitrates)
        self.n = len(bitrates)
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.specs = {}
        self.values = {}
        self.counts = {}

        self.add_metric("Bitrate(bps)", mode="none", kind="text")
        self.add_metric("Selected", mode="none", kind="number", fmt=".0f")
        self.values["Bitrate(bps)"] = self.bitrates

        if metrics:
            for spec in metrics:
                self.add_metric(**spec)

    def add_metric(
        self,
        name: str,
        mode: str = "decrease",
        kind: str = "number",
        scale: float = 1.0,
        fmt: str = ".3f",
    ) -> None:
        """
        mode:
            "decrease" -> lower is green
            "increase" -> higher is green
            "none"     -> no comparison
        kind:
            "number" or "text"
        """
        self.specs[name] = dict(mode=mode, kind=kind, scale=scale, fmt=fmt)
        if name not in self.values:
            self.values[name] = [0.0 if kind == "number" else "" for _ in range(self.n)]
            self.counts[name] = [0 for _ in range(self.n)]

    def set(self, name: str, bw_idx: int, value: Any) -> None:
        """Set a metric value for one bitrate index."""
        if name not in self.specs:
            self.add_metric(name)
        self.values[name][bw_idx] = value

    def add(self, name: str, bw_idx: int, value: float, count: int = 1) -> None:
        """Accumulate a numeric metric value for later averaging."""
        if name not in self.specs:
            self.add_metric(name)
        self.values[name][bw_idx] += float(value)
        self.counts[name][bw_idx] += int(count)

    def average(self, name: str, bw_idx: int, denom: int | None = None) -> None:
        """Average one accumulated metric value in-place."""
        spec = self.specs[name]
        if spec["kind"] != "number":
            return
        if denom is None:
            denom = max(1, self.counts[name][bw_idx])
        self.values[name][bw_idx] = self.values[name][bw_idx] * spec["scale"] / max(1, denom)

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable representation of metric state."""
        return {
            "bitrates": self.bitrates,
            "specs": self.specs,
            "values": self.values,
            "counts": self.counts,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], run_dir: str | Path = ".") -> "EpochMetrics":
        """Rebuild an `EpochMetrics` object from saved state."""
        obj = cls(state["bitrates"], run_dir=run_dir)
        obj.specs = state["specs"]
        obj.values = state["values"]
        obj.counts = state["counts"]
        return obj

    def _format_cell(self, name: str, current: Any, previous: Any) -> str:
        """Format one table cell and optionally color-code its change."""
        spec = self.specs[name]

        if spec["kind"] == "text":
            return str(current)

        if current is None or (isinstance(current, float) and math.isnan(current)):
            return "nan"

        text = format(float(current), spec["fmt"])

        if previous is None or spec["mode"] == "none":
            return text
        if previous == 0 or previous is None:
            return text

        change = 100.0 * (float(current) - float(previous)) / float(previous)
        if change == 0:
            return text

        good = (
            (spec["mode"] == "decrease" and change < 0) or
            (spec["mode"] == "increase" and change > 0)
        )

        color = GREEN if good else RED
        arrow = "↑" if change > 0 else "↓"
        return f"{text}{color} {arrow}%{abs(change):.1f}{RESET}"

    def print(self, previous: "EpochMetrics | None" = None) -> PrettyTable:
        """Print and return a PrettyTable for the current metric values."""
        headers = list(self.values.keys())
        rows = []

        for i in range(self.n):
            row = []
            for name in headers:
                cur = self.values[name][i]
                prev = None
                if previous is not None and name in previous.values:
                    prev = previous.values[name][i]
                row.append(self._format_cell(name, cur, prev))
            rows.append(row)

        table = PrettyTable(headers, align="l")
        for r in rows:
            table.add_row(r)
        print(table)
        return table

    def save_epoch(self, epoch: int) -> None:
        """Write the current epoch metrics to JSON and history CSV."""
        out = self.state_dict()
        out["epoch"] = int(epoch)

        with open(self.metrics_dir / f"epoch_{epoch:04d}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        csv_path = self.metrics_dir / "history.csv"
        headers = ["epoch", "bw_idx"] + list(self.values.keys())

        old_rows = []
        if csv_path.exists():
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                old_rows = [r for r in reader if int(r["epoch"]) != int(epoch)]

        new_rows = []
        for bw_idx in range(self.n):
            row = {"epoch": int(epoch), "bw_idx": bw_idx}
            for name, vals in self.values.items():
                row[name] = vals[bw_idx]
            new_rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerows(new_rows)

class CodebookUsageTracker:
    """
    Quantizer-dropout-aware codebook usage tracker for ResidualVQ.

    Assumptions:
      - indices is LongTensor returned by ResidualVQ.
      - Shape can be [num_q, B, T] OR [B, T, num_q]. We auto-detect and permute to [num_q, B, T].
      - Dropped quantizer levels either:
          a) fill indices with -1 (preferred), or
          b) produce no valid tokens (we'll detect empty valid set and mark the level inactive).
    """
    def __init__(
        self,
        num_quantizers: int,
        codebook_sizes: int | Sequence[int],
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if isinstance(codebook_sizes, int):
            codebook_sizes = [codebook_sizes] * num_quantizers
        assert isinstance(codebook_sizes, (list, tuple)) and len(codebook_sizes) == num_quantizers

        self.num_q = num_quantizers
        self.cb_sizes = list(codebook_sizes)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        self.reset()

    def reset(self) -> None:
        """Reset all usage histograms and counters."""
        # Usage counts per level (histograms)
        self.usage = [
            torch.zeros(size=(self.cb_sizes[q],), dtype=torch.long, device=self.device)
            for q in range(self.num_q)
        ]
        # How many steps a level was active (had any valid tokens)
        self.level_active_counts = torch.zeros(self.num_q, dtype=torch.long, device=self.device)
        # Total number of update() calls seen
        self.num_updates = 0

    @staticmethod
    def _to_qbt(indices: torch.Tensor, num_q: int) -> torch.Tensor:
        """
        Ensure indices shape is [num_q, B, T].
        Accept [num_q, B, T] or [B, T, num_q].
        """
        if indices.dim() != 3:
            raise ValueError(f"indices must be 3D, got shape {tuple(indices.shape)}")
        if indices.shape[0] == num_q:
            return indices  # [num_q, B, T]
        if indices.shape[-1] == num_q:
            return indices.permute(2, 0, 1).contiguous()  # [B, T, num_q] -> [num_q, B, T]
        raise ValueError(f"indices shape {tuple(indices.shape)} incompatible with num_q={num_q}")

    def update(self, indices: torch.Tensor) -> None:
        """
        Update histograms using current batch indices (may include dropped levels).
        We treat valid tokens as 0 <= idx < codebook_size. All others are ignored.
        """
        indices = indices.to(self.device, non_blocking=True)
        qbt = self._to_qbt(indices, self.num_q)  # [num_q, B, T]
        self.num_updates += 1

        for q in range(self.num_q):
            idx_q = qbt[q]  # [B, T]
            # valid = indices within [0, cb_size_q)
            valid = (idx_q >= 0) & (idx_q < self.cb_sizes[q])
            if valid.any():
                flat = idx_q[valid].view(-1)
                hist = torch.bincount(flat, minlength=self.cb_sizes[q])
                # Accumulate
                self.usage[q][:hist.numel()] += hist.to(self.usage[q].dtype)
                self.level_active_counts[q] += 1  # level was active for this update
            # else: level dropped for this batch; no update to usage or active count

    @staticmethod
    def _perplexity_from_counts(counts: torch.Tensor) -> float:
        total = counts.sum().item()
        if total == 0:
            return 0.0
        p = counts.float() / float(total)
        # Avoid log(0) for unused entries
        p = torch.clamp(p, min=1e-12)
        H = -(p * torch.log(p)).sum().item()
        return float(torch.exp(torch.tensor(H)).item())

    def stats(self) -> list[dict[str, float | int]]:
        """
        Returns a list of dicts, one per level, with:
          - level, total, used, perplexity, active_ratio
        """
        out = []
        for q in range(self.num_q):
            counts = self.usage[q]
            used = int((counts > 0).sum().item())
            total = int(counts.numel())
            ppl = self._perplexity_from_counts(counts)
            active_ratio = float(self.level_active_counts[q].item() / max(1, self.num_updates))
            out.append({
                "level": q,
                "total": total,
                "used": used,
                "perplexity": ppl,
                "active_ratio": active_ratio,  # fraction of steps where this level was active
            })
        return out

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

class LogTracker:
    """Mirror writes to multiple streams while optionally stripping ANSI color."""

    def __init__(
        self,
        *files: TextIO,
        force_color_streams: Sequence[TextIO] | None = None,
        strip_non_tty: bool = True,
    ) -> None:
        force = set(force_color_streams or [])
        self.targets = []
        for f in files:
            # True only if this handle is actually a TTY
            is_tty = False
            try:
                is_tty = getattr(f, "isatty", lambda: False)()
            except Exception:
                pass

            # Color is allowed ONLY if:
            #  - it is a TTY, or
            #  - you explicitly forced this exact stream
            supports_ansi = is_tty or (f in force)

            # NOTE: do NOT use PYCHARM_HOSTED here; we explicitly pass stdout/stderr in 'force'
            self.targets.append((f, supports_ansi, strip_non_tty))

    def write(self, obj: str) -> None:
        """Write one string to every target stream."""
        for f, supports_ansi, strip_non_tty in self.targets:
            s = obj if supports_ansi else (_ANSI_RE.sub('', obj) if strip_non_tty else obj)
            f.write(s)
            f.flush()

    def flush(self) -> None:
        """Flush every target stream."""
        for f, *_ in self.targets:
            f.flush()

def start_logging(
    hyper: dict[str, dict[str, Any]],
    config: dict[str, Any],
    name: str,
    disable_warnings: bool = True,
) -> None:
    """Redirect stdout/stderr to console plus a timestamped log file."""
    if disable_warnings:
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=(
                r"(?:.*mel filterbank has all zero values.*"
                r"|.*In 2\.9, this function's implementation will be changed to use `?torchaudio\.load_with_torchcodec`?.*)"
            ),
            category=UserWarning,
        )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_dir = Path("logs/") / f"{name}_{timestamp}.txt"
    log_file_dir.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_file_dir, "w", encoding="utf-8")

    # Force color ONLY on the interactive console streams
    sys.stdout = LogTracker(sys.__stdout__, log_file, force_color_streams=[sys.__stdout__])
    sys.stderr = LogTracker(sys.__stderr__, log_file, force_color_streams=[sys.__stderr__])

    for section, params in hyper.items():
        print(f"------------ {section} parameters --------------")
        for pname, value in params.items():
            print(f"{pname}: {value}")

    print("----------------- model config -------------------")
    pprint(config, sort_dicts=False,)
    print("--------------------------------------------------")
