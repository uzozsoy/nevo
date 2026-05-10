import torch
import math
import inspect
from pathlib import Path
from typing import Any

# ------------------------------------------------------------
# 0) Utility: flatten / unflatten nested (tuple/list) of tensors
# ------------------------------------------------------------
def flatten_tensors(tree: Any) -> list[torch.Tensor]:
    """Flatten a nested tuple/list tensor state tree into a list."""
    flat = []

    def rec(x: Any) -> None:
        if torch.is_tensor(x):
            flat.append(x)
        elif isinstance(x, (tuple, list)):
            for v in x:
                rec(v)
        else:
            raise TypeError(f"Non-tensor leaf in state tree: {type(x)} -> {x}")
    rec(tree)
    return flat

def unflatten_like(spec_tree: Any, flat_list: list[torch.Tensor]) -> Any:
    """Rebuild a nested tensor tree using `spec_tree` and a flat tensor list."""
    it = iter(flat_list)

    def rec(spec: Any) -> Any:
        if torch.is_tensor(spec):
            return next(it)
        elif isinstance(spec, tuple):
            return tuple(rec(s) for s in spec)
        elif isinstance(spec, list):
            return [rec(s) for s in spec]
        else:
            raise TypeError(f"Non-tensor leaf in spec tree: {type(spec)} -> {spec}")
    out = rec(spec_tree)
    # ensure consumed
    try:
        next(it)
        raise RuntimeError("Too many flat tensors provided for the spec structure.")
    except StopIteration:
        pass
    return out

# ------------------------------------------------------------
# 1) Hierarchical naming (encoder / decoder)
# These names assume your state structure is:
# Encoder states: (init_conv, enc_block_states, lstm_state(h,c), final_conv)
#   each EncoderBlock state: (ru0_state, ru1_state, ..., ruN_state, downsample_conv_state)
#   each ResidualUnit state: (conv0_state, conv1_state, ...)
#
# Decoder states: (init_conv, dec_block_states, lstm_state(h,c), final_conv)
#   each DecoderBlock state: (convt_state, ru0_state, ru1_state, ...)
#   each ResidualUnit state: (conv0_state, conv1_state, ...)
# ------------------------------------------------------------
def encoder_io_names(enc: torch.nn.Module) -> tuple[list[str], list[str]]:
    """Return flat ONNX input/output names for an exported encoder."""
    in_names = ["x_audio"]
    out_names = ["y_latent"]

    # initial conv state
    in_names.append("enc/initial_conv/state")
    out_names.append("enc/initial_conv/state_out")

    # blocks
    for bi, blk in enumerate(enc.encoder_blocks):
        # res units conv states
        for ri, ru in enumerate(blk.res_units):
            for ci, _ in enumerate(ru.convs):
                in_names.append(f"enc/block{bi}/res{ri}/conv{ci}/state")
                out_names.append(f"enc/block{bi}/res{ri}/conv{ci}/state_out")
        # block downsample conv state
        in_names.append(f"enc/block{bi}/downsample_conv/state")
        out_names.append(f"enc/block{bi}/downsample_conv/state_out")

    # lstm
    if enc.lstm is not None:
        in_names += ["enc/lstm/h", "enc/lstm/c"]
        out_names += ["enc/lstm/h_out", "enc/lstm/c_out"]

    # final conv
    in_names.append("enc/final_conv/state")
    out_names.append("enc/final_conv/state_out")

    return in_names, out_names

def decoder_io_names(dec: torch.nn.Module) -> tuple[list[str], list[str]]:
    """Return flat ONNX input/output names for an exported decoder."""
    in_names = ["x_latent"]
    out_names = ["y_audio"]

    # initial conv
    in_names.append("dec/initial_conv/state")
    out_names.append("dec/initial_conv/state_out")

    # blocks FIRST (matches state_spec flatten order)
    for bi, blk in enumerate(dec.decoder_blocks):
        in_names.append(f"dec/block{bi}/convt/state")
        out_names.append(f"dec/block{bi}/convt/state_out")

        for ri, ru in enumerate(blk.res_units):
            for ci, _ in enumerate(ru.convs):
                in_names.append(f"dec/block{bi}/res{ri}/conv{ci}/state")
                out_names.append(f"dec/block{bi}/res{ri}/conv{ci}/state_out")

    # LSTM AFTER blocks even though the logical order is different
    if dec.lstm is not None:
        in_names += ["dec/lstm/h", "dec/lstm/c"]
        out_names += ["dec/lstm/h_out", "dec/lstm/c_out"]

    # final conv
    in_names.append("dec/final_conv/state")
    out_names.append("dec/final_conv/state_out")

    return in_names, out_names

# ------------------------------------------------------------
# 3) Wrapper modules (so ONNX I/O is flat tensors with nice names)
# ------------------------------------------------------------
class EncoderONNXWrapper(torch.nn.Module):
    """Flatten explicit encoder state for ONNX export."""

    def __init__(self, enc: torch.nn.Module, state_spec: Any) -> None:
        super().__init__()
        self.enc = enc
        self.state_spec = state_spec

    def forward(self, x: torch.Tensor, *flat_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run encoder export path and return flat output state tensors."""
        # flat_states correspond to flatten_tensors(state_spec)
        nested = unflatten_like(self.state_spec, list(flat_states))
        y, s_init, s_blk, s_lstm, s_final = self.enc.export_forward(x, *nested)
        out_states = (s_init, s_blk, s_lstm, s_final)
        flat_out = flatten_tensors(out_states)
        return (y, *flat_out)

class DecoderONNXWrapper(torch.nn.Module):
    """Flatten explicit decoder state for ONNX export."""

    def __init__(self, dec: torch.nn.Module, state_spec: Any) -> None:
        super().__init__()
        self.dec = dec
        self.state_spec = state_spec

    def forward(self, x: torch.Tensor, *flat_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run decoder export path and return flat output state tensors."""
        nested = unflatten_like(self.state_spec, list(flat_states))
        y, s_init, s_blk, s_lstm, s_final = self.dec.export_forward(x, *nested)
        out_states = (s_init, s_blk, s_lstm, s_final)
        flat_out = flatten_tensors(out_states)
        return (y, *flat_out)

