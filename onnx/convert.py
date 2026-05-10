import torch
import math
from pathlib import Path
import numpy as np
from model import NevoModel
import re
import zipfile

from io_helpers import (
    encoder_io_names, decoder_io_names,
    EncoderONNXWrapper, DecoderONNXWrapper,
    flatten_tensors
)


######################
run="main/nevo"
######################

use_ema = True
use_best = True

if not use_best: #checkpoint epoch, 0 for final model, and -1 for custom directory
    checkpoint = 50
    if checkpoint==-1:
        custom_dir = r""

######################################################################################

device = "cpu"

onnx_dir = Path(__file__).resolve().parent
root_dir = onnx_dir.parent

run_dir = root_dir / "runs"

parts = re.split(r"[\\/]", run)
name = parts[-1]
sub_dir = Path(*parts[:-1])  # empty Path(".") if no folders

if use_best:
    saveloc = run_dir / sub_dir / name / ("models/" + name + "_best.pt")
elif checkpoint>0:
    saveloc = run_dir / sub_dir / name/ ("models/checkpoints/"+name+"_"+str(checkpoint)+".pt")
elif checkpoint==0:
    saveloc = run_dir / sub_dir / name/ ("models/"+name+".pt")
elif checkpoint == -1:
    saveloc = custom_dir

allstates=torch.load(saveloc,weights_only=False)
config=allstates["config"]
hyper=allstates["hyper"]

model = NevoModel(config)

if use_ema:
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.99), use_buffers=True)
    ema.load_state_dict(allstates["ema_gen_model"])  # contains n_averaged + module.*
    model.load_state_dict(ema.module.state_dict())
else:
    model.load_state_dict(allstates["gen_model"])

encoder_model = model.encoder
decoder_model = model.decoder

encoder_model.export_mode()
decoder_model.export_mode()

#####------onnx-----#####

frame_length=int(math.prod(config["strides"]))
latent_dim = int(config["VQ"]["latent_dim"])

x_audio = torch.zeros(1, 1, frame_length, device=device, dtype=torch.float32)
x_latent = torch.zeros(1, latent_dim, 1, device=device, dtype=torch.float32)

# ---------- build states (no args) ----------
enc_in_names, enc_out_names = encoder_io_names(encoder_model)
dec_in_names, dec_out_names = decoder_io_names(decoder_model)

enc_state_spec = encoder_model.export_initial_state()
dec_state_spec = decoder_model.export_initial_state()

enc_state_flat = flatten_tensors(enc_state_spec)
dec_state_flat = flatten_tensors(dec_state_spec)

enc_wrap = EncoderONNXWrapper(encoder_model, enc_state_spec).eval()
dec_wrap = DecoderONNXWrapper(decoder_model, dec_state_spec).eval()

# ---------- export paths ----------
onnx_out_dir = run_dir / sub_dir / name/ "onnx"
onnx_out_dir.mkdir(parents=True, exist_ok=True)

enc_onnx_path = onnx_out_dir / f"encoder.onnx"
dec_onnx_path = onnx_out_dir / f"decoder.onnx"
codebook_path = onnx_out_dir / "codebook.npy"

torch.onnx.export(
    enc_wrap,
    (x_audio, *enc_state_flat),
    str(enc_onnx_path),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=enc_in_names,
    output_names=enc_out_names,
    dynamo=False
)

torch.onnx.export(
    dec_wrap,
    (x_latent, *dec_state_flat),
    str(dec_onnx_path),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=dec_in_names,
    output_names=dec_out_names,
    dynamo=False
)

codebooks=model.vq.codebooks
np.save(codebook_path, codebooks.numpy())

nevo_path = onnx_out_dir / f"{name}.nevo"

with zipfile.ZipFile(nevo_path, "w", compression=zipfile.ZIP_STORED) as z:
    z.write(enc_onnx_path, arcname="encoder.onnx")
    z.write(dec_onnx_path, arcname="decoder.onnx")
    z.write(codebook_path, arcname="codebook.npy")

print("Saved NEVO bundle:", nevo_path)
