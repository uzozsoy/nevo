import re

import torch,torchaudio,math
import torch.nn.functional as F
from model import NevoModel
from pesq import pesq
from helpers.utils import stoi_aligned
from pathlib import Path


######################
run="stabilize/nevo_wo_mel_mask"
######################

use_ema = True
use_best = True

if not use_best: #checkpoint epoch, 0 for final model, and -1 for custom directory
    checkpoint = -1
    if checkpoint==-1:
        custom_dir = r""

######################################################################################

clips_directory=r".\comparison\from_melpe_website\reference"

save=True
score=True

######################################################################################

device = "cpu"

eval_dir = Path(__file__).resolve().parent
root_dir = eval_dir.parent

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

print("Epoch:",allstates["last_epoch_idx"])

model = NevoModel(config)

if use_ema:
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.99), use_buffers=True)
    ema.load_state_dict(allstates["ema_gen_model"])  # contains n_averaged + module.*
    model.load_state_dict(ema.module.state_dict())
else:
    model.load_state_dict(allstates["gen_model"])

sample_rate=config["sample_rate"]
frame_length=int(math.prod(config["strides"]))

rows=[]
upper_dir = Path(clips_directory).parent
clips_root = Path(clips_directory)
for clip_loc in clips_root.rglob("*.wav"):
    clip_name = clip_loc.stem
    waveform, sr = torchaudio.load(clip_loc)
    if waveform.shape[0] != 1: waveform = waveform.mean(dim=0, keepdim=True)
    audiotocompress = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
    audiodata = F.pad(audiotocompress, [320 - (len(audiotocompress[-1]) % frame_length), 0]).flatten()
    for idx, q_lvl in enumerate(hyper["bandwidth"]["quantizer_levels"]):
        with torch.no_grad():
            decoded = torch.zeros(len(audiodata) // frame_length, frame_length)
            model.eval()
            model.stream()
            for i in range((len(audiodata) // frame_length)):
                audiotocompress = audiodata[i * frame_length:(i + 1) * frame_length]
                decoded[i] = model.evalforward(audiotocompress.unsqueeze(0).unsqueeze(0),
                                               quantizer_limit=q_lvl).flatten()
            bandwidth = hyper["bandwidth"]["bw_per_q_level(bps)"][idx]
            save_name = f"{clip_name}_{name}_{bandwidth}bps.wav"
            save_loc= upper_dir / run / save_name
            save_loc.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(save_loc, decoded.flatten().unsqueeze(0), sample_rate)
            if score:
                reference = audiodata.flatten().numpy()
                decoded = decoded.flatten().numpy()
            rows.append({
                "clip": clip_name,
                "bitrate_bps": bandwidth,
                "estoi": float(stoi_aligned(reference, decoded, sample_rate, True,debug=True)) if score else 0,
                "pesq": float(pesq(sample_rate, reference, decoded, "nb")) if score else 0,
            })

text_loc = upper_dir / run / "results.txt"

w_clip = max(len("clip"), *(len(r["clip"]) for r in rows)) if rows else len("clip")

with open(text_loc, "w", encoding="utf-8") as f:
    f.write(f"{'clip':<{w_clip}}  {'bitrate_bps':>11}  {'estoi':>7}  {'pesq':>7}\n")
    f.write(f"{'-'*w_clip}  {'-'*11}  {'-'*7}  {'-'*7}\n")

    for r in rows:
        f.write(
            f"{r['clip']:<{w_clip}}  "
            f"{int(r['bitrate_bps']):>11}  "
            f"{r['estoi']:>7.3f}  "
            f"{r['pesq']:>7.3f}\n"
        )