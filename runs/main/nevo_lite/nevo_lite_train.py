"""Training entrypoint for the default NEVO model.

You can copy this file and alter any hyperparameter or model config for your own run
Place the training script under runs/**/run-name/run-name_train.py
"""

import torch,torchaudio,os,math,random
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pystoi import stoi
from pesq import pesq
from encodec import balancer as blc
from model import MSSTFTDiscriminator, NevoModel
from helpers import info,utils,losses
from dataset import AudioDataset
from datetime import datetime

from pathlib import Path
from typing import Any

#############################################

TEST_CLIPS_DIR = r"D:\PycharmProjects\Datasets\Custom\test"

DATASETS: dict[str, float] = { # key=dataset_location : value=sampling_probability
    r"D:\PycharmProjects\Datasets\LibriSpeech": 1,
    r"D:\PycharmProjects\Datasets\LibriTTS": 1,
}

NOISE_DIR = r"D:\PycharmProjects\Datasets\DEMAND"
RIR_DIR = r"D:\PycharmProjects\Datasets\Room Impulse Response and Noise Database\rirs"

##############################################

hyper: dict[str, dict[str, Any]] = {

    "training":{
        "batch_size": 64,
        "accum_steps": 1,
        "updates_per_epoch": 3000,
        "epoch_count": 300,

        "gen_num_warmup": 15,
        "gen_learn_rate": 3e-4,
        "gen_min_lr_coeff": 0.01,

        "dis_num_warmup": 5,
        "dis_learn_rate": 2e-4,
        "dis_min_lr_coeff": 0.01,
        "dis_update_prob": 1
    },

    "loss":{
        "commit_coeff":1,"commit_rampinit":0,"commit_rampstart":0,"commit_rampend":0,
        "time_weight":0.1,
        "freq_weight":2.0,
        "adv_weight":4.0,"adv_rampinit":0.0,"adv_rampstart":0,"adv_rampend":0,
        "feat_weight":4.0,"feat_rampinit":0.0,"feat_rampstart":0,"feat_rampend":0,
    },

    "bandwidth":{
        "quantizer_levels":[2,4,8],
        "level_selection_weights":[1,2,3],
    },

    "dataset":{
        "add_rir_prob":0.1,
        "add_noise_prob":0.2,
        "snr_range_db":[-10,25],
    },

    "misc":{
        "frame_count":25, #1 second
        "ema_decay": 0.99,
        "checkpoint_steps": 50,
        "test_steps": 1,
        "test_clips_directory" : TEST_CLIPS_DIR,
        "discard_for_test":10,
        "bw_idxs_for_best":[0,1,2],
        "rng_seed":42,
        "final_save_location": "./models/" + os.path.basename(__file__)[:-9] + ".pt",
        "script_name": os.path.basename(__file__)[:-9],
    }
}

config: dict[str, Any] = {
    "sample_rate": 8000,
    "channels":24,
    "strides":[2,4,5,8],
    "n_resunits":1,
    "res_kernelsize":3,
    "res_compression":2,
    "dilation_base":2,
    "edge_kernelsize":7,
    "lstm_layers":1,
    "VQ": {
        "latent_dim": 16,
        "num_quantizer": 8,
        "codebook_size": 2 ** 6,
        "kmeans_iters": 50,
        "th_ema_dead": 2,
        "decay": 0.99,
    },
    "MSSTFTD": {
    "enable": True,
    "nffts":[1024, 512, 256, 128],
    "windows":[1024, 512, 256, 128],
    "hops":[256, 128, 64, 32],
    "channels":32,
    "max_channels":32,
    "logmag":False,
    },
    "melspec_scales":[6,7,8,9,10],
}

resumefromcheckpoint: int = 0  #-1 for last temp save, 0 for fresh training

if __name__ == '__main__':
    utils.set_rng_seeds(hyper["misc"]["rng_seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu" #only tested with cuda

    gen_model = NevoModel(config).to(device)

    use_ema=bool(hyper["misc"]["ema_decay"])
    if use_ema : ema_gen_model = torch.optim.swa_utils.AveragedModel(gen_model , torch.device(device) , multi_avg_fn= torch.optim.swa_utils.get_ema_multi_avg_fn(hyper["misc"]["ema_decay"]), use_buffers=True)

    num_bw_lvls = len(hyper["bandwidth"]["quantizer_levels"])
    bw_per_q_lvl = []
    frame_length = math.prod(config["strides"])
    for idx, q_lvl in enumerate(hyper["bandwidth"]["quantizer_levels"]):
        bw_per_q_lvl.append((config["sample_rate"] / frame_length) * q_lvl * math.log2(config["VQ"]["codebook_size"]))
        bw_per_q_lvl[idx] = int(bw_per_q_lvl[idx])  # /1000 #for kbps

    hyper["bandwidth"].update({"bw_per_q_level(bps)":bw_per_q_lvl})

    bw_weights = torch.tensor(hyper["bandwidth"]["level_selection_weights"], device=device, dtype=torch.float)
    bw_probs = bw_weights / bw_weights.sum()  # normalize the weights for probs

    info.start_logging(hyper,config,hyper["misc"]["script_name"])

    gen_optimizer = optim.Adam(gen_model.parameters(), lr=hyper["training"]["gen_learn_rate"], betas=(0.5, 0.9))
    gen_warmupschedular = utils.WarmupCosineLR_Scheduler(num_epochs=hyper["training"]["epoch_count"], num_warmup=hyper["training"]["gen_num_warmup"], min_lr_coeff=hyper["training"]["gen_min_lr_coeff"], linear=True)
    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, gen_warmupschedular.lrlambda)

    dis_model = []
    dis_optimizer = []
    dis_warmupschedular = []
    dis_scheduler = []

    if config["MSSTFTD"]["enable"]:
        for b in range(num_bw_lvls):
            dis_model.append(MSSTFTDiscriminator(
                n_ffts=config["MSSTFTD"]["nffts"],
                window_sizes=config["MSSTFTD"]["windows"],
                hop_lengths=config["MSSTFTD"]["hops"],
                channels=config["MSSTFTD"]["channels"],
                max_n_channels=config["MSSTFTD"]["max_channels"],
                log_mag=config["MSSTFTD"]["logmag"],
            ).to(device))
            dis_optimizer.append(optim.Adam(dis_model[-1].parameters(), lr=hyper["training"]["dis_learn_rate"], betas=(0.5, 0.9)))
            dis_warmupschedular.append(utils.WarmupCosineLR_Scheduler(num_epochs=hyper["training"]["epoch_count"],
                                                                  num_warmup=hyper["training"]["dis_num_warmup"],
                                                                  min_lr_coeff=hyper["training"]["dis_min_lr_coeff"],
                                                                  linear=True))
            dis_scheduler.append(torch.optim.lr_scheduler.LambdaLR(dis_optimizer[-1], dis_warmupschedular[-1].lrlambda))

    metric_specs = [
        {"name": "CodebookUsage", "mode": "none", "kind": "text"},
        {"name": "CodeUtil(%)", "mode": "increase", "fmt": ".2f"},
        {"name": "TimeLoss(e-3)", "mode": "decrease", "scale": 1e3, "fmt": ".2f"},
        {"name": "FreqLoss(e-3)", "mode": "decrease", "scale": 1e3, "fmt": ".2f"},
        {"name": "FeatLoss(e-3)", "mode": "decrease", "scale": 1e3, "fmt": ".2f"},
        {"name": "AdvLoss(e-3)", "mode": "none", "scale": 1e3, "fmt": ".2f"},
        {"name": "DisLoss(e-3)", "mode": "none", "scale": 1e3, "fmt": ".2f"},
        {"name": "CommitLoss(e-6)", "mode": "decrease", "scale": 1e6, "fmt": ".2f"},
        {"name": "PESQ", "mode": "increase", "fmt": ".3f"},
        {"name": "ESTOI", "mode": "increase", "fmt": ".3f"},
        {"name": "TotalScore", "mode": "increase", "fmt": ".3f"},
    ]

    root_dir: Path = Path(__file__).resolve().parent
    runs_dir: Path = root_dir.parent
    project_dir: Path = runs_dir.parent
    run_dir = root_dir

    epoch_info = info.EpochMetrics(bw_per_q_lvl, run_dir=run_dir, metrics=metric_specs)
    prev_info = None

    bw_selection_counter = [0]*num_bw_lvls
    d_selection_counter = [0]*num_bw_lvls

    tracker = info.CodebookUsageTracker(num_quantizers=config["VQ"]["num_quantizer"], codebook_sizes=config["VQ"]["codebook_size"],device=device)

    SpectralLoss = losses.MelSpecLoss(config["sample_rate"], config["melspec_scales"], 64)

    accum_scaler=1/hyper["training"]["accum_steps"]

    balancer = blc.Balancer(weights={"time": hyper["loss"]["time_weight"], "freq": hyper["loss"]["freq_weight"], "feat": hyper["loss"]["feat_weight"],"adv": hyper["loss"]["adv_weight"]}, total_norm=accum_scaler,ema_decay=0.999,per_batch_item=True, monitor=False,epsilon=1e-12)

    featramp=utils.Ramp(init=hyper["loss"]["feat_rampinit"],start=hyper["loss"]["feat_rampstart"],end=hyper["loss"]["feat_rampend"])
    advramp=utils.Ramp(init=hyper["loss"]["adv_rampinit"],start=hyper["loss"]["adv_rampstart"],end=hyper["loss"]["adv_rampend"])
    commitramp=utils.Ramp(init=hyper["loss"]["commit_rampinit"],start=hyper["loss"]["commit_rampstart"],end=hyper["loss"]["commit_rampend"])

    best_score = 0

    # Resume path expects checkpoints saved by this same script layout.
    if resumefromcheckpoint:
        if resumefromcheckpoint == -1:
            ckpt = torch.load("./models/temp/" + hyper["misc"]["script_name"] +".pt", map_location="cpu",weights_only=False)
            resumefromcheckpoint = ckpt["last_epoch_idx"]+1
        else:
            ckpt = torch.load("./models/checkpoints/" + hyper["misc"]["script_name"]+"_" + str(resumefromcheckpoint) + ".pt", map_location="cpu",weights_only=False)

        rs = ckpt.get("rng_state", None)
        if rs is not None:
            random.setstate(rs["python"])
            np.random.set_state(rs["numpy"])
            torch.set_rng_state(rs["torch"])
            if torch.cuda.is_available() and rs["cuda"] is not None:
                torch.cuda.set_rng_state_all(rs["cuda"])

        gen_model.load_state_dict(ckpt["gen_model"])
        if ckpt["config"]["MSSTFTD"]["enable"]:
            for m, sd in zip(dis_model, ckpt["dis_models"]):
                m.load_state_dict(sd)

        gen_optimizer.load_state_dict(ckpt["gen_optimizer"])
        if ckpt["config"]["MSSTFTD"]["enable"]:
            for opt, sd in zip(dis_optimizer, ckpt["dis_optimizers"]):
                opt.load_state_dict(sd)

        gen_scheduler.load_state_dict(ckpt["gen_scheduler"])
        if ckpt["config"]["MSSTFTD"]["enable"]:
            for sch, sd in zip(dis_scheduler, ckpt["dis_schedulers"]):
                sch.load_state_dict(sd)

        balancer.load_state_dict(ckpt["balancer"])

        if use_ema: ema_gen_model.load_state_dict(ckpt["ema_gen_model"])

        prev_state = ckpt["last_epoch_info"]
        prev_info = info.EpochMetrics.from_state_dict(prev_state, run_dir=run_dir) if prev_state else None

        epochlist= range(ckpt["last_epoch_idx"]+1,hyper["training"]["epoch_count"])

        best_score = float(ckpt["best_score"])

        del ckpt
        import gc; gc.collect()

    else:
        epochlist=range(hyper["training"]["epoch_count"])

    test_clips = []
    wav_root= Path(hyper["misc"]["test_clips_directory"])
    for wav in wav_root.rglob("*.wav"):
        waveform, sr = torchaudio.load(wav)
        if waveform.shape[0] != 1: waveform = waveform.mean(dim=0, keepdim=True)
        audio = torchaudio.functional.resample(waveform, sr, config["sample_rate"]).flatten()

        padding = (frame_length - (len(audio) % frame_length)) % frame_length
        test_clips.append(F.pad(audio, (0, padding)))

    dataset = AudioDataset(dataset_directories=list(DATASETS.keys()),
                           dataset_weights=list(DATASETS.values()),
                           frames_in_chunk=hyper["misc"]["frame_count"],
                           sample_rate=config["sample_rate"],
                           frame_length=frame_length,
                           samples_per_epoch=hyper["training"]["updates_per_epoch"] * hyper["training"]["batch_size"],
                           noise_prob=hyper["dataset"]["add_noise_prob"],
                           snr_range=hyper["dataset"]["snr_range_db"],
                           noise_directory=NOISE_DIR,
                           preload_noise=True,# use only if the noise dataset is small enough (depending on available ram), enabling this improves the dataset efficiency drastically
                           rir_prob=hyper["dataset"]["add_rir_prob"],
                           rir_directory=RIR_DIR,
                           )

    data_loader = DataLoader(
        dataset,
        batch_size=hyper["training"]["batch_size"],
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
    )

    timer = utils.TrainingTimer(len(epochlist))
    timer.start()

    if resumefromcheckpoint:
        print("Resuming from Checkpoint, Epoch: "+str(resumefromcheckpoint))
    else:
        print("Training Begin...")

    for i in epochlist:
        gen_model.train()
        if config["MSSTFTD"]["enable"]:
            for b in range(num_bw_lvls):
                dis_model[b].train()
        tracker.reset()

        balancer.weights["feat"] = hyper["loss"]["feat_weight"] * featramp.val(i+1)
        balancer.weights["adv"] = hyper["loss"]["adv_weight"] * advramp.val(i+1)
        c_ramp = 1 * commitramp.val(i + 1)

        nan_counter = 0

        gen_optimizer.zero_grad(set_to_none=True)

        step=0

        for step, frames in enumerate(data_loader, start=1):

            frames = frames.to(device, non_blocking=True)

            input = frames[:, 0, :]
            desired_output = frames[:, 1, :]

            input_wch = input.unsqueeze(1)  # converting from [B, F*320] to [B,1,F*320] (frames with 1 channel)

            #select a bandwidth
            bw_index = torch.multinomial(bw_probs, num_samples=1).item()
            bw_selection_counter[bw_index]+=1

            # single generator pass
            quantized_latents, indices, decoded_wch, commitment_loss = gen_model(input_wch,quantizer_limit=hyper["bandwidth"]["quantizer_levels"][bw_index])
            decoded = decoded_wch.squeeze(1)

            tracker.update(indices)

            # discriminator step
            update_D = bool(torch.rand(()) < hyper["training"]["dis_update_prob"])  # update with a probability to avoid the discriminator from overpowering
            if update_D and config["MSSTFTD"]["enable"]:
                d_selection_counter[bw_index]+=1
                for p in dis_model[bw_index].parameters(): p.requires_grad_(True)
                dis_model[bw_index].train()
                gen_model.eval()

                real_logits_d, _ = dis_model[bw_index](desired_output)
                fake_logits_d, _ = dis_model[bw_index](decoded.detach())
                real_loss, fake_loss = losses.DiscriminatorLoss(real_logits_d, fake_logits_d)
                discriminator_loss = (real_loss + fake_loss)/2
                epoch_info.add("DisLoss(e-3)", bw_index, discriminator_loss.detach().item())

                dis_optimizer[bw_index].zero_grad(set_to_none=True)
                discriminator_loss.backward()
                dis_optimizer[bw_index].step()
            else:
                empty_loss=(decoded * 0).sum()
                discriminator_loss = empty_loss

            # generator step
            gen_model.train()

            if config["MSSTFTD"]["enable"]:
                dis_model[bw_index].eval()
                for p in dis_model[bw_index].parameters(): p.requires_grad_(False)

                with torch.no_grad():
                    _, real_feats_g = dis_model[bw_index](desired_output)

                fake_logits_g, fake_feats_g = dis_model[bw_index](decoded)

                feature_loss = losses.FeatureLoss(real_feats_g, fake_feats_g)
                adversarial_loss = losses.AdversarialLoss(fake_logits_g)
            else:
                feature_loss = empty_loss
                adversarial_loss = empty_loss

            recontime_loss = F.l1_loss(decoded, desired_output)
            reconfreq_loss = SpectralLoss(decoded, desired_output)

            balance_loss = {
                "time": recontime_loss.float(),
                "freq": reconfreq_loss.float(),
                "feat": feature_loss.float(),
                "adv": adversarial_loss.float(),
            }

            (hyper["loss"]["commit_coeff"] * c_ramp * commitment_loss.float()*accum_scaler).backward(retain_graph=True)
            balancer.backward(balance_loss, decoded)

            if step % hyper["training"]["accum_steps"] == 0:
                if not all(torch.isfinite(p.grad).all() for p in gen_model.parameters() if p.grad is not None):
                    #no step
                    gen_optimizer.zero_grad(set_to_none=True)
                    nan_counter+=1
                    #print("[WARNING] Non-Finite Grads, Skipped Generator Step")
                else:
                    gen_optimizer.step()
                    if use_ema: ema_gen_model.update_parameters(gen_model)
                    gen_optimizer.zero_grad(set_to_none=True)

            epoch_info.add("TimeLoss(e-3)", bw_index, recontime_loss.detach().item())
            epoch_info.add("FreqLoss(e-3)", bw_index, reconfreq_loss.detach().item())
            epoch_info.add("FeatLoss(e-3)", bw_index, feature_loss.detach().item())
            epoch_info.add("AdvLoss(e-3)", bw_index, adversarial_loss.detach().item())
            epoch_info.add("CommitLoss(e-6)", bw_index, commitment_loss.detach().item())

        if (step % hyper["training"]["accum_steps"]) != 0:
            # only if we actually saw at least one batch this epoch
            if step > 0 and all(torch.isfinite(p.grad).all() for p in gen_model.parameters()if p.grad is not None):
                gen_optimizer.step()
                if use_ema: ema_gen_model.update_parameters(gen_model)
            else: nan_counter+=1
            gen_optimizer.zero_grad(set_to_none=True)

        gen_scheduler.step()
        if config["MSSTFTD"]["enable"]:
            for d_sch in dis_scheduler:
                d_sch.step()

        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

        print(
            "Epoch " + str(i + 1) + " ------------------" +
            timer.convert(timer.step(returnremaining=True)) +
            "left" +
            "------------------ " +
            timestamp
        )

        if nan_counter:
            print("WARNING, "+str(nan_counter)+" of "+str(len(data_loader))+" steps are non-finite!")


        stats = tracker.stats()  # list ordered by level: 0..num_q-1
        for idx, k in enumerate(hyper["bandwidth"]["quantizer_levels"]):
            subset = stats[:k]
            used_sum = sum(s["used"] for s in subset)
            total_sum = sum(s["total"] for s in subset)
            n_codes = [s["total"] for s in subset]
            entropy = [math.log2(max(s["perplexity"], 1e-12)) for s in subset]

            c = max(1, bw_selection_counter[idx])
            d = max(1, d_selection_counter[idx])

            epoch_info.set("CodebookUsage", idx, f"{used_sum}/{total_sum}")
            epoch_info.set("CodeUtil(%)", idx, 100 * sum(entropy) / sum(math.log2(n) for n in n_codes))
            epoch_info.set("Selected", idx, bw_selection_counter[idx])

            for name in ["TimeLoss(e-3)", "FreqLoss(e-3)", "FeatLoss(e-3)", "AdvLoss(e-3)", "CommitLoss(e-6)"]:
                epoch_info.average(name, idx, denom=c)

            epoch_info.average("DisLoss(e-3)", idx, denom=d)


        if hyper["misc"]["test_steps"] and (i + 1) >= hyper["misc"]["discard_for_test"]:
            if (i + 1) % hyper["misc"]["test_steps"] == 0:
                with torch.no_grad():
                    total_score = 0.0

                    for idx, (bw, q_lvl) in enumerate(
                            zip(bw_per_q_lvl, hyper["bandwidth"]["quantizer_levels"])
                    ):
                        estoi_scores = []
                        pesq_scores = []

                        for clip in test_clips:
                            reference = clip.cpu().numpy()

                            if use_ema:
                                output = ema_gen_model.module.evalforward(
                                    clip.to(device).unsqueeze(0).unsqueeze(0),
                                    quantizer_limit=q_lvl,
                                )
                            else:
                                output = gen_model.evalforward(
                                    clip.to(device).unsqueeze(0).unsqueeze(0),
                                    quantizer_limit=q_lvl,
                                )

                            generated = output.flatten().cpu().numpy()

                            estoi_scores.append(
                                stoi(reference, generated, config["sample_rate"], extended=True)
                            )
                            pesq_scores.append(
                                pesq(config["sample_rate"], reference, generated, "nb")
                            )

                        mean_estoi = float(np.mean(estoi_scores))
                        mean_pesq = float(np.mean(pesq_scores))

                        epoch_info.set("PESQ", idx, mean_pesq)
                        epoch_info.set("ESTOI", idx, mean_estoi)

                        print(f"{bw:>4}bps  PESQ: {mean_pesq:.3f}   ESTOI: {mean_estoi:.3f}")

                        if idx in hyper["misc"]["bw_idxs_for_best"]:
                            total_score += mean_estoi
                            total_score += mean_pesq / 5

                    total_score /= len(hyper["misc"]["bw_idxs_for_best"]) * 2
                    rounded_total_score = round(float(total_score), 3)

                    for idx in range(num_bw_lvls):
                        epoch_info.set("TotalScore", idx, rounded_total_score)

                    if rounded_total_score >= best_score:
                        best_score = rounded_total_score

                        if (i + 1) != hyper["training"]["epoch_count"]:
                            utils.save_allstates(
                                hyper,
                                config,
                                i,
                                gen_model,
                                dis_model,
                                gen_optimizer,
                                dis_optimizer,
                                gen_scheduler,
                                dis_scheduler,
                                balancer,
                                savelocation="./models/" + hyper["misc"]["script_name"] + "_best.pt",
                                ema_gen_model=ema_gen_model if use_ema else None,
                                last_epoch_info=epoch_info.state_dict(),
                                best_score=best_score,
                                save_rng_state=True,
                            )

                    print(f"Total Score: {rounded_total_score:.3f}   Best Score: {best_score:.3f}")

        epoch_info.print(prev_info)
        epoch_info.save_epoch(i + 1)

        prev_info = epoch_info
        epoch_info = info.EpochMetrics(bw_per_q_lvl, run_dir=run_dir, metrics=metric_specs)

        bw_selection_counter = [0] * num_bw_lvls
        d_selection_counter = [0] * num_bw_lvls

        nan_counter = 0

        if hyper["misc"]["checkpoint_steps"]:
            if ((i + 1) % hyper["misc"]["checkpoint_steps"] == 0) and ((i + 1) != hyper["training"]["epoch_count"]):
                utils.save_allstates(hyper,
                                     config,
                                     i,
                                     gen_model,
                                     dis_model,
                                     gen_optimizer,
                                     dis_optimizer,
                                     gen_scheduler,
                                     dis_scheduler,
                                     balancer,
                                     savelocation= "./models/checkpoints/" + hyper["misc"]["script_name"]+"_" + str(i+1) + ".pt",
                                     ema_gen_model=ema_gen_model if use_ema else None,
                                     last_epoch_info=prev_info.state_dict() if prev_info is not None else None,
                                     best_score=best_score,
                                     save_rng_state=True)
                print("Checkpoint Saved")

        # autosave
        if (i + 1) != hyper["training"]["epoch_count"]:
            utils.save_allstates(hyper,
                                 config,
                                 i,
                                 gen_model,
                                 dis_model,
                                 gen_optimizer,
                                 dis_optimizer,
                                 gen_scheduler,
                                 dis_scheduler,
                                 balancer,
                                 savelocation="./models/temp/" + hyper["misc"]["script_name"] + ".pt",
                                 ema_gen_model=ema_gen_model if use_ema else None,
                                 last_epoch_info=prev_info.state_dict() if prev_info is not None else None,
                                 best_score=best_score,
                                 save_rng_state=True)

    utils.save_allstates(hyper,
                         config,
                         i,
                         gen_model,
                         dis_model,
                         gen_optimizer,
                         dis_optimizer,
                         gen_scheduler,
                         dis_scheduler,
                         balancer,
                         savelocation=hyper["misc"]["final_save_location"],
                         ema_gen_model=ema_gen_model if use_ema else None,
                         last_epoch_info=prev_info.state_dict() if prev_info is not None else None,
                         best_score=best_score,
                         save_rng_state=True)
    print("Training Complete")
