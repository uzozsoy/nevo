from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helpers.modules import NevoMelSpectogram
import torch_pesq

mel_masking_values_8khz = [
 5.97818494e-05, 7.59511427e-04, 2.77077232e-03, 6.37054728e-03,
 1.15707758e-02, 1.82501280e-02, 2.62284643e-02, 3.53058351e-02,
 4.52827242e-02, 5.59697829e-02, 6.71925976e-02, 7.87930628e-02,
 9.06299027e-02, 1.02577477e-01, 1.14525197e-01, 1.26376008e-01,
 1.38045300e-01, 1.49459825e-01, 1.60556211e-01, 1.71280531e-01,
 1.81586878e-01, 1.91436755e-01, 2.00798388e-01, 2.09645829e-01,
 2.17958656e-01, 2.25721074e-01, 2.32921851e-01, 2.39553438e-01,
 2.45611962e-01, 2.51096558e-01, 2.56009329e-01, 2.60354741e-01,
 2.64139609e-01, 2.67372795e-01, 2.70064899e-01, 2.72228118e-01,
 2.73876037e-01, 2.75023474e-01, 2.75686306e-01, 2.75881327e-01,
 2.75626097e-01, 2.74938836e-01, 2.73838273e-01, 2.72343580e-01,
 2.70474232e-01, 2.68249934e-01, 2.65690531e-01, 2.62815929e-01,
 2.59646036e-01, 2.56200572e-01, 2.52499168e-01, 2.48561395e-01,
 2.44406223e-01, 2.40052560e-01, 2.35518993e-01, 2.30823443e-01,
 2.25983599e-01, 2.21016665e-01, 2.15939196e-01, 2.10767476e-01,
 2.05516890e-01, 2.00202456e-01, 1.94838684e-01, 1.89439420e-01]


def linear_increase_coeff(difference: float) -> list[float]: #unused
    """Return 64 linearly spaced coefficients around 1.0."""
    return np.linspace(1-difference,1+difference,64,dtype=float).tolist()

class MelSpecLoss(nn.Module):
    """Multi-scale mel-spectrogram reconstruction loss.

    Args:
        sample_rate: Audio sample rate.
        stft_scales: Powers of two used as FFT sizes.
        n_mels: Number of mel bands.
        mel_mask_L2: Optional per-band mask for the L2 term.
        mel_mask_L1: Optional per-band mask for the L1 term.
        log_L2: Use log mel features for the L2 term.
        log_L1: Use log mel features for the L1 term.
        log_floor: Numeric floor before logarithms.
        alpha: Weight applied to the L2 term.
        f_min: Lowest mel frequency.
        normalized: Average loss by the number of scales when true.
        device: Device used for masks and submodules.
    """

    def __init__(
        self,
        sample_rate: int,
        stft_scales: Sequence[int],
        n_mels: int,
        mel_mask_L2: Sequence[float] | None = None,
        mel_mask_L1: Sequence[float] | None = None,
        log_L2: bool = True,
        log_L1: bool = False,
        log_floor: float = 1e-5,
        alpha: float = 1,
        f_min: float = 0,
        normalized: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.n_of_scales = len(stft_scales)
        self.log = log_L2
        self.eps = log_floor
        self.alpha = alpha
        self.normalized = normalized

        mel_mask_tensor_L1 = torch.tensor(mel_mask_L1, dtype=torch.float32) if mel_mask_L1 is not None else torch.ones(n_mels,dtype=torch.float32)
        mel_mask_tensor_L2 = torch.tensor(mel_mask_L2, dtype=torch.float32) if mel_mask_L2 is not None else torch.ones(n_mels,dtype=torch.float32)
        assert n_mels == len(mel_mask_tensor_L1), "L1 mel mask should be a coefficient list with length equal to n_mels!"
        assert n_mels == len(mel_mask_tensor_L2), "L2 mel mask should be a coefficient list with length equal to n_mels!"
        self.mask_coeffs_L1 = (mel_mask_tensor_L1/mel_mask_tensor_L1.mean()).unsqueeze(0).unsqueeze(-1).to(device)
        self.mask_coeffs_L2 = (mel_mask_tensor_L2/mel_mask_tensor_L2.mean()).unsqueeze(0).unsqueeze(-1).to(device)

        l1s = [];l2s = []
        for i in stft_scales:
            l1s.append(NevoMelSpectogram(n_fft=2 ** i, n_mels=n_mels, sample_rate=sample_rate, f_min=f_min,
                                        normalized=normalized, log=log_L1, floor_level=self.eps))
            l2s.append(NevoMelSpectogram(n_fft=2 ** i, n_mels=n_mels, sample_rate=sample_rate, f_min=f_min,
                                        normalized=normalized, log=log_L2, floor_level=self.eps))
        self.l1s = nn.ModuleList(l1s).to(device)
        self.l2s = nn.ModuleList(l2s).to(device)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar mel reconstruction loss for `output` vs `target`."""
        total_f_loss = torch.tensor(0.0, device=output.device)
        for i in range(self.n_of_scales):
            mell1_output = self.l1s[i](output)
            mell1_target = self.l1s[i](target)
            mell2_output = self.l2s[i](output)
            mell2_target = self.l2s[i](target)
            l1_loss = (F.l1_loss(mell1_output, mell1_target, reduction='none') * self.mask_coeffs_L1).mean()
            l2_loss = (F.mse_loss(mell2_output, mell2_target, reduction='none') * self.mask_coeffs_L2).mean()
            total_f_loss += l1_loss + self.alpha * l2_loss
        freqloss = total_f_loss / (self.n_of_scales * 2) if self.normalized else total_f_loss
        return  freqloss

class PESQLoss(nn.Module): #unused
    """Multi-scale differentiable PESQ proxy loss.

    Args:
        sample_rate: Audio sample rate.
        stft_scales: Powers of two used as FFT sizes.
        n_barks: Bark filter count used by `torch_pesq`.
        normalized: Average loss by number of scales when true.
        device: Device used for submodules.
    """

    def __init__(
        self,
        sample_rate: int,
        stft_scales: Sequence[int],
        n_barks: int = 49,
        normalized: bool = True,
        device: str = "cuda",
    ) -> None:
        #note that stft scales should be given as if the signal is sampled in 16khz. This is due to the workings of torch_pesq where the signal is resampled to 16khz before calculations.
        super().__init__()
        self.normalized = normalized
        pesqs=[]
        for i in stft_scales:
            pesqs.append(torch_pesq.PesqLoss(factor=1,sample_rate=sample_rate,nbarks=n_barks,win_length=2**i,n_fft=2**i,hop_length=(2**i)//2)) #50 percent overlap is advised in pesq
        self.pesqs = nn.ModuleList(pesqs).to(device)
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar PESQ proxy loss for `output` vs `target`."""
        total_pesq_loss = torch.tensor(0.0, device=output.device)
        for pesq in self.pesqs:
            total_pesq_loss += pesq(target,output).mean()
        pesq_loss = total_pesq_loss / len(self.pesqs) if self.normalized else total_pesq_loss
        return pesq_loss

def DiscriminatorLoss(
    reallogits: Sequence[torch.Tensor],
    fakelogits: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hinge discriminator losses for real and generated logits."""
    n_of_discriminators = len(reallogits)
    realloss = torch.tensor(0.0, device=reallogits[0].device)
    fakeloss = torch.tensor(0.0, device=fakelogits[0].device)
    for i in range(n_of_discriminators):
        realloss += -torch.min(reallogits[i]-1,torch.zeros_like(reallogits[i])).mean()
        fakeloss += -torch.min(-fakelogits[i]-1,torch.zeros_like(fakelogits[i])).mean()
    return realloss/n_of_discriminators,fakeloss/n_of_discriminators

def AdversarialLoss(logits: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return generator adversarial loss from discriminator logits."""
    n_of_discriminators = len(logits)
    advloss = torch.tensor(0.0, device=logits[0].device)
    for i in range(n_of_discriminators):
        advloss += -logits[i].mean()
    return advloss/n_of_discriminators

def FeatureLoss(
    realfeatures: Sequence[Sequence[torch.Tensor]],
    fakefeatures: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    """Return average feature matching L1 loss across discriminators/layers."""
    #features are indexed like features[-discriminator-][-layer-]
    n_of_discriminators = len(realfeatures)
    n_of_layers = len(realfeatures[0])
    totalloss = torch.tensor(0.0, device=realfeatures[0][0].device)
    for d in range(n_of_discriminators):
        disloss = torch.tensor(0.0, device=realfeatures[0][0].device)
        for l in range(n_of_layers):
            disloss += F.l1_loss(fakefeatures[d][l], realfeatures[d][l],reduction='mean')
        totalloss += disloss
    return totalloss/(n_of_discriminators*n_of_layers)

if __name__ == "__main__":
    pass
