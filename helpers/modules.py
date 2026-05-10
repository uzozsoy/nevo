import torch,math
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F
import torchaudio.transforms as T

class NevoLSTM(nn.Module):
    """Residual LSTM block for `[batch, channels, time]` tensors.

    Args:
        feature_size: Number of channel/features passed through the LSTM.
        num_layers: Number of recurrent layers.
        keep_state: Preserve recurrent state between forward calls for streaming.
    """

    def __init__(self, feature_size: int, num_layers: int, keep_state: bool = False) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=feature_size, num_layers=num_layers)
        self.feature_size = feature_size
        self.num_layers = num_layers

        self.state=None
        self.keep_state=keep_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual LSTM and return `[batch, channels, time]`."""
        x = x.permute(2,0,1)
        if self.keep_state:
            y, state = self.lstm(x,self.state)
            self.state=state
        else:
            y, _ = self.lstm(x)
        y = y + x
        y = y.permute(1,2,0)
        return y

    def export_forward(
        self,
        x: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Streaming/export forward pass with explicit LSTM state."""
        x = x.permute(2, 0, 1)
        y, lstm_state = self.lstm(x, lstm_state)
        y = y + x
        y = y.permute(1, 2, 0)
        return y, lstm_state

    def export_initial_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero `(h, c)` state tensors for export."""
        h0 = torch.zeros(self.num_layers, 1, self.feature_size)
        c0 = torch.zeros(self.num_layers, 1, self.feature_size)
        return (h0,c0)

    def export_mode(self) -> None:
        """Route `forward` calls through the explicit-state export path."""
        self.forward = self.export_forward


class NevoConv1d(nn.Module):
    """Causal 1D convolution wrapper with optional streaming state.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dilation: Convolution dilation.
        pad_mode: Padding mode passed to `torch.nn.functional.pad`.
        pad_with_previous: Use saved samples instead of fresh left padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
        pad_with_previous: bool = False,
    ) -> None:
        super().__init__()
        self.pad_mode = pad_mode
        self.conv = weight_norm(nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding=0,dilation=dilation)) #requires [B,C,T] input

        self.eff_kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        self.stride = stride
        self.padding_total=self.eff_kernel_size - self.stride #left padding

        self.pad_with_previous = pad_with_previous
        self.previous_samples=torch.zeros(size=(1,in_channels, self.padding_total))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution to `[batch, channels, time]` input."""
        if self.pad_with_previous:
            if self.padding_total == 0:
                output = self.conv(x)
            else:
                concat = torch.cat((self.previous_samples,x),dim=-1)
                output = self.conv(concat)
                self.previous_samples = concat[...,-self.padding_total:]
        else:
            output= self.conv(F.pad(x, (self.padding_total, 0), mode=self.pad_mode))  # right padding value is used to be extra_padding
        return output

    def export_forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply convolution with explicit left-context state."""
        if self.padding_total == 0:
            output = self.conv(x)
        else:
            concat = torch.cat((conv_state, x), dim=-1)
            output = self.conv(concat)
            conv_state = concat[..., -self.padding_total:]
        return output, conv_state

    def export_initial_state(self) -> torch.Tensor:
        """Return a zero left-context buffer for export."""
        return torch.zeros_like(self.previous_samples)

    def export_mode(self) -> None:
        """Route `forward` calls through the explicit-state export path."""
        self.forward = self.export_forward


class NevoConvTranspose1d(nn.Module):
    """Transposed convolution wrapper with overlap carry for streaming decode.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Transposed convolution kernel size.
        stride: Transposed convolution stride.
        save_trimmed: Preserve the trimmed tail for the next streaming frame.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        save_trimmed: bool = False,
    ) -> None:
        super().__init__()
        self.convt = weight_norm(nn.ConvTranspose1d(in_channels,out_channels,kernel_size,stride,padding=0,output_padding=0))
        self.stride = stride

        self.trim_total = kernel_size - stride
        assert self.trim_total != 0

        self.save_trimmed = save_trimmed
        self.previous_samples = torch.zeros(size=(1,out_channels, self.trim_total))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transposed convolution and remove overlap tail."""
        y=self.convt(x)
        if self.save_trimmed:
            bias = self.convt.bias
            y[..., :self.trim_total]+=self.previous_samples - bias.view(1, -1, 1)
            self.previous_samples = y[...,-self.trim_total:]
        return y[..., : -self.trim_total]

    def export_forward(
        self,
        x: torch.Tensor,
        convt_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply transposed convolution with explicit overlap state."""
        y = self.convt(x)
        bias = self.convt.bias
        y[..., :self.trim_total] += convt_state - bias.view(1, -1, 1)
        convt_state = y[..., -self.trim_total:]
        return y[..., : -self.trim_total], convt_state

    def export_initial_state(self) -> torch.Tensor:
        """Return a zero overlap buffer for export."""
        return torch.zeros_like(self.previous_samples)

    def export_mode(self) -> None:
        """Route `forward` calls through the explicit-state export path."""
        self.forward = self.export_forward


class NevoMelSpectogram(nn.Module):
    """Mel spectrogram module used by reconstruction losses.

    Args:
        n_fft: FFT/window size.
        n_mels: Number of mel bands; defaults to `n_fft // 8`.
        sample_rate: Audio sample rate.
        f_min: Lowest mel frequency.
        f_max: Highest mel frequency, or `None` for Nyquist.
        log: Return log10-scaled mel energies when true.
        normalized: Pass through to torchaudio's spectrogram normalization.
        floor_level: Numeric floor added before logarithm.
    """

    def __init__(
        self,
        n_fft: int,
        n_mels: int | None = None,
        sample_rate: int = 24000,
        f_min: float = 0.0,
        f_max: float | None = None,
        log: bool = True,
        normalized: bool = True,
        floor_level: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.log=log
        self.eps=floor_level
        self.mel_transform = T.MelSpectrogram(n_mels=n_mels if n_mels else n_fft // 8,sample_rate=sample_rate,window_fn=torch.hann_window,
                                              n_fft=n_fft,hop_length=self.hop_length,win_length=n_fft,f_min=f_min if f_min else 0.0,f_max=f_max,normalized=normalized,center=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return mel features shaped `[batch, channels * mels, frames]`."""
        # Framing/padding logic adapted from Meta EnCodec; see THIRD_PARTY_NOTICES.md.
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (p, p), "reflect")
        # Make sure that all the frames are full.
        # The combination of `pad_for_conv1d` and the above padding
        # will make the output of size ceil(T / hop).
        length = x.shape[-1]
        n_frames = (length - self.n_fft) / self.hop_length + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.hop_length + (self.n_fft)
        x=F.pad(x, (0, ideal_length - length))

        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        B, C, freqs, frame = mel_spec.shape
        if self.log:
            mel_spec = torch.log10(self.eps + mel_spec)
        return mel_spec.reshape(B, C * freqs, frame)

# Taken from Descript Audio Codec (DAC) implementation
# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    """Learnable Snake activation over 1D feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Snake activation."""
        return snake(x, self.alpha)
