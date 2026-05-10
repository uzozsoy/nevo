import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.nn.utils.parametrizations import weight_norm
from typing import Any

from vq.residual_vq import ResidualVQ
from helpers.modules import NevoLSTM,NevoConv1d,NevoConvTranspose1d,Snake1d #snake is unused

def get_2d_padding(
    kernel_size: tuple[int, int] ,
    dilation: tuple[int, int] = (1, 1),
) -> tuple[int, int]:
    """Return symmetric 2D padding for a kernel/dilation pair."""
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)

class STFTDModule(nn.Module):
    """Single-scale STFT discriminator branch.

    Args:
        n_fft: FFT size.
        windowsize: STFT window length.
        hop_length: STFT hop length.
        normalize: Enable torchaudio spectrogram normalization.
        channels: Base convolution channel count.
        max_n_channels: Upper channel cap.
        log_mag: Use log magnitude input; otherwise use real/imag channels.
        log_floor: Numeric floor before logarithm.
    """

    def __init__(
        self,
        n_fft: int,
        windowsize: int,
        hop_length: int,
        normalize: bool = True,
        channels: int = 32,
        max_n_channels: int = 32,
        log_mag: bool = False,
        log_floor: float = 1e-5,
    ) -> None:
        super().__init__()
        self.stft = T.Spectrogram(n_fft=n_fft, win_length=windowsize, hop_length=hop_length,window_fn=torch.hann_window,center=False, normalized=normalize,power=None)
        self.log_mag = log_mag
        self.eps = log_floor
        c=channels
        max_c=max_n_channels
        self.convolutions=nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels=1 if log_mag else 2, out_channels=min(max_c,c), kernel_size=(9, 3), padding=get_2d_padding((9, 3)))),
            weight_norm(nn.Conv2d(in_channels=min(max_c,c), out_channels=min(max_c,2*c), kernel_size=(9, 3), stride=(2, 1), dilation=(1, 1),padding=get_2d_padding((9, 3), (1, 1)))),
            weight_norm(nn.Conv2d(in_channels=min(max_c,2*c), out_channels=min(max_c,4*c), kernel_size=(9, 3), stride=(2, 1), dilation=(1, 2),padding=get_2d_padding((9, 3), (1, 2)))),
            weight_norm(nn.Conv2d(in_channels=min(max_c,4*c), out_channels=min(max_c,8*c), kernel_size=(9, 3), stride=(2, 1), dilation=(1, 4),padding=get_2d_padding((9, 3), (1, 4)))),
            weight_norm(nn.Conv2d(in_channels=min(max_c,8*c), out_channels=min(max_c,2*c), kernel_size=(3, 3), padding=get_2d_padding((3, 3)))),
            weight_norm(nn.Conv2d(in_channels=min(max_c,2*c), out_channels=1, kernel_size=(3, 3), padding=get_2d_padding((3, 3))))        ])
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return intermediate feature maps and final logits for `[B, T]` audio."""
        # input shape is [B,F*320]
        x = x.unsqueeze(1)  # [B,1,F*320]
        stftx = self.stft(x)  # [B,1,N,T]
        if self.log_mag:
            stftx = torch.log(stftx.abs() + self.eps)
        else:
            stftx = torch.cat([stftx.real, stftx.imag], dim=1)  # [B,2,N,T]
        features = [stftx]  # for 6 convs length is 7
        for i, conv in enumerate(self.convolutions):
            if i == len(self.convolutions) - 1:  # for the last element no activation is applied
                features.append(conv(features[i])); break
            features.append(self.act(conv(features[i])))
        return features  # last element is the logits in shape [B,1,?,?]

class MSSTFTDiscriminator(nn.Module):
    """Multi-scale STFT discriminator.

    Args:
        n_ffts: FFT sizes for each discriminator branch.
        window_sizes: Window sizes paired with `n_ffts`.
        hop_lengths: Hop sizes paired with `n_ffts`.
        channels: Base channel count for each branch.
        max_n_channels: Upper channel cap for each branch.
        log_mag: Use log magnitude input in each branch.
        log_floor: Numeric floor before logarithm.
        normalize: Enable torchaudio spectrogram normalization.
    """

    def __init__(
        self,
        n_ffts: list[int] | tuple[int, ...],
        window_sizes: list[int] | tuple[int, ...],
        hop_lengths: list[int] | tuple[int, ...],
        channels: int = 32,
        max_n_channels: int = 32,
        log_mag: bool = False,
        log_floor: float = 1e-5,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList()
        for nfft,winsize,hop in zip(n_ffts,window_sizes,hop_lengths):
            self.discriminators.append(STFTDModule(nfft,winsize,hop,normalize=normalize,channels=channels,max_n_channels=max_n_channels,log_mag=log_mag,log_floor=log_floor))
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """Return branch logits and feature maps for `[B, T]` audio."""
        # input shape is [B,F*320]
        logits=[]
        features = []
        for discriminator in self.discriminators:
            allfeatures = discriminator.forward(x)
            logits.append(allfeatures[-1])
            features.append(allfeatures[1:-1])
        return logits,features

class ResidualUnit(nn.Module):
    """Residual stack of causal convolution layers.

    Args:
        N: Input and output channel count.
        dilations: Dilation per convolution.
        kernel_sizes: Kernel size per convolution.
        compression: Inner channel compression factor.
        act: Activation name, either `"elu"` or `"snake"`.
    """

    def __init__(
        self,
        N: int,
        dilations: list[int] | tuple[int, ...],
        kernel_sizes: list[int] | tuple[int, ...],
        compression: int = 2,
        act: str = "elu",
    ) -> None:
        super().__init__()
        acts = []
        convs = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = N if i == 0 else N//compression #only inner layers are compressed
            out_chs = N if i == len(kernel_sizes) - 1 else N//compression
            acts.append((Snake1d(in_chs) if act == "snake" else nn.ELU()))
            convs.append(NevoConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,pad_with_previous=False))
        self.acts= nn.ModuleList(acts)
        self.convs = nn.ModuleList(convs)
        self.skip = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual unit to `[B, C, T]` input."""
        y = x.clone()
        for act, conv in zip(self.acts, self.convs):
            y = act(y)
            y = conv(y)
        return y+self.skip(x)

    def stream(self) -> None:
        """Enable internal convolution state reuse for streaming."""
        for conv in self.convs:
            conv.pad_with_previous = True

    def export_forward(
        self,
        x: torch.Tensor,
        res_unit_state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Forward pass with explicit convolution state tensors."""
        y = x
        new_states = []
        for i, (act, conv) in enumerate(zip(self.acts, self.convs)):
            y = act(y)
            y, st = conv(y, res_unit_state[i])   # UNPACK
            new_states.append(st)
        return y + self.skip(x), tuple(new_states)

    def export_initial_state(self) -> tuple[torch.Tensor, ...]:
        """Return explicit initial states for all internal convolutions."""
        return tuple(conv.export_initial_state() for conv in self.convs)

    def export_mode(self) -> None:
        """Route `forward` through the explicit-state export path."""
        self.forward = self.export_forward
        for conv in self.convs:
            conv.export_mode()

class EncoderBlock(nn.Module):
    """Downsampling encoder block with residual preprocessing.

    Args:
        N: Output channel count for the block.
        S: Downsampling stride.
        n_of_resunits: Number of residual units before downsampling.
        dilation_base: Base used to grow residual dilations.
        res_kernelsize: Residual convolution kernel size.
        res_compression: Residual inner-channel compression factor.
        act: Activation name, either `"elu"` or `"snake"`.
    """

    def __init__(
        self,
        N: int,
        S: int,
        n_of_resunits: int = 1,
        dilation_base: int = 2,
        res_kernelsize: int = 3,
        res_compression: int = 2,
        act: str = "elu",
    ) -> None:
        super().__init__()
        res_units = []
        for n in range(n_of_resunits):
            res_units.append(ResidualUnit( N//2,dilations= [dilation_base ** n,1],
                                     kernel_sizes= [res_kernelsize,1],
                                     compression= res_compression, act=act) )
        self.res_units = nn.ModuleList(res_units)
        self.act = Snake1d(N) if act == "snake" else nn.ELU()
        self.conv = NevoConv1d(N//2 ,N,kernel_size=2*S,stride=S,pad_with_previous=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample `[B, N/2, T]` input to `[B, N, T/S]` """
        for r, res_unit in enumerate(self.res_units):
            x = res_unit(x)
        x = self.act(x)
        x = self.conv(x)
        return x

    def stream(self) -> None:
        """Enable streaming state reuse for this block."""
        self.conv.pad_with_previous = True
        for r in self.res_units:
            r.stream()

    def export_forward(
        self,
        x: torch.Tensor,
        encoder_block_state: tuple[Any, ...],
    ) -> tuple[torch.Tensor, tuple[Any, ...]]:
        """Forward pass with explicit block state."""
        new_states = []
        for r, res_unit in enumerate(self.res_units):
            x, state = res_unit(x, encoder_block_state[r])
            new_states.append(state)
        x = self.act(x)
        x, state = self.conv(x, encoder_block_state[-1])
        new_states.append(state)
        return x, tuple(new_states)

    def export_initial_state(self) -> tuple[Any, ...]:
        """Return initial states for residual units and downsample convolution."""
        ru_states = tuple(ru.export_initial_state() for ru in self.res_units)
        conv_state = self.conv.export_initial_state()
        return tuple(list(ru_states) + [conv_state])

    def export_mode(self) -> None:
        """Route `forward` through the explicit-state export path."""
        self.forward = self.export_forward
        self.conv.export_mode()
        for r in self.res_units:
            r.export_mode()

class DecoderBlock(nn.Module):
    """Upsampling decoder block with residual post-processing.

    Args:
        N: Input channel count for the block.
        S: Upsampling stride.
        n_of_resunits: Number of residual units after upsampling.
        dilation_base: Base used to grow residual dilations.
        res_kernelsize: Residual convolution kernel size.
        res_compression: Residual inner-channel compression factor.
        act: Activation name, either `"elu"` or `"snake"`.
    """

    def __init__(
        self,
        N: int,
        S: int,
        n_of_resunits: int = 1,
        dilation_base: int = 2,
        res_kernelsize: int = 3,
        res_compression: int = 2,
        act: str = "elu",
    ) -> None: #res_compression = 1 for no compression
        super().__init__()
        self.act = Snake1d(N) if act == "snake" else nn.ELU()
        self.convt = NevoConvTranspose1d(N, N // 2 , kernel_size=2*S, stride=S,save_trimmed=False)
        res_units = []
        for n in range(n_of_resunits):
            res_units.append(ResidualUnit( N//2,dilations= [dilation_base ** n,1],
                                     kernel_sizes= [res_kernelsize,1],
                                     compression= res_compression, act=act) )
        self.res_units = nn.ModuleList(res_units)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample `[B, N, T]` input to `[B, N/2, T*S]` approximately."""
        x = self.act(x)
        x = self.convt(x)
        for r,res_unit in enumerate(self.res_units):
            x = res_unit(x)
        return x

    def stream(self) -> None:
        """Enable streaming overlap/state reuse for this block."""
        self.convt.save_trimmed = True
        for r in self.res_units:
            r.stream()

    def export_forward(
        self,
        x: torch.Tensor,
        decoder_block_state: tuple[Any, ...],
    ) -> tuple[torch.Tensor, tuple[Any, ...]]:
        """Forward pass with explicit block state."""
        x = self.act(x)
        new_states = []
        x, state = self.convt(x , decoder_block_state[0])
        new_states.append(state)
        for r,res_unit in enumerate(self.res_units):
            x, state = res_unit(x, decoder_block_state[r+1])
            new_states.append(state)
        return x, tuple(new_states)

    def export_initial_state(self) -> tuple[Any, ...]:
        """Return initial states for transposed conv and residual units."""
        convt_state = self.convt.export_initial_state()
        ru_states = tuple(ru.export_initial_state() for ru in self.res_units)
        return tuple([convt_state] + list(ru_states))

    def export_mode(self) -> None:
        """Route `forward` through the explicit-state export path."""
        self.forward = self.export_forward
        self.convt.export_mode()
        for r in self.res_units:
            r.export_mode()

class NevoEncoder(nn.Module):
    """Waveform encoder producing latent frames.

    Args:
        channels: Channel schedule from input edge conv through blocks.
        strides: Downsampling stride per encoder block.
        n_of_resunits: Residual units per block.
        res_kernelsize: Residual convolution kernel size.
        res_compression: Residual inner-channel compression factor.
        dilation_base: Base used to grow residual dilations.
        edge_kernelsize: Kernel size for first and last encoder convs.
        lstm_layers: Optional LSTM layer count at the bottleneck.
        latent_dim: Output latent channel count.
        act: Activation name, either `"elu"` or `"snake"`.
    """

    def __init__(
        self,
        channels: list[int],
        strides: list[int] | tuple[int, ...],
        n_of_resunits: int = 1,
        res_kernelsize: int = 3,
        res_compression: int = 2,
        dilation_base: int = 2,
        edge_kernelsize: int = 7,
        lstm_layers: int = 3,
        latent_dim: int = 16,
        act: str = "elu",
    ) -> None:
        super().__init__()
        self.initial_conv = NevoConv1d(1, out_channels=channels[0], kernel_size=edge_kernelsize)
        encoder_blocks = []
        for s,c in zip(strides,channels[1:]):
            encoder_blocks += [EncoderBlock(N=c, S=s, n_of_resunits=n_of_resunits,
                                     dilation_base = dilation_base,
                                     res_kernelsize = res_kernelsize,
                                     res_compression = res_compression)]
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.lstm = NevoLSTM(feature_size=channels[-1], num_layers=lstm_layers) if lstm_layers > 0 else None
        self.act = (Snake1d(channels[-1]) if act == "snake" else nn.ELU())
        self.final_conv = NevoConv1d(channels[-1], latent_dim, kernel_size=edge_kernelsize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio shaped `[B, 1, T]` into `[B, latent_dim, frames]`."""

        x= self.initial_conv(x)
        for e,encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
        if self.lstm is not None:
            x = self.lstm(x)
        x = self.act(x)
        x = self.final_conv(x)
        return x

    def export_forward(
        self,
        x: torch.Tensor,
        initial_conv_state: torch.Tensor,
        encoder_block_state: tuple[Any, ...],
        lstm_state: Any,
        final_conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[Any, ...], Any, torch.Tensor]:
        """Encode one streaming frame with explicit recurrent/convolution state."""
        x,initial_conv_state = self.initial_conv(x,initial_conv_state)
        new_states = []
        for e,encoder_block in enumerate(self.encoder_blocks):
            x, state = encoder_block(x,encoder_block_state[e])
            new_states.append(state)
        encoder_block_state = tuple(new_states)
        if self.lstm is not None:
            x, lstm_state = self.lstm(x, lstm_state)
        x = self.act(x)
        x, final_conv_state = self.final_conv(x,final_conv_state)
        return x, initial_conv_state, encoder_block_state, lstm_state, final_conv_state

    def export_mode(self) -> None:
        """Route submodules through explicit-state export paths."""
        self.forward = self.export_forward
        for module in self.modules():
            if isinstance(module, NevoConv1d): module.export_mode()
            if isinstance(module, EncoderBlock): module.export_mode()
            if isinstance(module, NevoLSTM): module.export_mode()

    def export_initial_state(self) -> tuple[Any, ...]:
        """Return initial export state tuple for the full encoder."""
        init_conv = self.initial_conv.export_initial_state()
        blk_states = tuple(blk.export_initial_state() for blk in self.encoder_blocks)
        if self.lstm is not None:
            lstm_state = self.lstm.export_initial_state()
        else:
            lstm_state = ()  # or omit from signature entirely
        final_conv = self.final_conv.export_initial_state()
        return (init_conv, blk_states, lstm_state, final_conv)

class NevoDecoder(nn.Module):
    """Latent decoder reconstructing waveform frames.

    Args:
        channels: Channel schedule matching the encoder.
        strides: Upsampling stride per decoder block.
        n_of_resunits: Residual units per block.
        res_kernelsize: Residual convolution kernel size.
        res_compression: Residual inner-channel compression factor.
        dilation_base: Base used to grow residual dilations.
        edge_kernelsize: Kernel size for first and last decoder convs.
        lstm_layers: Optional LSTM layer count at the bottleneck.
        latent_dim: Input latent channel count.
        act: Activation name, either `"elu"` or `"snake"`.
    """

    def __init__(
        self,
        channels: list[int],
        strides: list[int] | tuple[int, ...],
        n_of_resunits: int = 1,
        res_kernelsize: int = 3,
        res_compression: int = 2,
        dilation_base: int = 2,
        edge_kernelsize: int = 7,
        lstm_layers: int = 3,
        latent_dim: int = 16,
        act: str = "elu",
    ) -> None:
        super().__init__()
        self.initial_conv = NevoConv1d(latent_dim, out_channels=channels[-1], kernel_size=edge_kernelsize)
        self.lstm = NevoLSTM(feature_size=channels[-1], num_layers=lstm_layers) if lstm_layers > 0 else None
        decoder_blocks = []
        for s,c in zip(reversed(strides),reversed(channels[1:])):
            decoder_blocks += [DecoderBlock(N=c, S=s, n_of_resunits=n_of_resunits,
                                     dilation_base = dilation_base,
                                     res_kernelsize = res_kernelsize,
                                     res_compression = res_compression)]
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.act = (Snake1d(channels[0]) if act == "snake" else nn.ELU())
        self.final_conv = NevoConv1d(channels[0], 1, kernel_size=edge_kernelsize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latents shaped `[B, latent_dim, frames]` into `[B, 1, T]`."""
        x = self.initial_conv(x)
        if self.lstm is not None:
            x = self.lstm(x)
        for d,decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x)
        x = self.act(x)
        x = self.final_conv(x)
        return x

    def export_forward(
        self,
        x: torch.Tensor,
        initial_conv_state: torch.Tensor,
        decoder_block_state: tuple[Any, ...],
        lstm_state: Any,
        final_conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[Any, ...], Any, torch.Tensor]:
        """Decode one streaming latent frame with explicit state."""
        x, initial_conv_state = self.initial_conv(x,initial_conv_state)
        if self.lstm is not None:
            x, lstm_state = self.lstm(x, lstm_state)
        new_states = []
        for d,decoder_block in enumerate(self.decoder_blocks):
            x, state = decoder_block(x,decoder_block_state[d])
            new_states.append(state)
        decoder_block_state = tuple(new_states)
        x = self.act(x)
        x, final_conv_state = self.final_conv(x,final_conv_state)
        return x, initial_conv_state, decoder_block_state, lstm_state, final_conv_state

    def export_initial_state(self) -> tuple[Any, ...]:
        """Return initial export state tuple for the full decoder."""
        init_conv = self.initial_conv.export_initial_state()
        blk_states = tuple(blk.export_initial_state() for blk in self.decoder_blocks)
        if self.lstm is not None:
            lstm_state = self.lstm.export_initial_state()
        else:
            lstm_state = ()
        final_conv = self.final_conv.export_initial_state()
        return (init_conv, blk_states, lstm_state, final_conv)

    def export_mode(self) -> None:
        """Route submodules through explicit-state export paths."""
        self.forward = self.export_forward
        for module in self.modules():
            if isinstance(module, NevoConv1d): module.export_mode()
            if isinstance(module, DecoderBlock): module.export_mode()
            if isinstance(module, NevoLSTM): module.export_mode()

class NevoModel(nn.Module):
    """End-to-end encoder, residual VQ, and decoder model.

    Args:
        config: Training/model configuration dictionary. Expected keys include
            `channels`, `strides`, residual settings, `lstm_layers`, and a `VQ`
            sub-dictionary with latent/codebook settings.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.channels = [config["channels"] * (2**i) for i in range(len(config["strides"]) + 1)]
        act = config.get("activation", "elu")
        self.encoder = NevoEncoder(channels=self.channels,strides=config["strides"]
                                   ,n_of_resunits=config["n_resunits"],res_kernelsize=config["res_kernelsize"],res_compression=config["res_compression"],
                                   dilation_base=config["dilation_base"],edge_kernelsize=config["edge_kernelsize"],lstm_layers=config["lstm_layers"],
                                   latent_dim=config["VQ"]["latent_dim"],act=act)
        self.vq = ResidualVQ(
            dim=config["VQ"]["latent_dim"],
            num_quantizers=config["VQ"]["num_quantizer"],  # specify number of quantizers
            codebook_size=config["VQ"]["codebook_size"],  # codebook size
            kmeans_init=True if config["VQ"]["kmeans_iters"] else False,  # set to True
            kmeans_iters=config["VQ"]["kmeans_iters"],# number of kmeans iterations to calculate the centroids for the codebook on init
            threshold_ema_dead_code=config["VQ"]["th_ema_dead"],# should actively replace any codes that have an exponential moving average cluster size less than 2
            decay=config["VQ"]["decay"],
        )

        self.decoder = NevoDecoder(channels=self.channels, strides=config["strides"]
                                   , n_of_resunits=config["n_resunits"], res_kernelsize=config["res_kernelsize"],
                                   res_compression=config["res_compression"],
                                   dilation_base=config["dilation_base"], edge_kernelsize=config["edge_kernelsize"],
                                   lstm_layers=config["lstm_layers"],
                                   latent_dim=config["VQ"]["latent_dim"], act=act)

    def forward(
        self,
        x: torch.Tensor,
        quantizer_limit: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, quantize, and decode audio `[B, 1, T]` during training."""
        z = self.encoder(x)
        z = z.transpose(1, 2)
        quantized_latents, indices, commitment_loss = self.vq(z,
                                                              use_levels=quantizer_limit,
                                                              freeze_codebook=False)
        quantized_latents = quantized_latents.transpose(1, 2)
        decoded = self.decoder(quantized_latents)
        return quantized_latents, indices, decoded, commitment_loss.mean()

    def stream(self) -> None:
        """Enable stateful streaming behavior in encoder and decoder modules."""
        for module in self.encoder.modules():
            if isinstance(module, NevoConv1d): module.pad_with_previous = True
            if isinstance(module, EncoderBlock): module.stream()
            if isinstance(module, NevoLSTM): module.keep_state = True
        for module in self.decoder.modules():
            if isinstance(module, NevoConv1d): module.pad_with_previous = True
            if isinstance(module, DecoderBlock): module.stream()
            if isinstance(module, NevoLSTM): module.keep_state = True

    def evalforward(self, x: torch.Tensor, quantizer_limit: int | None = None) -> torch.Tensor:
        """Inference-only forward pass returning decoded audio."""
        with torch.no_grad():
            z = self.encoder(x)
            z = z.transpose(1, 2)
            quantized_latents, indices, commitment_loss = self.vq(z.float(),
                                                                  use_levels=quantizer_limit,
                                                                  freeze_codebook=True)
            quantized_latents = quantized_latents.transpose(1, 2)
            decoded = self.decoder(quantized_latents)
        return decoded

if __name__ == "__main__":
    #default nevo config
    config = {
        "sample_rate": 8000,
        "channels": 32,
        "strides": [2, 4, 5, 8],
        "n_resunits": 1,
        "res_kernelsize": 3,
        "res_compression": 2,
        "dilation_base": 2,
        "edge_kernelsize": 7,
        "lstm_layers": 3,
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
            "nffts": [1024, 512, 256, 128],
            "windows": [1024, 512, 256, 128],
            "hops": [256, 128, 64, 32],
            "channels": 64,
            "max_channels": None,
            "logmag": False,
        },
        "melspec_scales": [6, 7, 8, 9, 10],
    }

    model = NevoModel(config)

    def count_params(model: torch.nn.Module) -> tuple[int, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable


    total_params, trainable_params = count_params(model)
    print(f"Params: total={total_params:,} | trainable={trainable_params:,}")
