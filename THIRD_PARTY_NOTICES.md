# Third-Party Notices

This project includes or references code and model assets derived from the following upstream projects.

## EnCodec

- Source: https://github.com/facebookresearch/encodec
- License: MIT
- Copyright: Copyright (c) Meta Platforms, Inc. and affiliates.
- Local files: `encodec/balancer.py`, `encodec/distrib.py`, and the mel-spectrogram framing logic in `helpers/modules.py`.
- Citation:

```bibtex
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Defossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```

## vector-quantize-pytorch

- Source: https://github.com/lucidrains/vector-quantize-pytorch
- License: MIT
- Copyright: Copyright (c) 2020 Phil Wang
- Local files: `vq/residual_vq.py`, `vq/vector_quantize.py`.
- Local changes: residual quantizer level limiting for custom bandwidth training and stripped unsupported branches.

## GTCRN

- Source: https://github.com/Xiaobin-Rong/gtcrn
- License: MIT
- Copyright: Copyright (c) 2024 Rong Xiaobin
- Local files: `onnx/benchmark/nevo_gtcrn_stream.py` references a GTCRN ONNX denoiser model.
- Citation:

```bibtex
@inproceedings{rong2024gtcrn,
  title={GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources},
  author={Rong, Xiaobin and Sun, Tianchi and Zhang, Xu and Hu, Yuxiang and Zhang, Changbao and Lu, Jing},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  doi={10.1109/ICASSP48485.2024.10448310}
}
```

## MIT License Text

The third-party components listed above are distributed under the MIT License:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
