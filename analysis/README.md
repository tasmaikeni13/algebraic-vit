# Vision analysis suite (training-free)

Five scripts benchmark the Algebraic Vision Transformer against Standard ViT
(softmax attention, learned absolute positions, GELU MLP) and convolution-style
mixing (ResNet family) without training any network. Each script writes a JSON
report with a fingerprint, a scope disclaimer, and a pass/FAIL gate to
`results/analysis/`.

| Script | Question | Headline result |
|---|---|---|
| `vit_link_analysis.py` | Which algebraic routing kernel? C1 rho-power+sink vs C2 clipped binomial vs C3 Cauchy gap vs S0 softmax | C1 sharpest (top-1 0.38 vs 0.21); all falsifier curves pass |
| `vit_primitives_analysis.py` | Activation beta, 2D positional code, QK normalization | ALU beta 1.2 matches GELU moments within ~1%; Cayley code is shift-invariant with 0 params; QK-norm prevents entropy collapse |
| `vit_expressivity.py` | Frozen-weight retrieval, background suppression, feature rank | Combined +36pp over softmax, +86pp over local mixing; rank 1.12x |
| `vit_efficiency.py` | FLOP + special-function audit, two analytic TPU models, CPU kernel round | Zero transcendental ops; no-slower in every analytic cell (1-5% time-model edge, roofline parity); decisive wall-clock proof needs silicon |
| `vit_certificates.py` | float64 numerical certificates of every operator identity | Max identity error < 3e-15 |

```bash
python3 analysis/vit_link_analysis.py
python3 analysis/vit_primitives_analysis.py
python3 analysis/vit_expressivity.py
python3 analysis/vit_efficiency.py
python3 analysis/vit_certificates.py
```

TPU wall-clock proof requires hardware; the CPU microbenchmark round in
`vit_efficiency.py` measures kernel-structure overhead only and is labeled
as such in its report.
