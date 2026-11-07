# Algebraic Vision Transformer

Can you train a competitive Vision Transformer without a single
transcendental function?

In standard vision models, non-linearities rely heavily on transcendental
operations: attention uses `exp` in softmax, feed-forward blocks use `GELU`
(`erf`), and the output head computes log-sum-exp cross-entropy.

The **Algebraic Vision Transformer** replaces all of them with purely
algebraic operations: functions of field arithmetic (+, -, *, /) and
reciprocal square roots (`rsqrt`). Its exact log-loss gradient evaluates
without logarithms, exponentials, or trigonometric primitives.

## Architecture comparison

| Component | Standard ViT | Algebraic ViT |
|---|---|---|
| **Attention** | Softmax (`exp(QK^T / sqrt(d))`) | Algebraic asinh power (`rho(S/8)^16`) + learned sink |
| **Positions** | Learned absolute embeddings | 2D factorized Cayley rotary code (zero params) |
| **Feed-forward** | GELU MLP | ALU-GLU (`beta = 1.2`, matched to GELU moments) |
| **Normalization** | LayerNorm | RMSNorm + QK normalization |
| **Head** | Softmax cross-entropy | Algebraic normalized power (`rho(z/2)^8`), exact gradient |

Here `rho(x) = x + sqrt(1 + x^2)`. The third baseline is a canonical
**ResNet** (convolution, batch normalization, ReLU, SGD with momentum).

## Repository layout

| Path | Contents |
|---|---|
| `theory.md` | Standalone mathematical specification |
| `analysis/` | Training-free benchmark suite (routing, primitives, expressivity, efficiency, certificates) |
| `formal/` | Lean 4 machine-checked proofs (95 theorems, zero `sorry`) |
| `src/vision/` | Algebraic ViT, Standard ViT, ResNet, data loading, distributed trainer |
| `src/vision/kernels/` | Custom Pallas kernels (flash attention, fused head, QK norm) |
| `scripts/` | Data preparation, sanity tier, sweeps, pod orchestration, algebraic audit |
| `configs/` | Sanity, sweep, and main-run tier configurations |
| `tests/` | CPU numerical test suite |

## Quick start

```bash
pip install -r requirements.txt
python scripts/prepare_cifar100.py --data-dir ~/data/cifar100
python scripts/sanity_cifar100.py --data-dir ~/data/cifar100 --steps 20 --out runs/sanity
python scripts/audit_algebraic.py
pytest tests/ -q
```

## Experiment tiers

| Tier | Models | Data | Budget |
|---|---|---|---|
| Sanity | Algebraic ViT-Ti, Standard ViT-Ti, ResNet-18 | CIFAR-100 | 150 epochs |
| Small sweep | ViT-S x2, ResNet-50 | ImageNet-1k | LR x WD grid, 50 epochs |
| Small main | ViT-S x2, ResNet-50 | ImageNet-1k | 300 epochs, 3 seeds |
| Base sweep | ViT-B x2 | ImageNet-1k | LR x WD grid, 50 epochs |
| Base main | ViT-B x2 (no ResNet) | ImageNet-1k | 300 epochs, 1 seed |

Sweeps assign one trial per worker (4 concurrent, one wave at a time); main runs span the
full v4-32 pod. See `scripts/run_pod.py` and `configs/`.

## Scope of the algebraic claim

The claim covers the model objective and its gradient during training.
Initialization, the cosine schedule, telemetry, and evaluation run outside
that boundary. Training-free analysis establishes decisive expressivity
superiority; wall-clock superiority is established by fused-kernel runs on
silicon (see `results/analysis/vit_efficiency.json` for the analytic bounds).
