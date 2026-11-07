# Algebraic Vision Transformer: Mathematical Specification

A standalone, publication-grade specification of the **Algebraic Vision
Transformer (Algebraic ViT)**, a purely algebraic image classifier whose
training objective and its gradient evaluate without a single
transcendental operation. Companion baselines: **Standard ViT**
(softmax attention, learned absolute positions, GELU) and **ResNet**
(convolution, batch normalization, ReLU).

## 1. Primitive: the positive link

Define

$$\rho(x) = x + \sqrt{1 + x^2}, \qquad \rho(-x) = \rho(x)^{-1}.$$

Implementations use reciprocal square roots and multiplication only.
The link obeys the group law

$$\rho(x)\rho(y) = \rho\left(x\sqrt{1+y^2} + y\sqrt{1+x^2}\right),$$

so normalization by a shared maximum needs no logarithm or exponential,
and the global bound $\rho(x) \le 2\lvert x\rvert + 1$. Positive integer
powers use binary exponentiation. The identity
$\rho(x) = \exp(\operatorname{asinh} x)$ is a reference-calculation tool
only; `exp` and `arcsinh` never appear in the model or its gradient.

## 2. Patch attention with sink

For bidirectional patch scores $s_{ij} = q_i^\top k_j / \sqrt{d_h}$ over
$T = 196$ patches plus one CLS token, head dimension $d_h = 64$, and a
learned per-head sink logit $b$,

$$a_{ij} = \frac{\rho(s_{ij}/8)^{16}}{\sum_k \rho(s_{ik}/8)^{16} + \rho(b/8)^{16}}.$$

The sink carries a zero value vector, so visible weights may sum to less
than one: the head can route background mass to nowhere instead of
averaging distractors. Queries and keys receive learned-gain RMS
normalization before rotation, which pins score variance and prevents
the depth-driven entropy collapse measured in
`results/analysis/vit_primitives_analysis.json`. The kernel slope at the
origin is $n/c = 2$, giving sharper selection than unit-slope softmax
(top-1 mass 0.38 vs 0.21 on matched synthetic scores).

## 3. Two-dimensional factorized Cayley positions

Head channels split into four groups $[r_1, r_2, c_1, c_2]$. Row and
column frequencies $w^R_k = w^C_k = \tfrac12 10000^{-k/(d_h/4)}$ define
unit rotations

$$C(w) = \frac{1-w^2}{1+w^2}, \qquad S(w) = \frac{2w}{1+w^2},$$

with position tables built by binary exponentiation and reciprocal-root
renormalization. Row tables rotate $(r_1, r_2)$ by row index, column
tables rotate $(c_1, c_2)$ by column index. Pairwise scores are exactly
invariant under joint grid shifts, the tables use zero parameters
(against $196 \times d$ learned embeddings), and no trigonometric
function is evaluated.

## 4. Normalization and activation

Learned RMS normalization $g \odot x / \sqrt{\mathrm{mean}(x^2) +
\epsilon}$ appears pre-attention, on queries and keys, pre-MLP, and
before the head. The gated MLP uses

$$\mathrm{ALU}_{1.2}(x) = \frac{x}{2}\left(1 +
\frac{1.2\,x}{\sqrt{1 + 1.44\,x^2}}\right),$$

with $\mathrm{ALU}'(x) = \tfrac12(1+u) + \tfrac12 u r^2$,
$u = 1.2x/\sqrt{1+1.44x^2}$, $r^2 = 1/(1+1.44x^2)$. Mean-field matching
against GELU selects $\beta = 1.2$: Gaussian $L_2$ distance is minimal
there, and both the gradient gain $E[{\phi'}^2]$ and the forward energy
$E[\phi^2]$ agree with GELU within about 1% at unit variance.

## 5. Classification head

With logits $z_j$ over 100 (CIFAR-100) or 1000 (ImageNet-1k) classes,

$$p_j = \frac{\rho(z_j/2)^8}{\sum_k \rho(z_k/2)^8}, \qquad
\frac{\partial[-\sum_k y_k \log p_k]}{\partial z_j} =
\frac{4\,(p_j - y_j)}{\sqrt{1 + (z_j/2)^2}}.$$

A custom vector-Jacobian product returns this exact log-loss gradient
with no logarithm in the differentiated path. Forward telemetry converts
stop-gradient target probabilities to NLL outside the model.

## 6. Training-free evidence

All claims below are measured without training, in `analysis/`:

- Routing shootout: C1 (this specification) is sharper than softmax and
  two alternative algebraic links, with every falsifier curve passing.
- Primitives: activation match, shift invariance to $< 10^{-8}$,
  unit-norm drift $< 10^{-12}$, QK-norm entropy protection.
- Expressivity: frozen-weight combined retrieval/suppression score
  exceeds softmax by 36pp and local mixing by 86pp; effective rank 1.12x.
- Efficiency: zero transcendental ops; analytic models show the
  algebraic stack no-slower under every cost assumption (1-5% edge in
  the time model, exact parity in the traffic-bound roofline). A
  decisive wall-clock separation is not established training-free and
  requires fused kernels measured on silicon (see `vit_efficiency.json`,
  `decisive_wallclock_proven: false`).
- Certificates: every operator identity holds in float64 to $< 3e-15.

## 7. Formal verification

`formal/` machine-checks the real-arithmetic theorems behind this
specification in Lean 4: link positivity, reciprocal and group laws,
global bound, max-shift invariance, bidirectional normalization and
tile decomposition with sink, normalized Jacobians, Cayley unit norm
and composition, and the exact head gradient. There are no `sorry`
statements and no additional axioms.

## 8. Scope of the algebraic claim

The claim covers the model objective and its gradient during training.
Parameter initialization, the AdamW learning-rate schedule, metric
telemetry, and evaluation scoring run outside that boundary. In
particular, NLL and perplexity reporting use logarithms, and the shared
cosine schedule uses a cosine.
