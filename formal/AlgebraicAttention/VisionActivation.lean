import AlgebraicAttention.Radical
import Mathlib.Tactic

namespace AlgebraicVision

open AlgebraicAttention

/-- Algebraic linear unit with slope parameter `b`. -/
noncomputable def alu (b x : ℝ) : ℝ :=
  x / 2 * (1 + b * x / Real.sqrt (1 + (b * x) ^ 2))

/-- The gate stays strictly inside (-1, 1). -/
theorem alu_gate_lt_one (b x : ℝ) :
    |b * x / Real.sqrt (1 + (b * x) ^ 2)| < 1 := by
  have hsqrt : 0 < Real.sqrt (1 + (b * x) ^ 2) :=
    Real.sqrt_pos.2 (by positivity)
  rw [abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
  exact (div_lt_one hsqrt).2 (abs_lt_radical (b * x))

theorem alu_zero (b : ℝ) : alu b 0 = 0 := by
  simp [alu]

/-- The activation preserves signs: no spurious background excitation. -/
theorem alu_nonneg_of_nonneg (b x : ℝ) (hx : 0 ≤ x) : 0 ≤ alu b x := by
  have hu := alu_gate_lt_one b x
  have hlo : -1 < b * x / Real.sqrt (1 + (b * x) ^ 2) :=
    neg_lt_of_abs_lt hu
  unfold alu
  exact mul_nonneg (by linarith) (by linarith)

/-- The activation preserves signs on the negative side. -/
theorem alu_nonpos_of_nonpos (b x : ℝ) (hx : x ≤ 0) : alu b x ≤ 0 := by
  have hu := alu_gate_lt_one b x
  have hlo : -1 < b * x / Real.sqrt (1 + (b * x) ^ 2) :=
    neg_lt_of_abs_lt hu
  unfold alu
  exact mul_nonpos_of_nonpos_of_nonneg (by linarith) (by linarith)

/-- The activation never magnifies magnitudes: a stability bound. -/
theorem alu_abs_le (b x : ℝ) : |alu b x| ≤ |x| := by
  have hu := alu_gate_lt_one b x
  have hpair := abs_lt.mp hu
  have h1u : |1 + b * x / Real.sqrt (1 + (b * x) ^ 2)| ≤ 2 := by
    apply abs_le.mpr
    constructor <;> linarith [hpair.1, hpair.2]
  unfold alu
  rw [abs_mul]
  have hx2 : |x / 2| = |x| / 2 := by
    rw [abs_div]
    norm_num
  rw [hx2]
  nlinarith [mul_nonneg (abs_nonneg x) (sub_nonneg.mpr h1u),
    abs_nonneg x, abs_nonneg (1 + b * x / Real.sqrt (1 + (b * x) ^ 2))]

end AlgebraicVision
