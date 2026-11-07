import AlgebraicAttention.Radical
import AlgebraicAttention.Normalization
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic

namespace AlgebraicVision

open AlgebraicAttention
open Finset
open scoped BigOperators

/-- Shifted algebraic patch weight with explicit shift `m` (the row maximum). -/
noncomputable def patchWeight (s m c : ℝ) (n : ℕ) : ℝ :=
  (rho (s / c) * rho (-m)) ^ n

theorem patch_weight_nonneg (s m c : ℝ) (n : ℕ) :
    0 ≤ patchWeight s m c n := by
  unfold patchWeight
  exact pow_nonneg (mul_nonneg (le_of_lt (rho_pos _)) (le_of_lt (rho_pos _))) n

theorem patch_weight_pos (s m c : ℝ) (n : ℕ) :
    0 < patchWeight s m c n := by
  unfold patchWeight
  exact pow_pos (mul_pos (rho_pos _) (rho_pos _)) n

/-- Scores below the shift normalize to weights in (0, 1]. -/
theorem patch_shifted_le_one (s m c : ℝ) (n : ℕ) (h : s / c ≤ m) :
    patchWeight s m c n ≤ 1 := by
  unfold patchWeight
  exact (shifted_weight_le_one (s / c) m n h).2

/-- Online rescaling preserves patch normalization, including a sink. -/
theorem patch_rescaling_invariant (a numerator denominator : ℝ) (ha : a ≠ 0) :
    (a * numerator) / (a * denominator) = numerator / denominator :=
  rescaling_invariant a numerator denominator ha

/-- Patch-block reduction: splitting keys into two blocks preserves the total. -/
theorem patch_block_split {ι : Type*} (s : Finset ι) (f : ι → ℝ)
    (p : ι → Prop) [DecidablePred p] :
    (∑ x ∈ s.filter p, f x) + ∑ x ∈ s.filter (fun x => ¬ p x), f x
      = ∑ x ∈ s, f x :=
  Finset.sum_filter_add_sum_filter_not s p f

/-- With a sink, visible patch weights sum to at most one. -/
theorem weights_sum_le_one_with_sink {ι : Type*} (s : Finset ι) (w : ι → ℝ)
    (sink : ℝ) (hw : ∀ i ∈ s, 0 ≤ w i) (hs : 0 ≤ sink)
    (hpos : 0 < (∑ i ∈ s, w i) + sink) :
    (∑ i ∈ s, w i) / ((∑ i ∈ s, w i) + sink) ≤ 1 := by
  have hsum : 0 ≤ ∑ i ∈ s, w i := Finset.sum_nonneg hw
  exact (div_le_one hpos).2 (le_add_of_nonneg_right hs)

/-- Without a sink, patch weights sum exactly to one. -/
theorem weights_sum_eq_one_no_sink {ι : Type*} (s : Finset ι) (w : ι → ℝ)
    (hpos : 0 < ∑ i ∈ s, w i) :
    (∑ i ∈ s, w i) / (∑ i ∈ s, w i) = 1 :=
  div_self (ne_of_gt hpos)

/-- No zero-row collapse: every nonempty patch row has positive mass. -/
theorem rank_noncollapse {ι : Type*} (s : Finset ι) (x : ι → ℝ) (m c : ℝ)
    (n : ℕ) (hne : s.Nonempty) :
    0 < ∑ i ∈ s, patchWeight (x i) m c n := by
  exact Finset.sum_pos (fun i _ => patch_weight_pos _ _ _ _) hne

/-- Equal-key sink identity: uniform patch background splits evenly with the sink. -/
theorem cls_sink_identity {ι : Type*} (s : Finset ι) (w : ι → ℝ) (u sink : ℝ)
    (h : ∀ i ∈ s, w i = u) :
    (∑ i ∈ s, w i) / ((∑ i ∈ s, w i) + sink)
      = (s.card * u) / ((s.card * u) + sink) := by
  have hsum : ∑ i ∈ s, w i = s.card * u := by
    rw [Finset.sum_congr rfl (fun i hi => h i hi)]
    rw [Finset.sum_const, nsmul_eq_mul]
  rw [hsum]

/-- QK-RMS mechanism: normalizing by the root-mean-square kills per-row scale. -/
theorem qk_scale_invariance (c q S : ℝ) (hc : 0 < c) :
    (c * q) / Real.sqrt (c ^ 2 * S) = q / Real.sqrt S := by
  have hsq : Real.sqrt (c ^ 2 * S) = c * Real.sqrt S := by
    rw [Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq (le_of_lt hc)]
  rw [hsq]
  exact mul_div_mul_left q (Real.sqrt S) (ne_of_gt hc)

end AlgebraicVision
