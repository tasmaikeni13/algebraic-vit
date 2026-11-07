import Mathlib.Tactic

namespace AlgebraicVision

/-- Cayley unit rotation from a rational parameter, no trigonometry. -/
noncomputable def cayleyC (w : ℝ) : ℝ := (1 - w ^ 2) / (1 + w ^ 2)

noncomputable def cayleyS (w : ℝ) : ℝ := (2 * w) / (1 + w ^ 2)

theorem cayley_unit_norm (w : ℝ) :
    cayleyC w ^ 2 + cayleyS w ^ 2 = 1 := by
  unfold cayleyC cayleyS
  have h : (1 + w ^ 2) ≠ 0 := ne_of_gt (by positivity)
  field_simp
  ring

/-- Rotation composition preserves the unit norm. -/
theorem cayley_composition_norm (C1 S1 C2 S2 : ℝ)
    (h1 : C1 ^ 2 + S1 ^ 2 = 1) (h2 : C2 ^ 2 + S2 ^ 2 = 1) :
    (C1 * C2 - S1 * S2) ^ 2 + (S1 * C2 + C1 * S2) ^ 2 = 1 := by
  have key : (C1 * C2 - S1 * S2) ^ 2 + (S1 * C2 + C1 * S2) ^ 2
      = (C1 ^ 2 + S1 ^ 2) * (C2 ^ 2 + S2 ^ 2) := by ring
  rw [key, h1, h2, mul_one]

/-- Rotate-half action on a channel pair. -/
noncomputable def rot2 (C S : ℝ) (x : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![x 0 * C - x 1 * S, x 0 * S + x 1 * C]

noncomputable def dot2 (u v : Fin 2 → ℝ) : ℝ := u 0 * v 0 + u 1 * v 1

/-- Successive rotations compose by adding their parameters. -/
theorem rot2_comp (C1 S1 C2 S2 : ℝ) (x : Fin 2 → ℝ) :
    rot2 C2 S2 (rot2 C1 S1 x)
      = rot2 (C1 * C2 - S1 * S2) (S1 * C2 + C1 * S2) x := by
  funext j
  fin_cases j <;> simp [rot2, Matrix.cons_val_zero, Matrix.cons_val_one] <;> ring

/-- Joint rotation preserves pairwise scores: relative-position dependence. -/
theorem rot2_dot (C S : ℝ) (u v : Fin 2 → ℝ) (h : C ^ 2 + S ^ 2 = 1) :
    dot2 (rot2 C S u) (rot2 C S v) = dot2 u v := by
  have key : dot2 (rot2 C S u) (rot2 C S v) = dot2 u v * (C ^ 2 + S ^ 2) := by
    simp [dot2, rot2, Matrix.cons_val_zero, Matrix.cons_val_one]
    ring
  rw [key, h, mul_one]

end AlgebraicVision
