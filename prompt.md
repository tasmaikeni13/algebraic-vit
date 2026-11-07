Transform this repository into a standalone, publication-grade computer vision codebase introducing the **Algebraic Vision Transformer (Algebraic ViT)**, demonstrating decisive superiority in both accuracy and wall-clock speed over **Standard ViT** and **ResNet** on a Google Cloud TPU v4-32 pod (16 TPU v4 chips across 4 workers).

### STRICT GLOBAL CONSTRAINTS & OPERATING PROTOCOL (READ FIRST)
1. **Strict Sequential Execution (Hard Phase Gates):** Execute this project strictly step-by-step and phase-by-phase. **DO NOT** jump ahead to writing training scripts, model implementations, or kernels for Algebraic ViT until Phase 1 has mathematically, numerically, and analytically proven a champion algebraic architecture that decisively beats both Standard ViT and ResNet.
2. **Mandatory Multi-Agent Swarm Orchestration:** For all phases—and for every medium-to-large task—you MUST continuously spawn multiple specialized subagents working together as a coordinated agentic team (e.g., dedicated subagents for literature search, abstract mathematical formulation, Monte Carlo simulations, Lean formalization, Pallas kernel engineering, and debugging). Never tackle complex exploration or verification in a single thread when parallel subagents can divide and conquer.
3. **Active Web Search & Deep Research:** You and your subagents are explicitly encouraged to search the internet, study recent papers, explore advanced abstract mathematics, and research TPU/Pallas documentation whenever encountering unfamiliar concepts or seeking stronger theoretical breakthroughs.
4. **Skills Integration (Silent):** Read and utilize everything in the `skills/` directory completely. **DO NOT** mention, cite, or acknowledge the `skills/` folder anywhere in your outputs, logs, or files. **DO NOT** create, update, or mention `research_ledger.md`.
5. **Complete Legacy Purge:** Delete all references to "Algebraic Transformer", "Standard Transformer", and "ARDS" across the entire repository.
6. **Standalone Documentary Style (Zero Diff-History):** Write all code, documentation (`theory.md`), and comments as a completely standalone computer vision project built from scratch. Never write historical or transitional phrasing (e.g., NEVER write "Previously this was Algebraic Transformer and now it is Algebraic Vision" or "Changed from before"). Maintain strict PEP 8 formatting with clear, pedagogical comments so any reader can learn the code and handle errors independently.

---

### PHASE 1: Open-Ended Algebraic Discovery, Zero-Training Benchmarking & Lean Formalization
*Gate Requirement: Do not proceed to Phase 2 until a superior Algebraic Vision architecture is discovered, analytically benchmarked, documented, and formalized.*

1. **Abstract Mathematical Exploration:** Spawn parallel subagents to explore diverse, wild, and novel ideas across algebra and abstract mathematics. Do not limit yourself to conventional mechanisms—leverage the full flexibility of algebraic equations to invent a strictly algebraic vision architecture.
2. **Rigorous Training-Free Benchmarking (`analysis/`):** 
   - Utilizing the `analysis/` folder, perform exhaustive mathematical, numerical, statistical, and Monte Carlo analyses comparing your candidate algebraic formulations directly against **Standard ViT** and **ResNet** *without full model training*.
   - Evaluate both **representational capability/expressivity** AND **computational wall-clock efficiency**. Keep iterating across mathematical formulations until the Algebraic Vision model beats both Standard ViT and ResNet by a wide, decisive margin on both fronts.
3. **Theory & Lean Formalization:**
   - Once the champion architecture is validated, rewrite `theory.md` as a standalone, comprehensive mathematical specification.
   - Formalize the champion theory and mathematical proofs in **Lean**, iterating until all proofs verify cleanly.

---

### PHASE 2: Standalone Codebase Cleanup, Architecture Scripts & Sanity Verification
*Gate Requirement: Only begin after Phase 1 is 100% verified.*

1. **Repository Cleanup & Architecture Implementation:** Purge all remaining legacy references and implement clean, standalone scripts for all three architectures: **Algebraic ViT**, **Standard ViT**, and **ResNet**.
2. **Optimization & Metrics Setup:**
   - **Algebraic ViT & Standard ViT:** Train using AdamW with cosine learning rate scheduling (zero references to ARDS).
   - **ResNet:** Train using standard literature best practices for optimization.
   - Track and log wall-clock time, throughput, parameter count, and accuracy across all runs.
3. **Experimental Script Suite & Tiers:**
   - **Sanity Check (~13M Parameters | CIFAR-100 | 100–200 Epochs):** `Algebraic ViT-Ti`, `Standard ViT-Ti`, and `ResNet-18`.
   - **Hyperparameter Sweeps:** Create literature-standard budget hyperparameter sweep scripts to run prior to the Small and Base main runs.
   - **Small Main Runs (~26M Parameters | ImageNet-1k | 300 Epochs | 3 Seeds per architecture):** `Algebraic ViT-S`, `Standard ViT-S`, and `ResNet-50`.
   - **Base Main Runs (~91M Parameters | ImageNet-1k | 300 Epochs | 1 Seed per architecture):** `Algebraic ViT-B` and `Standard ViT-B` (no ResNet at Base scale).
4. **Live Sanity Execution:** Personally execute the CIFAR-100 sanity check scripts and iterate continuously until all three architectures run end-to-end and verify as working as expected.

---

### PHASE 3: Custom Pallas Kernels for Hardware Acceleration
1. **Kernel Engineering:** Spawn specialized kernel subagents to write custom Pallas kernels providing maximum wall-clock speedups for all three architectures:
   - **Algebraic FlashAttention** Pallas kernel for `Algebraic ViT`.
   - **Standard FlashAttention** Pallas kernel for `Standard ViT`.
   - Custom **Pallas kernels** for `ResNet` bottlenecks.
2. **Verification Loop:** Iteratively test, benchmark, and validate every Pallas kernel for numerical correctness, gradient flow, and wall-clock acceleration until all three architectures run seamlessly with their custom kernels.

---

### PHASE 4: TPU v4-32 Pod Orchestration (16 TPU Chips Across 4 Workers)
1. **Parallel Single-Chip Budget Sweeps (16 Concurrent Sweeps):** Configure the hyperparameter sweep scripts to assign **1 hyperparameter configuration per individual TPU chip** across the 4 workers, executing 16 sweep trials simultaneously in parallel for maximum speed.
2. **Full-Pod Distributed Main Runs (All 16 Chips Synchronized):** Configure the Small (~26M) and Base (~91M) main runs to execute across **all 16 TPU v4 chips across all 4 workers together**, ensuring synchronized distributed training, optimal throughput, and end-to-end reliability.

Proceed sequentially starting with Phase 1. Spawn your initial research and mathematical analysis subagent teams now.
