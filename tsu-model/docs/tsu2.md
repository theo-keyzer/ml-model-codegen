# Guide to the TSU Domain Generator: Building Custom Thermodynamic AI Schemas

## Introduction

Welcome to the **TSU Domain Generator**, a schema-driven framework for defining and generating code for Thermodynamic Sampling Units (TSU) and related probabilistic AI systems. Designed for experts in thermodynamic computing (e.g., Extropic TSU chips), this tool enables rapid prototyping of energy-based models, sampling algorithms, and optimizers for problems like Max-Cut, QUBO, and probabilistic graphical models (PGMs).

### Why Use This for TSU Work?
- **Hardware-Centric Design**: Directly models TSU constraints (e.g., entropy budgets, thermal relaxation, control voltages) from schemas like `tsu.unit` and `tsu-auto.unit`.
- **Probabilistic AI Focus**: Supports energy functions, Gibbs/block sampling, and annealing schedules optimized for stochastic hardware.
- **Multi-Target Code Generation**: Outputs JAX/THRML code for TSU (production/emulated), GPU baselines, and hybrids—ideal for validation against real hardware.
- **Auto-Tuning Integration**: Evolutionary optimizers (`tsu-auto.unit`) tune hyperparameters for rugged energy landscapes, respecting physical limits like Landauer's principle.
- **Extensibility**: Build custom domains (e.g., for protein folding or Ising models) using a self-describing schema language inspired by `unit.unit`.

This guide assumes familiarity with TSU concepts (e.g., Boltzmann sampling, kT noise) and Python/JAX. If you're new to schema-based generation, see the [unit.md guide](unit.md) for basics.

**Version**: Based on schema v1.0 (from `tsu.unit`). Last updated: [Current Date].

## System Overview

The domain generator is a bootstrap system:
1. **Schemas** (`*.unit` files): Define domain knowledge (e.g., `TSU` hardware, `EnergyFunction`, `EvolutionStrategy`) using components (Comp), elements (fields), and options (enums).
2. **Definitions** (`*.net` files): User-specified models (e.g., `max-cut.net` for a QUBO solver).
3. **Actors** (`*.act` files): Templates that process definitions to generate code (e.g., Python/JAX for inference and tuning).
4. **Bootstrap Process**: Use a Go-based loader/generator (per `unit.md`) to parse schemas, validate definitions, and emit code (e.g., `structs.go`, `generated_maxcut.py`).

Key Files in This Repo:
- `tsu.unit`: Core TSU schema (hardware, kernels, PGMs).
- `tsu-auto.unit`: Auto-tuning extensions (evolution, meta-learning).
- `max-cut.net`: Example definition for Max-Cut/QUBO on TSU.
- `max-cut.act` / `max-cut-auto.act`: Code generation actors.
- `unit.md`: Meta-guide to building schemas.

**Workflow Diagram**:
```
[Schema: tsu.unit] → [Bootstrap Loader] → [Validated Structs]
                          ↓
[Definition: max-cut.net] → [Actor Scripts: max-cut.act] → [Generated Code: JAX/THRML]
                          ↓
[Run]: python generated_solver.py → [TSU Emulation / Hardware Output]
```

## Prerequisites

- **Languages/Tools**:
  - Go 1.20+ (for bootstrap: `go run gen/*.go`).
  - Python 3.10+ with JAX (e.g., `pip install jax jaxlib numpy optax`).
  - THRML/JAX (if using features like `jit_compile=true`).
- **Hardware/Emulation**:
  - TSU access (e.g., Extropic dev kit) or emulator (set `emulation=noisy` in `Hardware`).
  - NVIDIA GPU for baselines (CUDA 12+).
- **Setup**:
  1. Clone repo: `git clone <repo> && cd <repo>`.
  2. Bootstrap schemas: `make bootstrap` (or manually: `go run gen/*.go tsu.unit tsu-auto.unit > src/structs.go`).
  3. Install deps: `pip install -r requirements.txt`.

## Quick Start: Running the Max-Cut Example

This example generates and runs a TSU-tuned Max-Cut solver for 100-vertex graphs.

### Step 1: Generate Code
```bash
# Parse schemas and definitions
go run gen/*.go tsu.unit tsu-auto.unit max-cut.net

# Run actors to generate Python
./generate.sh  # Or: python actor_runner.py max-cut.act max-cut-auto.act
```

Output: `./build/maxcut_qubo.py` (solver) and `./build/autotuner.py` (optimizer).

### Step 2: Run Inference (No Tuning)
```bash
cd build
python maxcut_qubo.py --target tsu_production --num-vertices 100
```
- Emulates TSU annealing: Generates random graph, runs Gibbs sampling with geometric cooling, outputs best cut (e.g., ~45% edges cut).
- Metrics: Time, energy (emulated via `ThermodynamicSimulation`), cut quality.

Example Output:
```
==================================================
MAX-CUT/QUBO Solver - Target: tsu_production
==================================================
Hardware: tsu_extropic_1
Mode: production
Best cut found: 47.23
Time elapsed: 2.45s
Samples/sec: 408163
==================================================
```

### Step 3: Run Auto-Tuning
```bash
python autotuner.py --num-vertices 100 --graph-density 0.5 --generations 50 --train-meta-learner
```
- Evolves annealing params (e.g., `T_initial=150.2, cooling_rate=0.96`).
- Uses GA (`population=50`), meta-learning from prior runs, diagnostics for issues like slow convergence.
- Outputs tuned config: Improves cut by 5-10% vs. defaults.

For real TSU: Set `emulation=false` in `Hardware` and deploy via CXL/PCIe interface.

## Deep Dive: Building Custom Schemas

Schemas use a declarative syntax (from `unit.unit`) to define hierarchies. Follow `unit.md` for details.

### Core Syntax
- **Components** (`Comp`): Entities like `Model`, `TSU`.
  ```
  Comp MyTSUOp parent SamplingOp FindIn
  ----------------------------------------------------------------
  * Custom TSU operation for Ising models
  ----------------------------------------------------------------
      Element op_name key . + Operation ID
      Element energy_fn ref EnergyFunction + Linked energy landscape
  ```
- **Elements**: Fields with types (`key`, `ref`, `word`) and checks (`+` required, `*` optional).
- **Options** (`Opt`): Enums for elements.
  ```
  Opt ising . * Ising spin model
  Opt potts . * Potts model
  ```
- **Hierarchy**: Use `parent` for nesting (e.g., `Kernel parent Domain FindIn`); `Find`/`FindIn` for searchability.

### TSU-Specific Components (from `tsu.unit`)
| Component | Purpose | Key Elements |
|-----------|---------|--------------|
| `TSU` | Hardware description | `tsu_id`, `operating_T`, `entropy_budget`, `programmable_energy` |
| `EnergyFunction` | Boltzmann energies | `expression` (e.g., `E(x) = -∑ J_ij s_i s_j`), `variables` |
| `SamplingOp` / `TSUSamplingOp` | Algorithms | `distribution=boltzmann`, `tsu_hint=force_tsu`, `energy_fn_ref` |
| `TSUCompilation` | Model → Hardware mapping | `mapping` (e.g., `spins → physical_qubits`), `fidelity` |
| `ThermodynamicSimulation` | Physics tracking | `T`, `entropy_traced`, `landauer_limit` |

For auto-tuning (from `tsu-auto.unit`):
- `SearchSpace`: Define params (e.g., `T_initial: continuous [10,1000]`).
- `EvolutionStrategy`: GA/CMA-ES with `population=50`, `multi_objective=NSGA-II`.
- `MetaLearning`: Surrogates from past runs (e.g., GP on graph features → params).

### Step-by-Step: Create a Custom Ising Domain
1. **Extend Schema** (`ising.unit`):
   ```
   // Inherit from tsu.unit
   Comp IsingModel parent Model Find
   ----------------------------------------------------------------
   * 2D Ising for TSU magnetic simulation
   ----------------------------------------------------------------
       Element lattice_size word . + Grid dims (e.g., 32x32)
       Element couplings ref Tensor + J_ij matrix
       Element field ref Tensor * External H field

   Comp IsingKernel parent TSUKernel FindIn
       Element energy_fn text . + "E = -∑ J_ij s_i s_j - ∑ H_i s_i"
       Element thermal_relax text . * τ=50ns for spin flips
   ```

2. **Define Model** (`ising.net`):
   ```
   Model {
     model = 2DIsing
     type = energy_based
     hardware = tsu_extropic_1
     pgm_schema = ising_grid
     desc = "2D Ising phase transition sampler"
   }

   EnergyFunction {
     energy_fn = ising_energy
     expression = "E(s) = -J ∑_<i,j> s_i s_j - H ∑ s_i"  // Ferromagnetic
     variables = "s [ -1, +1 ]"
     params = couplings, field
     source = symbolic
   }

   // Add TSU compilation, config, etc. (similar to max-cut.net)
   ```

3. **Generate Code** (Extend `max-cut.act` to `ising.act`):
   - Add actors for Ising-specific energy (e.g., `@jax.jit def energy_ising(spins, J, H): ...`).
   - Auto-tune: Search over `J` strength and `T_critical`.

4. **Run & Validate**:
   ```bash
   go run gen/*.go tsu.unit ising.unit ising.net
   python generated_ising.py --target tsu_emulated --lattice 32
   ```
   - Validates: Phase transition (magnetization vs. T), entropy production.
   - Use `PhysicalConstraint` for TSU limits (e.g., `min_energy_gap >= 0.1 kT`).

5. **Optimize** (via `tsu-auto.unit`):
   - Add `SearchSpace` for `beta=1/T` schedules.
   - Run tuning: Evolves to optimal sampling for critical point detection.

## TSU-Specific Features

- **Hardware Modeling**:
  - `TSUSubstrateModel`: Limits like `max_vars=10000`, `kT_per_step=1.5 kT`.
  - Emulation Modes: `ideal` (noiseless), `noisy` (Gaussian thermal noise), `limited_entropy` (Landauer tracking).
- **Thermodynamic Fidelity**:
  - Energy functions must be `thermodynamic=true` for TSU offload.
  - Simulations compute dissipation: `total_dissipation = ∑ |ΔE|` vs. Landauer limit (`2.85e-21 J/bit`).
- **Compilation & Mapping**:
  - `TSUCompilation`: Auto-maps variables to spins (e.g., `control_lines="V_ctrl[0:99] = map_qubo_weights"`).
  - Fusion: `Fusion` combines ops (e.g., anneal + evaluate) to reduce latency.
- **Auto-Tuning for TSU**:
  - Constraints: Power/time budgets via `ResourceBudget`.
  - Diagnostics: Detects TSU failures (e.g., `tsu_calibration_drift`) with recovery (re-calibrate every 500 samples).
  - Pareto Optimization: Trade cut quality vs. entropy (e.g., `objectives="cut_quality, energy_consumption"`).

**Tip**: For real TSU, set `backend=tsu` and `emulation=false`; use `control_voltage` for programming (e.g., [-1.2V, +1.2V] for weights).

## Example: Extending Max-Cut for Custom Graphs

The provided `max-cut.net` solves QUBO via TSU annealing. To adapt:
- Change `num_vertices=200`, add `graph_type=planar` in `PGMSchema`.
- Tune via `max-cut-auto.net`: Focus on `cooling_rate` (most sensitive per `SensitivityAnalysis`).
- Generated Code Snippet (from `max-cut.act`):
  ```python
  @jax.jit
  def energy_maxcut(partition: jnp.ndarray, edge_weights: jnp.ndarray) -> float:
      # MAX-CUT: E(x) = -sum w_ij * (x_i XOR x_j)
      n = partition.shape[0]
      # Vectorized XOR contributions...
      return energy  # Minimized for max cut
  ```

Benchmark: On TSU emulation, achieves ~95% of theoretical bound in 1M samples; tuning boosts to 98%.

## Advanced Topics

- **Multi-Target Generation** (`Project` / `TargetConfig`):
  - Define: `targets="tsu_prod,gpu_sim,hybrid"`.
  - Generate: Per-target code with validation (`validate_against=gpu_baseline` compares cuts).
- **Validation & Constraints**:
  - `Validation`: Rules like `found_cut >= 0.90 * upper_bound`.
  - Physics Checks: `PhysicalConstraint` halts if `power > 5W`.
- **Meta-Learning & Transfer**:
  - Train surrogates: `MetaLearning` uses past runs to predict params (10x speedup).
  - Transfer: Map Ising couplings to Max-Cut edges.
- **Scaling**:
  - JAX-Parallel: Use `jax.pmap` for population eval in tuning.
  - Large Graphs: Fuse kernels (`thermodynamic_fusion`) for >10k vars.

## Troubleshooting & Best Practices

### Common Issues
- **Unresolved Refs**: Error like `?Model:10`. **Fix**: Ensure `ref` targets exist; order matters (define parents first).
- **Bootstrap Fails**: Schema syntax error. **Fix**: Validate with `unit.md` (e.g., tabs for alignment).
- **Emulation Mismatch**: Noisy TSU underperforms. **Fix**: Tune `thermal_noise=gaussian_0.01`; compare vs. `ideal`.
- **Tuning Stalls**: Premature convergence. **Fix**: Increase `mutation_rate=0.3`; enable `adaptive_acceptance`.

### Best Practices
- **Start Small**: Define 5-10 components; test with a tiny `.net` (e.g., 10 vars).
- **Document**: Use `desc` fields; add math (e.g., `E(x;θ)` in LaTeX-like).
- **TSU Optimization**: Always include `energy_model` (e.g., `kT * ln(2) * vars`); test fidelity ≥98%.
- **Version Control**: Schema changes? Re-bootstrap everything.
- **Performance**: Vectorize (JAX `vmap`); profile entropy via `ThermodynamicSimulation`.
- **Extensions**: Add Opt for new distributions (e.g., `Opt quantum . * Quantum Boltzmann`).

### Pitfalls to Avoid
- Over-Nesting: Keep <3 levels to avoid parse errors.
- Ignoring Physics: Set `thermodynamic=true` for kernels, or TSU offload fails.
- No Validation: Always run `PhysicalConstraint`—TSU violations (e.g., ΔE < min_gap) cause unreliable samples.

## References
- **Schemas**: `tsu.unit` (core), `tsu-auto.unit` (tuning), `max-cut.net` (example).
- **Guides**: [unit.md](unit.md) for schema basics; THRML docs for JAX integration.
- **Papers**: "Thermodynamic Computing" (Extropic); "Evolutionary Algorithms for Combinatorial Optimization".
- **Community**: Join TSU forums (e.g., Extropic Discord) for hardware tips.
- **Contribute**: Fork and add components (e.g., optical TSU via `connectivity=optical`).

For questions, open an issue or email [your-contact]. Happy sampling—may your energies minimize efficiently! 🚀
