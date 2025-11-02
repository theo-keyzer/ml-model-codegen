
## **Model-Level Variables**

```
Model {
  model = <VARIABLE: IsingPGM, MaxCutQUBO, ProteinFolding, TSP_QUBO>
  type = <VARIABLE: energy_based, probabilistic, pgm>
  framework = <VARIABLE: thrml_jax, pytorch_pgm, tensorflow_prob>
  pgm_schema = <VARIABLE: ising_2d_lattice, maxcut_graph, crf_linear_chain>
  tsu_use = <VARIABLE: required, preferred, avoided>
}
```

## **PGM Schema Variables**

```
PGMSchema {
  schema = <VARIABLE: ising_2d_lattice, maxcut_graph, rbm_100x50>
  variables = <VARIABLE: "s_0...s_1023", "x_0...x_99", "y_0...y_T">
  domains = <VARIABLE: binary, discrete_K, continuous>
  factors = <VARIABLE: "pairwise: J*s_i*s_j", "clique_3", "neural_potential">
  graph_sparsity = <VARIABLE: sparse, dense, tree, grid>
}
```

## **Tensor Shape & Size Variables**

```
Tensor {
  tensor = <VARIABLE: spin_state, partition_state, hidden_units>
  shape = <VARIABLE: [100], [32, 1024], [batch, height, width]>
  dtype = <VARIABLE: binary, fp32, fp16, int8>
  layout = <VARIABLE: dense, sparse_csr, sparse_coo>
  role = <VARIABLE: latent, observed, factor_param>
}
```

## **Hardware Configuration Variables**

```
Hardware {
  hardware = <VARIABLE: tsu_extropic_1, gpu_a100, cpu_x86, tpu_v4>
  backend = <VARIABLE: tsu, gpu, cpu, tpu>
  constraints = <VARIABLE: "sample_rate=1e12/s, T=300K", "memory=80GB">
}

TSU {
  tsu_id = <VARIABLE: tsu0, tsu1, tsu_cluster_0>
  process_node = <VARIABLE: 3nm, 5nm, 7nm>
  operating_T = <VARIABLE: 300, 77, 4>  // Kelvin: room temp, liquid N2, liquid He
  sample_rate = <VARIABLE: 1e12, 1e9, 1e15>  // samples/sec
  precision = <VARIABLE: 8, 12, 16>  // effective bits
  connectivity = <VARIABLE: CXL-3, PCIe-5, optical>
  programmable_energy = <VARIABLE: true, false>
  physical_implementation = <VARIABLE: CMOS-stochastic, spintronics, photonic>
}
```

## **Configuration & Schedule Variables**

```
Config {
  config = <VARIABLE: inference_tsu, training_gpu, validation_cpu>
  target = <VARIABLE: tsu_extropic_1, gpu_a100, cpu_x86>
  batch = <VARIABLE: 1, 32, 64, 256>
  sample_budget = <VARIABLE: 1000, 1000000, 1e9>
}

Schedule {
  seq = <VARIABLE: 1, 2, 3, ...>  // Execution order
  layer = <VARIABLE: spin_lattice_layer, tsu_sampling_layer>
  op = <VARIABLE: block_gibbs_update, tsu_native_sample>
}
```

## **Sampling Algorithm Variables**

```
SamplingOp {
  sampling_op = <VARIABLE: gibbs_sweep, langevin_dynamics, hmc_step>
  distribution = <VARIABLE: boltzmann, gaussian, categorical>
  temperature = <VARIABLE: 1.0, variable, "schedule">
  algorithm = <VARIABLE: block_gibbs, metropolis, langevin>
  tsu_hint = <VARIABLE: force_tsu, prefer_tsu, avoid_tsu>
}

BlockGibbsOp {
  block_op = <VARIABLE: lattice_block_32, sequential_sweep, parallel_async>
  resampling = <VARIABLE: sequential, parallel, async>
  energy_cost = <VARIABLE: "1.2 kT", "0.8 kT", "variable">
}
```

## **Energy Function Variables**

```
EnergyFunction {
  energy_fn = <VARIABLE: ising_energy, maxcut_energy, rbm_free_energy>
  expression = <VARIABLE: "-sum J*s_i*s_j", "x^T Q x", "custom_formula">
  source = <VARIABLE: symbolic, learned, diffusion_prior>
}

EnergyFactor {
  factor = <VARIABLE: pairwise_coupling, unary_field, clique_potential>
  type = <VARIABLE: pairwise, clique, unary, neural>
  learned = <VARIABLE: true, false>
  sparse = <VARIABLE: true, false>
  tsu_native = <VARIABLE: true, false>
}
```

## **TSU Compilation Variables**

```
TSUCompilation {
  plan_id = <VARIABLE: ising_to_tsu0, maxcut_mapping_v2>
  mapping = <VARIABLE: "spin_state -> physical_spins[1024]", "custom_map">
  clocking = <VARIABLE: event_driven, synchronous, async>
  control_bus = <VARIABLE: 64, 128, 256>  // Number of control lines
  calibration = <VARIABLE: "per_inference", "background", "one_time">
  fidelity = <VARIABLE: "99.7%", "95%", "99.99%">
}

TSUSubstrateModel {
  substrate = <VARIABLE: extropic_substrate, prototype_v2>
  max_vars = <VARIABLE: 10000, 100000, 1000>
  max_factors = <VARIABLE: 50000, 500000, 5000>
  connectivity = <VARIABLE: sparse, programmable, fixed_grid>
  interaction = <VARIABLE: Ising, QUBO, Potts, XY>
  kT_per_step = <VARIABLE: "1.2 kT", "0.5 kT", "2.0 kT">
  min_energy_gap = <VARIABLE: "0.5 kT", "0.1 kT", "1.0 kT">
}
```

## **Thermodynamic Simulation Variables**

```
ThermodynamicSimulation {
  sim_id = <VARIABLE: tsu_thermo_sim, cooling_analysis, entropy_study>
  T = <VARIABLE: 300, 77, 4, variable>  // Temperature in Kelvin
  kT = <VARIABLE: "4.11e-21", "1.06e-21 * T", "variable">
  entropy_traced = <VARIABLE: true, false>
  landauer_limit = <VARIABLE: "2.85e-21 J/bit", "kT * ln(2)">
}
```

## **Validation Variables**

```
Validation {
  rule = <VARIABLE: spin_convergence, solution_quality, energy_bounds>
  condition = <VARIABLE: "autocorr < 0.01", "cut >= 0.90 * bound", "E_min < E < E_max">
}

PhysicalConstraint {
  constraint = <VARIABLE: tsu_power_cap, landauer_bound, thermal_limit>
  target = <VARIABLE: tsu0, block_gibbs_update, entire_model>
  condition = <VARIABLE: "power < 3W", "E >= kT*ln(2)", "T < T_max">
  severity = <VARIABLE: warning, error, halt>
  thermodynamic = <VARIABLE: true, false>
}
```

## **Kernel Variables**

```
Kernel {
  kernel = <VARIABLE: tsu_gibbs_kernel, energy_cuda, maxcut_cpu>
  hardware = <VARIABLE: tsu_extropic_1, gpu_a100, cpu_x86>
  signature = <VARIABLE: "void tsu_gibbs(...)", "float compute_energy(...)">
  body = <VARIABLE: actual_implementation_code>
  stochastic = <VARIABLE: true, false>
  thermodynamic = <VARIABLE: true, false>
}

TSUKernel {
  tsu_kernel = <VARIABLE: tsu_ising_native, tsu_maxcut_native>
  sample_shape = <VARIABLE: "batch=32,spins=1024", "batch=64,variables=100">
  thermal_relax = <VARIABLE: "tau = 100ns", "tau = 1us">
  control_voltage = <VARIABLE: "V = 0.8V", "V_range = [-1.2V, +1.2V]">
}
```

## **Optimization Variables**

```
Optimization {
  target = <VARIABLE: tsu_extropic_1, gpu_a100>
  type = <VARIABLE: thermodynamic_fusion, entropy_coding, weight_compression>
  params = <VARIABLE: "merge_blocks=true", "precision=12bit", "ratio=0.95">
}

Fusion {
  fusion = <VARIABLE: gibbs_energy_fusion, anneal_evaluate_fusion>
  pattern = <VARIABLE: "block_gibbs + energy_eval", "sample + validate">
  hardware = <VARIABLE: tsu_extropic_1, gpu_a100>
}
```

## **Framework Variables**

```
Framework {
  framework = <VARIABLE: thrml_jax, pytorch_pgm, pyro>
  language = <VARIABLE: jax, python, cpp>
  paradigm = <VARIABLE: probabilistic, energy_based, gradient_based>
  runtime = <VARIABLE: gpu, tpu, tsu_emulated, cpu>
}

THRML {
  thrml_id = <VARIABLE: thrml_v0_3, thrml_v0_4, thrml_beta>
  version = <VARIABLE: "0.3.1", "0.4.0", "1.0.0-rc1">
  features = <VARIABLE: "block_gibbs,sparse_factors", "qubo_solver,annealing">
  hardware_accel = <VARIABLE: gpu, tpu, tsu_proto>
  tsu_emulation = <VARIABLE: ideal, noisy, limited_entropy>
  jit_compile = <VARIABLE: true, false>
}
```

## **Project & Build Variables**

```
Project {
  project = <VARIABLE: IsingMultiTarget, MaxCutBenchmark, ProteinFoldingSuite>
  domain = <VARIABLE: tsu_kernels, qubo_kernels, pgm_kernels>
  model = <VARIABLE: IsingPGM, MaxCutQUBO, CRF_NER>
}

TargetConfig {
  target_id = <VARIABLE: tsu_production, gpu_emulation, cpu_simulation>
  hardware = <VARIABLE: tsu_extropic_1, gpu_a100, cpu_x86>
  mode = <VARIABLE: production, simulation, emulation, validation>
  priority = <VARIABLE: primary, fallback, validation>
  codegen = <VARIABLE: true, false>
}

BuildRule {
  build_id = <VARIABLE: generate_all, tsu_only, gpu_baseline>
  targets = <VARIABLE: "tsu_prod,gpu_sim", "tsu_only", "all">
  output_dir = <VARIABLE: "./build/ising_multi", "./build/tsu">
  template = <VARIABLE: multi_target, tsu, cuda, jax>
}
```

## **Temperature Schedule Variables** (Special Case)

```
// As you noted, this is particularly important:
temperature_schedule = <VARIABLE: geometric, linear, exponential, adaptive>
T_initial = <VARIABLE: 100.0, 1000.0, 10.0>
T_final = <VARIABLE: 0.01, 0.001, 0.1>
num_steps = <VARIABLE: 1000, 10000, 100>
cooling_rate = <VARIABLE: 0.95, 0.99, 0.90>

// Affects:
// - Convergence speed (too fast → local minima)
// - Solution quality (too slow → computational waste)
// - Energy dissipation (thermodynamic cost)
// - TSU hardware utilization
```

## **Problem-Size Variables** (Critical for Scaling)

```
num_vertices = <VARIABLE: 100, 1000, 10000>
num_spins = <VARIABLE: 1024, 4096, 16384>
num_factors = <VARIABLE: 5000, 50000, 500000>
lattice_size = <VARIABLE: 32x32, 64x64, 128x128>
sequence_length = <VARIABLE: 100, 1000, 10000>
```

These variables capture the entire configuration space of your TSU system, from low-level hardware parameters to high-level algorithm choices!
