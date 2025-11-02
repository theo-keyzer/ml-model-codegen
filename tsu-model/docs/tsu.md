# TSU/JAX Framework Documentation for TSU Experts

## Overview

This document provides a comprehensive guide to the TSU/JAX framework for experts familiar with TSU (Thermodynamic State Unit) hardware but new to this specific modeling framework. The system generates optimized Python/JAX inference code from high-level network descriptions for execution on various target platforms including TSU hardware, GPUs, and CPUs.

## Architecture Components

### 1. Model Definition (.net Files)
The `.net` files define the probabilistic graphical model structure using a hierarchical composition approach:

```
Model (e.g., IsingPGM)
├── Layers (logical processing units)
│   ├── spin_lattice_layer
│   ├── energy_computation
│   └── tsu_sampling_layer
├── Operations (computational primitives)
│   ├── block_gibbs_update
│   ├── energy_eval
│   └── tsu_native_sample
├── Sampling Operations (TSU-specific sampling)
│   ├── gibbs_sweep
│   └── gibbs_direct
├── Energy Functions (energy computation)
│   └── ising_energy
└── Configurations (inference scenarios)
    ├── inference_tsu
    ├── inference_gpu_emulated
    └── inference_cpu_sim
```

### 2. Actor System (.act Files)
Generates platform-specific code from model definitions using pattern-matching actors that traverse the model hierarchy:

```act
Actor generate_config_implementation Config
C def inference_${config:l}(state: jnp.ndarray, key: Any, params: Dict = None) -> jnp.ndarray:
C     # Implementation that executes schedule steps
```

### 3. Generated JAX Code Structure
The framework produces optimized JAX code that can run on multiple backends:

- **TSU Hardware**: Direct execution using native TSU sampling operations
- **GPU Emulation**: JAX-compiled operations for NVIDIA GPUs
- **CPU Simulation**: Pure NumPy/Python execution for verification

## TSU-Specific Features

### Native TSU Sampling Operations

The framework provides specialized sampling operations that map directly to TSU hardware capabilities:

```python
def sampling_op_gibbs_direct(state: jnp.ndarray, key: Any, params: Dict) -> jnp.ndarray:
    # Executes on TSU hardware with 1,000,000 sample budget
    # Utilizes TSU's native thermodynamic sampling capabilities
```

**Key Capabilities:**
- **Massive Parallelism**: 1M+ concurrent spin state evaluations
- **Native Thermodynamic Sampling**: Hardware-accelerated Gibbs sampling
- **Hardware-Accelerated Updates**: Direct memory mapping to TSU registers
- **Configurable Budget**: Sample budget from 10K-1M depending on configuration

### Energy Computation Engine

TSU-optimized energy computation with sparse matrix handling:

```python
def energy_function_ising_energy(state: jnp.ndarray, params: Dict) -> float:
    # Horizontal interactions: JAX-compiled for GPU execution
    horizontal_energy = jnp.sum(state[:, :-1] * state[:, 1:])
    # Vertical interactions: Optimized tensor contractions
    vertical_energy = jnp.sum(state[:-1, :] * state[1:, :])
    energy = -(horizontal_energy + vertical_energy)
    return float(energy)
```

## Execution Platforms

### 1. TSU Native Mode (`tsu_extropic_1`)

Direct hardware execution with:
- **Sample Budget**: 1,000,000 samples per batch
- **Batch Size**: 32 concurrent configurations
- **Hardware Acceleration**: Direct TSU instruction mapping
- **Use Case**: Production inference with maximum throughput

```python
# Direct TSU sampling operations
state = sampling_op_gibbs_direct(state, key, params)
```

### 2. GPU Emulation Mode (`gpu_a100`)

JAX-compiled execution for NVIDIA GPUs:
- **Sample Budget**: 100,000 samples (10% of native TSU)
- **Batch Size**: 32 concurrent configurations
- **JIT Compilation**: Automatic GPU kernel generation
- **Use Case**: Development and TSU verification

```python
# GPU-emulated sampling 
state = sampling_op_gibbs_sweep(state, key, params)  # Compiled for GPU
```

### 3. CPU Simulation Mode (`cpu_x86`)

Pure Python execution for verification:
- **Sample Budget**: 10,000 samples (1% of native TSU)
- **Batch Size**: 4 concurrent configurations
- **Pure JAX**: No hardware dependencies
- **Use Case**: Debugging and algorithm development

```python
# CPU simulation with reduced parameters
params.setdefault('batch', 4)
params.setdefault('sample_budget', 10000)
```

## Tensor Management System

### Static Tensor Declarations

All tensors are pre-allocated with specific properties:

```
# spin_state: 1, 1024 dense binary static
# j_matrix: 1024, 1024 sparse_csr fp32 static
# h_vector: 1024 dense fp32 static
```

**Properties:**
- **Shape**: Tensor dimensions (e.g., 1024 spin states)
- **Layout**: Dense vs. sparse storage formats
- **Datatype**: Binary, fp32 for different memory efficiency
- **Static**: Compile-time allocation for hardware optimization

### Memory Mapping to TSU Hardware

The tensor system directly maps to TSU memory architecture:

| Tensor | TSU Memory Map | Size | Usage |
|--------|---------------|------|-------|
| `spin_state` | Register file | 1KB | Current spin configuration |
| `j_matrix` | Sparse memory | 4MB | Interaction weights |
| `h_vector` | Vector memory | 4KB | External magnetic field |
| `tsu_samples` | Sample buffer | 128KB | Batch sampling results |

## Scheduling and Execution Flow

### Multi-Platform Scheduling

Each configuration defines platform-specific execution schedules:

```
Step 1: Direct TSU sampling (native hardware)
  └── tsu_sampling_layer
      ├── op_tsu_sampling_layer_tsu_native_sample
      └── sampling_op_gibbs_direct

Step 2: Compute energy on GPU for validation
  └── energy_computation
      ├── op_energy_computation_energy_eval
      └── energy_function_ising_energy
```

### Hardware-Aware Optimization

The scheduler automatically optimizes for target platforms:
- **TSU**: Prioritizes sampling operations and minimizes host communication
- **GPU**: Batches operations for maximum occupancy
- **CPU**: Sequential execution with debugging hooks

## Advanced Features for TSU Experts

### Thermodynamic Parameter Control

Runtime control of thermodynamic properties:

```python
params.setdefault('temperature', 1.0)  # Beta parameter mapping
params.setdefault('sample_budget', 1000000)  # TSU cycle budget
```

### Hardware Register Interface

Direct mapping to TSU hardware registers through JAX primitives:

```python
# TSU register update mapping
flat_state = flat_state.at[indices].set(updates)  # Direct register write
```

### Performance Calibration

Platform-specific performance characteristics:

| Platform | Sample Rate | Energy Computation | Memory Bandwidth |
|----------|-------------|-------------------|------------------|
| TSU Native | 1M samples/ms | Hardware accelerated | 512 GB/s |
| GPU A100 | 100K samples/ms | Tensor cores | 1.5 TB/s |
| CPU x86 | 10K samples/ms | Scalar execution | 50 GB/s |

## Development Workflow

### 1. Model Design
Create `.net` files defining the probabilistic model structure

### 2. Actor Script Development
Write `.act` files to generate platform-specific implementations

### 3. Code Generation
Run the actor system to produce JAX code for all target platforms

### 4. Platform Testing
Verify correctness across CPU → GPU → TSU hardware

### 5. Production Deployment
Deploy native TSU code for maximum performance

## Common Patterns and Best Practices

### Efficient TSU Utilization

```python
# Maximize TSU throughput with large batch sizes
batch_size = params.get('batch', 32)  # Optimal for TSU pipeline

# Leverage TSU's native sampling for 1M+ samples
sample_budget = params.get('sample_budget', 1000000)
```

### Cross-Platform Compatibility

```python
# Parameter sets that work across all platforms
params.setdefault('batch', 32)      # Works on TSU/GPU
params.setdefault('temperature', 1.0)  # Universal beta mapping
```

### Hardware-Specific Optimizations

```python
# TSU-native operations
state = sampling_op_gibbs_direct(state, key, params)  # Hardware accelerated

# GPU-compiled operations  
state = sampling_op_gibbs_sweep(state, key, params)   # JAX JIT compilation
```

## Debugging and Verification

### Multi-Level Verification
1. **CPU Simulation**: Algorithmic correctness
2. **GPU Emulation**: Performance modeling 
3. **TSU Hardware**: Production validation

### Performance Monitoring
```python
# Energy tracking for convergence monitoring
energy = energy_function_ising_energy(state, params)
print(f"Energy at step: {energy}")
```

This framework provides a complete ecosystem for developing, testing, and deploying TSU-accelerated probabilistic models while maintaining compatibility with traditional computing platforms for development and verification.
