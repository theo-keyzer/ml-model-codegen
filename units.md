Based on the schema files provided, here's a markdown document explaining what each file is used for:

# Schema Documentation for Code Generator

## Overview
These files define a comprehensive schema system for a code generator that handles multiple computing paradigms including classical ML, thermodynamic sampling, neuromorphic computing, quantum computing, and more.

## Schema Files

### 1. `act_unit.txt` - **Actor System & Code Generation Control**
**Purpose**: Defines the core code generation engine and control flow mechanisms.

**Key Components**:
- **Actor**: Base component for code generation routines
- **Control Commands**: 
  - `All` - Call actors for each component
  - `Du` - Conditional actor calls
  - `New` - Create new components
  - `Var` - Set variables
  - `Its`/`This` - Process children or collection items
  - `C`/`Cs` - Output code snippets
  - `Out` - Output control (delay, normal, off)
  - `Break` - Break out of loops
  - `Add` - Add to collections
  - `Replace` - String replacement

**Usage**: Controls how code generation templates are executed and how the generator processes component hierarchies.

### 2. `net5_unit.txt` - **Traditional Neural Network Domain**
**Purpose**: Defines components for conventional deep learning and neural network operations.

**Key Areas**:
- **Domain Knowledge**: Kernels, optimization strategies, memory layouts
- **User Specifications**: Models, layers, tensors, configurations
- **Operation Types**:
  - Attention mechanisms (self, cross, multi-head)
  - Graph neural networks
  - Recurrent/stateful operations
  - Differential equations (ODE, SDE, PDE)
  - Memory operations
  - Expert routing (Mixture of Experts)
  - Neural architecture search
  - Continuous depth networks

**Usage**: For generating code for traditional ML models running on GPUs/TPUs.

### 3. `nexus_unit.txt` - **Unified Multi-Paradigm Computing**
**Purpose**: Flat architecture supporting classical, neuromorphic, analog, quantum, photonic, and molecular computing.

**Key Features**:
- **Unified Operation Graph**: No nested hierarchy, direct dependencies
- **Multiple Paradigms**:
  - Classical (conv2d, matmul)
  - Spiking neural networks
  - Analog computing (memristor crossbars)
  - Quantum gates and circuits
  - Photonic computing
  - Molecular/DNA computing
- **Hardware Abstraction**: Supports all computing paradigms
- **Auto-optimization**: Evolutionary and Bayesian optimization
- **Fault Tolerance**: Redundancy strategies

**Usage**: For heterogeneous computing systems that combine multiple computing technologies.

### 4. `tsu.txt` - **Thermodynamic Sampling Unit Domain**
**Purpose**: Specialized domain for probabilistic AI and thermodynamic computing (Extropic hardware).

**Key Components**:
- **TSU Hardware**: Thermodynamic Sampling Unit specifications
- **Probabilistic Models**: Energy-based models, PGMs
- **Sampling Operations**: Gibbs sampling, block sampling
- **Thermodynamic Simulation**: Energy functions, physical constraints
- **Framework Support**: THRML (JAX-based thermodynamic AI library)

**Usage**: For generating code for probabilistic models and sampling algorithms on thermodynamic hardware.

### 5. `tsu-auto.txt` - **Evolutionary Optimization & Auto-Tuning**
**Purpose**: Defines auto-tuning and optimization strategies for the TSU domain.

**Key Components**:
- **Search Spaces**: Parameter spaces for optimization
- **Evolution Strategies**: Genetic algorithms, CMA-ES, Bayesian optimization
- **Fitness Functions**: Multi-objective evaluation
- **Performance Metrics**: Tracking optimization progress
- **Checkpointing**: Resumable optimization runs

**Usage**: For automatically tuning thermodynamic sampling parameters and optimization strategies.

### 6. `tsu-ext.txt` - **TSU Domain Extensions**
**Purpose**: Extends TSU domain with advanced features for complex probabilistic models.

**Extensions**:
- **Multi-state Models**: Potts models, q-ary variables
- **Loopy PGMs**: Belief propagation for cyclic graphs
- **Variational Inference**: Hybrid MCMC/VI approaches
- **Cluster Sampling**: Swendsen-Wang algorithm
- **Advanced TSU Features**: Multi-voltage support, noise models

**Usage**: For complex probabilistic graphical models requiring advanced sampling techniques.

### 7. `unit_unit.txt` - **Schema Definition Language**
**Purpose**: Meta-schema that defines the schema language itself.

**Key Components**:
- **Comp**: Defines component classes and their hierarchy
- **Element**: Defines fields within components with types and constraints
- **Opt**: Defines enumeration options for fields

**Usage**: The foundation that all other schema files are built upon - defines the structure and validation rules for the schema system.

## Relationship Between Files

```
unit_unit.txt (foundation)
    ↓
act_unit.txt (code generation engine)
    ↓
net5_unit.txt (traditional ML) ←→ nexus_unit.txt (multi-paradigm)
    ↓
tsu.txt (thermodynamic computing) ←→ tsu-ext.txt (advanced features)
    ↓
tsu-auto.txt (optimization)
```

This schema system enables a sophisticated code generator that can target everything from traditional neural networks to cutting-edge thermodynamic and quantum computing platforms, with built-in optimization and auto-tuning capabilities.
