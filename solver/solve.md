# Nuclear PBit Optimizer Suite Documentation

## Overview

The Nuclear PBit Optimizer Suite is a JAX-based optimization framework that combines probabilistic computing with advanced escape mechanisms for constrained and unconstrained optimization problems.

## Module Overview

### 1. `pb.py` - Path-Based PBit Core
**Core probabilistic optimizer with momentum and native jump support**

#### Purpose
- Main optimization engine using probabilistic bits (PBits)
- Implements momentum-based gradient descent with physical noise modeling
- Includes native quantum jump capabilities for escaping local minima

#### Key Features
- Physical PBit differential pair modeling
- Linear momentum with avoidance mechanisms
- Shared memory support for multi-instance optimization
- Built-in quantum jumps without external wrappers

#### Configuration (`PBitConfig`)
```python
@dataclass
class PBitConfig:
    momentum_beta: float = 0.9                    # Momentum coefficient
    momentum_decay_on_stuck: float = 0.1          # Momentum decay when near stuck points
    avoidance_threshold: float = 0.3              # Distance threshold for avoidance
    learning_rate: float = 0.002                  # Base learning rate
    noise_scale: float = 0.12                     # Noise intensity
    clip_params: Tuple[float, float] = (-2.5, 2.5) # Parameter clipping
    clip_velocity: Tuple[float, float] = (-0.25, 0.25) # Velocity clipping
    clip_delta: Tuple[float, float] = (-0.4, 0.4) # Step size clipping
    
    # Jump integration
    enable_quantum_jumps: bool = False            # Enable native jumps
    jump_consecutive_stuck_threshold: int = 50    # Stuck steps before jump
    post_jump_momentum_factor: float = 0.0        # Momentum preservation after jumps
    
    # Physical constants
    alpha: float = 2.0
    beta: float = 1.0
    gamma: float = 0.12
    i_tail: float = 5e-5
    seed: int = 42
```

#### Basic Usage
```python
from pb import PathBasedPBit, PBitConfig

# Create optimizer
config = PBitConfig(learning_rate=0.01, enable_quantum_jumps=True)
pbit = PathBasedPBit(problem_dim=10, config=config)

# Optimization loop
for step in range(1000):
    result = pbit.step(gradient_fn, objective_fn)
    if result['best_objective'] < target:
        break
```

---

### 2. `mr.py` - SAR Memory Manager
**Stuck-point Avoidance and Reset memory system**

#### Purpose
- Tracks stuck points in optimization landscape
- Manages reset strategies for escaping local minima
- Provides quantum and nuclear jump mechanisms

#### Key Features
- FIFO stuck-point memory
- Multiple reset strategies
- Strategy effectiveness tracking
- Quantum and nuclear jump support

#### Configuration (`SARConfig`)
```python
@dataclass
class SARConfig:
    # Basic memory configuration
    spf_depth: int = 25                           # Stuck-point FIFO size
    avoidance_threshold: float = 0.3              # Distance threshold
    avoidance_strength: float = 0.6               # Push strength from stuck points
    effectiveness_decay: float = 0.99             # Strategy effectiveness decay
    seed: int = 42
    
    # Jump configuration
    enable_jumps: bool = False                    # Enable quantum/nuclear jumps
    quantum_jump_range: float = 5.0               # Range for quantum jumps
    nuclear_reset_strength: float = 2.0           # Noise scale for nuclear jumps
    min_jump_distance: float = 1.0                # Minimum jump displacement
    jump_severity_threshold: float = 0.7          # Severity threshold for jumps
    
    # Strategy weights (0-6: PERTURB_BEST, BEST_PARAMS, RANDOM_RESTART, 
    # GRADIENT_ESCAPE, AVOIDANCE_RESTART, QUANTUM_JUMP, NUCLEAR_JUMP)
    strategy_weights: Optional[Dict[str, float]] = None
```

#### Reset Strategies
1. **PERTURB_BEST** - Small perturbation around best solution
2. **BEST_PARAMS** - Return to best parameters with noise
3. **RANDOM_RESTART** - Complete random restart
4. **GRADIENT_ESCAPE** - Move opposite to gradient
5. **AVOIDANCE_RESTART** - Move away from nearest stuck point
6. **QUANTUM_JUMP** - Random uniform or opposite vector jump
7. **NUCLEAR_JUMP** - To best + huge noise or full random

#### Basic Usage
```python
from mr import SARMemoryManager, SARConfig

# Create memory manager
config = SARConfig(spf_depth=20, enable_jumps=True)
memory = SARMemoryManager(problem_dim=10, config=config)

# Use with PBit
pbit = PathBasedPBit(problem_dim=10, memory_manager=memory)
```

---

### 3. `w.py` - Nuclear PBit Wrapper
**Lightweight wrapper for nuclear jump triggers**

#### Purpose
- Zero-overhead wrapper for nuclear escape mechanisms
- Triggers nuclear jumps based on stuck detection
- Manual nuclear jump capability

#### Key Features
- Minimal performance overhead
- Automatic stuck detection
- Manual nuclear jump API
- Compatible with existing PBit infrastructure

#### Configuration (`NuclearWrapperConfig`)
```python
@dataclass
class NuclearWrapperConfig:
    enable: bool = True                    # Enable nuclear jumps
    stuck_threshold: int = 50              # Steps without improvement to trigger
    manual_nuke_strength: float = 1.0      # Strength for manual jumps
    verbose: bool = False                  # Output verbosity
```

#### Basic Usage
```python
from w import NuclearPBitWrapper, NuclearWrapperConfig

# Wrap existing PBit
wrapper_config = NuclearWrapperConfig(stuck_threshold=30)
wrapper = NuclearPBitWrapper(pbit, wrapper_config)

# Use wrapper instead of raw PBit
result = wrapper.step(gradient_fn, objective_fn)

# Manual nuclear jump
wrapper.nuke(gradient_fn)
```

---

### 4. `c.py` - Constraint Handling
**Advanced constraint handling for optimization problems**

#### Purpose
- Handle various constraint types in optimization
- Integrate constraints with nuclear PBit system
- Support for mixed-integer and categorical variables

#### Key Features
- Multiple constraint types (box, linear, nonlinear, integer)
- Various handling methods (penalty, barrier, projection, repair)
- Adaptive penalty coefficients
- Feasibility tracking

#### Constraint Types
```python
class ConstraintType(Enum):
    BOX = "box"                    # lb <= x <= ub
    LINEAR_EQUALITY = "linear_eq"  # Ax = b
    LINEAR_INEQUALITY = "linear_ineq"  # Ax <= b
    NONLINEAR_EQUALITY = "nonlinear_eq"  # g(x) = 0
    NONLINEAR_INEQUALITY = "nonlinear_ineq"  # g(x) <= 0
    INTEGER = "integer"            # x must be integer
    CATEGORICAL = "categorical"    # x in {cat1, cat2, ...}
```

#### Handling Methods
```python
class ConstraintHandlingMethod(Enum):
    PENALTY = "penalty"            # Add penalty to objective
    BARRIER = "barrier"            # Interior point barrier
    PROJECTION = "projection"      # Project onto feasible set
    REPAIR = "repair"              # Repair infeasible solutions
    HYBRID = "hybrid"              # Adaptive combination
```

#### Configuration (`ConstrainedPBitConfig`)
```python
@dataclass
class ConstrainedPBitConfig:
    # Problem definition
    problem_dim: int
    constraints: List[Constraint]
    
    # PBit/SAR configuration
    max_steps: int = 5000
    reset_patience: int = 50
    initial_params: Optional[jnp.ndarray] = None
    
    # SAR configuration
    sar_spf_depth: int = 20
    sar_avoidance_threshold: float = 0.8
    sar_avoidance_strength: float = 1.5
    
    # PBit configuration  
    pbit_momentum_decay_on_stuck: float = 0.3
    pbit_learning_rate: float = 0.01
    pbit_noise_scale: float = 0.2
    
    # Nuclear wrapper configuration
    enable_nuclear: bool = True
    nuclear_stuck_threshold: int = 50
    nuclear_manual_strength: float = 1.0
    
    # Quantum jumps
    enable_quantum_jumps: bool = True
    jump_consecutive_stuck_threshold: int = 50
    
    # Constraint handling
    handling_method: ConstraintHandlingMethod = ConstraintHandlingMethod.HYBRID
    penalty_coefficient: float = 100.0
    penalty_growth_rate: float = 1.5
    barrier_coefficient: float = 0.1
    barrier_decay_rate: float = 0.95
    repair_max_iterations: int = 20
    projection_step_size: float = 0.1
    constraint_tolerance: float = 1e-6
    use_adaptive_penalties: bool = True
    integer_rounding_threshold: float = 0.3
    feasibility_emphasis: float = 0.8
    
    seed: int = 42
```

#### Basic Usage
```python
from c import ConstrainedNuclearPBitWrapper, ConstrainedPBitConfig, Constraint, ConstraintType

# Define constraints
box_constraint = Constraint(
    name="bounds",
    constraint_type=ConstraintType.BOX,
    bounds=(-2.0, 2.0)
)

# Create constrained optimizer
config = ConstrainedPBitConfig(
    problem_dim=2,
    constraints=[box_constraint],
    handling_method=ConstraintHandlingMethod.HYBRID
)
optimizer = ConstrainedNuclearPBitWrapper(config)

# Run constrained optimization
results = optimizer.optimize(objective_fn, gradient_fn)
```

---

### 5. `i.py` - Fully Integrated Optimizer
**Complete integration of all components**

#### Purpose
- Combine all advanced features into single optimizer
- Provide high-level API for complex optimization problems
- Advanced visualization and analysis tools

#### Key Features
- Nuclear PBit with quantum tunneling
- SAR memory with constraint awareness
- Adaptive constraint handling
- Multi-start with constraint-aware initialization
- Comprehensive visualization

#### Configuration
Uses `ConstrainedPBitConfig` from `c.py` with all integrated features.

#### Basic Usage
```python
from i import FullyIntegratedConstrainedOptimizer

# Create integrated optimizer
optimizer = FullyIntegratedConstrainedOptimizer(config)

# Run comprehensive optimization
results = optimizer.optimize(
    objective_fn,
    gradient_fn,
    max_steps=5000,
    enable_quantum_tunneling=True,
    verbose=True
)

# Visualize results
optimizer.visualize_constrained_optimization(objective_fn)
```

---

## Typical Workflow

### 1. Simple Unconstrained Problems
```python
from pb import PathBasedPBit, PBitConfig

config = PBitConfig(enable_quantum_jumps=True)
pbit = PathBasedPBit(problem_dim=10, config=config)

for step in range(1000):
    result = pbit.step(gradient_fn, objective_fn)
```

### 2. Complex Constrained Problems
```python
from i import FullyIntegratedConstrainedOptimizer
from c import ConstrainedPBitConfig, Constraint, ConstraintType

# Define constraints
constraints = [
    Constraint("bounds", ConstraintType.BOX, bounds=(-5, 5)),
    # ... more constraints
]

config = ConstrainedPBitConfig(
    problem_dim=10,
    constraints=constraints,
    enable_nuclear=True,
    enable_quantum_jumps=True
)

optimizer = FullyIntegratedConstrainedOptimizer(config)
results = optimizer.optimize(objective_fn, gradient_fn)
```

### 3. Custom Strategy Development
```python
from mr import SARMemoryManager, SARConfig

def custom_strategy(curr, best, grad, strength, key):
    # Custom reset logic
    return new_params, new_key

memory = SARMemoryManager(problem_dim=10, config=SARConfig())
memory.add_custom_strategy(0, custom_strategy)  # Override PERTURB_BEST
```

## Performance Tips

1. **Use JAX JIT compilation** for objective and gradient functions
2. **Enable jumps** for problems with many local minima
3. **Adjust reset_patience** based on problem complexity
4. **Use appropriate constraint handling** method for your problem type
5. **Monitor convergence** with the provided history tracking

## Common Configurations

### For Smooth Convex Problems
```python
config = PBitConfig(
    learning_rate=0.01,
    enable_quantum_jumps=False,  # No jumps needed
    momentum_beta=0.9
)
```

### For Rugged Landscapes
```python
config = ConstrainedPBitConfig(
    enable_quantum_jumps=True,
    enable_nuclear=True,
    jump_consecutive_stuck_threshold=30,
    reset_patience=100
)
```

### For Tight Constraints
```python
config = ConstrainedPBitConfig(
    handling_method=ConstraintHandlingMethod.PROJECTION,
    constraint_tolerance=1e-8,
    feasibility_emphasis=0.9
)
```

This suite provides a comprehensive optimization framework suitable for everything from simple convex problems to complex constrained optimization with multiple local minima.
