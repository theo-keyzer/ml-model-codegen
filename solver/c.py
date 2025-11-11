# c.py: Constraint Handling for Nuclear PBit Optimizer
# Updated for w.py wrapper and pb/mr v2.1
# Supports: Box constraints, linear constraints, nonlinear constraints, mixed-integer variables
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import sys

try:
    from pb import PathBasedPBit, PBitConfig
    from mr import SARMemoryManager, SARConfig
    from w import NuclearPBitWrapper, NuclearWrapperConfig
except ImportError:
    print("Warning: pb.py, mr.py, or w.py not found. Using standalone mode.")

class ConstraintType(Enum):
    """Types of constraints supported."""
    BOX = "box"  # Simple bounds: lb <= x <= ub
    LINEAR_EQUALITY = "linear_eq"  # Ax = b
    LINEAR_INEQUALITY = "linear_ineq"  # Ax <= b
    NONLINEAR_EQUALITY = "nonlinear_eq"  # g(x) = 0
    NONLINEAR_INEQUALITY = "nonlinear_ineq"  # g(x) <= 0
    INTEGER = "integer"  # x must be integer
    CATEGORICAL = "categorical"  # x in {cat1, cat2, ...}

class ConstraintHandlingMethod(Enum):
    """Methods for handling constraint violations."""
    PENALTY = "penalty"  # Add penalty to objective
    BARRIER = "barrier"  # Interior point barrier
    PROJECTION = "projection"  # Project onto feasible set
    REPAIR = "repair"  # Repair infeasible solutions
    HYBRID = "hybrid"  # Adaptive combination

@dataclass
class Constraint:
    """Generic constraint specification."""
    name: str
    constraint_type: ConstraintType
    function: Optional[Callable] = None  # For nonlinear constraints
    matrix: Optional[jnp.ndarray] = None  # For linear constraints (A)
    vector: Optional[jnp.ndarray] = None  # For linear constraints (b)
    bounds: Optional[Tuple[float, float]] = None  # For box constraints
    indices: Optional[List[int]] = None  # Variable indices affected
    tolerance: float = 1e-6  # Equality tolerance
    
    def __post_init__(self):
        """Validate constraint specification."""
        if self.constraint_type in [ConstraintType.LINEAR_EQUALITY, ConstraintType.LINEAR_INEQUALITY]:
            if self.matrix is None or self.vector is None:
                raise ValueError(f"Linear constraints require matrix and vector")
        elif self.constraint_type in [ConstraintType.NONLINEAR_EQUALITY, ConstraintType.NONLINEAR_INEQUALITY]:
            if self.function is None:
                raise ValueError(f"Nonlinear constraints require function")
        elif self.constraint_type == ConstraintType.BOX:
            if self.bounds is None:
                raise ValueError(f"Box constraints require bounds")

@dataclass
class ConstraintViolation:
    """Information about constraint violations."""
    total_violation: Union[float, jnp.ndarray]
    individual_violations: Dict[str, float]
    feasible: Optional[bool] = None
    violation_gradient: Optional[jnp.ndarray] = None

@dataclass
class ConstrainedPBitConfig:
    """Configuration for constrained optimization (updated for w.py)."""
    problem_dim: int
    constraints: List[Constraint]
    
    # PBit/SAR config
    max_steps: int = 5000
    reset_patience: int = 50
    initial_params: Optional[jnp.ndarray] = None
    
    # SAR config
    sar_spf_depth: int = 20
    sar_avoidance_threshold: float = 0.8
    sar_avoidance_strength: float = 1.5
    
    # PBit config
    pbit_momentum_decay_on_stuck: float = 0.3
    pbit_learning_rate: float = 0.01
    pbit_noise_scale: float = 0.2
    
    # Nuclear wrapper config (w.py)
    enable_nuclear: bool = True
    nuclear_stuck_threshold: int = 50
    nuclear_manual_strength: float = 1.0
    
    # Quantum jumps (pb.py native)
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

class ConstraintHandler:
    """
    Handles constraint evaluation, violation measurement, and repair mechanisms.
    Integrates with Nuclear PBit's escape mechanisms for constraint-aware optimization.
    """
    
    def __init__(self, config: ConstrainedPBitConfig):
        self.config = config
        self.constraints = config.constraints
        self.penalty_coeff = config.penalty_coefficient
        self.barrier_coeff = config.barrier_coefficient
        self.violation_history = []
        self.penalty_history = []
        self.best_feasible_params = None
        
        # Compile constraint functions
        self._compile_constraint_functions()
        
        print(f"🔒 ConstraintHandler initialized:")
        print(f"   Constraints: {len(self.constraints)}")
        print(f"   Method: {config.handling_method.value}")
        print(f"   Penalty coefficient: {config.penalty_coefficient}")
    
    def _compile_constraint_functions(self):
        """Compile JAX functions for efficient constraint evaluation."""
        
        # Box constraints
        box_constraints = [c for c in self.constraints if c.constraint_type == ConstraintType.BOX]
        if box_constraints:
            @jax.jit
            def evaluate_box_violation(params):
                total = 0.0
                for c in box_constraints:
                    if c.indices:
                        values = params[jnp.array(c.indices)]
                    else:
                        values = params
                    lb, ub = c.bounds
                    lower_violation = jnp.maximum(0.0, lb - values)
                    upper_violation = jnp.maximum(0.0, values - ub)
                    total += jnp.sum(lower_violation**2 + upper_violation**2)
                return total
            self._box_violation = evaluate_box_violation
        else:
            self._box_violation = lambda p: 0.0
        
        # Linear constraints
        linear_eq = [c for c in self.constraints if c.constraint_type == ConstraintType.LINEAR_EQUALITY]
        linear_ineq = [c for c in self.constraints if c.constraint_type == ConstraintType.LINEAR_INEQUALITY]
        
        if linear_eq:
            matrices = [c.matrix for c in linear_eq]
            vectors = [c.vector for c in linear_eq]
            @jax.jit
            def evaluate_linear_eq_violation(params):
                total = 0.0
                for A, b in zip(matrices, vectors):
                    residual = A @ params - b
                    total += jnp.sum(residual**2)
                return total
            self._linear_eq_violation = evaluate_linear_eq_violation
        else:
            self._linear_eq_violation = lambda p: 0.0
        
        if linear_ineq:
            matrices = [c.matrix for c in linear_ineq]
            vectors = [c.vector for c in linear_ineq]
            @jax.jit
            def evaluate_linear_ineq_violation(params):
                total = 0.0
                for A, b in zip(matrices, vectors):
                    violation = jnp.maximum(0.0, A @ params - b)
                    total += jnp.sum(violation**2)
                return total
            self._linear_ineq_violation = evaluate_linear_ineq_violation
        else:
            self._linear_ineq_violation = lambda p: 0.0
    
    def evaluate_constraints(self, params: jnp.ndarray) -> ConstraintViolation:
        """
        Evaluate all constraints and return violation information.
        
        Args:
            params: Current parameter vector
            
        Returns:
            ConstraintViolation object with detailed violation info
        """
        # Box constraints
        box_viol = self._box_violation(params)
        
        # Linear constraints
        lin_eq_viol = self._linear_eq_violation(params)
        lin_ineq_viol = self._linear_ineq_violation(params)
        
        total_viol = box_viol + lin_eq_viol + lin_ineq_viol
        
        # Nonlinear constraints
        for c in self.constraints:
            if c.constraint_type in [ConstraintType.NONLINEAR_EQUALITY, ConstraintType.NONLINEAR_INEQUALITY]:
                g_val = c.function(params)
                if c.constraint_type == ConstraintType.NONLINEAR_EQUALITY:
                    viol = jnp.sum(g_val**2)
                else:  # inequality
                    viol = jnp.sum(jnp.maximum(0.0, g_val)**2)
                total_viol += viol
        
        feasible = total_viol <= self.config.constraint_tolerance
        
        return ConstraintViolation(
            total_violation=total_viol,
            individual_violations={},
            feasible=feasible
        )
    
    def compute_penalty(self, params: jnp.ndarray, objective_value: float) -> float:
        """
        Compute penalty-based augmented objective.
        
        Args:
            params: Current parameters
            objective_value: Original objective value
            
        Returns:
            Augmented objective with penalty terms
        """
        violation_info = self.evaluate_constraints(params)
        total_viol = violation_info.total_violation
        
        if violation_info.feasible:
            return objective_value
        
        penalty = self.penalty_coeff * total_viol
        alpha = self.config.feasibility_emphasis
        aug = (1 - alpha) * objective_value + alpha * penalty
        return aug
    
    def compute_barrier(self, params: jnp.ndarray, objective_value: float) -> float:
        """
        Compute barrier function (interior point method).
        Prevents optimizer from leaving feasible region.
        """
        barrier_value = 0.0
        
        # Box constraint barriers
        for c in self.constraints:
            if c.constraint_type == ConstraintType.BOX:
                lb, ub = c.bounds
                if c.indices:
                    values = params[jnp.array(c.indices)]
                else:
                    values = params
                
                # Log barrier: -log(x - lb) - log(ub - x)
                lower_barrier = -jnp.sum(jnp.log(jnp.maximum(values - lb, 1e-10)))
                upper_barrier = -jnp.sum(jnp.log(jnp.maximum(ub - values, 1e-10)))
                barrier_value += float(lower_barrier + upper_barrier)
        
        # Inequality barriers
        for c in self.constraints:
            if c.constraint_type == ConstraintType.NONLINEAR_INEQUALITY:
                g_val = c.function(params)
                barrier_value += float(-jnp.sum(jnp.log(jnp.maximum(-g_val, 1e-10))))
        
        augmented = objective_value + self.barrier_coeff * barrier_value
        
        return augmented
    
    def project_to_feasible(self, params: jnp.ndarray, max_iter: int = None) -> jnp.ndarray:
        """
        Project infeasible point onto feasible set.
        Uses gradient-based projection for general constraints.
        """
        max_iter = max_iter or self.config.repair_max_iterations
        current = params.copy()
        tol = self.config.constraint_tolerance
        
        for iteration in range(max_iter):
            violation_info = self.evaluate_constraints(current)
            total = violation_info.total_violation
            
            if total <= tol:
                break
            
            # Project box constraints (hard clipping)
            for c in self.constraints:
                if c.constraint_type == ConstraintType.BOX:
                    lb, ub = c.bounds
                    if c.indices:
                        current = current.at[c.indices].set(
                            jnp.clip(current[c.indices], lb, ub)
                        )
                    else:
                        current = jnp.clip(current, lb, ub)
            
            # Project linear equality constraints (orthogonal projection)
            for c in self.constraints:
                if c.constraint_type == ConstraintType.LINEAR_EQUALITY:
                    A, b = c.matrix, c.vector
                    residual = A @ current - b
                    # Projection: x - A^T(AA^T)^{-1}(Ax - b)
                    try:
                        correction = A.T @ jnp.linalg.solve(A @ A.T, residual)
                        current = current - correction
                    except:
                        # Singular matrix - use pseudo-inverse
                        correction = A.T @ jnp.linalg.lstsq(A @ A.T, residual)[0]
                        current = current - correction
            
            # Gradient-based projection for nonlinear constraints
            for c in self.constraints:
                if c.constraint_type in [ConstraintType.NONLINEAR_EQUALITY, ConstraintType.NONLINEAR_INEQUALITY]:
                    def violation_fn(x):
                        g = c.function(x)
                        return jnp.sum(g**2)
                    g_grad = jax.grad(violation_fn)(current)
                    
                    # Move in direction to reduce violation
                    step_size = self.config.projection_step_size
                    current = current - step_size * g_grad
        
        return current
    
    def repair_solution(self, params: jnp.ndarray, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Repair infeasible solution using problem-specific heuristics.
        Falls back to projection if no specific repair strategy available.
        """
        # First try projection
        repaired = self.project_to_feasible(params)
        
        violation_info = self.evaluate_constraints(repaired)
        total = violation_info.total_violation
        tol = self.config.constraint_tolerance
        if total <= tol:
            return repaired, key
        
        # If still infeasible, try random perturbations
        key, subkey = jax.random.split(key)
        
        # Find feasible starting point via random search
        for _ in range(10):
            key, subkey = jax.random.split(key)
            
            # Sample near current best feasible point if available
            if self.best_feasible_params is not None:
                candidate = self.best_feasible_params + jax.random.normal(subkey, params.shape) * 0.5
            else:
                # Random sample in domain
                candidate = jax.random.uniform(subkey, params.shape, minval=-3.0, maxval=3.0)
            
            candidate = self.project_to_feasible(candidate)
            candidate_violation = self.evaluate_constraints(candidate)
            c_total = candidate_violation.total_violation
            
            if c_total <= tol:
                repaired = candidate
                break
        
        return repaired, key
    
    def handle_constraint_violation(self, 
                                   params: jnp.ndarray, 
                                   objective_value: float,
                                   key: jnp.ndarray) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
        """
        Main constraint handling dispatch based on configured method.
        
        Returns:
            (augmented_objective, modified_params, new_key)
        """
        method = self.config.handling_method
        
        if method == ConstraintHandlingMethod.PENALTY:
            aug_obj = self.compute_penalty(params, objective_value)
            mod_params = params
            
        elif method == ConstraintHandlingMethod.BARRIER:
            aug_obj = self.compute_barrier(params, objective_value)
            mod_params = params
            
        elif method == ConstraintHandlingMethod.HYBRID:
            aug_obj = self.compute_penalty(params, objective_value)
            mod_params = params
            
        elif method == ConstraintHandlingMethod.PROJECTION:
            aug_obj = objective_value
            mod_params = params
            
        elif method == ConstraintHandlingMethod.REPAIR:
            violation_info = self.evaluate_constraints(params)
            total = violation_info.total_violation
            if total > self.config.constraint_tolerance:
                mod_params, key = self.repair_solution(params, key)
            else:
                mod_params = params
            aug_obj = objective_value
            
        else:
            aug_obj = objective_value
            mod_params = params
        
        return aug_obj, mod_params, key

class ConstrainedNuclearPBitWrapper:
    """
    Nuclear PBit wrapper (w.py) with constraint handling.
    Integrates constraint-aware escapes and feasibility-driven optimization.
    """
    
    def __init__(self, config: ConstrainedPBitConfig):
        self.config = config
        self.constraint_handler = ConstraintHandler(config)
        self.problem_dim = config.problem_dim
        
        # Create SAR memory manager
        sar_config = SARConfig(
            spf_depth=config.sar_spf_depth,
            avoidance_threshold=config.sar_avoidance_threshold,
            avoidance_strength=config.sar_avoidance_strength,
            enable_jumps=config.enable_quantum_jumps,
            seed=config.seed
        )
        memory_manager = SARMemoryManager(config.problem_dim, sar_config)
        
        # Create PBit
        pbit_config = PBitConfig(
            momentum_decay_on_stuck=config.pbit_momentum_decay_on_stuck,
            learning_rate=config.pbit_learning_rate,
            noise_scale=config.pbit_noise_scale,
            enable_quantum_jumps=config.enable_quantum_jumps,
            jump_consecutive_stuck_threshold=config.jump_consecutive_stuck_threshold,
            seed=config.seed
        )
        
        # Initialize at feasible point if possible
        initial_params = config.initial_params
        if initial_params is not None:
            initial_params = self.constraint_handler.project_to_feasible(initial_params)
        
        pbit = PathBasedPBit(
            problem_dim=config.problem_dim,
            config=pbit_config,
            initial_params=initial_params,
            memory_manager=memory_manager
        )
        
        # Wrap with Nuclear wrapper (w.py)
        wrapper_config = NuclearWrapperConfig(
            enable=config.enable_nuclear,
            stuck_threshold=config.nuclear_stuck_threshold,
            manual_nuke_strength=config.nuclear_manual_strength,
            verbose=False
        )
        self.wrapper = NuclearPBitWrapper(pbit, wrapper_config)
        
        # Tracking
        self.feasible_solutions = []
        self.best_feasible_objective = float('inf')
        self.best_feasible_params = None
        self.infeasibility_history = []
        
        print(f"🔒 ConstrainedNuclearPBitWrapper initialized (using w.py)")
        print(f"   Problem dimension: {self.problem_dim}")
        print(f"   Constraints: {len(config.constraints)}")
    
    def _constrained_objective(self, params: jnp.ndarray, original_obj_fn: Callable) -> float:
        """Wrap objective function with constraint handling."""
        original_value = original_obj_fn(params)
        
        method = self.config.handling_method
        if method in [ConstraintHandlingMethod.PENALTY, 
                      ConstraintHandlingMethod.BARRIER, 
                      ConstraintHandlingMethod.HYBRID]:
            key = jax.random.PRNGKey(0)
            aug_obj, _, _ = self.constraint_handler.handle_constraint_violation(
                params, original_value, key
            )
            return aug_obj
        else:
            return original_value
    
    def optimize(self, 
                objective_fn: Callable,
                gradient_fn: Callable,
                max_steps: int = 5000,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Run constrained optimization using w.py wrapper.
        
        Args:
            objective_fn: Original objective function (will be augmented)
            gradient_fn: Gradient of original objective
            max_steps: Maximum optimization steps
            verbose: Print progress
            
        Returns:
            Dictionary with optimization results
        """
        if verbose:
            print(f"\n🔒 CONSTRAINED NUCLEAR OPTIMIZATION (w.py)")
            print("=" * 70)
            print(f"   Method: {self.config.handling_method.value}")
            print(f"   Constraints: {len(self.config.constraints)}")
        
        # Create augmented objective and gradient
        def augmented_objective(params):
            return self._constrained_objective(params, objective_fn)
        
        augmented_gradient = jax.grad(augmented_objective)
        
        # Track initial feasibility
        current_params = self.wrapper.pbit.state.params
        violation_info = self.constraint_handler.evaluate_constraints(current_params)
        feasible = violation_info.feasible
        
        if feasible:
            current_obj = objective_fn(current_params)
            self.feasible_solutions.append((current_obj, current_params.copy()))
            self.best_feasible_objective = current_obj
            self.best_feasible_params = current_params.copy()
            self.constraint_handler.best_feasible_params = current_params.copy()
        
        self.infeasibility_history.append(violation_info.total_violation)
        
        for step in range(max_steps):
            # Take step with wrapper
            result = self.wrapper.step(augmented_gradient, augmented_objective, 
                                      reset_patience=self.config.reset_patience)
            
            current_params = result['params']
            
            # Handle constraints based on method
            method = self.config.handling_method
            if method == ConstraintHandlingMethod.PROJECTION:
                current_params = self.constraint_handler.project_to_feasible(current_params)
                self.wrapper.pbit.state = self.wrapper.pbit.state.replace(params=current_params)
            elif method == ConstraintHandlingMethod.REPAIR:
                violation_info = self.constraint_handler.evaluate_constraints(current_params)
                if violation_info.total_violation > self.config.constraint_tolerance:
                    key = jax.random.PRNGKey(step)
                    current_params, _ = self.constraint_handler.repair_solution(current_params, key)
                    self.wrapper.pbit.state = self.wrapper.pbit.state.replace(params=current_params)
            
            # Evaluate constraints and original objective
            violation_info = self.constraint_handler.evaluate_constraints(current_params)
            current_obj = objective_fn(current_params)
            feasible = violation_info.feasible
            
            # Track feasible solutions
            if feasible:
                self.feasible_solutions.append((current_obj, current_params.copy()))
                if current_obj < self.best_feasible_objective:
                    self.best_feasible_objective = current_obj
                    self.best_feasible_params = current_params.copy()
                    self.constraint_handler.best_feasible_params = current_params.copy()
            
            self.infeasibility_history.append(violation_info.total_violation)
            self.constraint_handler.violation_history.append(violation_info.total_violation)
            
            # Adaptive penalty updates
            if self.config.use_adaptive_penalties and len(self.constraint_handler.violation_history) > 10:
                recent_violations = self.constraint_handler.violation_history[-10:]
                if all(v > self.config.constraint_tolerance for v in recent_violations):
                    self.constraint_handler.penalty_coeff *= self.config.penalty_growth_rate
            
            if method == ConstraintHandlingMethod.BARRIER:
                self.constraint_handler.barrier_coeff *= self.config.barrier_decay_rate
            
            # Progress reporting
            if verbose and step % 500 == 0:
                feasible_marker = "✅" if feasible else "❌"
                nuclear_marker = "💥" if result.get('nuclear_triggered', False) else "  "
                print(f"  Step {step:4d}: Obj={current_obj:8.4f}, "
                      f"Viol={violation_info.total_violation:8.4f} {feasible_marker} {nuclear_marker}")
            
            # Early stopping
            if feasible and current_obj < 0.01:
                if verbose:
                    print(f"🎯 Converged at step {step}!")
                break
        
        # Final results
        wrapper_stats = self.wrapper.read()
        results = {
            'best_feasible_objective': self.best_feasible_objective,
            'best_feasible_params': self.best_feasible_params,
            'num_feasible_solutions': len(self.feasible_solutions),
            'final_params': current_params,
            'final_objective': current_obj,
            'final_violation': violation_info.total_violation,
            'infeasibility_history': self.infeasibility_history,
            'nuclear_triggers': wrapper_stats.get('nuclear_stats', {}).get('triggers', 0),
            'total_jumps': wrapper_stats.get('metrics', {}).get('total_jumps', 0)
        }
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print optimization summary."""
        print(f"\n🔒 CONSTRAINED OPTIMIZATION SUMMARY:")
        print(f"   {'Best Feasible Obj:':<30} {results['best_feasible_objective']:.8f}")
        print(f"   {'Feasible Solutions Found:':<30} {results['num_feasible_solutions']}")
        print(f"   {'Final Violation:':<30} {results['final_violation']:.8f}")
        print(f"   {'Nuclear Triggers:':<30} {results['nuclear_triggers']}")
        print(f"   {'Total Jumps:':<30} {results['total_jumps']}")
        
        if results['final_violation'] <= self.config.constraint_tolerance:
            print("   ✅ FINAL SOLUTION IS FEASIBLE!")
        else:
            print("   ⚠️  FINAL SOLUTION VIOLATES CONSTRAINTS")


# =======================
# Example Usage & Tests
# =======================

def test_box_constrained_rastrigin():
    """Test box-constrained Rastrigin optimization."""
    print("\n" + "="*80)
    print("TEST 1: Box-Constrained 2D Rastrigin")
    print("="*80)
    
    # Define Rastrigin
    @jax.jit
    def rastrigin(params):
        x, y = params[0], params[1]
        A = 10
        return A * 2 + (x**2 - A * jnp.cos(2 * jnp.pi * x)) + (y**2 - A * jnp.cos(2 * jnp.pi * y))
    
    rastrigin_grad = jax.jit(jax.grad(rastrigin))
    
    # Box constraints: -2 <= x,y <= 2
    box_constraint = Constraint(
        name="box",
        constraint_type=ConstraintType.BOX,
        bounds=(-2.0, 2.0)
    )
    
    config = ConstrainedPBitConfig(
        problem_dim=2,
        constraints=[box_constraint],
        handling_method=ConstraintHandlingMethod.HYBRID,
        penalty_coefficient=50.0,
        max_steps=2000,
        pbit_learning_rate=0.01,
        pbit_noise_scale=0.2,
        seed=42
    )
    
    optimizer = ConstrainedNuclearPBitWrapper(config)
    results = optimizer.optimize(rastrigin, rastrigin_grad, max_steps=2000, verbose=True)
    
    print(f"\n✅ Test completed!")
    print(f"   Best feasible solution: {results['best_feasible_objective']:.6f}")
    print(f"   At position: {results['best_feasible_params']}")


def test_linear_constrained_optimization():
    """Test linear equality constrained optimization."""
    print("\n" + "="*80)
    print("TEST 2: Linear Equality Constrained Quadratic")
    print("="*80)
    
    # Minimize ||x||^2 subject to x1 + x2 = 1
    @jax.jit
    def quadratic(params):
        return jnp.sum(params**2)
    
    quadratic_grad = jax.jit(jax.grad(quadratic))
    
    # Constraint: x1 + x2 = 1
    linear_eq = Constraint(
        name="sum_constraint",
        constraint_type=ConstraintType.LINEAR_EQUALITY,
        matrix=jnp.array([[1.0, 1.0]]),
        vector=jnp.array([1.0])
    )
    
    config = ConstrainedPBitConfig(
        problem_dim=2,
        constraints=[linear_eq],
        handling_method=ConstraintHandlingMethod.PROJECTION,
        penalty_coefficient=100.0,
        max_steps=2000,
        pbit_learning_rate=0.01,
        pbit_noise_scale=0.2,
        seed=42
    )
    optimizer = ConstrainedNuclearPBitWrapper(config)
    results = optimizer.optimize(quadratic, quadratic_grad, max_steps=2000, verbose=True)
    
    print(f"\n✅ Test completed!")
    print(f"   Expected solution: [0.5, 0.5]")
    print(f"   Best feasible solution: {results['best_feasible_objective']:.6f}")
    print(f"   At position: {results['best_feasible_params']}")

def test_nonlinear_constrained():
    """Test nonlinear inequality constrained optimization."""
    print("\n" + "="*80)
    print("TEST 3: Nonlinear Inequality Constrained")
    print("="*80)
    
    # Minimize x^2 + y^2 subject to x^2 + y^2 >= 1 (annulus)
    @jax.jit
    def objective(params):
        return jnp.sum(params**2)
    
    objective_grad = jax.jit(jax.grad(objective))
    
    # Constraint: x^2 + y^2 - 1 >= 0 i.e. 1 - (x^2 + y^2) <= 0
    @jax.jit
    def circle_constraint(params):
        return jnp.array([1.0 - jnp.sum(params**2)])  # g <= 0 feasible (outside unit circle)
    
    nonlinear_ineq = Constraint(
        name="circle",
        constraint_type=ConstraintType.NONLINEAR_INEQUALITY,
        function=circle_constraint
    )
    
    
    config = ConstrainedPBitConfig(
        problem_dim=2,
        constraints=[nonlinear_ineq],
        handling_method=ConstraintHandlingMethod.PENALTY,
        penalty_coefficient=200.0,
        max_steps=2000,
        pbit_learning_rate=0.01,
        pbit_noise_scale=0.2,
        seed=42
    )
    
    optimizer = ConstrainedNuclearPBitWrapper(config)
    results = optimizer.optimize(objective, objective_grad, max_steps=2000, verbose=True)
    
    print(f"\n✅ Test completed!")
    print(f"   Solution should lie on unit circle")
    print(f"   Found solution: {results['best_feasible_params']}")
    print(f"   Distance from origin: {jnp.linalg.norm(results['best_feasible_params']):.6f}")

if __name__ == "__main__":
    print("🔒 CONSTRAINED NUCLEAR PBIT OPTIMIZER")
    print("=" * 80)
    print("Testing constraint handling capabilities...")
    
    test_box_constrained_rastrigin()
    test_linear_constrained_optimization()
    test_nonlinear_constrained()
    
    print("\n")

