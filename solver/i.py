# constraint_integration.py: Full Integration with Nuclear PBit System
# Combines constraint handling with quantum tunneling, nuclear resets, and SAR memory
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import sys

# Import existing infrastructure
try:
    from pb import PathBasedPBit, PBitConfig, PathBasedPBitState
    from mr import SARMemoryManager, SARConfig
    from w import NuclearPBitWrapper, NuclearWrapperConfig  # CHANGED: x -> w
    from c import (
        ConstraintHandler, ConstrainedPBitConfig, Constraint, 
        ConstraintType, ConstraintHandlingMethod, ConstraintViolation
    )
except ImportError:
    print("Warning: Required modules not found. Using standalone mode.")    #from constrained_pbit import (
    from c import (
        ConstraintHandler, ConstrainedPBitConfig, Constraint, 
        ConstraintType, ConstraintHandlingMethod, ConstraintViolation
    )
except ImportError:
    print("Warning: Required modules not found. Using standalone mode.")


class ConstraintAwareSARMemory(SARMemoryManager):
    """
    Enhanced SAR memory that tracks constraint violations at stuck points.
    Helps the optimizer learn which regions are infeasible.
    """
    
    def __init__(self, problem_dim: int, config: SARConfig, constraint_handler: ConstraintHandler):
        super().__init__(problem_dim, config)
        self.constraint_handler = constraint_handler
        
        # Additional tracking for constraint-aware optimization
        self.constraint_violations_at_stuck = []  # Store violations at each stuck point
        self.feasible_region_boundaries = []  # Learn boundary of feasible region
        self.infeasible_escape_count = 0  # Count escapes from infeasible regions
        
    def update_stuck_points_fifo_with_constraints(self, new_stuck_point: jnp.ndarray) -> None:
        """Enhanced stuck point tracking with constraint information."""
        # Store the point
        self.update_stuck_points_fifo(new_stuck_point)
        
        # Evaluate constraints at this stuck point
        violation_info = self.constraint_handler.evaluate_constraints(new_stuck_point)
        self.constraint_violations_at_stuck.append({
            'point': np.array(new_stuck_point),
            'violation': violation_info.total_violation,
            'feasible': violation_info.feasible,
            'details': violation_info.individual_violations
        })
        
        # If this is on the boundary (small violation), track it
        if 0 < violation_info.total_violation < 0.5:
            self.feasible_region_boundaries.append(np.array(new_stuck_point))
    
    def perform_constraint_aware_reset(self, 
                                      current_params: jnp.ndarray,
                                      best_params: jnp.ndarray,
                                      gradient: jnp.ndarray,
                                      steps_since_improvement: int,
                                      reset_patience: int,
                                      is_feasible: bool) -> Dict[str, Any]:
        """
        Perform reset with constraint awareness.
        If current point is infeasible, bias reset toward feasible regions.
        """
        # Standard reset
        reset_info = self.perform_reset(
            current_params, best_params, gradient, 
            steps_since_improvement, reset_patience
        )
        
        new_params = reset_info['new_params']
        
        # If we're infeasible, try to bias toward known feasible regions
        if not is_feasible and len(self.feasible_region_boundaries) > 5:
            # Pick a random boundary point and perturb toward feasible side
            boundary_idx = np.random.randint(len(self.feasible_region_boundaries))
            boundary_point = self.feasible_region_boundaries[boundary_idx]
            
            # Move partially toward boundary
            alpha = 0.3
            new_params = new_params * (1 - alpha) + jnp.array(boundary_point) * alpha
            
            self.infeasible_escape_count += 1
            reset_info['constraint_aware_escape'] = True
        
        return reset_info
    
    def get_constraint_statistics(self) -> Dict[str, Any]:
        """Get statistics about constraints at stuck points."""
        if not self.constraint_violations_at_stuck:
            return {'no_data': True}
        
        violations = [v['violation'] for v in self.constraint_violations_at_stuck]
        feasible_stuck = [v for v in self.constraint_violations_at_stuck if v['feasible']]
        
        return {
            'total_stuck_points': len(self.constraint_violations_at_stuck),
            'feasible_stuck_points': len(feasible_stuck),
            'infeasible_stuck_points': len(self.constraint_violations_at_stuck) - len(feasible_stuck),
            'mean_violation': np.mean(violations),
            'max_violation': np.max(violations),
            'boundary_points_found': len(self.feasible_region_boundaries),
            'infeasible_escapes': self.infeasible_escape_count
        }


class ConstraintAwareNuclearPBit(PathBasedPBit):
    """
    Nuclear PBit optimizer with integrated constraint handling.
    Combines momentum-based optimization with constraint awareness.
    """
    
    def __init__(self, 
                 problem_dim: int,
                 config: PBitConfig,
                 constraint_handler: ConstraintHandler,
                 initial_params: Optional[jnp.ndarray] = None,
                 memory_manager: Optional[SARMemoryManager] = None):
        
        self.constraint_handler = constraint_handler
        
        # Ensure initial params are feasible
        if initial_params is not None:
            initial_params = constraint_handler.project_to_feasible(initial_params)
        
        super().__init__(problem_dim, config, initial_params, memory_manager)
        
        # Constraint-specific tracking
        self.feasibility_history = []
        self.constraint_violation_history = []
        self.projection_count = 0
        self.feasible_improvements = 0
        self.infeasible_improvements = 0
        
        print(f"🔒 ConstraintAwareNuclearPBit initialized")
        print(f"   With {len(constraint_handler.constraints)} constraints")
    
    def step_with_constraints(self,
                             gradient_fn: Callable[[jnp.ndarray], jnp.ndarray],
                             objective_fn: Callable[[jnp.ndarray], float],
                             reset_patience: int = 100) -> Dict[str, Any]:
        """
        Enhanced step function with constraint handling.
        """
        # Evaluate constraints at current point
        current_violation = self.constraint_handler.evaluate_constraints(self.state.params)
        self.feasibility_history.append(current_violation.feasible)
        self.constraint_violation_history.append(current_violation.total_violation)
        
        # Get augmented objective (handles constraints internally)
        augmented_obj_fn = lambda p: self.constraint_handler.handle_constraint_violation(
            p, objective_fn(p), self.state.key
        )[0]
        
        # Compute gradient of augmented objective
        try:
            augmented_grad_fn = jax.grad(augmented_obj_fn)
        except:
            # Fallback to original gradient if augmented is not differentiable
            augmented_grad_fn = gradient_fn
        
        # Take standard step
        result = self.step(augmented_grad_fn, augmented_obj_fn, reset_patience)
        
        # Post-step constraint handling
        new_violation = self.constraint_handler.evaluate_constraints(self.state.params)
        
        # If severely infeasible, project back
        if new_violation.total_violation > 5.0:
            projected = self.constraint_handler.project_to_feasible(self.state.params)
            self.state = self.state.replace(params=projected)
            self.projection_count += 1
            result['projected'] = True
        else:
            result['projected'] = False
        
        # Track improvement type
        if result['objective'] < self.state.best_objective:
            if current_violation.feasible:
                self.feasible_improvements += 1
            else:
                self.infeasible_improvements += 1
        
        # Add constraint info to result
        result['constraint_violation'] = new_violation.total_violation
        result['feasible'] = new_violation.feasible
        result['constraint_details'] = new_violation.individual_violations
        
        return result


class FullyIntegratedConstrainedOptimizer:
    """
    Fully integrated constrained optimizer combining all advanced features:
    - Nuclear PBit with quantum tunneling
    - SAR memory with constraint awareness
    - Adaptive constraint handling
    - Multi-start with constraint-aware initialization
    """
    
    def __init__(self, config: ConstrainedPBitConfig):
        self.config = config
        self.constraint_handler = ConstraintHandler(config)
        self.problem_dim = config.problem_dim
        
        # Create constraint-aware memory manager
        sar_config = SARConfig(
            spf_depth=config.sar_spf_depth,
            avoidance_threshold=config.sar_avoidance_threshold,
            avoidance_strength=config.sar_avoidance_strength,
            seed=config.seed
        )
        self.memory_manager = ConstraintAwareSARMemory(
            self.problem_dim, sar_config, self.constraint_handler
        )
        
        # Create constraint-aware PBit
        pbit_config = PBitConfig(
            momentum_decay_on_stuck=config.pbit_momentum_decay_on_stuck,
            learning_rate=config.pbit_learning_rate,
            noise_scale=config.pbit_noise_scale,
            seed=config.seed
        )
        
        initial_params = config.initial_params
        if initial_params is None:
            # Generate random feasible initial point
            initial_params = self._generate_feasible_initial_point()
        
        self.pbit = ConstraintAwareNuclearPBit(
            self.problem_dim,
            pbit_config,
            self.constraint_handler,
            initial_params,
            self.memory_manager
        )
        
        # Tracking
        self.optimization_history = []
        self.best_feasible_objective = float('inf')
        self.best_feasible_params = None
        self.quantum_tunnel_count = 0
        self.constraint_aware_escape_count = 0
        
        print(f"🚀 FullyIntegratedConstrainedOptimizer ready!")
        print(f"   Combining: Nuclear PBit + SAR Memory + Constraint Handling")
    
    def _generate_feasible_initial_point(self) -> jnp.ndarray:
        """Generate a random feasible starting point."""
        max_attempts = 100
        
        for _ in range(max_attempts):
            candidate = jax.random.normal(jax.random.PRNGKey(np.random.randint(1000)), 
                                         (self.problem_dim,)) * 0.5
            
            violation = self.constraint_handler.evaluate_constraints(candidate)
            if violation.feasible:
                return candidate
            
            # Try projection
            projected = self.constraint_handler.project_to_feasible(candidate)
            violation = self.constraint_handler.evaluate_constraints(projected)
            if violation.feasible:
                return projected
        
        # Fallback: zero vector (often feasible for box constraints)
        return jnp.zeros(self.problem_dim)
    
    def _quantum_tunnel_to_feasible(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Quantum tunneling with constraint awareness.
        Tunnel to a feasible region, preferably unexplored.
        """
        tunnel_key, new_key = jax.random.split(key)
        
        # If we know feasible boundary points, tunnel near them
        if len(self.memory_manager.feasible_region_boundaries) > 3:
            boundary_points = self.memory_manager.feasible_region_boundaries
            # Pick random boundary and add large noise
            idx = np.random.randint(len(boundary_points))
            base = jnp.array(boundary_points[idx])
            noise = jax.random.normal(tunnel_key, base.shape) * 2.0
            candidate = base + noise
        else:
            # Random tunnel
            candidate = jax.random.uniform(tunnel_key, (self.problem_dim,), 
                                          minval=-3.0, maxval=3.0)
        
        # Project to ensure feasibility
        candidate = self.constraint_handler.project_to_feasible(candidate)
        self.quantum_tunnel_count += 1
        
        return candidate, new_key
    
    def optimize(self,
                objective_fn: Callable[[jnp.ndarray], float],
                gradient_fn: Callable[[jnp.ndarray], jnp.ndarray],
                max_steps: int = 10000,
                target_objective: float = 1e-6,
                enable_quantum_tunneling: bool = True,
                tunnel_threshold: int = 200,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Run fully integrated constrained optimization.
        
        Args:
            objective_fn: Objective to minimize
            gradient_fn: Gradient of objective
            max_steps: Maximum iterations
            target_objective: Early stopping threshold
            enable_quantum_tunneling: Use quantum escapes
            tunnel_threshold: Steps without improvement before tunneling
            verbose: Print progress
        
        Returns:
            Comprehensive results dictionary
        """
        if verbose:
            print(f"\n🚀 INTEGRATED CONSTRAINED OPTIMIZATION")
            print("="*80)
            print(f"   Max steps: {max_steps}")
            print(f"   Quantum tunneling: {enable_quantum_tunneling}")
            print(f"   Constraints: {len(self.config.constraints)}")
        
        key = jax.random.PRNGKey(self.config.seed)
        steps_without_improvement = 0
        
        for step in range(max_steps):
            # Take constrained step
            result = self.pbit.step_with_constraints(
                gradient_fn, 
                objective_fn,
                reset_patience=self.config.reset_patience
            )
            
            self.optimization_history.append(result)
            
            # Track best feasible solution
            if result['feasible']:
                current_obj = result['objective']
                if current_obj < self.best_feasible_objective:
                    self.best_feasible_objective = current_obj
                    self.best_feasible_params = result['params'].copy()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
            else:
                steps_without_improvement += 1
            
            # Quantum tunneling escape
            if (enable_quantum_tunneling and 
                steps_without_improvement > tunnel_threshold and
                self.quantum_tunnel_count < 5):
                
                new_pos, key = self._quantum_tunnel_to_feasible(key)
                self.pbit.state = self.pbit.state.replace(params=new_pos)
                
                if verbose:
                    print(f"   🌌 QUANTUM TUNNEL #{self.quantum_tunnel_count} at step {step}")
                    print(f"      Jumped to feasible region")
                
                steps_without_improvement = 0
            
            # Progress reporting
            if verbose and step % 500 == 0:
                feasible_mark = "✅" if result['feasible'] else "❌"
                projected_mark = "📐" if result.get('projected', False) else "  "
                
                print(f"  Step {step:5d}: Obj={result['objective']:9.4f}, "
                      f"Best={self.best_feasible_objective:9.4f}, "
                      f"Viol={result['constraint_violation']:7.4f} "
                      f"{feasible_mark} {projected_mark}")
            
            # Early stopping
            if (self.best_feasible_objective <= target_objective and 
                result['feasible']):
                if verbose:
                    print(f"\n🎯 TARGET ACHIEVED at step {step}!")
                break
        
        # Compile results
        results = self._compile_results(objective_fn, verbose)
        return results
    
    def _compile_results(self, objective_fn: Callable, verbose: bool) -> Dict[str, Any]:
        """Compile comprehensive optimization results."""
        feasible_count = sum(1 for r in self.optimization_history if r['feasible'])
        
        constraint_stats = self.memory_manager.get_constraint_statistics()
        
        results = {
            'best_feasible_objective': self.best_feasible_objective,
            'best_feasible_params': self.best_feasible_params,
            'total_steps': len(self.optimization_history),
            'feasible_steps': feasible_count,
            'infeasible_steps': len(self.optimization_history) - feasible_count,
            'feasibility_rate': feasible_count / max(len(self.optimization_history), 1),
            'projection_count': self.pbit.projection_count,
            'quantum_tunnels': self.quantum_tunnel_count,
            'constraint_aware_escapes': self.memory_manager.infeasible_escape_count,
            'feasible_improvements': self.pbit.feasible_improvements,
            'infeasible_improvements': self.pbit.infeasible_improvements,
            'constraint_statistics': constraint_stats,
            'optimization_history': self.optimization_history,
            'penalty_history': self.constraint_handler.penalty_history,
            'violation_history': self.pbit.constraint_violation_history
        }
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print comprehensive optimization summary."""
        print(f"\n🏆 OPTIMIZATION COMPLETE!")
        print("="*80)
        print(f"{'Metric':<40} {'Value':<20}")
        print("-"*80)
        print(f"{'Best Feasible Objective:':<40} {results['best_feasible_objective']:.8f}")
        print(f"{'Total Steps:':<40} {results['total_steps']}")
        print(f"{'Feasibility Rate:':<40} {results['feasibility_rate']:.1%}")
        print(f"{'Projections Performed:':<40} {results['projection_count']}")
        print(f"{'Quantum Tunnels:':<40} {results['quantum_tunnels']}")
        print(f"{'Constraint-Aware Escapes:':<40} {results['constraint_aware_escapes']}")
        print(f"{'Improvements (Feasible):':<40} {results['feasible_improvements']}")
        print(f"{'Improvements (Infeasible):':<40} {results['infeasible_improvements']}")
        
        if not results['constraint_statistics'].get('no_data', False):
            stats = results['constraint_statistics']
            print(f"\nConstraint Statistics:")
            print(f"  Total Stuck Points: {stats['total_stuck_points']}")
            print(f"  Feasible Stuck Points: {stats['feasible_stuck_points']}")
            print(f"  Boundary Points Found: {stats['boundary_points_found']}")
        
        if results['best_feasible_params'] is not None:
            final_violation = self.constraint_handler.evaluate_constraints(
                results['best_feasible_params']
            )
            if final_violation.feasible:
                print(f"\n✅ FINAL SOLUTION IS FEASIBLE!")
            else:
                print(f"\n⚠️  Warning: Best solution has violation: {final_violation.total_violation:.6f}")
    
    def visualize_constrained_optimization(self, 
                                          objective_fn: Callable,
                                          save_path: str = "integrated_constrained_opt.png"):
        """Create comprehensive visualization of constrained optimization."""
        if self.problem_dim != 2:
            print(f"⚠️  Visualization only supports 2D problems (current: {self.problem_dim}D)")
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # Extract trajectory
        trajectory = np.array([r['params'] for r in self.optimization_history])
        objectives = np.array([r['objective'] for r in self.optimization_history])
        feasibility = np.array([r['feasible'] for r in self.optimization_history])
        violations = np.array([r['constraint_violation'] for r in self.optimization_history])
        
        # 1. Optimization landscape with constraints
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        x = np.linspace(-4, 4, 300)
        y = np.linspace(-4, 4, 300)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        Feasible = np.ones_like(X, dtype=bool)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = jnp.array([X[i, j], Y[i, j]])
                Z[i, j] = objective_fn(point)
                violation = self.constraint_handler.evaluate_constraints(point)
                Feasible[i, j] = violation.feasible
        
        # Plot objective contours
        contour = ax1.contourf(X, Y, Z, levels=50, alpha=0.7, cmap='viridis')
        
        # Overlay feasible region
        ax1.contour(X, Y, Feasible.astype(float), levels=[0.5], colors='red', 
                   linewidths=3, linestyles='--', label='Feasible boundary')
        
        # Plot trajectory (color by feasibility)
        feasible_traj = trajectory[feasibility]
        infeasible_traj = trajectory[~feasibility]
        
        if len(feasible_traj) > 0:
            ax1.plot(feasible_traj[:, 0], feasible_traj[:, 1], 'g-', 
                    alpha=0.6, linewidth=2, label='Feasible trajectory')
        if len(infeasible_traj) > 0:
            ax1.plot(infeasible_traj[:, 0], infeasible_traj[:, 1], 'r--', 
                    alpha=0.4, linewidth=1, label='Infeasible trajectory')
        
        # Mark special points
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='blue', s=200, 
                   marker='o', edgecolors='black', linewidths=2, label='Start', zorder=10)
        if self.best_feasible_params is not None:
            ax1.scatter(self.best_feasible_params[0], self.best_feasible_params[1], 
                       c='gold', s=300, marker='*', edgecolors='black', 
                       linewidths=2, label='Best feasible', zorder=11)
        
        ax1.set_xlabel('x₁', fontsize=12)
        ax1.set_ylabel('x₂', fontsize=12)
        ax1.set_title('Constrained Optimization Landscape', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(contour, ax=ax1, label='Objective Value')
        
        # 2. Objective convergence
        ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        ax2.semilogy(objectives, 'b-', alpha=0.7, linewidth=1.5, label='Objective')
        ax2.axhline(y=self.best_feasible_objective, color='r', linestyle='--', 
                   linewidth=2, label=f'Best feasible: {self.best_feasible_objective:.4f}')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Objective Value (log scale)', fontsize=12)
        ax2.set_title('Convergence History', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feasibility over time
        ax3 = plt.subplot2grid((3, 3), (0, 2))
        ax3.fill_between(range(len(feasibility)), 0, feasibility.astype(float), 
                        alpha=0.5, color='green', label='Feasible')
        ax3.fill_between(range(len(feasibility)), feasibility.astype(float), 1, 
                        alpha=0.5, color='red', label='Infeasible')
        ax3.set_xlabel('Iteration', fontsize=10)
        ax3.set_ylabel('Feasibility', fontsize=10)
        ax3.set_title('Feasibility Timeline', fontsize=12, fontweight='bold')
        ax3.set_ylim(-0.1, 1.1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Constraint violation
        ax4 = plt.subplot2grid((3, 3), (1, 2))
        ax4.semilogy(violations, 'r-', linewidth=1.5)
        ax4.axhline(y=self.config.constraint_tolerance, color='g', 
                   linestyle='--', linewidth=2, label='Tolerance')
        ax4.set_xlabel('Iteration', fontsize=10)
        ax4.set_ylabel('Violation (log)', fontsize=10)
        ax4.set_title('Constraint Violation', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Statistics summary
        ax5 = plt.subplot2grid((3, 3), (2, 2))
        ax5.axis('off')
        
        stats_text = f"""
        OPTIMIZATION STATISTICS
        {'='*25}
        Total Steps: {len(self.optimization_history)}
        Feasible Rate: {feasibility.mean():.1%}
        Projections: {self.pbit.projection_count}
        Quantum Tunnels: {self.quantum_tunnel_count}
        
        FINAL RESULT
        {'='*25}
        Best Objective: {self.best_feasible_objective:.6f}
        Final Violation: {violations[-1]:.6f}
        Status: {'✅ Feasible' if feasibility[-1] else '❌ Infeasible'}
        """
        
        ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved: {save_path}")
        plt.close()


# =======================
# DEMONSTRATION
# =======================

def demo_integrated_constrained_rastrigin():
    """Demonstrate fully integrated constrained optimization on Rastrigin."""
    print("\n" + "="*80)
    print("DEMONSTRATION: Integrated Constrained Rastrigin Optimization")
    print("="*80)
    
    # 2D Rastrigin with circular constraint
    @jax.jit
    def rastrigin(params):
        x, y = params[0], params[1]
        A = 10
        return A * 2 + (x**2 - A * jnp.cos(2 * jnp.pi * x)) + \
               (y**2 - A * jnp.cos(2 * jnp.pi * y))
    
    rastrigin_grad = jax.jit(jax.grad(rastrigin))
    
    # Constraints: Stay within circle of radius 3, avoid small circle at origin
    @jax.jit
    def outer_circle(params):
        return jnp.array([jnp.sum(params**2) - 9.0])  # x² + y² <= 9
    
    @jax.jit  
    def inner_circle(params):
        return jnp.array([0.5 - jnp.sum(params**2)])  # x² + y² >= 0.5 (inverted)
    
    outer = Constraint(
        name="outer_bound",
        constraint_type=ConstraintType.NONLINEAR_INEQUALITY,
        function=outer_circle
    )
    
    inner = Constraint(
        name="inner_exclusion",
        constraint_type=ConstraintType.NONLINEAR_INEQUALITY,
        function=inner_circle
    )
    
    # NEW FLAT CONFIG STRUCTURE (matching c.py)
    config = ConstrainedPBitConfig(
        problem_dim=2,
        constraints=[outer, inner],
        handling_method=ConstraintHandlingMethod.HYBRID,
        penalty_coefficient=200.0,
        feasibility_emphasis=0.75,
        # PBit/SAR config
        max_steps=5000,
        reset_patience=50,
        initial_params=jnp.array([2.0, 2.0]),
        sar_spf_depth=20,
        sar_avoidance_threshold=0.8,
        sar_avoidance_strength=1.5,
        pbit_momentum_decay_on_stuck=0.3,
        pbit_learning_rate=0.01,
        pbit_noise_scale=0.2,
        # Nuclear wrapper config
        enable_nuclear=True,
        nuclear_stuck_threshold=50,
        nuclear_manual_strength=1.0,
        # Quantum jumps
        enable_quantum_jumps=True,
        jump_consecutive_stuck_threshold=50,
        seed=100
    )
    
    optimizer = FullyIntegratedConstrainedOptimizer(config)
    
    results = optimizer.optimize(
        rastrigin,
        rastrigin_grad,
        max_steps=5000,
        target_objective=0.01,
        enable_quantum_tunneling=True,
        tunnel_threshold=150,
        verbose=True
    )
    
    # Create visualization
    optimizer.visualize_constrained_optimization(
        rastrigin,
        save_path="demo_constrained_rastrigin.png"
    )
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print(f"   Check 'demo_constrained_rastrigin.png' for visualization")

if __name__ == "__main__":
    print("🚀 FULLY INTEGRATED CONSTRAINED NUCLEAR PBIT OPTIMIZER")
    print("="*80)
    print("Combining:")
    print("  ✓ Nuclear P")
    demo_integrated_constrained_rastrigin()

