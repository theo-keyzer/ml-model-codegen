# w.py: Lightweight NuclearPBitWrapper for PathBasedPBit (No Overhead)
# API Version: 2.1 (Pairs with pb.py/mr.py v2.1)
# Simple wrapper to enable nuclear jumps (strategy=6) on demand or stuck detection.
# Zero overhead: No history, plots, or debug—pure functional API.
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
from enum import Enum

# Assume updated pb.py and mr.py are available
try:
    from pb import PathBasedPBit, PBitConfig, ResetStrategy  # From updated pb.py (v2.1)
    from mr import SARMemoryManager, SARConfig  # For tuning, but not required here
except ImportError:
    raise ImportError("Require updated pb.py (v2.1) and mr.py (v2.1) for imports.")

__version__ = "2.1.1"  # Bumped for bug fix

@dataclass
class NuclearWrapperConfig:
    """Minimal config for NuclearPBitWrapper (zero overhead)."""
    enable: bool = True  # Enable nuclear jumps
    stuck_threshold: int = 50  # Steps_since_improvement > this triggers nuclear in step()
    manual_nuke_strength: float = 1.0  # Strength for manual nuke() (passed to MR)
    verbose: bool = False  # No output if False (default)
    
    def __post_init__(self):
        if self.stuck_threshold < 10:
            raise ValueError("stuck_threshold must be >=10")
        if not 0 <= self.manual_nuke_strength <= 2.0:
            raise ValueError("manual_nuke_strength must be in [0, 2.0]")

class NuclearPBitWrapper:
    """
    Zero-overhead wrapper for PathBasedPBit: Adds nuclear jumps (MR strategy=6) on stuck/manual trigger.
    - No history tracking, no plots, no debug—delegates to pbit.
    - Uses MR's native NUCLEAR_JUMP (full random or best + huge noise).
    - API: step(), nuke(), read() (with minimal nuclear_stats).
    
    Example:
        wrapper = NuclearPBitWrapper(pbit, NuclearWrapperConfig(enable=True, stuck_threshold=30))
        result = wrapper.step(grad_fn, obj_fn, reset_patience=100)  # Triggers nuclear if stuck
        wrapper.nuke()  # Manual nuclear reset
        stats = wrapper.read()['nuclear_stats']  # {'triggers': 2, 'last_step': 150}
    """

    def __init__(self, pbit: PathBasedPBit, config: NuclearWrapperConfig = NuclearWrapperConfig()):
        if not isinstance(pbit, PathBasedPBit):
            raise ValueError("pbit must be PathBasedPBit instance")
        self.pbit = pbit
        self.config = config
        self.nuclear_triggers = 0  # Minimal counter (int, no array)
        self.last_nuclear_step = 0  # Minimal tracker (int)
        if config.verbose:
            print(f"✅ NuclearPBitWrapper v{__version__} attached (Enable: {config.enable}, Threshold: {config.stuck_threshold})")

    def step(self, gradient_fn: Callable[[jnp.ndarray], jnp.ndarray],
             objective_fn: Optional[Callable[[jnp.ndarray], float]] = None,
             reset_patience: int = 100) -> Dict[str, Any]:
        """
        Step wrapped pbit; trigger nuclear if enable=True and stuck (steps_since_improvement > threshold).
        
        Args:
            gradient_fn, objective_fn, reset_patience: Passed to pbit.step().
        
        Returns:
            pbit result + 'nuclear_triggered': bool, 'nuclear_nuke_type': str or None.
        """
        if not self.config.enable:
            return self.pbit.step(gradient_fn, objective_fn, reset_patience)
        
        # Normal step
        result = self.pbit.step(gradient_fn, objective_fn, reset_patience)
        current_step = result['step']
        
        # Check for stuck and trigger nuclear
        nuclear_triggered = False
        nuke_type = None
        if result['steps_since_improvement'] > self.config.stuck_threshold:
            if self.config.verbose:
                print(f"🔥 Nuclear trigger at step {current_step} (stuck: {result['steps_since_improvement']})")
            
            # Force nuclear reset via MR (strategy=6, high strength)
            mr = self.pbit.memory_manager
            best_params = self.pbit.state.best_params
            current_params = self.pbit.state.params
            gradient = gradient_fn(current_params)
            steps_since = result['steps_since_improvement']
            
            # FIX: Ensure at least one stuck point exists to avoid distance computation errors
            ptr_value = int(mr.stuck_points_ptr) if mr.stuck_points_ptr.ndim == 0 else int(mr.stuck_points_ptr[0])
            if ptr_value == 0:
                mr.update_stuck_points_fifo(current_params)
            
            # Call MR.perform_reset with forced strategy=6 (NUCLEAR_JUMP)
            high_severity_key, _ = jax.random.split(mr.key)
            strength = self.config.manual_nuke_strength
            new_params, new_key, is_jump = mr._sar_parameter_reset(  # Internal call for force (zero overhead)
                current_params, best_params, gradient, jnp.array(6), strength,  # strategy=6 forced
                mr.stuck_points_fifo, mr.config.avoidance_threshold, mr.config.avoidance_strength,
                mr.problem_dim, high_severity_key, mr.config.enable_jumps, 
                mr.config.quantum_jump_range, mr.config.nuclear_reset_strength, mr.config.min_jump_distance
            )
            
            # Update MR state (mimic perform_reset internals)
            mr.key = new_key
            mr.update_stuck_points_fifo(current_params)  # Use public method instead
            mr.strategy_effectiveness = mr.strategy_effectiveness.at[6].add(1)  # Nuclear effectiveness
            mr.strategy_effectiveness = mr.strategy_effectiveness * mr.config.effectiveness_decay
            mr.jump_counts = mr.jump_counts.at[1].add(1)  # Nuclear count (index 1)
            mr.avoidance_active = False  # Reset avoidance
            
            # Update pbit state post-nuclear (set params, reset velocity/stuck counters)
            self.pbit.state = self.pbit.state.replace(
                params=new_params,
                velocity=jnp.zeros(mr.problem_dim),  # Zero momentum post-nuke (configurable if needed)
                steps_since_improvement=0,  # Reset stuck counters
                consecutive_stuck=0
            )
            
            nuclear_triggered = True
            nuke_type = 'NUCLEAR'  # From is_jump=True and strategy=6
            
            self.nuclear_triggers += 1
            self.last_nuclear_step = current_step
            if self.config.verbose:
                disp = jnp.linalg.norm(new_params - current_params)
                #print(f"   💥 Nuclear jump: Disp={disp:.2f}, New Params={[p:.2f for p in new_params]}")
        
        # Augment result
        result['nuclear_triggered'] = nuclear_triggered
        result['nuclear_nuke_type'] = nuke_type
        return result

    def nuke(self, gradient_fn: Callable[[jnp.ndarray], jnp.ndarray] = None) -> Dict[str, Any]:
        """
        Manual nuclear jump (ignores stuck check; uses current state/gradient).
        
        Args:
            gradient_fn: Optional; if provided, computes gradient for MR.
        
        Returns:
            {'new_params': array, 'displacement': float, 'nuclear_type': 'NUCLEAR'}.
        """
        if not self.config.enable:
            raise ValueError("Nuclear disabled; set config.enable=True")
        
        mr = self.pbit.memory_manager
        current_params = self.pbit.state.params
        best_params = self.pbit.state.best_params
        gradient = gradient_fn(current_params) if gradient_fn is not None else jnp.zeros(self.pbit.problem_dim)
        
        # FIX: Ensure at least one stuck point exists to avoid distance computation errors
        ptr_value = int(mr.stuck_points_ptr) if mr.stuck_points_ptr.ndim == 0 else int(mr.stuck_points_ptr[0])
        if ptr_value == 0:
            mr.update_stuck_points_fifo(current_params)
        
        # Force nuclear reset
        high_severity_key, _ = jax.random.split(mr.key)
        strength = self.config.manual_nuke_strength
        new_params, new_key, is_jump = mr._sar_parameter_reset(
            current_params, best_params, gradient, jnp.array(6), strength,
            mr.stuck_points_fifo, mr.config.avoidance_threshold, mr.config.avoidance_strength,
            mr.problem_dim, high_severity_key, mr.config.enable_jumps, 
            mr.config.quantum_jump_range, mr.config.nuclear_reset_strength, mr.config.min_jump_distance
        )
        
        # Update MR as above
        mr.key = new_key
        mr.update_stuck_points_fifo(current_params)  # Use public method
        mr.strategy_effectiveness = mr.strategy_effectiveness.at[6].add(1)
        mr.strategy_effectiveness = mr.strategy_effectiveness * mr.config.effectiveness_decay
        mr.jump_counts = mr.jump_counts.at[1].add(1)
        
        # Update pbit
        self.pbit.state = self.pbit.state.replace(
            params=new_params,
            velocity=jnp.zeros(mr.problem_dim),
            steps_since_improvement=0,
            consecutive_stuck=0
        )
        
        self.nuclear_triggers += 1
        self.last_nuclear_step = self.pbit.state.step_count
        
        disp = jnp.linalg.norm(new_params - current_params)
        #if self.config.verbose:
        #print(f"💥 Manual nuke at step {self.last_nuclear_step}: Disp={disp:.2f}, New Params={[p:.2f for p in new_params]}")
        
        return {
            'new_params': new_params,
            'displacement': float(disp),
            'nuclear_type': 'NUCLEAR' if is_jump else None
        }

    def read(self, what: str = "all") -> Dict[str, Any]:
        """
        Read wrapped state: Delegates to pbit.read(), adds minimal nuclear_stats.
        
        Args:
            what: Passed to pbit.read() ('all', 'metrics', etc.).
        
        Returns:
            pbit read() + 'nuclear_stats': {'triggers': int, 'last_step': int}.
        """
        pbit_read = self.pbit.read(what)
        if what == 'all':
            pbit_read['nuclear_stats'] = {
                'triggers': self.nuclear_triggers,
                'last_step': self.last_nuclear_step
            }
        return pbit_read

    def detach(self) -> PathBasedPBit:
        """Return unwrapped pbit (for direct use)."""
        return self.pbit

# Simple test (demos wrapper with quadratic; no overhead)
def test_nuclear_wrapper():
    print("\n🚀 Test: NuclearPBitWrapper API (Zero Overhead) - Bug Fixed")
    
    # Setup: Enable native jumps in MR, but wrapper adds nuclear force
    sar_config = SARConfig(enable_jumps=True, quantum_jump_range=3.0, nuclear_reset_strength=4.0)
    pbit_config = PBitConfig(enable_quantum_jumps=True, jump_consecutive_stuck_threshold=100)  # High to demo wrapper trigger
    pbit = PathBasedPBit(problem_dim=2, config=pbit_config, memory_manager=SARMemoryManager(2, sar_config))
    
    wrapper_config = NuclearWrapperConfig(enable=True, stuck_threshold=20, verbose=True)  # Verbose for test
    wrapper = NuclearPBitWrapper(pbit, wrapper_config)
    
    # Quadratic test
    @jax.jit
    def quadratic(params):
        return jnp.sum(params**2)
    
    quadratic_grad = jax.jit(jax.grad(quadratic))
    max_steps = 100
    for step in range(max_steps):
        result = wrapper.step(quadratic_grad, quadratic, reset_patience=80)  # High patience to hit stuck_threshold
        
        if step % 20 == 0 or result.get('nuclear_triggered', False):
            print(f"Step {step:3d}: Params=[{result['params'][0]:.2f}, {result['params'][1]:.2f}], "
                  f"Obj={result['objective']:.3f}, Stuck={result['steps_since_improvement']}, "
                  f"Nuclear={result.get('nuclear_triggered', False)}, Triggers={wrapper.nuclear_triggers}")
        
        if result['best_objective'] < 0.01:
            print(f"🎯 Converged at step {step} (Best Obj: {result['best_objective']:.4f})")
            break
    
    print(f"\n📊 Final: Best Obj={result['best_objective']:.4f}, Nuclear Triggers={wrapper.nuclear_triggers}")
    print(f"Nuclear Stats: {wrapper.read()['nuclear_stats']}")

if __name__ == "__main__":
    test_nuclear_wrapper()
