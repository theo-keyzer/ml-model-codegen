# mr.py: SAR Memory Manager for Stuck-Point Avoidance and Resets (JUMPS INTEGRATED)
# API Version: 2.1
# Native support for jump strategies (quantum/nuclear) to eliminate external wrappers.
import jax
import jax.numpy as jnp
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Callable, Any
import numpy as np

__version__ = "2.1.0"

@dataclass
class SARConfig:
    """Configuration for SARMemoryManager (now with jump support)."""
    spf_depth: int = 25  # Size of FIFO stuck-points queue
    avoidance_threshold: float = 0.3  # Distance threshold to trigger avoidance
    avoidance_strength: float = 0.6  # Push strength away from stuck points
    strategy_weights: Optional[Dict[str, float]] = None  # Custom weights (defaults provided)
    effectiveness_decay: float = 0.99  # Decay factor for strategy effectiveness (0-1)
    seed: int = 42  # RNG seed
    
    # NEW: Jump-specific configs (enable via enable_jumps)
    enable_jumps: bool = False  # Enable quantum/nuclear jumps
    quantum_jump_range: float = 5.0  # Uniform range for quantum jumps (e.g., [-range, range])
    nuclear_reset_strength: float = 2.0  # Noise scale for nuclear (to best or full random)
    min_jump_distance: float = 1.0  # Enforce min displacement for jumps (avoids tiny hops)
    jump_severity_threshold: float = 0.7  # Severity > this favors jumps (0-1)

    def __post_init__(self):
        if self.spf_depth < 1:
            raise ValueError("spf_depth must be >= 1")
        if self.avoidance_threshold <= 0:
            raise ValueError("avoidance_threshold must be > 0")
        if not 0 <= self.avoidance_strength <= 2.0:
            raise ValueError("avoidance_strength must be in [0, 2.0]")
        if self.enable_jumps:
            if self.quantum_jump_range <= 0 or self.nuclear_reset_strength <= 0:
                raise ValueError("Jump params must be > 0 when enable_jumps=True")
            if not 0 <= self.min_jump_distance <= self.quantum_jump_range:
                raise ValueError("min_jump_distance must be in [0, quantum_jump_range]")
            if not 0 <= self.jump_severity_threshold <= 1.0:
                raise ValueError("jump_severity_threshold must be in [0, 1]")
        if self.strategy_weights is None:
            # Default weights: Balanced, with room for jumps
            self.strategy_weights = {
                "PERTURB_BEST": 0.1, "BEST_PARAMS": 0.1, "RANDOM_RESTART": 0.2,
                "GRADIENT_ESCAPE": 0.15, "AVOIDANCE_RESTART": 0.2,
                "QUANTUM_JUMP": 0.1, "NUCLEAR_JUMP": 0.1  # NEW: Default low, boosted by severity
            }
        if sum(self.strategy_weights.values()) <= 0:
            raise ValueError("strategy_weights must sum to > 0")

class ResetStrategy(Enum):
    PERTURB_BEST = 0
    BEST_PARAMS = 1
    RANDOM_RESTART = 2
    GRADIENT_ESCAPE = 3
    AVOIDANCE_RESTART = 4
    QUANTUM_JUMP = 5      # NEW: Random uniform or opposite vector jump
    NUCLEAR_JUMP = 6      # NEW: To best + huge noise or full random

class SARMemoryManager:
    """
    Manages memory and reset logic, now with native jumps (quantum/nuclear) for bold escapes.
    Jumps eliminate need for external wrappers like NuclearPBitWrapper.
    Shared across multiple PathBasedPBit instances.
    """

    def __init__(self, problem_dim: int, config: SARConfig):
        if problem_dim < 1:
            raise ValueError("problem_dim must be >= 1")
        self.problem_dim = problem_dim
        self.config = config
        self.key = jax.random.PRNGKey(config.seed)

        num_strategies = len(ResetStrategy)  # Now 7
        self.stuck_points_fifo = jnp.zeros((config.spf_depth, problem_dim))
        self.stuck_points_ptr = jnp.zeros(1, dtype=jnp.int32)
        self.strategy_effectiveness = jnp.zeros(num_strategies)
        self.avoidance_active = jnp.bool_(False)
        self.jump_counts = jnp.zeros(2)  # NEW: [quantum, nuclear] counts (indices 0,1)

        self.custom_strategies: Dict[int, Callable] = {}
        self._compile_functions()
        print(f"✅ SARMemoryManager v{__version__} initialized (Dim: {problem_dim}, SPF: {config.spf_depth}, Jumps: {config.enable_jumps})")

    def add_custom_strategy(self, strategy_idx: int, fn: Callable) -> None:
        """
        Add a custom reset strategy.

        Args:
            strategy_idx: Index (0-6) to override.
            fn: Callable taking (current_params, best_params, gradient, strength, key) -> (new_params, new_key).

        Raises:
            ValueError: Invalid fn (must return (array, key)).
        """
        if strategy_idx < 0 or strategy_idx > 6:
            raise ValueError("strategy_idx must be in [0, 6]")
        # Quick test
        test_params = jnp.zeros(self.problem_dim)
        test_key = jax.random.PRNGKey(0)
        try:
            result = fn(test_params, test_params, test_params, 0.5, test_key)
            if len(result) != 2 or result[0].shape != (self.problem_dim,):
                raise ValueError("Custom fn must return (jnp.ndarray of shape (dim,), key)")
        except Exception as e:
            raise ValueError(f"Custom fn validation failed: {e}")
        self.custom_strategies[strategy_idx] = fn

    def _compile_functions(self):
        """Compile JAX functions for memory and resets."""
        def compute_min_stuck_distance(current_params, stuck_points_fifo, threshold):
            distances = jnp.linalg.norm(stuck_points_fifo - current_params, axis=1)
            min_distance = jnp.min(distances)
            too_close = min_distance < threshold
            return min_distance, too_close

        self._compute_min_stuck_distance = jax.jit(compute_min_stuck_distance)

        def update_stuck_points_fifo(stuck_points_fifo, ptr, new_stuck_point, depth):
            idx = jnp.int32(ptr % depth)
            new_fifo = stuck_points_fifo.at[idx].set(new_stuck_point)
            new_ptr = (ptr + 1) % depth
            return new_fifo, new_ptr

        self._update_stuck_points_fifo = jax.jit(update_stuck_points_fifo, static_argnames=['depth'])

        def sar_parameter_reset(current_params, best_params, gradient, strategy, strength,
                                stuck_points_fifo, threshold, avoidance_strength, problem_dim, key,
                                enable_jumps, quantum_range, nuclear_strength, min_jump_dist):
            # Safeguard NaN/Inf
            current_params = jnp.nan_to_num(current_params, nan=0.0, posinf=1.0, neginf=-1.0)
            best_params = jnp.nan_to_num(best_params, nan=0.0, posinf=1.0, neginf=-1.0)
            gradient = jnp.nan_to_num(gradient, nan=0.0, posinf=0.1, neginf=-0.1)

            min_distance, too_close = self._compute_min_stuck_distance(current_params, stuck_points_fifo, threshold)
            effective_strategy = jnp.where(too_close, 4, strategy)  # Force avoidance if too close
            effective_strategy = jnp.clip(effective_strategy, 0, 6)

            # FIXED: xs tuple EXCLUDES static problem_dim (static_arg); 10 elements
            # Branches take xs, unpack, use static problem_dim for shapes (concrete in tracing)
            xs = (current_params, best_params, gradient, strength, stuck_points_fifo, avoidance_strength, 
                  quantum_range, nuclear_strength, min_jump_dist, key)

            # Branches: Take single 'xs', unpack to 10 args + use static problem_dim for RNG shapes
            def perturb_best(xs):
                curr, best_p, grad, stren, stuck_p, avoid_s, q_range, n_stren, min_dist, kk = xs
                noise = jax.random.normal(kk, (problem_dim,)) * stren * 0.5
                base = curr * (1 - stren) + best_p * stren
                new_params = jnp.clip(base + noise, -5.0, 5.0)
                return new_params, kk, False

            def best_params_fn(xs):
                curr, best_p, grad, stren, stuck_p, avoid_s, q_range, n_stren, min_dist, kk = xs
                perturbation = jax.random.normal(kk, (problem_dim,)) * stren * 0.1
                new_params = jnp.clip(best_p + perturbation, -5.0, 5.0)
                return new_params, kk, False

            def random_restart(xs):
                curr, best_p, grad, stren, stuck_p, avoid_s, q_range, n_stren, min_dist, kk = xs
                new_params = jax.random.uniform(kk, (problem_dim,), minval=-3.0, maxval=3.0)
                return new_params, kk, False

            def gradient_escape(xs):
                curr, best_p, grad, stren, stuck_p, avoid_s, q_range, n_stren, min_dist, kk = xs
                grad_norm = jnp.linalg.norm(grad)
                safe_gradient = jnp.where(grad_norm < 1e-10,
                                          jax.random.normal(kk, (problem_dim,)) * 0.1,
                                          grad / (grad_norm + 1e-10))
                escape_direction = -safe_gradient * stren * 2.0
                new_params = jnp.clip(curr + escape_direction, -5.0, 5.0)
                return new_params, kk, False

            def avoidance_restart(xs):
                curr, best_p, grad, stren, stuck_p, avoid_s, q_range, n_stren, min_dist, kk = xs
                base_key, sub_key = jax.random.split(kk)
                distances = jnp.linalg.norm(stuck_p - curr, axis=1)
                closest_idx = jnp.argmin(distances)
                closest_stuck = stuck_p[closest_idx]
                avoidance_dir = curr - closest_stuck
                avoidance_norm = jnp.linalg.norm(avoidance_dir)
                rand_dir = jax.random.normal(sub_key, (problem_dim,))
                rand_norm = jnp.linalg.norm(rand_dir) + 1e-10
                chosen_dir = jnp.where(avoidance_norm > 1e-10,
                                       avoidance_dir / avoidance_norm,
                                       rand_dir / rand_norm)
                avoidance_push = chosen_dir * avoid_s * stren
                new_params = jnp.clip(curr + avoidance_push, -5.0, 5.0)
                return new_params, base_key, False

            def quantum_jump(xs):
                curr, best_p, grad, stren, stuck_p, avoid_s, q_range, n_stren, min_dist, kk = xs
                sub_key, split_key = jax.random.split(kk)
                is_uniform = jax.random.bernoulli(sub_key, 0.5)

                def uniform_branch(k3):
                    return jax.random.uniform(k3, (problem_dim,), minval=-q_range, maxval=q_range), k3

                def opposite_branch(k3):
                    noise = jax.random.normal(k3, (problem_dim,)) * stren
                    opp_dir = -curr + noise
                    new_pos = curr + opp_dir
                    return new_pos, k3

                new_pos_pre, new_kk = jax.lax.cond(
                    is_uniform,
                    lambda k: uniform_branch(k),
                    lambda k: opposite_branch(k),
                    split_key
                )

                jump_dist = jnp.linalg.norm(new_pos_pre - curr)
                scale = jnp.where(jump_dist < min_dist,
                                  min_dist / (jump_dist + 1e-10),
                                  1.0)
                new_pos = curr + (new_pos_pre - curr) * scale
                return jnp.clip(new_pos, -5.0, 5.0), new_kk, True

            def nuclear_jump(xs):
                curr, best_p, grad, stren, stuck_p, avoid_s, q_range, n_stren, min_dist, kk = xs
                sub_key, split_key = jax.random.split(kk)
                is_to_best = jax.random.bernoulli(sub_key, 0.5)

                def to_best_branch(k3):
                    noise = jax.random.normal(k3, (problem_dim,)) * n_stren
                    return best_p + noise, k3

                def full_random_branch(k3):
                    return jax.random.uniform(k3, (problem_dim,), minval=-q_range * 1.2, maxval=q_range * 1.2), k3

                new_pos_pre, new_kk = jax.lax.cond(
                    is_to_best,
                    lambda k: to_best_branch(k),
                    lambda k: full_random_branch(k),
                    split_key
                )

                jump_dist = jnp.linalg.norm(new_pos_pre - curr)
                scale = jnp.where(jump_dist < min_dist,
                                  min_dist / (jump_dist + 1e-10),
                                  1.0)
                new_pos = curr + (new_pos_pre - curr) * scale
                return jnp.clip(new_pos, -5.0, 5.0), new_kk, True

            # FIXED: Switch with xs (10 elems); branches use static problem_dim for concrete shapes
            branches = [perturb_best, best_params_fn, random_restart, gradient_escape, 
                        avoidance_restart, quantum_jump, nuclear_jump]
            new_params, new_key, raw_is_jump = jax.lax.switch(effective_strategy, branches, xs)

            # Mask jumps if not enabled
            is_jump = jnp.logical_and(raw_is_jump, enable_jumps)
            return new_params, new_key, is_jump

        # FIXED: Add problem_dim to static_argnames (concrete for RNG shapes)
        self._sar_parameter_reset = jax.jit(
            sar_parameter_reset, 
            static_argnames=['enable_jumps', 'problem_dim']
        )

        def select_strategy(steps_since_improvement, reset_patience, strategy_effectiveness, key, enable_jumps, jump_thresh):
            severity = jnp.minimum(steps_since_improvement / jnp.maximum(reset_patience, 1.0), 1.0)
            base_weights = jnp.array([0.1, 0.1, 0.2, 0.15, 0.2, 0.1, 0.1])  # 0-6
            
            severity_weights = jnp.where(
                severity > jump_thresh, 
                base_weights * jnp.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.3, 0.3]),  
                base_weights * jnp.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1])   
            )
            
            jump_mask = jnp.array([False, False, False, False, False, enable_jumps, enable_jumps])
            adjusted_weights = jnp.where(
                jump_mask, 
                severity_weights + strategy_effectiveness * 0.2, 
                severity_weights
            )
            adjusted_weights = adjusted_weights / (jnp.sum(adjusted_weights) + 1e-10)
            
            select_key, new_key = jax.random.split(key)
            chosen_idx = jax.random.categorical(select_key, jnp.log(adjusted_weights + 1e-10))
            strength = jnp.clip(0.3 * (1.0 + severity * 0.7), 0.1, 1.0)
            return jnp.int32(chosen_idx), strength, new_key

        self._select_strategy = jax.jit(select_strategy, static_argnames=['enable_jumps'])

    def update_stuck_points_fifo(self, new_stuck_point: jnp.ndarray) -> None:
        """
        Add a point to the avoidance list (FIFO). Updates are visible to all sharing PBs.

        Args:
            new_stuck_point: JAX array of shape (problem_dim,).

        Raises:
            ValueError: If shape mismatches.
        """
        if new_stuck_point.shape != (self.problem_dim,):
            raise ValueError(f"new_stuck_point must be shape {(self.problem_dim,)}, got {new_stuck_point.shape}")
        self.stuck_points_fifo, new_ptr = self._update_stuck_points_fifo(
            self.stuck_points_fifo, self.stuck_points_ptr[0], new_stuck_point, self.config.spf_depth
        )
        self.stuck_points_ptr = jnp.array([new_ptr])

    def perform_reset(self, current_params: jnp.ndarray, best_params: jnp.ndarray,
                      gradient: jnp.ndarray, steps_since_improvement: int, reset_patience: int) -> Dict[str, Any]:
        """
        Perform reset using selected strategy; update memory (shared).

        Args:
            current_params: Current parameters (shape: (problem_dim,)).
            best_params: Best parameters seen (shape: (problem_dim,)).
            gradient: Current gradient (shape: (problem_dim,)).
            steps_since_improvement: Steps without improvement.
            reset_patience: Max steps before reset.

        Returns:
            Dict with 'new_params', 'strategy' (str), 'strength', etc. + 'is_jump'.

        Raises:
            ValueError: Shape mismatches.
        """
        for arr, name in [(current_params, "current_params"), (best_params, "best_params"), (gradient, "gradient")]:
            if arr.shape != (self.problem_dim,):
                raise ValueError(f"{name} must be shape {(self.problem_dim,)}, got {arr.shape}")

        self.key, strategy_key = jax.random.split(self.key)
        strategy, strength, new_key = self._select_strategy(
            jnp.array(steps_since_improvement), jnp.array(reset_patience),
            self.strategy_effectiveness, strategy_key, self.config.enable_jumps, self.config.jump_severity_threshold
        )
        strategy_int = int(strategy)  # Python int for custom check

        # Custom strategy check OUTSIDE JIT
        if strategy_int in self.custom_strategies:
            custom_fn = self.custom_strategies[strategy_int]
            new_params, reset_key = custom_fn(current_params, best_params, gradient, float(strength), new_key)
            is_jump = False  # Custom isn't jump by default
            reset_key = jax.random.PRNGKey(0) if reset_key is None else reset_key
        else:
            new_params, reset_key, is_jump = self._sar_parameter_reset(
                current_params, best_params, gradient, strategy, strength,
                self.stuck_points_fifo, self.config.avoidance_threshold, self.config.avoidance_strength,
                self.problem_dim, new_key, self.config.enable_jumps, self.config.quantum_jump_range,
                self.config.nuclear_reset_strength, self.config.min_jump_distance
            )

        # Update FIFO with pre-reset params
        self.update_stuck_points_fifo(current_params)

        # Update effectiveness with decay
        self.strategy_effectiveness = self.strategy_effectiveness.at[strategy_int].add(1)
        self.strategy_effectiveness = self.strategy_effectiveness * self.config.effectiveness_decay

        if is_jump:
            jump_type = 0 if strategy_int == 5 else 1  # Quantum=0, Nuclear=1
            self.jump_counts = self.jump_counts.at[jump_type].add(1)

        min_distance, too_close = self._compute_min_stuck_distance(
            current_params, self.stuck_points_fifo, self.config.avoidance_threshold
        )
        self.avoidance_active = too_close
        self.key = reset_key

        strategy_names = [s.name for s in ResetStrategy]
        return {
            'new_params': new_params,
            'strategy': strategy_names[strategy_int],
            'strength': float(strength),
            'min_stuck_distance': float(min_distance),
            'avoidance_triggered': bool(too_close),
            'is_jump': bool(is_jump),
            'jump_type': 'QUANTUM' if strategy_int == 5 else 'NUCLEAR' if strategy_int == 6 else None,
            'new_ptr': int(self.stuck_points_ptr[0])
        }

    def get_unique_stuck_points(self, tolerance: float = 0.1) -> List[jnp.ndarray]:
        """
        Get unique stuck points (de-duped with tolerance).

        Args:
            tolerance: Euclidean distance for considering points "unique".

        Returns:
            List of unique points (not all FIFO contents, to avoid duplicates).
        """
        active = self.stuck_points_fifo[:int(self.stuck_points_ptr[0])]
        if len(active) == 0:
            return []
        # Simple CPU de-dup (for API simplicity; optimize with JAX if needed)
        active_np = np.array(active)
        unique = []
        for p in active_np:
            if not any(np.linalg.norm(p - u) < tolerance for u in unique):
                unique.append(p)
        return [jnp.array(u) for u in unique]

    def clear_memory(self) -> None:
        """Clear the stuck points FIFO and reset pointer."""
        self.stuck_points_fifo = jnp.zeros_like(self.stuck_points_fifo)
        self.stuck_points_ptr = jnp.zeros_like(self.stuck_points_ptr)
        self.avoidance_active = jnp.bool_(False)
        self.jump_counts = jnp.zeros_like(self.jump_counts)  # Clear jumps too
        print("🧹 SARMemoryManager memory cleared.")

    def clone(self) -> 'SARMemoryManager':
        """Create a deep copy for safe, independent use."""
        import copy
        clone = copy.deepcopy(self)
        clone.key = jax.random.PRNGKey(self.config.seed + 1)  # Different seed
        return clone

    def read(self) -> Dict[str, Any]:
        """Read memory state (shared)."""
        unique_stuck = self.get_unique_stuck_points()
        return {
            'strategy_effectiveness': np.array(self.strategy_effectiveness),
            'stuck_points_fifo': np.array(self.stuck_points_fifo),
            'stuck_points_ptr': int(self.stuck_points_ptr[0]),
            'avoidance_active': bool(self.avoidance_active),
            'unique_stuck_points': [np.array(p) for p in unique_stuck],
            'jump_counts': np.array(self.jump_counts),  # [quantum, nuclear]
            'config': self.config  # Expose config for inspection
        }

# Enhanced test (covers custom strategy, unique points, clear, and jumps)
def test_sar_memory():
    print("\n🧠 Enhanced Test: SARMemoryManager API v2.1 (with Jumps)")
    config = SARConfig(spf_depth=3, avoidance_strength=0.8, enable_jumps=True, quantum_jump_range=4.0)
    manager = SARMemoryManager(problem_dim=2, config=config)

    # Custom strategy example (override QUANTUM_JUMP)
    def custom_quantum(curr, best, grad, strength, key):
        new_key, sub_key = jax.random.split(key)
        # Custom: Always opposite jump
        opp = -curr * strength * 2.0
        return curr + opp, new_key
    manager.add_custom_strategy(5, custom_quantum)  # Override QUANTUM_JUMP

    # Add stuck points
    manager.update_stuck_points_fifo(jnp.array([1.0, 1.0]))
    manager.update_stuck_points_fifo(jnp.array([0.5, 0.5]))
    print(f"Added 2 stuck points. FIFO ptr: {manager.stuck_points_ptr[0]}")
    print(f"Unique stuck points: {[p.tolist() for p in manager.get_unique_stuck_points()]}")

    # Simulate reset (high steps_since_improvement -> severity > threshold -> likely jump)
    current_params = jnp.array([0.9, 0.9])
    best_params = jnp.array([0.0, 0.0])
    gradient = jnp.array([0.1, 0.1])
    reset_info = manager.perform_reset(current_params, best_params, gradient, 100, 50)
    print(f"Reset performed: Strategy={reset_info['strategy']}, New Params={reset_info['new_params'][0]:.2f},{reset_info['new_params'][1]:.2f}, "
          f"Is Jump={reset_info['is_jump']}, Jump Type={reset_info['jump_type']}, "
          f"Avoidance Triggered={reset_info['avoidance_triggered']}, New Ptr={reset_info['new_ptr']}")

    # Test clear
    manager.clear_memory()
    print(f"After clear: Unique stuck points: {len(manager.get_unique_stuck_points())}, Jump Counts: {manager.jump_counts}")

if __name__ == "__main__":
    test_sar_memory()
