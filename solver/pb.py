# pb.py: Path-Based PBit with Linear Momentum Path, Avoidance, and Native Jumps
# API Version: 2.1
# Jumps now integrated: No need for external x.py wrappers for quantum/nuclear escapes.
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Optional, Callable, Any, List
from dataclasses import dataclass, asdict
from functools import partial
import sys

try:
    from mr import SARMemoryManager, SARConfig, ResetStrategy  # Updated MR
except ImportError:
    print("Error: mr.py not found.")
    sys.exit(1)

__version__ = "2.1.0"

@dataclass(frozen=True)
class PathBasedPBitState:
    params: jnp.ndarray
    velocity: jnp.ndarray
    key: jnp.ndarray
    step_count: int
    best_objective: float
    best_params: jnp.ndarray
    steps_since_improvement: int
    consecutive_stuck: int  # NEW: Track stuck steps (0 on improvement/jump)
    memory_manager: SARMemoryManager

    def replace(self, **changes) -> 'PathBasedPBitState':
        data = asdict(self)
        data.update(changes)
        return PathBasedPBitState(**data)

@dataclass
class PBitConfig:
    """Configuration for PathBasedPBit (now with jump support)."""
    momentum_beta: float = 0.9
    momentum_decay_on_stuck: float = 0.1
    avoidance_threshold: float = 0.3
    learning_rate: float = 0.002
    noise_scale: float = 0.12
    clip_params: Tuple[float, float] = (-2.5, 2.5)
    clip_velocity: Tuple[float, float] = (-0.25, 0.25)
    clip_delta: Tuple[float, float] = (-0.4, 0.4)
    seed: int = 42
    alpha: float = 2.0
    beta: float = 1.0
    gamma: float = 0.12
    i_tail: float = 5e-5

    # NEW: Jump integration (syncs with SARConfig)
    enable_quantum_jumps: bool = False  # Enable native jumps in step()
    jump_consecutive_stuck_threshold: int = 50  # Trigger jump after this many stuck steps
    post_jump_momentum_factor: float = 0.0  # 0=zero velocity; 0.5=preserve half direction (0-1)

    def __post_init__(self):
        # Old checks...
        for val, name in [(self.momentum_beta, "momentum_beta"), (self.noise_scale, "noise_scale")]:
            if not 0 <= val <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        # NEW checks
        if self.enable_quantum_jumps:
            if self.jump_consecutive_stuck_threshold < 10:
                raise ValueError("jump_consecutive_stuck_threshold must be >=10")
            if not 0 <= self.post_jump_momentum_factor <= 1.0:
                raise ValueError("post_jump_momentum_factor must be in [0, 1]")

class PathBasedPBit:
    """
    PathBasedPBit with native jumps for extreme escapes.
    Triggers jumps on consecutive stuck steps; reports in results.
    Shared MR handles jump logic (no external x.py needed).
    """

    def __init__(self, problem_dim: int, config: PBitConfig = PBitConfig(),
                 initial_params: Optional[jnp.ndarray] = None,
                 memory_manager: Optional[SARMemoryManager] = None,
                 on_step: Optional[Callable] = None,
                 on_reset: Optional[Callable] = None):
        if problem_dim < 1:
            raise ValueError("problem_dim must be >= 1")
        self.problem_dim = problem_dim
        self.config = config
        self.key = jax.random.PRNGKey(config.seed)
        self.on_step = on_step or (lambda *args: None)
        self.on_reset = on_reset or (lambda *args: None)

        # Sync MR config with PB jumps
        sar_config = SARConfig(
            avoidance_threshold=config.avoidance_threshold,
            enable_jumps=config.enable_quantum_jumps,
            seed=config.seed
        )
        if memory_manager is None:
            self.memory_manager = SARMemoryManager(problem_dim, sar_config)
        else:
            if not memory_manager.config.enable_jumps and config.enable_quantum_jumps:
                print("⚠️ Warning: Shared MR jumps disabled; enabling...")
                memory_manager.config.enable_jumps = True  # Mutable update
            self.memory_manager = memory_manager

        if initial_params is None or initial_params.shape != (problem_dim,):
            self.key, init_key = jax.random.split(self.key)
            initial_params = jax.random.normal(init_key, (problem_dim,)) * 0.1
        else:
            initial_params = jnp.array(initial_params)

        # Physical constants unchanged...
        BOLTZMANN = 1.3806e-23
        TEMPERATURE = 300
        GAMMA_NOISE = 2/3
        V_T = 25.85e-3
        KAPPA = 0.85
        self.noise_power = 4 * BOLTZMANN * TEMPERATURE * GAMMA_NOISE * (100e-6) * 50e6
        self.vt_kappa = KAPPA * V_T
        self._physical_params = {
            'i_tail': config.i_tail, 'alpha': config.alpha, 'beta': config.beta, 'gamma': config.gamma,
            'noise_power': self.noise_power, 'vt_kappa': self.vt_kappa
        }

        self.state = PathBasedPBitState(
            params=initial_params,
            velocity=jnp.zeros(problem_dim),
            key=self.key,
            step_count=0,
            best_objective=float('inf'),
            best_params=initial_params.copy(),
            steps_since_improvement=0,
            consecutive_stuck=0,  # NEW
            memory_manager=self.memory_manager
        )

        self._compile_functions()
        jump_str = " + Quantum Jumps" if config.enable_quantum_jumps else ""
        print(f"✅ PathBasedPBit v{__version__} created (Dim: {problem_dim}{jump_str})")

    def _compile_functions(self):
        # Unchanged (differential_pair, compute_min_stuck_distance, random_normal_vmap)...
        def differential_pair_single(v_diff, i_tail, alpha, beta, gamma, noise_key, noise_power, vt_kappa):
            eta = jax.random.normal(noise_key) * jnp.sqrt(noise_power)
            v_eff = alpha * v_diff + beta * i_tail + gamma * eta
            arg = jnp.clip(-v_eff / vt_kappa, -500.0, 500.0)
            prob_1 = 1.0 / (1.0 + jnp.exp(arg))
            return prob_1

        inner_vmap = jax.vmap(differential_pair_single, in_axes=(0, None, None, None, None, 0, None, None))
        self._differential_pair_probability_vmap = jax.jit(
            inner_vmap, static_argnames=['noise_power', 'vt_kappa']
        )

        @jax.jit
        def compute_min_stuck_distance(current_params, stuck_points_fifo, threshold):
            distances = jnp.linalg.norm(stuck_points_fifo - current_params, axis=1)
            min_distance = jnp.min(distances)
            too_close = min_distance < threshold
            return min_distance, too_close

        self._compute_min_stuck_distance = jax.jit(
            partial(compute_min_stuck_distance, threshold=self.config.avoidance_threshold)
        )

        self._random_normal_vmap = jax.jit(jax.vmap(lambda k: jax.random.normal(k), in_axes=0))

    def _lose_momentum_on_stuck(self, velocity: jnp.ndarray, min_distance: float) -> jnp.ndarray:
        decay = jnp.where(min_distance < self.config.avoidance_threshold, self.config.momentum_decay_on_stuck, 1.0)
        return velocity * decay

    def step(self, gradient_fn: Callable[[jnp.ndarray], jnp.ndarray],
             objective_fn: Optional[Callable[[jnp.ndarray], float]] = None,
             reset_patience: int = 100) -> Dict[str, Any]:
        """
        Single step: Now with native jump triggers on consecutive stuck.
        """
        grad_val = gradient_fn(self.state.params)
        if grad_val.shape != (self.problem_dim,):
            raise ValueError(f"Gradient must be shape {(self.problem_dim,)}")

        objective_value = objective_fn(self.state.params) if callable(objective_fn) else float('inf')
        current_obj = float(objective_value)
        current_best = float(self.state.best_objective)
        has_improved = current_obj < current_best - 1e-8
        new_best_objective = current_obj if has_improved else current_best
        new_best_params = self.state.params if has_improved else self.state.best_params
        new_steps_since_improvement = 0 if has_improved else self.state.steps_since_improvement + 1

        # NEW: Consecutive stuck tracking (resets on improvement)
        new_consecutive_stuck = 0 if has_improved else self.state.consecutive_stuck + 1

        # Gradient/momentum/noise/avoidance unchanged...
        grad_val = jnp.nan_to_num(grad_val, nan=0.0, posinf=1e-3, neginf=-1e-3)
        grad_norm = float(jnp.linalg.norm(grad_val))

        self.key, update_key = jax.random.split(self.state.key)
        v_diffs = jnp.abs(grad_val) / 25.0
        key_dims = jax.random.split(update_key, self.problem_dim)
        probs = self._differential_pair_probability_vmap(
            v_diffs, self.config.i_tail, self.config.alpha, self.config.beta, self.config.gamma,
            key_dims, self.noise_power, self.vt_kappa
        )
        step_sizes = probs * self.config.learning_rate * jnp.abs(grad_val)
        delta_params_det = -step_sizes * jnp.sign(grad_val)

        self.key, noise_key_global = jax.random.split(update_key)
        noise_keys_final = jax.random.split(noise_key_global, self.problem_dim)
        eta_random_vector = self._random_normal_vmap(noise_keys_final)
        eta_noise = eta_random_vector * self.config.gamma * 0.03 * step_sizes
        delta_params = delta_params_det + eta_noise
        delta_params = jnp.clip(delta_params, *self.config.clip_delta)

        new_velocity = (self.config.momentum_beta * self.state.velocity +
                        (1 - self.config.momentum_beta) * delta_params)
        new_velocity = jnp.clip(new_velocity, *self.config.clip_velocity)

        min_distance, too_close = self._compute_min_stuck_distance(
            self.state.params, self.state.memory_manager.stuck_points_fifo
        )
        new_velocity = self._lose_momentum_on_stuck(new_velocity, float(min_distance))

        new_params = jnp.clip(self.state.params + new_velocity, *self.config.clip_params)

        if bool(too_close):
            self.state.memory_manager.update_stuck_points_fifo(self.state.params)

        # Reset logic: Enhanced with jump forcing
        needs_reset = new_steps_since_improvement > reset_patience
        force_jump = self.config.enable_quantum_jumps and new_consecutive_stuck > self.config.jump_consecutive_stuck_threshold
        if force_jump:
            needs_reset = True  # Force reset as jump opportunity
            # Optionally tweak patience for jumps (lower for more aggression)
            effective_patience = reset_patience // 2

        reset_info = None
        if needs_reset:
            reset_info = self.state.memory_manager.perform_reset(
                self.state.params, new_best_params, grad_val, new_steps_since_improvement, 
                effective_patience if force_jump else reset_patience
            )
            new_params = reset_info['new_params']
            
            # NEW: Post-jump momentum (directional if factor >0)
            if reset_info.get('is_jump', False) and self.config.post_jump_momentum_factor > 0:
                jump_dir = new_params - self.state.params
                jump_norm = jnp.linalg.norm(jump_dir) + 1e-10
                momentum_dir = jump_dir / jump_norm * self.config.post_jump_momentum_factor
                new_velocity = jnp.clip(momentum_dir, *self.config.clip_velocity)
            else:
                new_velocity = jnp.zeros(self.problem_dim)
            
            new_consecutive_stuck = 0  # Reset on any reset/jump
            self.on_reset(self.state, reset_info)

        new_state = self.state.replace(
            params=new_params,
            velocity=new_velocity,
            key=noise_key_global,
            step_count=self.state.step_count + 1,
            best_objective=new_best_objective,
            best_params=new_best_params,
            steps_since_improvement=new_steps_since_improvement,
            consecutive_stuck=new_consecutive_stuck  # NEW
        )
        self.state = new_state

        result = {
            'params': new_params,
            'velocity': new_velocity,
            'objective': current_obj,
            'gradient_norm': grad_norm,
            'best_objective': new_best_objective,
            'steps_since_improvement': new_steps_since_improvement,
            'consecutive_stuck': new_consecutive_stuck,  # NEW
            'min_stuck_distance': float(min_distance),
            'too_close_to_stuck': bool(too_close),
            'avoidance_list_size': int(self.state.memory_manager.stuck_points_ptr[0]),
            'needs_reset': needs_reset,
            'force_jump': force_jump,  # NEW
            'is_jump': reset_info.get('is_jump', False) if reset_info else False,
            'jump_type': reset_info.get('jump_type') if reset_info else None,
            'step': self.state.step_count,
            'unique_stuck_points': len(self.state.memory_manager.get_unique_stuck_points())
        }
        self.on_step(new_state, result)
        return result

    def read(self, what: str = "all") -> Dict[str, Any]:
        if what == "all":
            mr_read = self.state.memory_manager.read()
            return {
                'params': np.array(self.state.params),
                'velocity': np.array(self.state.velocity),
                'best_params': np.array(self.state.best_params),
                'memory': mr_read,
                'metrics': {
                    'step_count': self.state.step_count,
                    'best_objective': float(self.state.best_objective),
                    'steps_since_improvement': self.state.steps_since_improvement,
                    'consecutive_stuck': self.state.consecutive_stuck,  # NEW
                    'unique_stuck_count': len(self.state.memory_manager.get_unique_stuck_points()),
                    'total_jumps': int(np.sum(mr_read['jump_counts'])),  # NEW
                    'quantum_jumps': int(mr_read['jump_counts'][0]),
                    'nuclear_jumps': int(mr_read['jump_counts'][1])
                }
            }
        # Other modes unchanged (add consecutive_stuck to "metrics" if needed)...

    @classmethod
    def create_ensemble(cls, num_instances: int, problem_dim: int, shared_memory: Optional[SARMemoryManager] = None,
                        base_config: Optional[PBitConfig] = None, seeds: Optional[List[int]] = None,
                        initial_params_list: Optional[List[jnp.ndarray]] = None, enable_jumps: bool = False) -> List['PathBasedPBit']:
        """
        Factory: Now supports jump-enabled configs per instance.
        """
        if shared_memory is None:
            shared_memory = SARMemoryManager(problem_dim, SARConfig(enable_jumps=enable_jumps))
        base_config = base_config or PBitConfig(enable_quantum_jumps=enable_jumps)
        seeds = seeds or list(range(num_instances))
        initial_params_list = initial_params_list or [None] * num_instances

        def create_config_with_seed_and_jumps(base: PBitConfig, seed: int, instance_jumps: bool) -> PBitConfig:
            data = asdict(base)
            data['seed'] = seed
            data['enable_quantum_jumps'] = instance_jumps
            return PBitConfig(**data)

        return [
            cls(problem_dim, create_config_with_seed_and_jumps(base_config, seeds[i], enable_jumps), 
                initial_params_list[i], shared_memory)
            for i in range(num_instances)
        ]

# Tests (now demo native jumps)
def test_multi_pb_shared_mr():
    print("\n🚀 Testing MULTIPLE PathBasedPBit v2.1 (with Native Jumps)")
    # Enable jumps in shared MR
    shared_mr = SARMemoryManager(problem_dim=2, config=SARConfig(spf_depth=5, avoidance_threshold=0.4, enable_jumps=True, quantum_jump_range=3.0))
    shared_mr.update_stuck_points_fifo(jnp.array([1.0, 1.0]))
    print("📝 Shared initial stuck point added.")

    # Ensemble with jumps
    base_config = PBitConfig(momentum_decay_on_stuck=0.5, enable_quantum_jumps=True, 
                             jump_consecutive_stuck_threshold=20, post_jump_momentum_factor=0.3)
    pb_ensemble = PathBasedPBit.create_ensemble(
        num_instances=2, problem_dim=2, shared_memory=shared_mr, enable_jumps=True,
        base_config=base_config, seeds=[42, 43],
        initial_params_list=[jnp.array([1.5, 1.0]), jnp.array([-1.0, -1.5])]
    )
    pb1, pb2 = pb_ensemble

    def on_reset_callback(old_state, reset_info):
        if reset_info.get('is_jump'):
            print(f"   🌌 PB reset as JUMP: {reset_info['jump_type']} (strength: {reset_info['strength']:.2f}, "
                  f"disp: {np.linalg.norm(reset_info['new_params'] - old_state.params):.2f})")

    pb1.on_reset = on_reset_callback

    @jax.jit
    def test_quadratic(params):
        return jnp.sum(params**2)

    test_quadratic_grad = jax.jit(jax.grad(test_quadratic))

    max_steps = 100
    for step in range(max_steps):
        result1 = pb1.step(test_quadratic_grad, test_quadratic, reset_patience=30)
        result2 = pb2.step(test_quadratic_grad, test_quadratic, reset_patience=30)

        if step % 10 == 0 or result1['too_close_to_stuck'] or result2['too_close_to_stuck'] or result1['needs_reset'] or result2['needs_reset'] or result1['force_jump'] or result2['force_jump']:
            too_close1 = "🔴" if result1['too_close_to_stuck'] else "⚪"
            too_close2 = "🔴" if result2['too_close_to_stuck'] else "⚪"
            jump1 = "🌌" if result1['is_jump'] else "   "
            jump2 = "🌌" if result2['is_jump'] else "   "
            reset1 = "🔄" if result1['needs_reset'] else "   "
            reset2 = "🔄" if result2['needs_reset'] else "   "
            unique_count = len(shared_mr.get_unique_stuck_points())

            print(f"{step:4d} | PB1 | {result1['params'][0]:.2f},{result1['params'][1]:.2f} | "
                  f"{result1['velocity'][0]:.2f},{result1['velocity'][1]:.2f} | "
                  f"{result1['objective']:5.2f} | {result1['best_objective']:5.2f} | "
                  f"{result1['min_stuck_distance']:6.2f} | {too_close1} | {unique_count:2d} | "
                  f"{result1['consecutive_stuck']:3d} | {jump1}{reset1}")
            print(f"     | PB2 | {result2['params'][0]:.2f},{result2['params'][1]:.2f} | "
                  f"{result2['velocity'][0]:.2f},{result2['velocity'][1]:.2f} | "
                  f"{result2['objective']:5.2f} | {result2['best_objective']:5.2f} | "
                  f"{result2['min_stuck_distance']:6.2f} | {too_close2} | {unique_count:2d} | "
                  f"{result2['consecutive_stuck']:3d} | {jump2}{reset2}")
            print()

        if (jnp.linalg.norm(result1['params']) < 0.1 or jnp.linalg.norm(result2['params']) < 0.1) and step > 50:
            print(f"🎯 Ensemble converged at step {step}!")
            break

    final_mr = shared_mr.read()
    print(f"\n📊 SHARED MR Final: Unique Stuck: {len(final_mr['unique_stuck_points'])}, "
          f"Ptr: {final_mr['stuck_points_ptr']}, Total Jumps: {int(np.sum(final_mr['jump_counts']))}, "
          f"Quantum: {int(final_mr['jump_counts'][0])}, Nuclear: {int(final_mr['jump_counts'][1])}")

    for i, pb in enumerate(pb_ensemble, 1):
        final_pb = pb.read("metrics")
        print(f"📊 PB{i} Best: {final_pb['best_objective']:.4f}, Steps: {final_pb['step_count']}, "
              f"Consec Stuck: {final_pb['consecutive_stuck']}, Total Jumps: {final_pb['total_jumps']}")

# Single test unchanged, but add enable_quantum_jumps=True for demo
def test_path_based_pbit():
    print("\n🚀 Testing SINGLE PathBasedPBit v2.1 (with Jumps Enabled)")
    config = PBitConfig(momentum_decay_on_stuck=0.2, enable_quantum_jumps=True, 
                        jump_consecutive_stuck_threshold=15, post_jump_momentum_factor=0.2)
    pbit = PathBasedPBit(problem_dim=2, config=config, initial_params=jnp.array([1.5, 1.5]))

    # Rest unchanged, but expect jumps after ~15 stuck steps...

if __name__ == "__main__":
    test_multi_pb_shared_mr()
    # test_path_based_pbit()
