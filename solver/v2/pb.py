# pb.py: Path-Based PBit with Hill-Climb Escape and Overshoot Detection
# API Version: 2.3 (Added AdaptiveConfig, MultiObjectivePBit, GradientFreePBit)
# Integrates user-provided enhancements for dynamic adaptation, multi-obj, and gradient-free modes.
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Optional, Callable, Any, List
from dataclasses import dataclass, asdict
from functools import partial
from enum import Enum
import sys

try:
    from mr import SARMemoryManager, SARConfig, ResetStrategy
except ImportError:
    print("Error: mr.py not found.")
    sys.exit(1)

__version__ = "2.3.0"

class ClimbState(Enum):
    """State machine for hill-climb escape."""
    NORMAL = 0          # Normal optimization (minimize objective)
    ESCAPE_CLIMB = 1    # Reward getting worse (climbing hill to escape)

@dataclass(frozen=True)
class PathBasedPBitState:
    params: jnp.ndarray
    velocity: jnp.ndarray
    key: jnp.ndarray
    step_count: int
    best_objective: float
    best_params: jnp.ndarray
    steps_since_improvement: int
    consecutive_stuck: int
    memory_manager: SARMemoryManager
    
    # NEW: Hill-climb state machine
    climb_state: int  # ClimbState enum value
    climb_start_objective: float  # Objective when entering ESCAPE_CLIMB
    climb_worst_objective: float  # Worst objective seen during climb
    climb_steps: int  # Steps spent in ESCAPE_CLIMB mode
    hill_climb_count: int = 0  # Total number of times entered ESCAPE_CLIMB

    def replace(self, **changes) -> 'PathBasedPBitState':
        data = asdict(self)
        data.update(changes)
        return PathBasedPBitState(**data)

@dataclass
class PBitConfig:
    """Configuration for PathBasedPBit."""
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

    # Jump configs
    enable_quantum_jumps: bool = False
    jump_consecutive_stuck_threshold: int = 50
    post_jump_momentum_factor: float = 0.0
    
    # NEW: Hill-climb escape configs
    enable_hill_climb: bool = True  # Enable hill-climb state machine
    hill_climb_trigger_stuck: int = 30  # Enter ESCAPE_CLIMB after this many stuck steps
    hill_climb_max_steps: int = 20  # Max steps to stay in ESCAPE_CLIMB before giving up
    hill_climb_improvement_threshold: float = 0.05  # Relative improvement to detect overshoot (5%)
    hill_climb_reward_scale: float = 1.5  # Scale factor for rewarding worse objectives

    def __post_init__(self):
        for val, name in [(self.momentum_beta, "momentum_beta"), (self.noise_scale, "noise_scale")]:
            if not 0 <= val <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.enable_quantum_jumps:
            if self.jump_consecutive_stuck_threshold < 10:
                raise ValueError("jump_consecutive_stuck_threshold must be >=10")
            if not 0 <= self.post_jump_momentum_factor <= 1.0:
                raise ValueError("post_jump_momentum_factor must be in [0, 1]")
        if self.enable_hill_climb:
            if self.hill_climb_trigger_stuck < 10:
                raise ValueError("hill_climb_trigger_stuck must be >= 10")
            if self.hill_climb_max_steps < 5:
                raise ValueError("hill_climb_max_steps must be >= 5")
            if not 0 <= self.hill_climb_improvement_threshold <= 1.0:
                raise ValueError("hill_climb_improvement_threshold must be in [0, 1]")

# NEW: User-Provided Enhancement 1 - AdaptiveConfig
@dataclass
class AdaptiveConfig:
    """Dynamic configuration that adapts during optimization."""
    base_config: PBitConfig
    adaptation_rate: float = 0.01
    performance_history_size: int = 100
    _performance_history: List[float] = None  # Internal tracking

    def __post_init__(self):
        if self._performance_history is None:
            object.__setattr__(self, '_performance_history', [])

    def adapt_based_on_performance(self, recent_objectives: List[float]) -> None:
        """Dynamically adjust parameters based on recent performance."""
        self._performance_history.extend(recent_objectives)
        if len(self._performance_history) > self.performance_history_size:
            self._performance_history = self._performance_history[-self.performance_history_size:]

        if len(self._performance_history) < 2:
            return

        improvement_rate = np.mean(np.diff(self._performance_history))  # Negative = improvement (minimize)

        if improvement_rate < -0.1:  # Good progress (negative diff)
            self.base_config.learning_rate *= (1 + self.adaptation_rate)
            self.base_config.noise_scale *= (1 - self.adaptation_rate)
        else:  # Stuck or worsening
            self.base_config.learning_rate *= (1 - self.adaptation_rate)
            self.base_config.noise_scale *= (1 + self.adaptation_rate)

        # Ensure bounds
        self.base_config.learning_rate = np.clip(self.base_config.learning_rate, 1e-5, 0.1)
        self.base_config.noise_scale = np.clip(self.base_config.noise_scale, 0.01, 0.5)

class PathBasedPBit:
    """
    PathBasedPBit with hill-climb escape state machine.
    
    State transitions:
    NORMAL -> ESCAPE_CLIMB: When stuck for too long
    ESCAPE_CLIMB: Reward getting worse (climbing hill)
    ESCAPE_CLIMB -> NORMAL: Timeout or significant improvement from start
    """

    def __init__(self, problem_dim: int, config: PBitConfig = PBitConfig(),
                 initial_params: Optional[jnp.ndarray] = None,
                 memory_manager: Optional[SARMemoryManager] = None,
                 on_step: Optional[Callable] = None,
                 on_reset: Optional[Callable] = None,
                 adaptive_config: Optional[AdaptiveConfig] = None):  # NEW: Support adaptive
        if problem_dim < 1:
            raise ValueError("problem_dim must be >= 1")
        self.problem_dim = problem_dim
        self.config = config if not adaptive_config else adaptive_config.base_config  # Use base if adaptive
        self.adaptive_config = adaptive_config  # NEW: Track for runtime adaptation
        self.key = jax.random.PRNGKey(config.seed)
        self.on_step = on_step or (lambda *args: None)
        self.on_reset = on_reset or (lambda *args: None)
        self.objective_history = []  # NEW: For adaptive tracking

        # Sync MR config (hill-climb handled locally in PBit, not in MR)
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
                memory_manager.config.enable_jumps = True
            self.memory_manager = memory_manager

        if initial_params is None or initial_params.shape != (problem_dim,):
            self.key, init_key = jax.random.split(self.key)
            initial_params = jax.random.normal(init_key, (problem_dim,)) * 0.1
        else:
            initial_params = jnp.array(initial_params)

        # Physical constants
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
            consecutive_stuck=0,
            memory_manager=self.memory_manager,
            climb_state=ClimbState.NORMAL.value,
            climb_start_objective=float('inf'),
            climb_worst_objective=float('inf'),
            climb_steps=0,
            hill_climb_count=0
        )

        self._compile_functions()
        hill_str = " + HillClimb" if config.enable_hill_climb else ""
        jump_str = " + Jumps" if config.enable_quantum_jumps else ""
        adapt_str = " + Adaptive" if adaptive_config else ""
        print(f"✅ PathBasedPBit v{__version__} created (Dim: {problem_dim}{hill_str}{jump_str}{adapt_str})")

    def _compile_functions(self):
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

    def _update_climb_state(self, current_obj: float) -> Tuple[int, float, float, int]:
        current_state = ClimbState(self.state.climb_state)
        
        if current_state == ClimbState.NORMAL:
            # Only enter climb if we're genuinely stuck (not just slow progress)
            if (self.config.enable_hill_climb and 
                self.state.consecutive_stuck >= self.config.hill_climb_trigger_stuck and
                self.state.steps_since_improvement > 50):
                return (ClimbState.ESCAPE_CLIMB.value, current_obj, current_obj, 0)
            return (current_state.value, self.state.climb_start_objective, 
                    self.state.climb_worst_objective, self.state.climb_steps)
        
        elif current_state == ClimbState.ESCAPE_CLIMB:
            new_climb_steps = self.state.climb_steps + 1
            new_worst = max(current_obj, self.state.climb_worst_objective)
            
            # Timeout or significant improvement from start
            timeout = new_climb_steps >= self.config.hill_climb_max_steps
            relative_improvement_from_start = (self.state.climb_start_objective - current_obj) / (abs(self.state.climb_start_objective) + 1e-10)
            significant_improvement = relative_improvement_from_start > 0.1
            
            if timeout or significant_improvement:
                return (ClimbState.NORMAL.value, float('inf'), float('inf'), 0)
            
            return (ClimbState.ESCAPE_CLIMB.value, self.state.climb_start_objective, 
                    new_worst, new_climb_steps)

    # NEW: Helper for MultiObjectivePBit
    def _compute_pareto_weights(self, objectives: List[float]) -> List[float]:
        """Simple weights based on normalized objectives (for Pareto approximation)."""
        norm_objs = [(obj - min(objectives)) / (max(objectives) - min(objectives) + 1e-10) for obj in objectives]
        return [1.0 / (len(objectives) * (norm + 1e-10)) for norm in norm_objs]

    # NEW: Helper for MultiObjectivePBit
    def _update_pareto_front(self, params: jnp.ndarray, objectives: List[float]) -> None:
        """Maintain non-dominated solutions (simple append for demo; use NSGA-II for production)."""
        solution = {'params': params.copy(), 'objectives': objectives.copy()}
        self.pareto_front.append(solution)  # In practice, filter non-dominated

    def step(self, gradient_fn: Callable[[jnp.ndarray], jnp.ndarray],
             objective_fn: Optional[Callable[[jnp.ndarray], float]] = None,
             reset_patience: int = 100) -> Dict[str, Any]:
        """
        Single step with hill-climb state machine and adaptive config update.
        Supports single or multi-objective via overloads in subclasses.
        """
        grad_val = gradient_fn(self.state.params)
        if grad_val.shape != (self.problem_dim,):
            raise ValueError(f"Gradient must be shape {(self.problem_dim,)}")

        objective_value = objective_fn(self.state.params) if callable(objective_fn) else float('inf')
        current_obj = float(objective_value)
        self.objective_history.append(current_obj)  # NEW: Track for adaptation

        # NEW: Adapt config if enabled (every 50 steps for efficiency)
        if self.adaptive_config and self.state.step_count % 50 == 0 and len(self.objective_history) >= 10:
            recent_objs = self.objective_history[-self.adaptive_config.performance_history_size:]
            self.adaptive_config.adapt_based_on_performance(recent_objs)
            # Update self.config reference (mutable via dataclass)
            self.config = self.adaptive_config.base_config

        current_best = float(self.state.best_objective)
        
        # Update climb state machine FIRST
        new_climb_state, new_climb_start, new_climb_worst, new_climb_steps = self._update_climb_state(
            current_obj
        )
        
        # Detect entry into ESCAPE_CLIMB to increment count
        entering_climb = (self.state.climb_state == ClimbState.NORMAL.value and 
                          new_climb_state == ClimbState.ESCAPE_CLIMB.value)
        new_hill_climb_count = self.state.hill_climb_count + 1 if entering_climb else self.state.hill_climb_count
        
        # FIXED: Only count real improvements, not climb steps
        if new_climb_state == ClimbState.ESCAPE_CLIMB.value:
            # During climb, only count improvements if we actually find a better solution
            # Don't reset counters artificially
            has_improved = False  # Changed from True
            new_best_objective = self.state.best_objective
            new_best_params = self.state.best_params
        else:
            # Normal mode: Check real improvement
            has_improved = current_obj < self.state.best_objective - 1e-8
            new_best_objective = current_obj if has_improved else self.state.best_objective
            new_best_params = self.state.params if has_improved else self.state.best_params
        
        # Reset counters on any "improvement" (including climb-mode pseudo-improvement)
        new_steps_since_improvement = 0 if has_improved else self.state.steps_since_improvement + 1
        new_consecutive_stuck = 0 if has_improved else self.state.consecutive_stuck + 1

        # Gradient/momentum/noise - FIXED for climb escape
        grad_val = jnp.nan_to_num(grad_val, nan=0.0, posinf=1e-3, neginf=-1e-3)
        grad_norm = float(jnp.linalg.norm(grad_val))

        self.key, update_key = jax.random.split(self.state.key)
        v_diffs = jnp.abs(grad_val) / 25.0
        key_dims = jax.random.split(update_key, self.problem_dim)
        probs = self._differential_pair_probability_vmap(
            v_diffs, self.config.i_tail, self.config.alpha, self.config.beta, self.config.gamma,
            key_dims, self.noise_power, self.vt_kappa
        )
        
        current_climb_state = ClimbState(new_climb_state)
        is_small_grad = grad_norm < 1e-6
        
        # Key management for climb direction
        dir_key = None
        noise_pre_key = update_key
        if current_climb_state == ClimbState.ESCAPE_CLIMB:
            dir_key, noise_pre_key = jax.random.split(update_key)
        
        # Compute delta_params_det - UPDATED WITH ADAPTIVE LEARNING RATE
        if current_climb_state == ClimbState.ESCAPE_CLIMB:
            # Better escape strategy: Combine gradient opposition with random exploration
            if is_small_grad:
                # Random exploration on flat regions
                rand_dir = jax.random.normal(dir_key, (self.problem_dim,))
                direction = rand_dir / (jnp.linalg.norm(rand_dir) + 1e-10)
            else:
                # Mix of uphill and orthogonal directions for better escape
                grad_dir = grad_val / (grad_norm + 1e-10)
                
                # Add orthogonal component for exploration
                ortho_key, _ = jax.random.split(dir_key)
                ortho = jax.random.normal(ortho_key, (self.problem_dim,))
                ortho = ortho - jnp.dot(ortho, grad_dir) * grad_dir  # Make orthogonal
                ortho_norm = jnp.linalg.norm(ortho) + 1e-10
                ortho = ortho / ortho_norm
                
                # Blend gradient follow and orthogonal exploration
                direction = 0.7 * grad_dir + 0.3 * ortho
                direction = direction / (jnp.linalg.norm(direction) + 1e-10)
            
            # FIXED: Larger step on flat regions for better exploration
            flat_multiplier = 1.5 if is_small_grad else 1.0
            base_step_size = self.config.learning_rate * self.config.hill_climb_reward_scale * flat_multiplier
            delta_params_det = direction * base_step_size
            step_sizes_for_noise = jnp.full((self.problem_dim,), base_step_size)
        else:
            # Normal: Standard descent with adaptive LR
            # NEW: Adaptive learning rate based on gradient and progress
            adaptive_lr = self.config.learning_rate
            if grad_norm < 1e-4 and current_obj < 1.0:
                adaptive_lr = self.config.learning_rate * 0.1  # Reduce when close to convergence
            elif self.state.steps_since_improvement > 100:
                adaptive_lr = self.config.learning_rate * 2.0  # Boost if stuck
            
            step_sizes = probs * adaptive_lr * jnp.abs(grad_val)
            delta_params_det = -step_sizes * jnp.sign(grad_val)
            step_sizes_for_noise = step_sizes

        # Noise (decoupled in climb for better exploration)
        self.key, noise_key_global = jax.random.split(noise_pre_key)
        noise_keys_final = jax.random.split(noise_key_global, self.problem_dim)
        eta_random_vector = self._random_normal_vmap(noise_keys_final)
        eta_noise = eta_random_vector * self.config.gamma * 0.03 * step_sizes_for_noise
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

        # Reset logic - FIXED: Don't reset during climb
        needs_reset = (new_steps_since_improvement > reset_patience and 
                       new_climb_state != ClimbState.ESCAPE_CLIMB.value)
        force_jump = self.config.enable_quantum_jumps and new_consecutive_stuck > self.config.jump_consecutive_stuck_threshold
        if force_jump:
            needs_reset = True
            effective_patience = reset_patience // 2

        reset_info = None
        if needs_reset:
            reset_info = self.state.memory_manager.perform_reset(
                self.state.params, new_best_params, grad_val, new_steps_since_improvement, 
                effective_patience if force_jump else reset_patience
            )
            new_params = reset_info['new_params']
            
            # FIXED: Only reset climb state on major disruptions (e.g., jumps or full restarts)
            strategy = reset_info.get('strategy', '')
            is_major_reset = any(term in strategy.upper() for term in ['RANDOM', 'NUCLEAR', 'RESTART'])
            if is_major_reset:
                new_climb_state = ClimbState.NORMAL.value
                new_climb_start = float('inf')
                new_climb_worst = float('inf')
                new_climb_steps = 0
            # Else: Keep the ongoing climb state (from _update_climb_state)
            
            if reset_info.get('is_jump', False) and self.config.post_jump_momentum_factor > 0:
                jump_dir = new_params - self.state.params
                jump_norm = jnp.linalg.norm(jump_dir) + 1e-10
                momentum_dir = jump_dir / jump_norm * self.config.post_jump_momentum_factor
                new_velocity = jnp.clip(momentum_dir, *self.config.clip_velocity)
            else:
                new_velocity = jnp.zeros(self.problem_dim)
            
            new_consecutive_stuck = 0
            
            self.on_reset(self.state, reset_info)
        # No else needed; use returned climb state vars unless overridden above

        new_state = self.state.replace(
            params=new_params,
            velocity=new_velocity,
            key=noise_key_global,
            step_count=self.state.step_count + 1,
            best_objective=new_best_objective,
            best_params=new_best_params,
            steps_since_improvement=new_steps_since_improvement,
            consecutive_stuck=new_consecutive_stuck,
            climb_state=new_climb_state,
            climb_start_objective=new_climb_start,
            climb_worst_objective=new_climb_worst,
            climb_steps=new_climb_steps,
            hill_climb_count=new_hill_climb_count
        )
        self.state = new_state

        result = {
            'params': new_params,
            'velocity': new_velocity,
            'objective': current_obj,
            'gradient_norm': grad_norm,
            'best_objective': new_best_objective,
            'steps_since_improvement': new_steps_since_improvement,
            'consecutive_stuck': new_consecutive_stuck,
            'min_stuck_distance': float(min_distance),
            'too_close_to_stuck': bool(too_close),
            'avoidance_list_size': int(self.state.memory_manager.stuck_points_ptr[0]),
            'needs_reset': needs_reset,
            'force_jump': force_jump,
            'is_jump': reset_info.get('is_jump', False) if reset_info else False,
            'jump_type': reset_info.get('jump_type') if reset_info else None,
            'step': self.state.step_count,
            'unique_stuck_points': len(self.state.memory_manager.get_unique_stuck_points()),
            # NEW: Hill-climb state info
            'climb_state': ClimbState(new_climb_state).name,
            'climb_steps': new_climb_steps,
            'in_escape_climb': new_climb_state == ClimbState.ESCAPE_CLIMB.value,
            # NEW: Adaptive info
            'adaptive_lr': float(self.config.learning_rate),  # Expose current LR
            'noise_scale': float(self.config.noise_scale),
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
                'adaptive_config': asdict(self.adaptive_config) if self.adaptive_config else None,  # NEW
                'pareto_front': getattr(self, 'pareto_front', []) if hasattr(self, 'pareto_front') else [],  # NEW
                'metrics': {
                    'step_count': self.state.step_count,
                    'best_objective': float(self.state.best_objective),
                    'steps_since_improvement': self.state.steps_since_improvement,
                    'consecutive_stuck': self.state.consecutive_stuck,
                    'unique_stuck_count': len(self.state.memory_manager.get_unique_stuck_points()),
                    'total_jumps': int(np.sum(mr_read['jump_counts'])),
                    'quantum_jumps': int(mr_read['jump_counts'][0]),
                    'nuclear_jumps': int(mr_read['jump_counts'][1]),
                    'hill_climb_count': self.state.hill_climb_count,
                    'climb_state': ClimbState(self.state.climb_state).name,
                    'climb_steps': self.state.climb_steps
                }
            }
        elif what == "metrics":
            mr_read = self.state.memory_manager.read()
            return {
                'step_count': self.state.step_count,
                'best_objective': float(self.state.best_objective),
                'steps_since_improvement': self.state.steps_since_improvement,
                'consecutive_stuck': self.state.consecutive_stuck,
                'unique_stuck_count': len(self.state.memory_manager.get_unique_stuck_points()),
                'total_jumps': int(np.sum(mr_read['jump_counts'])),
                'quantum_jumps': int(mr_read['jump_counts'][0]),
                'nuclear_jumps': int(mr_read['jump_counts'][1]),
                'hill_climb_count': self.state.hill_climb_count,
                'climb_state': ClimbState(self.state.climb_state).name,
                'current_lr': float(self.config.learning_rate)  # NEW: Expose adaptive LR
            }
        elif what == "params":
            return {'params': np.array(self.state.params)}
        elif what == "best":
            return {
                'best_params': np.array(self.state.best_params),
                'best_objective': float(self.state.best_objective)
            }
        else:
            raise ValueError(f"Unknown read mode: {what}")

    @classmethod
    def create_ensemble(cls, num_instances: int, problem_dim: int, shared_memory: Optional[SARMemoryManager] = None,
                        base_config: Optional[PBitConfig] = None, seeds: Optional[List[int]] = None,
                        initial_params_list: Optional[List[jnp.ndarray]] = None, 
                        enable_jumps: bool = False, enable_hill_climb: bool = True,
                        use_adaptive: bool = False) -> List['PathBasedPBit']:  # NEW: Adaptive support
        """Factory for creating ensemble with shared memory."""
        if shared_memory is None:
            shared_memory = SARMemoryManager(
                problem_dim, 
                SARConfig(enable_jumps=enable_jumps)
            )
        base = base_config or PBitConfig(
            enable_quantum_jumps=enable_jumps, 
            enable_hill_climb=enable_hill_climb
        )
        seeds = seeds or list(range(num_instances))
        initial_params_list = initial_params_list or [None] * num_instances

        def create_config_with_seed(base: PBitConfig, seed: int, jumps: bool, climb: bool) -> PBitConfig:
            data = asdict(base)
            data['seed'] = seed
            data['enable_quantum_jumps'] = jumps
            data['enable_hill_climb'] = climb
            return PBitConfig(**data)

        configs = [create_config_with_seed(base, seeds[i], enable_jumps, enable_hill_climb) for i in range(num_instances)]
        if use_adaptive:
            configs = [AdaptiveConfig(c) for c in configs]

        return [
            cls(problem_dim, configs[i], initial_params_list[i], shared_memory)
            for i in range(num_instances)
        ]

# NEW: User-Provided Enhancement 2 - MultiObjectivePBit
class MultiObjectivePBit(PathBasedPBit):
    """Extension for multi-objective optimization."""
    
    def __init__(self, problem_dim: int, num_objectives: int, config: PBitConfig = None, **kwargs):
        super().__init__(problem_dim, config, **kwargs)
        self.num_objectives = num_objectives
        self.pareto_front: List[Dict] = []  # Store non-dominated solutions
    
    def step(self, gradient_fns: List[Callable], objective_fns: List[Callable], reset_patience: int = 100):
        """Handle multiple objectives with Pareto dominance."""
        if len(gradient_fns) != self.num_objectives or len(objective_fns) != self.num_objectives:
            raise ValueError("Number of gradient/objective fns must match num_objectives")

        # Compute gradients and objectives for all
        grads = [grad_fn(self.state.params) for grad_fn in gradient_fns]
        objs = [obj_fn(self.state.params) for obj_fn in objective_fns]
        
        # Weighted combination (simple mean for demo; use epsilon-constraint in prod)
        weights = self._compute_pareto_weights(objs)
        combined_grad = sum(w * g for w, g in zip(weights, grads))
        combined_obj = np.mean(objs)
        
        # Update Pareto front
        self._update_pareto_front(self.state.params, objs)
        
        # Call base step with combined
        def combined_grad_fn(params): return combined_grad  # Static for this step
        def combined_obj_fn(params): return combined_obj
        return super().step(combined_grad_fn, combined_obj_fn, reset_patience)

    def read(self, what: str = "all") -> Dict[str, Any]:
        base_read = super().read(what)
        if what in ["all", "metrics"]:
            base_read['pareto_front_size'] = len(self.pareto_front)
            base_read['pareto_front'] = self.pareto_front if what == "all" else len(self.pareto_front)
        return base_read

# NEW: User-Provided Enhancement 4 - GradientFreePBit
class GradientFreePBit(PathBasedPBit):
    """Variant that doesn't require gradient information."""
    
    def step(self, objective_fn: Callable, num_samples: int = 10, reset_patience: int = 100):
        """Use finite differences or random sampling instead of gradients."""
        current_obj = float(objective_fn(self.state.params))
        self.objective_history.append(current_obj)  # Track for adaptive

        # Sample directions (unit vectors)
        self.key, sample_key = jax.random.split(self.key)
        directions = jax.random.normal(sample_key, (num_samples, self.problem_dim))
        norms = jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-10
        directions = directions / norms

        # Evaluate trials (vectorized for efficiency)
        trial_params = self.state.params + 0.1 * jnp.expand_dims(jnp.arange(num_samples), 1) * directions  # Vary step size slightly
        trial_objs = jax.vmap(objective_fn)(trial_params)
        improvements = current_obj - trial_objs  # Positive = better

        # Best direction as pseudo-gradient
        best_idx = jnp.argmax(improvements)
        best_direction = directions[best_idx]
        best_improvement = improvements[best_idx]

        def pseudo_gradient_fn(params):
            return -best_direction * float(best_improvement)  # Downhill pseudo-grad

        # Proceed with base step
        return super().step(pseudo_gradient_fn, objective_fn, reset_patience)

# NEW: Factory for Enhancements
def create_adaptive_pbit(problem_dim: int, base_config: PBitConfig, **kwargs) -> PathBasedPBit:
    """Create PBit with adaptive config."""
    adaptive = AdaptiveConfig(base_config)
    return PathBasedPBit(problem_dim, adaptive_config=adaptive, **kwargs)

def create_multiobjective_pbit(problem_dim: int, num_objectives: int, config: PBitConfig = None, **kwargs) -> MultiObjectivePBit:
    """Create multi-objective PBit."""
    return MultiObjectivePBit(problem_dim, num_objectives, config, **kwargs)

def create_gradientfree_pbit(problem_dim: int, config: PBitConfig = None, **kwargs) -> GradientFreePBit:
    """Create gradient-free PBit."""
    return GradientFreePBit(problem_dim, config, **kwargs)

# Test with hill-climb demo (unchanged)
def test_hill_climb_escape():
    print("\n🧗 Testing Hill-Climb Escape with Overshoot Detection")
    
    # Create a challenging landscape with local minimum
    @jax.jit
    def double_well(params):
        """Two wells with barrier - perfect for hill-climb testing."""
        x, y = params[0], params[1]
        # Local minimum at (1, 1), global at (-1, -1), barrier in between
        local_well = ((x - 1)**2 + (y - 1)**2) * 2.0
        global_well = ((x + 1)**2 + (y + 1)**2) * 0.5  # Make global slightly better
        barrier = jnp.exp(-((x**2 + y**2) / 0.5)) * 3.0
        return jnp.minimum(local_well, global_well) + barrier
    
    double_well_grad = jax.jit(jax.grad(double_well))
    
    config = PBitConfig(
        enable_hill_climb=True,
        hill_climb_trigger_stuck=20,
        hill_climb_max_steps=15,
        hill_climb_improvement_threshold=0.001,  # Lowered for similar mins
        hill_climb_reward_scale=2.0,
        momentum_decay_on_stuck=0.3
    )
    
    # Start in local minimum
    pb = PathBasedPBit(problem_dim=2, config=config, initial_params=jnp.array([0.9, 0.9]))
    
    print(f"Starting at params: {pb.state.params}, objective: {double_well(pb.state.params):.4f}")
    print(f"Local minimum: ~(1,1), Global minimum: ~(-1,-1)")
    print()
    best_step = 0
    for step in range(2800):
        result = pb.step(double_well_grad, double_well, reset_patience=40)
        if result['objective'] < result['best_objective']:
            best_step = step
        # Print on key events
        if (result['in_escape_climb'] or 
            result['climb_state'] != 'NORMAL' or step % 100 == 0):
            state_emoji = "🧗" if result['in_escape_climb'] else "✅"
            print(f"{step:3d} {state_emoji} | Pos: ({result['params'][0]:5.2f}, {result['params'][1]:5.2f}) | "
                  f"Obj: {result['objective']:6.3f} | Best: {result['best_objective']:6.3f} | "
                  f"State: {result['climb_state']:15s} | Steps: {result['climb_steps']:2d} | LR: {result['adaptive_lr']:.4f}")
        
        # Check if we found global minimum (adjusted for similar values)
        #if result['climb_state'] == 'NORMAL' and result['params'][0] < -0.5 and abs(result['params'][1] + 1.0) < 0.5:
        #print(f"\n🎉 Found global minimum at step {step}!")
        #break
    
    final = pb.read("metrics")
    print(f"\n📊 Final: Best={final['best_objective']:.4f} at step {best_step}, "
          f"HillClimbs={final['hill_climb_count']}, Steps={final['step_count']}, Final LR={final['current_lr']:.4f}")

# NEW: Quick test for enhancements
def test_enhancements():
    print("\n🔧 Testing Enhancements")
    
    # Test AdaptiveConfig
    base_conf = PBitConfig(learning_rate=0.01, noise_scale=0.1)
    adapt_conf = AdaptiveConfig(base_conf)
    dummy_objs = [1.0, 0.9, 0.8, 1.1, 0.7]  # Mixed progress
    adapt_conf.adapt_based_on_performance(dummy_objs)
    print(f"Adaptive: LR={adapt_conf.base_config.learning_rate:.4f}, Noise={adapt_conf.base_config.noise_scale:.4f}")
    
    # Test MultiObjectivePBit (dummy multi-obj)
    def obj1(params): return jnp.sum(params**2)
    def obj2(params): return jnp.sum((params - 1)**2)
    grad1 = jax.grad(obj1)
    grad2 = jax.grad(obj2)
    multi_pb = create_multiobjective_pbit(2, 2)
    result = multi_pb.step([grad1, grad2], [obj1, obj2])
    print(f"Multi-Obj: Best Obj={result['best_objective']:.4f}, Pareto Size={len(multi_pb.pareto_front)}")
    
    # Test GradientFreePBit (no grads)
    def blackbox_obj(params): return jnp.sum(params**2) + 0.1 * jnp.random.normal()
    free_pb = create_gradientfree_pbit(2)
    result = free_pb.step(blackbox_obj, num_samples=5)
    print(f"Gradient-Free: Best Obj={result['best_objective']:.4f}")

if __name__ == "__main__":
    test_hill_climb_escape()
    test_enhancements()
