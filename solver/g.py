import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Optional, Callable
import sys

# Import from w.py (lightweight NuclearPBitWrapper)
try:
    from w import NuclearPBitWrapper, NuclearWrapperConfig
    from pb import PathBasedPBit, PBitConfig
    from mr import SARMemoryManager, SARConfig
except ImportError:
    print("Error: w.py, pb.py, or mr.py not found. Please ensure they are in the same directory.")
    sys.exit(1)

# Define test functions
@jax.jit
def rastrigin(params):
    """Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))"""
    n = params.shape[0]
    return 10 * n + jnp.sum(params**2 - 10 * jnp.cos(2 * jnp.pi * params))

rastrigin_grad = jax.jit(jax.grad(rastrigin))


def test_rastrigin_2d():
    print("\n" + "="*100)
    print("🎯 TEST 1: 2D Rastrigin (NuclearPBitWrapper from w.py)")
    print("="*100)
    print("Global minimum: 0 at (0,0)")
    
    # Setup: Create PBit with native jumps, then wrap with Nuclear
    sar_config = SARConfig(
        spf_depth=8,
        avoidance_threshold=0.4,
        avoidance_strength=1.0,
        enable_jumps=True,
        quantum_jump_range=5.0,
        nuclear_reset_strength=2.0,
        seed=42
    )
    
    pbit_config = PBitConfig(
        momentum_decay_on_stuck=0.3,
        learning_rate=0.005,
        noise_scale=0.15,
        enable_quantum_jumps=True,
        jump_consecutive_stuck_threshold=30,
        seed=42
    )
    
    memory_manager = SARMemoryManager(problem_dim=2, config=sar_config)
    pbit = PathBasedPBit(
        problem_dim=2,
        config=pbit_config,
        initial_params=jnp.array([1.0, -0.5]),
        memory_manager=memory_manager
    )
    
    wrapper_config = NuclearWrapperConfig(
        enable=True,
        stuck_threshold=30,
        manual_nuke_strength=1.5,
        verbose=False
    )
    
    wrapper = NuclearPBitWrapper(pbit, wrapper_config)
    
    print(f"Initial objective: {rastrigin(wrapper.pbit.state.params):.4f}")
    print("\nStep | Objective | Best Obj | Params | Stuck Steps | Nuclear | Velocity")
    print("-" * 100)
    
    max_steps = 2000
    target_obj = 0.0001
    
    for step in range(max_steps):
        result = wrapper.step(rastrigin_grad, rastrigin, reset_patience=50)
        
        if step % 20 == 0:
            params = result['params']
            velocity_norm = float(jnp.linalg.norm(result['velocity']))
            nuclear_marker = "💥" if result.get('nuclear_triggered', False) else "  "
            
            print(f"{step:4d} | {result['objective']:9.4f} | {result['best_objective']:9.4f} | "
                  f"({params[0]:6.2f}, {params[1]:6.2f}) | {result['steps_since_improvement']:11d} | "
                  f"{nuclear_marker:3} | {velocity_norm:8.4f}")
        
        if result['best_objective'] < target_obj:
            print(f"\n🎯 Target reached at step {step}!")
            break
    
    final_stats = wrapper.read()
    print(f"\n📊 Final Results:")
    print(f"  Best objective: {final_stats['metrics']['best_objective']:.6e}")
    print(f"  Distance to (0,0): {jnp.linalg.norm(final_stats['best_params']):.6e}")
    print(f"  Total steps: {final_stats['metrics']['step_count']}")
    print(f"  Nuclear triggers: {final_stats['nuclear_stats']['triggers']}")
    print(f"  Quantum jumps: {final_stats['metrics']['quantum_jumps']}")
    print(f"  Nuclear jumps: {final_stats['metrics']['nuclear_jumps']}")


def test_rastrigin_10d():
    print("\n" + "="*100)
    print("🎯 TEST 2: 10D Rastrigin (NuclearPBitWrapper)")
    print("="*100)
    
    sar_config = SARConfig(
        spf_depth=12,
        avoidance_threshold=0.6,
        avoidance_strength=1.2,
        enable_jumps=True,
        quantum_jump_range=6.0,
        nuclear_reset_strength=2.5,
        seed=42
    )
    
    pbit_config = PBitConfig(
        momentum_decay_on_stuck=0.4,
        learning_rate=0.003,
        noise_scale=0.2,
        enable_quantum_jumps=True,
        jump_consecutive_stuck_threshold=50,
        seed=42
    )
    
    memory_manager = SARMemoryManager(problem_dim=10, config=sar_config)
    pbit = PathBasedPBit(
        problem_dim=10,
        config=pbit_config,
        memory_manager=memory_manager
    )
    
    wrapper_config = NuclearWrapperConfig(
        enable=True,
        stuck_threshold=50,
        manual_nuke_strength=1.8,
        verbose=False
    )
    
    wrapper = NuclearPBitWrapper(pbit, wrapper_config)
    
    print(f"Initial objective: {rastrigin(wrapper.pbit.state.params):.4f}")
    print("\nStep | Objective | Best Obj | Param Mean | Param Std | Stuck Steps | Nuclear")
    print("-" * 100)
    
    max_steps = 5000
    target_obj = 1.0
    
    for step in range(max_steps):
        result = wrapper.step(rastrigin_grad, rastrigin, reset_patience=80)
        
        if step % 100 == 0:
            params = result['params']
            param_mean = float(jnp.mean(params))
            param_std = float(jnp.std(params))
            nuclear_marker = "💥" if result.get('nuclear_triggered', False) else "  "
            
            print(f"{step:4d} | {result['objective']:9.4f} | {result['best_objective']:9.4f} | "
                  f"{param_mean:10.2f} | {param_std:9.2f} | {result['steps_since_improvement']:11d} | {nuclear_marker:3}")
        
        if result['best_objective'] < target_obj:
            print(f"\n🎯 Target reached at step {step}!")
            break
    
    final_stats = wrapper.read()
    print(f"\n📊 Final Results:")
    print(f"  Best objective: {final_stats['metrics']['best_objective']:.6e}")
    print(f"  Distance to origin: {jnp.linalg.norm(final_stats['best_params']):.6e}")
    print(f"  Total steps: {final_stats['metrics']['step_count']}")
    print(f"  Nuclear triggers: {final_stats['nuclear_stats']['triggers']}")
    print(f"  Total jumps: {final_stats['metrics']['total_jumps']}")


def test_rastrigin_50d():
    print("\n" + "="*100)
    print("🎯 TEST 3: 50D Rastrigin (NuclearPBitWrapper)")
    print("="*100)
    
    sar_config = SARConfig(
        spf_depth=20,
        avoidance_threshold=0.8,
        avoidance_strength=1.5,
        enable_jumps=True,
        quantum_jump_range=8.0,
        nuclear_reset_strength=3.0,
        seed=42
    )
    
    pbit_config = PBitConfig(
        momentum_decay_on_stuck=0.5,
        learning_rate=0.002,
        noise_scale=0.25,
        enable_quantum_jumps=True,
        jump_consecutive_stuck_threshold=80,
        seed=42
    )
    
    memory_manager = SARMemoryManager(problem_dim=50, config=sar_config)
    pbit = PathBasedPBit(
        problem_dim=50,
        config=pbit_config,
        memory_manager=memory_manager
    )
    
    wrapper_config = NuclearWrapperConfig(
        enable=True,
        stuck_threshold=80,
        manual_nuke_strength=2.0,
        verbose=False
    )
    
    wrapper = NuclearPBitWrapper(pbit, wrapper_config)
    
    print(f"Initial objective: {rastrigin(wrapper.pbit.state.params):.4f}")
    print("\nStep | Objective | Best Obj | Param Mean | Param Std | Stuck Steps | Nuclear")
    print("-" * 100)
    
    max_steps = 10000
    target_obj = 1.0
    
    for step in range(max_steps):
        result = wrapper.step(rastrigin_grad, rastrigin, reset_patience=120)
        
        if step % 200 == 0:
            params = result['params']
            param_mean = float(jnp.mean(params))
            param_std = float(jnp.std(params))
            nuclear_marker = "💥" if result.get('nuclear_triggered', False) else "  "
            
            print(f"{step:4d} | {result['objective']:9.4f} | {result['best_objective']:9.4f} | "
                  f"{param_mean:10.2f} | {param_std:9.2f} | {result['steps_since_improvement']:11d} | {nuclear_marker:3}")
        
        if result['best_objective'] < target_obj:
            print(f"\n🎯 Target reached at step {step}!")
            break
    
    final_stats = wrapper.read()
    print(f"\n📊 Final Results:")
    print(f"  Best objective: {final_stats['metrics']['best_objective']:.6e}")
    print(f"  Distance to origin: {jnp.linalg.norm(final_stats['best_params']):.6e}")
    print(f"  Total steps: {final_stats['metrics']['step_count']}")
    print(f"  Nuclear triggers: {final_stats['nuclear_stats']['triggers']}")
    print(f"  Total jumps: {final_stats['metrics']['total_jumps']}")


def test_multi_trap():
    """Test multi-trap function using the Nuclear wrapper."""
    print("\n" + "="*100)
    print("🛡️ TEST 4: Multi-Trap Function (Nuclear Wrapper)")
    print("="*100)
    
    # Define multi-trap function
    @jax.jit
    def multi_trap_function(params):
        """Multi-trap function with three deep traps and one global minimum."""
        n = params.shape[0]
        trap1_center = jnp.ones(n) * 2.0
        trap2_center = jnp.ones(n) * -1.0
        trap3_center = jnp.ones(n) * 0.5
        global_center = jnp.ones(n) * -3.0
        
        trap1_depth = 50.0 * jnp.exp(-jnp.linalg.norm(params - trap1_center) ** 2 / 2.0)
        trap2_depth = 40.0 * jnp.exp(-jnp.linalg.norm(params - trap2_center) ** 2 / 2.0)
        trap3_depth = 30.0 * jnp.exp(-jnp.linalg.norm(params - trap3_center) ** 2 / 2.0)
        global_value = 5.0 * jnp.linalg.norm(params - global_center) ** 2
        
        return trap1_depth + trap2_depth + trap3_depth + global_value

    multi_trap_grad = jax.jit(jax.grad(multi_trap_function))
    
    sar_config = SARConfig(
        spf_depth=10,
        avoidance_threshold=0.8,
        avoidance_strength=2.0,
        enable_jumps=True,
        quantum_jump_range=6.0,
        nuclear_reset_strength=3.0,
        seed=42
    )
    
    pbit_config = PBitConfig(
        momentum_decay_on_stuck=0.2,
        learning_rate=0.008,
        noise_scale=0.3,
        enable_quantum_jumps=True,
        jump_consecutive_stuck_threshold=20,
        seed=42
    )
    
    memory_manager = SARMemoryManager(problem_dim=6, config=sar_config)
    pbit = PathBasedPBit(
        problem_dim=6,
        config=pbit_config,
        initial_params=jnp.ones(6) * 2.0,  # Start in Trap 1
        memory_manager=memory_manager
    )
    
    wrapper_config = NuclearWrapperConfig(
        enable=True,
        stuck_threshold=20,
        manual_nuke_strength=2.0,
        verbose=False
    )
    
    wrapper = NuclearPBitWrapper(pbit, wrapper_config)
    
    print(f"Starting position: Trap 1 (near [2.0, 2.0, ...])")
    print(f"Initial objective: {multi_trap_function(wrapper.pbit.state.params):.2f}")
    
    print("\nStep | Objective | Best | Stuck Steps | Nuclear | Escapes")
    print("-" * 80)
    
    max_steps = 3000
    target_obj = 10.0
    
    for step in range(max_steps):
        result = wrapper.step(multi_trap_grad, multi_trap_function, reset_patience=40)
        
        if step % 150 == 0:
            nuclear_marker = "💥" if result.get('nuclear_triggered', False) else "  "
            
            print(f"{step:4d} | {result['objective']:9.2f} | {result['best_objective']:6.2f} | "
                  f"{result['steps_since_improvement']:11d} | {nuclear_marker:3} | "
                  f"{wrapper.nuclear_triggers:7d}")
        
        if result['best_objective'] < target_obj:
            print(f"\n🎯 Target reached at step {step}!")
            break
    
    # Check if reached global minimum
    final_params = wrapper.pbit.state.best_params
    norm_to_global = float(jnp.linalg.norm(final_params + 3.0))
    reached_global = norm_to_global < 2.0 and result['best_objective'] < 10.0
    
    if reached_global:
        print(f"\n🎉 SUCCESS! Reached global minimum region!")
        print(f"   Final objective: {result['best_objective']:.6f}")
        print(f"   Distance to global: {norm_to_global:.4f}")
    else:
        print(f"\n⏰ Multi-Trap test completed")
        print(f"   Best objective: {result['best_objective']:.2f}")
        print(f"   Distance to global: {norm_to_global:.4f}")
    
    final_stats = wrapper.read()
    print(f"\n📊 Final Multi-Trap Results:")
    print(f"  Best objective: {final_stats['metrics']['best_objective']:.6f}")
    print(f"  Total steps: {final_stats['metrics']['step_count']}")
    print(f"  Nuclear triggers: {final_stats['nuclear_stats']['triggers']}")
    print(f"  Total jumps: {final_stats['metrics']['total_jumps']}")


def test_nuclear_manual():
    """Test manual nuclear trigger using nuke() method."""
    print("\n" + "="*100)
    print("💥 TEST 5: Manual Nuclear Trigger")
    print("="*100)
    
    sar_config = SARConfig(
        spf_depth=10,
        enable_jumps=True,
        quantum_jump_range=5.0,
        nuclear_reset_strength=3.0,
        seed=42
    )
    
    pbit_config = PBitConfig(
        momentum_decay_on_stuck=0.3,
        learning_rate=0.01,
        enable_quantum_jumps=True,
        seed=42
    )
    
    memory_manager = SARMemoryManager(problem_dim=2, config=sar_config)
    pbit = PathBasedPBit(
        problem_dim=2,
        config=pbit_config,
        initial_params=jnp.array([2.5, 2.5]),
        memory_manager=memory_manager
    )
    
    wrapper_config = NuclearWrapperConfig(
        enable=True,
        stuck_threshold=100,  # High threshold to test manual trigger
        manual_nuke_strength=2.0,
        verbose=True
    )
    
    wrapper = NuclearPBitWrapper(pbit, wrapper_config)
    
    print(f"Initial position: [{wrapper.pbit.state.params[0]:.1f}, {wrapper.pbit.state.params[1]:.1f}]")
    print(f"Initial objective: {rastrigin(wrapper.pbit.state.params):.2f}")
    
    # Run for a bit
    for step in range(50):
        result = wrapper.step(rastrigin_grad, rastrigin, reset_patience=200)
    
    print(f"\nAfter 50 steps:")
    print(f"  Position: [{wrapper.pbit.state.params[0]:.2f}, {wrapper.pbit.state.params[1]:.2f}]")
    print(f"  Objective: {rastrigin(wrapper.pbit.state.params):.4f}")
    
    # Manual nuclear trigger
    print("\n💥 Triggering manual nuclear jump...")
    nuke_result = wrapper.nuke(rastrigin_grad)
    
    print(f"  Displacement: {nuke_result['displacement']:.2f}")
    print(f"  New position: [{nuke_result['new_params'][0]:.2f}, {nuke_result['new_params'][1]:.2f}]")
    print(f"  New objective: {rastrigin(nuke_result['new_params']):.4f}")
    
    # Continue optimization
    for step in range(50, 150):
        result = wrapper.step(rastrigin_grad, rastrigin, reset_patience=200)
        
        if result['best_objective'] < 0.01:
            break
    
    final_stats = wrapper.read()
    print(f"\n📊 Final Results:")
    print(f"  Best objective: {final_stats['metrics']['best_objective']:.6f}")
    print(f"  Total steps: {final_stats['metrics']['step_count']}")
    print(f"  Nuclear triggers (total): {final_stats['nuclear_stats']['triggers']}")
    print(f"  Last nuclear step: {final_stats['nuclear_stats']['last_step']}")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("🚀 NuclearPBitWrapper Test Suite (using w.py)")
    print("   Features: Lightweight nuclear wrapper with zero overhead")
    print("="*100)
    
    test_rastrigin_2d()
    test_rastrigin_10d() 
    test_rastrigin_50d()
    test_multi_trap()
    test_nuclear_manual()
    
    print("\n" + "="*100)
    print("✅ All Nuclear wrapper tests completed!")
    print("="*100)
