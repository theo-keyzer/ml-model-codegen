"""
Hardware-Accurate Enhanced Differential Pair P-bit Portfolio Optimization
============================================================================
Fixed version with JAX-compatible string handling for the enhanced differential pair circuit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Callable

# Set global JAX configuration
jax.config.update("jax_enable_x64", True)

# =======================================================================
# 1. HARDWARE-ACCURATE DIFFERENTIAL PAIR P-BIT MODEL
# =======================================================================

# Physical Constants (from the paper)
V_T = 25.85e-3  # Thermal voltage (kT/q at 300K)
KAPPA = 0.85    # Subthreshold slope factor
BOLTZMANN = 1.3806e-23
TEMPERATURE = 300
GAMMA_NOISE = 2/3  # MOSFET noise coefficient

@jax.jit
def differential_pair_probability(v_diff: jnp.ndarray, i_tail: jnp.ndarray, 
                                alpha: float, beta: float, gamma: float,
                                key: jnp.ndarray) -> jnp.ndarray:
    """
    Implements the actual differential pair circuit equations from the paper:
    P(1) = 1 / (1 + exp(-(α·V_diff + β·I_adaptive + γ·η)/V_T))
    
    This is the hardware-accurate version of Equation (3) from the paper.
    """
    # Generate physical thermal noise (η)
    key, noise_key = jax.random.split(key)
    transconductance = 100e-6  # Typical MOSFET gm
    bandwidth = 50e6          # Circuit bandwidth
    noise_power = 4 * BOLTZMANN * TEMPERATURE * GAMMA_NOISE * transconductance * bandwidth
    eta = jax.random.normal(noise_key, v_diff.shape) * jnp.sqrt(noise_power)
    
    # Circuit-accurate effective voltage (Equation 3 from paper)
    v_eff = (alpha * v_diff) + (beta * i_tail) + (gamma * eta)
    
    # Physical sigmoid response from differential pair
    prob_1 = 1.0 / (1.0 + jnp.exp(-v_eff / (KAPPA * V_T)))
    
    return prob_1, key

# Separate functions for different modes (JAX compatible)
@jax.jit
def programmable_current_continuous(step: int, total_steps: int) -> Tuple[float, float, float]:
    """Continuous optimization mode current sources."""
    T = 1.0 - (step / total_steps)
    I_0 = 1e-4  # Base bias current
    I_adaptive = 5e-5 * T  # Decreasing adaptive current
    noise_scale = 0.1 + 0.9 * (1 - T)  # Decreasing noise
    return I_0, I_adaptive, noise_scale

@jax.jit
def programmable_current_combinatorial(step: int, total_steps: int) -> Tuple[float, float, float]:
    """Combinatorial search mode current sources."""
    T = 1.0 - (step / total_steps)
    I_0 = 2e-4
    I_adaptive = 1e-4 * (1 - T**2)  # Non-linear decay
    noise_scale = 0.3 + 0.7 * (1 - T)
    return I_0, I_adaptive, noise_scale

@jax.jit
def programmable_current_bayesian(step: int, total_steps: int) -> Tuple[float, float, float]:
    """Bayesian inference mode current sources."""
    T = 1.0 - (step / total_steps)
    I_0 = 5e-5
    I_adaptive = 2e-5 * T
    noise_scale = 0.05 + 0.1 * (1 - T)
    return I_0, I_adaptive, noise_scale

@jax.jit
def programmable_current_portfolio(step: int, total_steps: int) -> Tuple[float, float, float]:
    """Portfolio optimization mode current sources."""
    T = 1.0 - (step / total_steps)
    I_0 = 1.5e-4
    I_adaptive = 8e-5 * jnp.sqrt(T)  # Square root decay for smoother transition
    noise_scale = 0.2 + 0.6 * (1 - T)
    return I_0, I_adaptive, noise_scale

@jax.jit
def multi_state_probability(v_diffs: jnp.ndarray, i_tails: jnp.ndarray,
                          alpha: float, beta: float, gamma: float,
                          key: jnp.ndarray) -> jnp.ndarray:
    """
    Implements multi-state probability computation from the enhanced architecture:
    P(k) = I_k / ΣI_j for k = 1,2,...,N
    
    This extends the basic differential pair to multiple outputs.
    """
    N = v_diffs.shape[0]
    probabilities = jnp.zeros(N)
    total_current = 0.0
    
    # Compute each branch current (simulating multiple differential pairs)
    for k in range(N):
        prob_k, key = differential_pair_probability(v_diffs[k], i_tails[k], 
                                                  alpha, beta, gamma, key)
        # Convert probability to current (I_k ∝ P(k))
        current_k = prob_k * i_tails[k]
        probabilities = probabilities.at[k].set(current_k)
        total_current += current_k
    
    # Normalize to get proper probabilities (Equation for multi-state)
    probabilities = probabilities / (total_current + 1e-12)
    
    return probabilities, key

# =======================================================================
# 2. PORTFOLIO OPTIMIZATION WITH HARDWARE-ACCURATE P-BITS
# =======================================================================

@jax.jit
def compute_portfolio_gradient(weights: jnp.ndarray, mu: jnp.ndarray, Sigma: jnp.ndarray,
                             risk_aversion: float, cost_multiplier: float,
                             initial_weights: jnp.ndarray, transaction_cost: jnp.ndarray) -> jnp.ndarray:
    """Gradient computation for portfolio optimization."""
    return_component = mu
    risk_component = 2 * jnp.dot(Sigma, weights) * risk_aversion
    weight_diff = weights - initial_weights
    cost_component = cost_multiplier * transaction_cost * jnp.sign(weight_diff)
    
    gradient = return_component - risk_component - cost_component
    return gradient

@jax.jit
def portfolio_objective(weights: jnp.ndarray, mu: jnp.ndarray, Sigma: jnp.ndarray,
                       risk_aversion: float, cost_multiplier: float,
                       initial_weights: jnp.ndarray, transaction_cost: jnp.ndarray) -> jnp.ndarray:
    """Portfolio objective function."""
    expected_return = jnp.dot(weights, mu)
    portfolio_variance = jnp.dot(weights.T, jnp.dot(Sigma, weights))
    risk_penalty = risk_aversion * portfolio_variance
    cost_penalty = cost_multiplier * jnp.sum(jnp.abs(weights - initial_weights) * transaction_cost)
    
    return expected_return - risk_penalty - cost_penalty

def create_portfolio_data(n_assets: int, key: jnp.ndarray):
    """Create realistic portfolio data."""
    key, mu_key, corr_key = jax.random.split(key, 3)
    
    mu = jax.random.uniform(mu_key, (n_assets,), minval=0.02, maxval=0.20)
    A = jax.random.normal(corr_key, (n_assets, n_assets)) * 0.3
    Sigma = jnp.dot(A, A.T) + jnp.eye(n_assets) * 0.01
    
    initial_weights = jnp.ones(n_assets) / n_assets
    transaction_cost = jnp.linspace(0.001, 0.01, n_assets)
    
    return mu, Sigma, initial_weights, transaction_cost

def run_hardware_accurate_optimization(n_assets: int, n_steps: int, key: jnp.ndarray):
    """Run optimization using hardware-accurate differential pair P-bits."""
    # Create portfolio data
    mu, Sigma, initial_weights, transaction_cost = create_portfolio_data(n_assets, key)
    risk_aversion = 0.3
    cost_multiplier = 0.5
    
    # Initialize weights
    key, init_key = jax.random.split(key)
    weights = initial_weights + jax.random.normal(init_key, (n_assets,)) * 0.01
    weights = jnp.clip(weights, 0.01, 0.35)
    weights = weights / jnp.sum(weights)
    
    history = []
    parameter_history = []
    
    print("Running hardware-accurate differential pair P-bit optimization...")
    print("Using enhanced architecture with programmable current sources")
    print("=" * 60)
    
    for step in range(n_steps):
        # Compute gradient (local field)
        grad = compute_portfolio_gradient(weights, mu, Sigma, risk_aversion,
                                        cost_multiplier, initial_weights, transaction_cost)
        
        # Normalize gradient for circuit compatibility
        grad_normalized = grad / (jnp.linalg.norm(grad) + 1e-8)
        
        # Get programmable current settings (from enhanced circuit)
        I_0, I_adaptive, noise_scale = programmable_current_portfolio(step, n_steps)
        
        # Circuit parameters (α, β, γ) from enhanced architecture
        T = 1.0 - (step / n_steps)
        alpha = 2.0 + 3.0 * T  # Programmable gain (V_diff scaling)
        beta = 1.0 + 2.0 * T   # Adaptive exploration coefficient  
        gamma = noise_scale     # Noise modulation factor
        
        # Update each weight using hardware-accurate differential pair
        new_weights = weights.copy()
        weight_updates = jnp.zeros(n_assets)
        
        for i in range(n_assets):
            key, prob_key = jax.random.split(key)
            
            # Use actual differential pair circuit equation
            prob_increase, _ = differential_pair_probability(
                v_diff=grad_normalized[i],
                i_tail=I_adaptive,
                alpha=alpha,
                beta=beta, 
                gamma=gamma,
                key=prob_key
            )
            
            # Convert probability to weight update (circuit-inspired)
            update_magnitude = I_0 * (1.0 + 2.0 * T)  # Larger steps early
            if jax.random.uniform(prob_key) < prob_increase:
                weight_updates = weight_updates.at[i].set(update_magnitude)
            else:
                weight_updates = weight_updates.at[i].set(-update_magnitude)
        
        # Apply updates
        new_weights = weights + weight_updates * jnp.abs(grad_normalized)
        
        # Circuit-inspired normalization (maintains physical consistency)
        min_w, max_w = 0.01, 0.35
        new_weights = jnp.clip(new_weights, min_w, max_w)
        weights = new_weights / jnp.sum(new_weights)
        
        # Record history
        obj = portfolio_objective(weights, mu, Sigma, risk_aversion,
                                cost_multiplier, initial_weights, transaction_cost)
        history.append(obj)
        parameter_history.append((alpha, beta, gamma, I_0, I_adaptive))
        
        if step % 500 == 0:
            print(f"Step {step}: Obj={obj:.6f}, α={alpha:.2f}, β={beta:.2f}, γ={gamma:.3f}")
    
    return (weights, jnp.array(history), jnp.array(parameter_history), 
            mu, Sigma, initial_weights, transaction_cost)

# =======================================================================
# 3. ANALYSIS AND VISUALIZATION
# =======================================================================

def analyze_circuit_performance(final_weights, history, parameter_history, mu, Sigma):
    """Analyze performance with circuit-specific metrics."""
    # Basic portfolio metrics
    expected_return = jnp.dot(final_weights, mu)
    portfolio_variance = jnp.dot(final_weights.T, jnp.dot(Sigma, final_weights))
    sharpe_ratio = expected_return / jnp.sqrt(portfolio_variance + 1e-8)
    
    # Circuit-specific analysis
    alphas, betas, gammas, I_0s, I_adaptives = jnp.array(parameter_history).T
    
    print("\n" + "="*70)
    print("HARDWARE-ACCURATE DIFFERENTIAL PAIR P-BIT RESULTS")
    print("="*70)
    print(f"Final Expected Return: {expected_return:.4f}")
    print(f"Portfolio Variance: {portfolio_variance:.6f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    print(f"\nCircuit Parameter Ranges:")
    print(f"  α (Gain): {jnp.min(alphas):.2f} to {jnp.max(alphas):.2f}")
    print(f"  β (Adaptive): {jnp.min(betas):.2f} to {jnp.max(betas):.2f}") 
    print(f"  γ (Noise): {jnp.min(gammas):.3f} to {jnp.max(gammas):.3f}")
    print(f"  I_0: {jnp.min(I_0s):.2e} to {jnp.max(I_0s):.2e} A")
    print(f"  I_adaptive: {jnp.min(I_adaptives):.2e} to {jnp.max(I_adaptives):.2e} A")
    
    return {
        'expected_return': expected_return,
        'portfolio_variance': portfolio_variance,
        'sharpe_ratio': sharpe_ratio,
        'circuit_params': (alphas, betas, gammas, I_0s, I_adaptives)
    }

def compare_operation_modes():
    """Compare different operation modes of the enhanced P-bit circuit."""
    n_steps = 1000
    steps = jnp.arange(n_steps)
    
    # Get parameters for each mode
    continuous_params = jnp.array([programmable_current_continuous(step, n_steps) for step in steps])
    combinatorial_params = jnp.array([programmable_current_combinatorial(step, n_steps) for step in steps])
    bayesian_params = jnp.array([programmable_current_bayesian(step, n_steps) for step in steps])
    portfolio_params = jnp.array([programmable_current_portfolio(step, n_steps) for step in steps])
    
    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot I_0 comparison
    ax1.plot(steps, continuous_params[:, 0] * 1e6, label='Continuous', linewidth=2)
    ax1.plot(steps, combinatorial_params[:, 0] * 1e6, label='Combinatorial', linewidth=2)
    ax1.plot(steps, bayesian_params[:, 0] * 1e6, label='Bayesian', linewidth=2)
    ax1.plot(steps, portfolio_params[:, 0] * 1e6, label='Portfolio', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('I_0 (μA)')
    ax1.set_title('Base Current I_0 by Operation Mode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot I_adaptive comparison
    ax2.plot(steps, continuous_params[:, 1] * 1e6, label='Continuous', linewidth=2)
    ax2.plot(steps, combinatorial_params[:, 1] * 1e6, label='Combinatorial', linewidth=2)
    ax2.plot(steps, bayesian_params[:, 1] * 1e6, label='Bayesian', linewidth=2)
    ax2.plot(steps, portfolio_params[:, 1] * 1e6, label='Portfolio', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('I_adaptive (μA)')
    ax2.set_title('Adaptive Current by Operation Mode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot noise scale comparison
    ax3.plot(steps, continuous_params[:, 2], label='Continuous', linewidth=2)
    ax3.plot(steps, combinatorial_params[:, 2], label='Combinatorial', linewidth=2)
    ax3.plot(steps, bayesian_params[:, 2], label='Bayesian', linewidth=2)
    ax3.plot(steps, portfolio_params[:, 2], label='Portfolio', linewidth=2)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Noise Scale')
    ax3.set_title('Noise Scale by Operation Mode')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot circuit response for different modes
    test_v_diff = jnp.linspace(-0.1, 0.1, 100)
    key = jax.random.PRNGKey(42)
    
    responses = []
    modes = [programmable_current_portfolio(500, 1000),  # Mid-optimization
             programmable_current_portfolio(100, 1000),  # Early
             programmable_current_portfolio(900, 1000)]  # Late
    
    for I_0, I_adaptive, noise_scale in modes:
        response = []
        for v in test_v_diff:
            prob, _ = differential_pair_probability(v, I_adaptive, 2.5, 1.5, noise_scale, key)
            response.append(prob)
        responses.append(response)
    
    for i, (response, label) in enumerate(zip(responses, ['Early', 'Mid', 'Late'])):
        ax4.plot(test_v_diff, response, label=f'{label} optimization', linewidth=2)
    
    ax4.set_xlabel('Differential Voltage V_diff')
    ax4.set_ylabel('Probability P(1)')
    ax4.set_title('Circuit Response at Different Stages')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_hardware_accurate_optimization_demo():
    N_ASSETS = 10
    N_STEPS = 2000
    
    key = jax.random.PRNGKey(42)
    
    # Run hardware-accurate optimization
    (final_weights, history, parameter_history, 
     mu, Sigma, initial_weights, transaction_cost) = run_hardware_accurate_optimization(N_ASSETS, N_STEPS, key)
    
    # Analyze results
    risk_aversion = 0.3
    cost_multiplier = 0.5
    final_objective = portfolio_objective(final_weights, mu, Sigma, risk_aversion,
                                        cost_multiplier, initial_weights, transaction_cost)
    
    analysis = analyze_circuit_performance(final_weights, history, parameter_history, mu, Sigma)
    
    print(f"\nOptimal Portfolio Weights:")
    sorted_indices = jnp.argsort(final_weights)[::-1]
    for i, idx in enumerate(sorted_indices):
        w = final_weights[idx]
        ret = mu[idx]
        print(f"  Asset {idx+1}: {w*100:5.1f}% (Return: {ret:.3f})")
    
    # Enhanced plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Optimization convergence
    ax1.plot(history, 'b-', linewidth=2)
    ax1.set_xlabel('Optimization Step')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Differential Pair P-bit Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Circuit parameters
    steps = jnp.arange(N_STEPS)
    alphas, betas, gammas, I_0s, I_adaptives = analysis['circuit_params']
    
    ax2.plot(steps, alphas, 'g-', label=r'$\alpha$ (Gain)', linewidth=2)
    ax2.plot(steps, betas, 'r--', label=r'$\beta$ (Adaptive)', linewidth=2)
    ax2.plot(steps, gammas, 'b:', label=r'$\gamma$ (Noise)', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Enhanced P-bit Circuit Parameters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Current sources
    ax3.plot(steps, I_0s * 1e6, 'purple', label='I_0 (μA)', linewidth=2)
    ax3.plot(steps, I_adaptives * 1e6, 'orange', label='I_adaptive (μA)', linewidth=2)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Current (μA)')
    ax3.set_title('Programmable Current Sources')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final weights
    assets = range(1, N_ASSETS + 1)
    ax4.bar(assets, final_weights * 100, color='skyblue', alpha=0.7)
    ax4.set_xlabel('Asset Index')
    ax4.set_ylabel('Weight (%)')
    ax4.set_title('Final Portfolio Allocation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare operation modes
    compare_operation_modes()
    
    print(f"\nFinal Objective: {final_objective:.6f}")
    print("Optimization completed using hardware-accurate differential pair P-bit circuit!")

if __name__ == "__main__":
    run_hardware_accurate_optimization_demo()
