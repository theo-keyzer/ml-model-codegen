"""
Enhanced C-Pbit Portfolio Optimization Under Market Uncertainty (JAX-Enabled)
============================================================================

This script implements the Continuous Probability Bit (C-Pbit) annealing
algorithm with advanced features for robust portfolio optimization.

Final Tuning Includes:
- Increased MIN/MAX weight bounds (2% to 25%) to force differentiation.
- C-Pbit update uses a 'velocity' (momentum) term to help escape uniform optima.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Callable

# Set global JAX configuration for stability
jax.config.update("jax_enable_x64", False)

# ========================================================================
# 1. Stochastic Portfolio Model
# ========================================================================

class StochasticPortfolioOptimizer:
    """
    Manages asset characteristics and computes the stochastic objective
    of a portfolio (weights) under market noise and regime shifts.
    """
    
    def __init__(self, n_assets: int, risk_aversion: float = 0.3, cost_multiplier: float = 0.5):
        """Initializes simulated market data and risk profile."""
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion 
        self.cost_multiplier = cost_multiplier
        
        # --- Simulated Market Data ---
        rng = jax.random.PRNGKey(0)
        
        # Expected returns (0.05 to 0.15)
        rng, k1 = jax.random.split(rng)
        self.expected_returns = jax.random.uniform(
            k1, (n_assets,), minval=0.05, maxval=0.15
        )
        
        # Base correlation matrix
        rng, k2 = jax.random.split(rng)
        self.base_correlation = self._generate_correlation_matrix(k2)
        
        # Transaction costs (non-linear with weight)
        self.base_transaction_costs = jnp.linspace(0.001, 0.01, n_assets)
        
    def _generate_correlation_matrix(self, rng_key: jnp.ndarray) -> jnp.ndarray:
        """Generates a realistic (positive semi-definite) correlation structure."""
        A = jax.random.normal(rng_key, (self.n_assets, self.n_assets))
        # Ensure it's positive semi-definite (Covariance matrix structure)
        return (A @ A.T) / jnp.sqrt(self.n_assets)
    
    def compute_portfolio_objective(
        self, 
        weights: jnp.ndarray, 
        time_step: int,
        rng_key: jnp.ndarray
    ) -> Tuple[float, jnp.ndarray]:
        """
        Computes the portfolio objective for a single, noisy realization.
        Maximize: Expected Return - λ * Risk - Transaction Costs
        """
        
        # 1. Market noise and realized returns
        rng_key, noise_key = jax.random.split(rng_key)
        market_noise = 0.03 * jax.random.normal(noise_key, (self.n_assets,))
        realized_returns = self.expected_returns + market_noise
        
        # 2. Time-varying correlations (Regime shift)
        regime_factor = jnp.sin(time_step * 0.05) * 0.3
        correlation = self.base_correlation * (1 + regime_factor)
        
        # --- Core Components ---
        portfolio_return = jnp.dot(weights, realized_returns)
        portfolio_variance = jnp.dot(weights, jnp.dot(correlation, weights))
        
        # Non-linear transaction cost (now multiplied by cost_multiplier)
        transaction_cost = self.cost_multiplier * jnp.sum(self.base_transaction_costs * weights ** 1.5) 
        
        # 3. Add rare extreme events (Crash risk)
        rng_key, extreme_key = jax.random.split(rng_key)
        extreme_prob = jax.random.uniform(extreme_key)
        crash_multiplier = jnp.where(extreme_prob < 0.05, 0.5, 1.0) 
        portfolio_return *= crash_multiplier
        
        # Objective (to MAXIMIZE): Return - Risk Penalty - Cost Penalty
        objective = (portfolio_return - 
                    self.risk_aversion * portfolio_variance - 
                    transaction_cost)
        
        return objective, rng_key
        
    def compute_portfolio_components(
        self, 
        weights: jnp.ndarray, 
        time_step: int,
        rng_key: jnp.ndarray
    ) -> Tuple[float, float, float]:
        """
        Computes and returns the raw components of the objective for diagnosis.
        Returns: (Return, Risk_Penalty, Cost_Penalty)
        """
        rng_key, noise_key = jax.random.split(rng_key)
        market_noise = 0.03 * jax.random.normal(noise_key, (self.n_assets,))
        realized_returns = self.expected_returns + market_noise
        
        regime_factor = jnp.sin(time_step * 0.05) * 0.3
        correlation = self.base_correlation * (1 + regime_factor)
        
        portfolio_return = jnp.dot(weights, realized_returns)
        portfolio_variance = jnp.dot(weights, jnp.dot(correlation, weights))
        
        transaction_cost_value = self.cost_multiplier * jnp.sum(self.base_transaction_costs * weights ** 1.5)
        risk_penalty_value = self.risk_aversion * portfolio_variance

        rng_key, extreme_key = jax.random.split(rng_key)
        crash_multiplier = jnp.where(jax.random.uniform(extreme_key) < 0.05, 0.5, 1.0) 
        portfolio_return *= crash_multiplier

        return portfolio_return, risk_penalty_value, transaction_cost_value


# ========================================================================
# 2. Robust C-Pbit Annealing Function
# ========================================================================

# --- Enhanced Weight Constraints (Improvement 2) ---
MIN_WEIGHT = 0.02  # Minimum 2% allocation
MAX_WEIGHT = 0.25  # Maximum 25% allocation

def project_weights(state_raw: jnp.ndarray) -> jnp.ndarray:
    """Project raw state to normalized weights satisfying min/max constraints."""
    weights = state_raw / jnp.sum(state_raw)
    weights = jnp.clip(weights, MIN_WEIGHT, MAX_WEIGHT)
    return weights / jnp.sum(weights)

# --- Initial Biased Weight Function ---
def initialize_weights_biased(expected_returns: jnp.ndarray, min_weight: float, max_weight: float) -> jnp.ndarray:
    """Initialize weights biased toward higher-return assets using Softmax."""
    temp = 5.0
    logits = expected_returns / temp
    weights = jax.nn.softmax(logits)
    
    weights = jnp.clip(weights, min_weight, max_weight)
    return weights / jnp.sum(weights)


# --- Enhanced Objective with CVaR ---
def objective_k_samples(weights: jnp.ndarray, time_step: int, keys: jnp.ndarray) -> float:
    """Computes K objectives and returns a blend of mean and CVaR."""
    K_SAMPLES = keys.shape[0]
    
    objectives, _ = jax.vmap(
        lambda k: optimizer.compute_portfolio_objective(weights, time_step, k)
    )(keys)
    
    alpha = 0.10  
    sorted_objectives = jnp.sort(objectives)
    cvar = jnp.mean(sorted_objectives[:int(K_SAMPLES * alpha)])
    
    # Blended Risk-Averse Objective (0.8 Mean / 0.2 CVaR)
    mean_obj = jnp.mean(objectives)
    return 0.8 * mean_obj + 0.2 * cvar

# --- Objective Wrapper for JAX Grad ---
def objective_to_differentiate(state_raw: jnp.ndarray, step: int, sample_keys: jnp.ndarray) -> float:
    """
    Wrapper function to compute the enhanced objective from the raw C-Pbit state.
    """
    weights = project_weights(state_raw)
    return objective_k_samples(weights, step, sample_keys)

# --- JAX Gradient Function ---
grad_fn = jax.jit(jax.grad(objective_to_differentiate, argnums=0))


def cpbit_portfolio_annealing(
    optimizer: StochasticPortfolioOptimizer,
    n_steps: int = 1000,
    temp_start: float = 50.0,
    temp_end: float = 0.1,
    noise_std: float = 1.5,
    k_samples: int = 100, 
    patience: int = 100,
    momentum_factor: float = 0.5, # Improvement 3: Momentum
    rng_key: jnp.ndarray = None
) -> Tuple[jnp.ndarray, float, list]:
    """
    Robust C-Pbit annealing using JAX Automatic Differentiation, CVaR, and Momentum.
    """
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
        
    n_vars = optimizer.n_assets
    
    # Initialize Biased Raw State and Velocity (Momentum)
    initial_weights = initialize_weights_biased(optimizer.expected_returns, MIN_WEIGHT, MAX_WEIGHT)
    state_raw = initial_weights.copy()
    state = project_weights(state_raw)
    velocity = jnp.zeros(n_vars) # Initialize velocity for momentum
    
    # Temperature schedule (log-linear decay)
    log_temp_schedule = np.linspace(
        np.log(temp_start), 
        np.log(temp_end), 
        n_steps
    )
    temp_schedule = np.exp(log_temp_schedule)
    
    best_state = state.copy()
    best_value = -jnp.inf
    history = []
    
    no_improvement_count = 0 
    min_improvement = 1e-5 

    
    for step in range(n_steps):
        temp = temp_schedule[step]
        beta = 1.0 / temp
        
        # 1. Generate K keys for objective evaluation
        rng_key, sample_key_start = jax.random.split(rng_key)
        sample_keys = jax.random.split(sample_key_start, k_samples)

        # 2. Evaluate current state 
        obj_mean = objective_k_samples(state, step, sample_keys)
        
        # --- Update Best Value and Check for Early Stopping ---
        improvement = obj_mean - best_value
        
        if improvement > min_improvement:
            best_value = obj_mean
            best_state = state.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Early Stopping
        if step > patience and no_improvement_count > patience:
            print(f"Early stopping triggered at step {step}/{n_steps} due to lack of improvement.")
            break
            
        # 3. Compute Local Fields (Gradient Estimate via JAX Autodiff)
        local_fields = grad_fn(state_raw, step, sample_keys)
        
        # 4. C-Pbit Update with Momentum (Velocity)
        
        # Adaptive noise
        adaptive_noise_level = noise_std * (temp / temp_start)
        rng_key, noise_key = jax.random.split(rng_key)
        noise = adaptive_noise_level * jax.random.normal(noise_key, (n_vars,))
        
        # C-Pbit Direction (The intended change in raw state)
        target_state = 1.0 / (1.0 + jnp.exp(-2.0 * beta * (local_fields + noise)))
        
        # Calculate update direction (target - current)
        update_direction = target_state - state_raw
        
        # Apply Momentum (Improvement 3)
        velocity = momentum_factor * velocity + (1.0 - momentum_factor) * update_direction
        
        # Apply velocity to raw state
        state_raw = state_raw + velocity
        
        # 5. Normalize and project state (weights)
        state = project_weights(state_raw)
        
        # 6. Record history
        history.append({
            'step': step,
            'objective': obj_mean.item(),
            'temperature': temp.item(),
            'best_so_far': best_value.item()
        })
        
        if step % 100 == 0:
            print(f"Step {step}/{n_steps}: Obj={obj_mean:.4f}, Best={best_value:.4f}, T={temp:.3f}")
    
    return best_state, best_value, history


# ========================================================================
# 3. Post-Optimization Analysis
# ========================================================================

def analyze_portfolio(weights: jnp.ndarray, optimizer: StochasticPortfolioOptimizer, n_simulations: int = 5000) -> np.ndarray:
    """Comprehensive portfolio analysis and component breakdown."""
    rng_key = jax.random.PRNGKey(42)
    
    rng_keys = jax.random.split(rng_key, n_simulations)
    
    # Run simulations for full objective value
    def run_simulation(key):
        obj, _ = optimizer.compute_portfolio_objective(weights, 0, key) 
        return obj

    returns = jax.vmap(run_simulation)(rng_keys)
    returns_np = np.asarray(returns)
    
    # Run simulations for component breakdown 
    def run_component_sim(key):
        ret, var_p, cost_p = optimizer.compute_portfolio_components(weights, 0, key)
        return ret, var_p, cost_p

    component_results = jax.vmap(run_component_sim)(rng_keys)
    
    mean_ret = np.mean(component_results[0])
    mean_risk_pen = np.mean(component_results[1])
    mean_cost_pen = np.mean(component_results[2])
    
    print("\n" + "=" * 70)
    print(f"Portfolio Analysis ({n_simulations} simulations):")
    print("-" * 70)
    print(f"Mean Objective (E[R] - E[Risk] - E[Cost]): {np.mean(returns_np):.4f}")
    print(f"  -> Breakdown: Return ({mean_ret:.4f}) - Risk Penalty ({mean_risk_pen:.4f}) - Cost Penalty ({mean_cost_pen:.4f})")
    print(f"Std Dev (Volatility): {np.std(returns_np):.4f}")
    
    sharpe_ratio = np.mean(returns_np) / np.std(returns_np) if np.std(returns_np) > 1e-6 else 0
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"5% VaR (Value at Risk): {np.percentile(returns_np, 5):.4f}")
    
    cvar_threshold = np.percentile(returns_np, 5)
    cvar_loss_subset = returns_np[returns_np < cvar_threshold]
    print(f"5% CVaR (Conditional VaR): {np.mean(cvar_loss_subset):.4f}")
    
    return returns_np


# ========================================================================
# 4. Execution and Visualization
# ========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Final C-Pbit Optimization (Momentum & Strict Constraints)")
    print("=" * 70)
    
    # --- Setup ---
    N_ASSETS = 15
    N_STEPS = 1000 
    K_SAMPLES = 100 
    MOMENTUM_FACTOR = 0.5 
    
    # Set lower risk aversion and cost sensitivity
    portfolio = StochasticPortfolioOptimizer(
        n_assets=N_ASSETS, 
        risk_aversion=0.3, 
        cost_multiplier=0.5
    )
    # Expose the global optimizer instance for the JAX functions to use
    globals()['optimizer'] = portfolio 
    rng_key = jax.random.PRNGKey(5678)

    print(f"Optimizing portfolio of {N_ASSETS} assets over max {N_STEPS} steps.")
    print(f"Aversion (λ): {portfolio.risk_aversion}, Cost Multiplier: {portfolio.cost_multiplier}")
    print(f"Constraints: Min {MIN_WEIGHT*100}%, Max {MAX_WEIGHT*100}% | Momentum: {MOMENTUM_FACTOR}")
    print("-" * 70)
    
    # --- Run Annealing ---
    best_weights, best_obj, history = cpbit_portfolio_annealing(
        optimizer=portfolio,
        n_steps=N_STEPS,
        k_samples=K_SAMPLES,
        temp_start=50.0, 
        temp_end=0.1,    
        noise_std=1.5,
        patience=100,      
        momentum_factor=MOMENTUM_FACTOR,
        rng_key=rng_key
    )

    # --- Results ---
    print("\n" + "=" * 70)
    print("Optimization Phase Complete.")
    print(f"Final Best Enhanced Objective Value (Mean/CVaR Blend): {best_obj:.4f}")
    print("-" * 70)
    
    # --- Post-Optimization Analysis ---
    simulation_returns = analyze_portfolio(best_weights, optimizer, n_simulations=5000)

    # Show final allocation and expected returns
    print("\nFinal Optimal Portfolio Allocation:")
    sorted_indices = jnp.argsort(best_weights)[::-1]
    
    for i in range(N_ASSETS):
        idx = sorted_indices[i].item()
        weight = best_weights[idx].item() * 100
        ret = optimizer.expected_returns[idx].item()
        cost = optimizer.base_transaction_costs[idx].item()
        
        # Only print non-zero weights
        if weight > 0.01:
            print(f"  Asset {idx+1}: {weight:.2f}% (Exp Return: {ret:.3f}, Base Cost: {cost:.4f})")
    print(f"Total Allocated: {jnp.sum(best_weights) * 100:.2f}% (Should be 100%)")
    
    # --- Visualization ---
    print("\nVisualizing Annealing History...")
    steps = [h['step'] for h in history]
    objectives = [h['objective'] for h in history]
    best_so_far = [h['best_so_far'] for h in history]
    temperatures = [h['temperature'] for h in history]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Objective
    color = '#1f77b4' # blue
    ax1.set_xlabel('Annealing Step', fontsize=12)
    ax1.set_ylabel('Enhanced Objective Value (Maximize)', color=color, fontsize=12)
    ax1.plot(steps, objectives, label='Current Enhanced Objective', color=color, alpha=0.5)
    ax1.plot(steps, best_so_far, label='Best Enhanced Objective So Far', color='#d62728', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left', frameon=False)
    ax1.grid(True, linestyle=':', alpha=0.6)


    # Plot Temperature
    ax2 = ax1.twinx()  # Shared X-axis
    color = '#2ca02c' # green
    ax2.set_ylabel('Temperature (T)', color=color, fontsize=12)
    ax2.plot(steps, temperatures[:len(steps)], label='Temperature Schedule', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right', frameon=False)

    plt.title(f'Enhanced C-Pbit Portfolio Optimization (K={K_SAMPLES}, Momentum={MOMENTUM_FACTOR})', fontsize=14)
    fig.tight_layout()
    plt.show()
    
    # --- Distribution Analysis ---
    plt.figure(figsize=(8, 5))
    plt.hist(simulation_returns, bins=50, density=True, color='#8c564b', alpha=0.7)
    plt.axvline(np.mean(simulation_returns), color='blue', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(simulation_returns):.4f}')
    plt.axvline(np.percentile(simulation_returns, 5), color='red', linestyle='dashed', linewidth=1, label=f'5% VaR: {np.percentile(simulation_returns, 5):.4f}')
    plt.title('Distribution of Final Portfolio Objective (5000 Simulations)')
    plt.xlabel('Objective Value (Return - Risk - Cost)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print("=" * 70)
    # Check if the problem is in the correlation structure
    print("Expected returns range:", jnp.min(portfolio.expected_returns), "to", jnp.max(portfolio.expected_returns))
    print("Expected returns std:", jnp.std(portfolio.expected_returns))

    corr_matrix = portfolio.base_correlation
    print("Correlation matrix mean:", jnp.mean(corr_matrix))
    print("Is correlation too high?", jnp.mean(corr_matrix) > 0.7)
