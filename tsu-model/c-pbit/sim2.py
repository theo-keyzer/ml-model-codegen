import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import jax
import jax.numpy as jnp

class DifferentialPBitSimulator:
    def __init__(self):
        self.thermal_voltage = 0.026  # V_T = kT/q at room temperature
        
    def differential_pbit_probability(self, v_plus, v_minus, tail_current, noise_std, rng_key):
        """Differential pair P-bit probability calculation"""
        # Differential input with added noise
        v_diff = (v_plus - v_minus) + noise_std * jax.random.normal(rng_key)
        
        # Current steering in differential pair
        i1 = tail_current / (1 + jnp.exp(-v_diff / self.thermal_voltage))
        i2 = tail_current - i1
        
        # Probability from current ratio
        probability = i1 / (i1 + i2)
        
        return probability, i1, i2, v_diff

def main():
    # Initialize simulator
    simulator = DifferentialPBitSimulator()
    
    # Create figure and subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Main probability plot
    ax_prob = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    ax_currents = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax_hist = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    ax_controls = plt.subplot2grid((3, 4), (2, 0), colspan=4)
    ax_controls.axis('off')
    
    # Initial parameters
    init_v_plus = 0.0
    init_v_minus = 0.0
    init_tail_current = 1e-6  # 1uA
    init_noise_std = 0.05
    init_n_samples = 1000
    
    # Create sliders
    slider_y = 0.9
    slider_height = 0.03
    slider_spacing = 0.05
    
    ax_v_plus = plt.axes([0.25, slider_y, 0.65, slider_height])
    ax_v_minus = plt.axes([0.25, slider_y - slider_spacing, 0.65, slider_height])
    ax_tail_current = plt.axes([0.25, slider_y - 2*slider_spacing, 0.65, slider_height])
    ax_noise_std = plt.axes([0.25, slider_y - 3*slider_spacing, 0.65, slider_height])
    ax_n_samples = plt.axes([0.25, slider_y - 4*slider_spacing, 0.65, slider_height])
    
    slider_v_plus = Slider(ax_v_plus, 'V_IN+ (V)', -0.2, 0.2, valinit=init_v_plus)
    slider_v_minus = Slider(ax_v_minus, 'V_IN- (V)', -0.2, 0.2, valinit=init_v_minus)
    slider_tail_current = Slider(ax_tail_current, 'I_TAIL (μA)', 0.1, 10.0, valinit=init_tail_current*1e6)
    slider_noise_std = Slider(ax_noise_std, 'Noise Std (V)', 0.0, 0.2, valinit=init_noise_std)
    slider_n_samples = Slider(ax_n_samples, 'Samples', 100, 10000, valinit=init_n_samples, valstep=100)
    
    # Add reset button
    reset_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
    
    def reset(event):
        slider_v_plus.reset()
        slider_v_minus.reset()
        slider_tail_current.reset()
        slider_noise_std.reset()
        slider_n_samples.reset()
    
    reset_button.on_clicked(reset)
    
    # JIT compile the probability function for speed
    jitted_prob_func = jax.jit(simulator.differential_pbit_probability)
    
    def update(val):
        # Get current slider values
        v_plus = slider_v_plus.val
        v_minus = slider_v_minus.val
        tail_current = slider_tail_current.val * 1e-6  # Convert to Amps
        noise_std = slider_noise_std.val
        n_samples = int(slider_n_samples.val)
        
        # Generate samples
        rng_key = jax.random.PRNGKey(42)
        rng_keys = jax.random.split(rng_key, n_samples)
        
        # Vectorized computation
        probabilities, currents_i1, currents_i2, v_diffs = jax.vmap(
            lambda key: jitted_prob_func(v_plus, v_minus, tail_current, noise_std, key)
        )(rng_keys)
        
        probabilities = np.array(probabilities)
        currents_i1 = np.array(currents_i1)
        currents_i2 = np.array(currents_i2)
        v_diffs = np.array(v_diffs)
        
        # Clear previous plots
        ax_prob.clear()
        ax_currents.clear()
        ax_hist.clear()
        
        # Plot 1: Probability distribution
        ax_prob.hist(probabilities, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax_prob.axvline(probabilities.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {probabilities.mean():.3f}')
        ax_prob.axvline(0.5, color='green', linestyle=':', linewidth=1, alpha=0.7, label='P=0.5')
        ax_prob.set_xlabel('Probability P(1)')
        ax_prob.set_ylabel('Density')
        ax_prob.set_title('Probability Distribution\n(Differential Pair P-bit Output)')
        ax_prob.legend()
        ax_prob.grid(True, alpha=0.3)
        
        # Plot 2: Current distribution
        ax_currents.hist(currents_i1*1e6, bins=50, alpha=0.7, label='I1', color='lightcoral', density=True)
        ax_currents.hist(currents_i2*1e6, bins=50, alpha=0.7, label='I2', color='lightgreen', density=True)
        ax_currents.set_xlabel('Current (μA)')
        ax_currents.set_ylabel('Density')
        ax_currents.set_title('Branch Current Distribution')
        ax_currents.legend()
        ax_currents.grid(True, alpha=0.3)
        
        # Plot 3: Input noise distribution
        ax_hist.hist(v_diffs, bins=50, alpha=0.7, color='orange', edgecolor='black', density=True)
        ax_hist.axvline(v_plus - v_minus, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean V_diff: {v_plus - v_minus:.3f}V')
        ax_hist.set_xlabel('Effective Differential Input (V)')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title('Input Voltage + Noise Distribution')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Update statistics display
        fig.suptitle(
            f'Differential P-bit Simulator | '
            f'V_diff = {v_plus-v_minus:.3f}V | '
            f'Noise = {noise_std:.3f}V | '
            f'P_mean = {probabilities.mean():.3f} ± {probabilities.std():.3f}',
            fontsize=14, y=0.95
        )
        
        plt.draw()
    
    # Register update function
    slider_v_plus.on_changed(update)
    slider_v_minus.on_changed(update)
    slider_tail_current.on_changed(update)
    slider_noise_std.on_changed(update)
    slider_n_samples.on_changed(update)
    
    # Initial update
    update(None)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
