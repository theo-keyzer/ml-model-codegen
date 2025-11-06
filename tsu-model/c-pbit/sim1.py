import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider

class SimpleDifferentialPBit:
    def __init__(self):
        self.Vt = 0.026  # Thermal voltage
        
    def compute_probability(self, v_plus, v_minus, noise_std, n_samples=1000):
        """Simple interactive P-bit simulation"""
        v_diff_nominal = v_plus - v_minus
        
        # Generate noisy differential inputs
        noisy_v_diff = v_diff_nominal + noise_std * np.random.randn(n_samples)
        
        # Differential pair current steering
        i1_relative = 1 / (1 + np.exp(-noisy_v_diff / self.Vt))
        i2_relative = 1 - i1_relative
        
        # Probabilities
        probabilities = i1_relative / (i1_relative + i2_relative)
        
        return probabilities, noisy_v_diff

def interactive_pbit_demo():
    pbit = SimpleDifferentialPBit()
    
    @interact(
        v_plus=FloatSlider(value=0.0, min=-0.2, max=0.2, step=0.01, description='V_IN+:'),
        v_minus=FloatSlider(value=0.0, min=-0.2, max=0.2, step=0.01, description='V_IN-:'),
        noise_std=FloatSlider(value=0.05, min=0.0, max=0.2, step=0.01, description='Noise:'),
        n_samples=IntSlider(value=1000, min=100, max=10000, step=100, description='Samples:')
    )
    def update_plot(v_plus, v_minus, noise_std, n_samples):
        probabilities, noisy_inputs = pbit.compute_probability(v_plus, v_minus, noise_std, n_samples)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Probability histogram
        ax1.hist(probabilities, bins=50, alpha=0.7, density=True)
        ax1.axvline(probabilities.mean(), color='red', linestyle='--', label=f'Mean: {probabilities.mean():.3f}')
        ax1.set_xlabel('P(1)')
        ax1.set_ylabel('Density')
        ax1.set_title('Output Probability Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Input noise distribution
        ax2.hist(noisy_inputs, bins=50, alpha=0.7, density=True, color='orange')
        ax2.axvline(v_plus - v_minus, color='red', linestyle='--', label=f'Nominal: {v_plus-v_minus:.3f}V')
        ax2.set_xlabel('Effective Input Voltage (V)')
        ax2.set_ylabel('Density')
        ax2.set_title('Input Voltage + Noise')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Probability vs Input scatter
        ax3.scatter(noisy_inputs, probabilities, alpha=0.5, s=1)
        ax3.set_xlabel('Input Voltage (V)')
        ax3.set_ylabel('P(1)')
        ax3.set_title('P(1) vs Input Voltage')
        ax3.grid(True, alpha=0.3)
        
        # Theoretical curve
        v_range = np.linspace(-0.3, 0.3, 1000)
        p_theoretical = 1 / (1 + np.exp(-v_range / pbit.Vt))
        ax4.plot(v_range, p_theoretical, 'r-', linewidth=2, label='Theoretical (no noise)')
        ax4.set_xlabel('Input Voltage (V)')
        ax4.set_ylabel('P(1)')
        ax4.set_title('Theoretical Transfer Function')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"Statistics: Mean P(1) = {probabilities.mean():.4f} ± {probabilities.std():.4f}")
        print(f"Input range: {noisy_inputs.min():.3f}V to {noisy_inputs.max():.3f}V")

# Run the interactive demo
if __name__ == "__main__":
    interactive_pbit_demo()
