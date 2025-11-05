import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

# --- 1. P-BIT CLASS DEFINITION ---

class PBit:
    """
    Simulates the stochastic behavior of a 1T1M-based p-bit using the Gaussian 
    noise and thresholding model, which yields a sigmoid activation function.
    
    The core probability P(Output=1) is derived from the Q-function (related to erf).
    """
    def __init__(self, V_READ=0.1, V_TH=0.05, k_scaling=1e4, sigma_noise=1e-8, random_seed=None):
        """
        Initializes the p-bit with key physical/circuit parameters.
        
        V_READ (V): Read voltage applied to the 1T1M cell.
        V_TH (V): Threshold voltage of the comparator/sense amplifier.
        k_scaling (V/A): Proportionality constant relating current to node voltage (V = k*I).
        sigma_noise (A): Standard deviation of the noise current (I_noise).
        random_seed (int): Seed for numpy's random number generator for reproducibility.
        """
        # Circuit constants
        self.V_READ = V_READ
        self.V_TH = V_TH
        self.k = k_scaling
        
        # Noise (controls the "inverse temperature" beta)
        self.sigma_noise = sigma_noise
        
        # Set seed for reproducible sampling
        if random_seed is not None:
            np.random.seed(random_seed)

        print(f"PBit initialized with V_READ={V_READ}V, V_TH={V_TH}V, Noise σ={sigma_noise:.2e}A")

    @property
    def beta(self):
        """
        Calculates the inverse temperature or gain factor (β).
        
        In this physical model, the gain is inversely proportional to the noise 
        standard deviation (β ∝ 1/σ_I). Higher β means a steeper sigmoid curve.
        We use the proportional factor 1/sigma_noise for a relative measure.
        """
        # Using a simple proportionality: k and V_READ are constants here.
        return 1.0 / self.sigma_noise

    def calculate_probability(self, G_M):
        """
        Calculates the probability P(Output=1) given the memristor conductance G_M (1/R_M).
        
        Args:
            G_M (S): Memristor Conductance (Siemens). The analog "weight" input.
            
        Returns:
            float: Probability P(Output=1).
        """
        # 1. Mean node voltage (mu_V) contribution from the memristor current (I_mem)
        mu_V = self.k * self.V_READ * G_M
        
        # 2. Standard deviation (sigma_V) of the node voltage from the noise current
        sigma_V = self.k * self.sigma_noise
        
        if sigma_V == 0:
             # Idealized deterministic case: P is 1 or 0
             return 1.0 if mu_V > self.V_TH else 0.0

        # 3. Argument for the erf function (standardized threshold)
        # This term is proportional to β * (x_0 - x)
        argument = (self.V_TH - mu_V) / (np.sqrt(2) * sigma_V)
        
        # 4. P(Output=1) using the derived Q-function equivalent
        P_out = 0.5 * (1 - erf(argument))
        
        return np.clip(P_out, 0.0, 1.0)

    
    def sample(self, G_M, n_samples=1):
        """
        Generates binary samples (0s or 1s) based on the calculated probability.
        Uses the internal numpy random state, initialized by the random_seed.
        """
        P = self.calculate_probability(G_M)
        # Compare a stream of random numbers to the probability P
        samples = np.random.rand(n_samples) < P
        return samples.astype(int)

# --- 2. DEMONSTRATION AND VISUALIZATION ---

def demonstrate_pbit():
    """Demonstrates the sigmoid curve and stochastic sampling."""
    
    # Define a range of memristor conductances (G_M, the input "weight")
    G_M_min = 1e-6 
    G_M_max = 2e-5 
    conductances = np.linspace(G_M_min, G_M_max, 200)
    
    # Set a fixed seed for reproducible graphs and samples
    FIXED_SEED = 42

    # 1. Low Noise Case (High Beta / High Gain - Steep Sigmoid)
    pbit_low_noise = PBit(sigma_noise=1e-10, random_seed=FIXED_SEED) 
    P_low_noise = [pbit_low_noise.calculate_probability(G) for G in conductances]
    
    # 2. High Noise Case (Low Beta / Low Gain - Shallow Sigmoid)
    pbit_high_noise = PBit(sigma_noise=5e-10, random_seed=FIXED_SEED) 
    P_high_noise = [pbit_high_noise.calculate_probability(G) for G in conductances]

    # --- Plotting the Sigmoid Curves ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Display beta in the label
    ax.plot(conductances * 1e6, P_high_noise, 
            label=f'High Noise ($\sigma_I=5e-10$ A) - Low Gain ($\mathbf{{\\beta}}={pbit_high_noise.beta:.1e}$)', 
            color='tab:red', linewidth=3, alpha=0.7)
    
    ax.plot(conductances * 1e6, P_low_noise, 
            label=f'Low Noise ($\sigma_I=1e-10$ A) - High Gain ($\mathbf{{\\beta}}={pbit_low_noise.beta:.1e}$)', 
            color='tab:blue', linewidth=3)
    
    ax.set_title('P-Bit Activation Function: Probability vs. Memristor Conductance ($G_M$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Memristor Conductance $G_M$ (µS - MicroSiemens)', fontsize=12)
    ax.set_ylabel('$P(\\text{Output}=1)$', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight the 50% bias point
    G_M_50 = pbit_low_noise.V_TH / (pbit_low_noise.V_READ)
    ax.axvline(G_M_50 * 1e6, color='gray', linestyle=':', label='Theoretical $P=0.5$ Bias Point')
    ax.axhline(0.5, color='gray', linestyle=':')
    
    # --- Demonstrate Stochastic Sampling ---
    test_G_M_us = 1.0 # Test conductance in microSiemens
    test_G_M = test_G_M_us * 1e-6
    N_SAMPLES = 10000
    
    P_test = pbit_low_noise.calculate_probability(test_G_M)
    # The sampling now uses the fixed seed, making the result reproducible
    samples = pbit_low_noise.sample(test_G_M, N_SAMPLES)
    empirical_P = np.mean(samples)
    
    print("\n--- Stochastic Sampling Demonstration (Seed 42) ---")
    print(f"1. P-Bit Model Used (Low Noise): β = {pbit_low_noise.beta:.1e}")
    print(f"2. Set Memristor Conductance G_M = {test_G_M_us} µS")
    print(f"3. Theoretical Probability P(Output=1) = {P_test:.4f}")
    print(f"4. Simulating {N_SAMPLES} physical clock cycles...")
    print(f"5. Empirical Probability (Mean of Samples) = {empirical_P:.4f}")
    print(f"   (Result is reproducible due to the fixed random seed.)")
    
    plt.show()

if __name__ == "__main__":
    demonstrate_pbit()
