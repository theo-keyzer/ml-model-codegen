# Replace your optimizer with SAM-Prime
model = YourModel()
params = list(model.parameters())

# Create hardware-accurate simulator
sam_prime = SAMPrimeSimulator(
    params,
    config=SAMPrimeConfig(rho_flat_target=0.62),
    learned_rho_flats={"layer1": 0.58, "layer2": 0.65}  # From meta-control plane
)

# Training loop
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    
    # SAM-Prime step (instead of optimizer.step())
    hw_state = sam_prime.step(lr=0.001, step=global_step)
    
    # Monitor hardware state
    if hw_state['boosting']:
        print(f"Boost active: factor={hw_state['boost_factor']:.2f}")

# Get final statistics
stats = sam_prime.get_stats()
print(f"Energy saved: {stats['energy_saved_est']:.2%}")
