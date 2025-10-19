"""
Visualization script for oxygen (O2) concentration changes over time.
This script simulates and animates the oxygen concentration in the tumor environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from concentrations import O2, update_O2_concentration, vessel_map
from visualization import animate_concentration, save_concentration_animation
from config import STEPS

def show_vessel_network():
    """Display the blood vessel network."""
    plt.figure(figsize=(8, 8))
    plt.imshow(vessel_map, cmap='Reds', interpolation='nearest')
    plt.colorbar(label='Vessel Density')
    plt.title('Blood Vessel Network')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

def main():
    """Run the O2 concentration visualization."""
    
    # Show vessel network first
    print("Displaying blood vessel network...")
    print(f"Total vessel coverage: {np.sum(vessel_map > 0) / vessel_map.size * 100:.1f}% of domain")
    show_vessel_network()
    
    print("\nStarting O2 concentration simulation...")
    print(f"Grid size: {O2.shape}")
    print(f"Number of steps: {STEPS}")
    print(f"Initial O2 concentration: 0.5*1e-3 mM (uniform)")
    print(f"Initial O2 range: {np.min(O2):.6e} - {np.max(O2):.6e} mM")
    print(f"\nVascular supply is LOCALIZED to blood vessels only")
    print(f"Set use_pulsatile=True for heartbeat simulation")
    
    # Store initial concentration for display
    initial_O2 = O2.copy()
    
    # Choose simulation mode
    USE_PULSATILE = True  # Set to True to simulate heartbeat
    
    # Create wrapper function with pulsatile parameter
    if USE_PULSATILE:
        def update_func():
            return update_O2_concentration(dt=1, use_pulsatile=True)
        title = "O2 Concentration Evolution (Pulsatile)"
    else:
        def update_func():
            return update_O2_concentration(dt=1, use_pulsatile=False)
        title = "O2 Concentration Evolution"
    
    # Option 1: Live animation (interactive)
    print("\nStarting live animation...")
    animate_concentration(
        initial_concentration=initial_O2,
        update_function=update_func,
        steps=STEPS,
        title=title,
        cmap='plasma',  # 'viridis', 'plasma', 'hot', 'coolwarm' are good options
        vmin=0,
        vmax=np.max(initial_O2) * 2,  # Allow some headroom
        pause_time=0.05  # Faster animation
    )
    
    # Option 2: Save animation to file (uncomment to use)
    # print("\nSaving animation to file...")
    # save_concentration_animation(
    #     initial_concentration=initial_O2,
    #     update_function=update_O2_concentration,
    #     steps=STEPS,
    #     filename="o2_concentration.gif",
    #     title="O2 Concentration Evolution",
    #     cmap='plasma',
    #     vmin=0,
    #     vmax=np.max(initial_O2) * 1.2,
    #     fps=10
    # )

if __name__ == "__main__":
    main()

