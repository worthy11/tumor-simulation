import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from config import *

cmap = ListedColormap(['white', 'red'])

def show_vessel_network():
    plt.figure(figsize=(8, 6))
    plt.imshow(VESSEL_MAP, cmap='Reds', interpolation='nearest')
    plt.colorbar(label='Vessel Density')
    plt.title('Blood Vessel Network')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

def animate(initial_concentration, update_function, title="Concentration", cmap='viridis', vmin=None, vmax=None):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    concentration = initial_concentration.copy()
    
    if vmin is None:
        vmin = np.min(concentration)
    if vmax is None:
        vmax = np.max(concentration)
    
    im = ax.imshow(concentration, cmap=cmap, interpolation='bilinear', 
                   vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(f"{title} - Step 0")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    
    for step in range(1, STEPS + 1):
        concentration = update_function()
        im.set_data(concentration)
        ax.set_title(f"{title} - Step {step}")
        plt.pause(DT)
    
    plt.ioff()
    plt.show()