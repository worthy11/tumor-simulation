import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from config import *
import grids

def cells_to_rgb(cells):
    """Convert CELLS tensor to RGB image for visualization."""
    rows, cols = cells.shape[1], cells.shape[2]
    rgb = np.ones((rows, cols, 3))
    
    colors = {
        0: np.array([0.0, 0.0, 1.0]),      # endothelial - blue (veins)
        1: np.array([1.0, 0.0, 0.0]),      # active tumor - red
        2: np.array([1.0, 0.647, 0.0]),    # quiescent tumor - orange
        3: np.array([1.0, 1.0, 0.0]),      # migrating tumor - yellow
        4: np.array([0.545, 0.271, 0.075]) # necrotic tumor - brown
    }
    
    for cell_type in [0, 1, 2, 3, 4]:
        mask = cells[cell_type] > 0
        rgb[mask] = colors[cell_type]
    
    return rgb

def animate(update_function, cmap='viridis', vmin=None, vmax=None):
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    titles = ['O2', 'Glucose', 'CO2', 'Cells']
    
    ims = []
    for idx, ax in enumerate(axes):
        if idx < 3:  # Environmental factors
            data = [grids.O2, grids.G, grids.CO2][idx]
            vmin_i = np.min(data) if vmin is None else (vmin[idx] if isinstance(vmin, (list, np.ndarray)) else vmin)
            vmax_i = np.max(data) if vmax is None else (vmax[idx] if isinstance(vmax, (list, np.ndarray)) else vmax)
            
            im = ax.imshow(data, cmap=cmap, interpolation='bilinear', 
                           vmin=vmin_i, vmax=vmax_i, origin='lower')
            plt.colorbar(im, ax=ax, label='Concentration (mM)')
        else:  # Cells
            cell_rgb = cells_to_rgb(grids.CELLS)
            im = ax.imshow(cell_rgb, interpolation='nearest', origin='lower')
        
        ax.set_title(f"{titles[idx]} - Step 0")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ims.append(im)
    
    plt.tight_layout()
    
    for step in range(1, STEPS + 1):
        update_function()
        
        # Update environmental factors
        for idx in range(3):
            data = [grids.O2, grids.G, grids.CO2][idx]
            ims[idx].set_data(data)
            axes[idx].set_title(f"{titles[idx]} - Step {step}")
        
        # Update cells
        cell_rgb = cells_to_rgb(grids.CELLS)
        ims[3].set_data(cell_rgb)
        axes[3].set_title(f'Cells - Step {step}')
        
        # Print cell statistics
        active_mask = grids.CELLS[1] > 0
        quiescent_mask = grids.CELLS[2] > 0
        necrotic_mask = grids.CELLS[4] > 0
        
        active_count = np.sum(active_mask)
        quiescent_count = np.sum(quiescent_mask)
        necrotic_count = np.sum(necrotic_mask)
        
        active_vitality = np.mean(grids.V[active_mask]) if active_count > 0 else 0
        quiescent_vitality = np.mean(grids.V[quiescent_mask]) if quiescent_count > 0 else 0
        necrotic_vitality = np.mean(grids.V[necrotic_mask]) if necrotic_count > 0 else 0
        
        active_energy = np.mean(grids.E[active_mask]) if active_count > 0 else 0
        quiescent_energy = np.mean(grids.E[quiescent_mask]) if quiescent_count > 0 else 0
        necrotic_energy = np.mean(grids.E[necrotic_mask]) if necrotic_count > 0 else 0
        
        # print(f"Step {step}:")
        # print(f"  Active: {active_count} (V: {active_vitality:.2e}, E: {active_energy:.2e})")
        # print(f"  Quiescent: {quiescent_count} (V: {quiescent_vitality:.2e}, E: {quiescent_energy:.2e})")
        # print(f"  Necrotic: {necrotic_count} (V: {necrotic_vitality:.2e}, E: {necrotic_energy:.2e})")
        
        plt.pause(DT)
    
    plt.ioff()
    plt.show()


