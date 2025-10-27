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
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    titles = ['O2', 'Glucose', 'CO2', 'MMP', 'ECM', 'Vitality', 'Energy', 'Cells']
    
    ims = []
    for idx, ax in enumerate(axes):
        if idx == 7:  # Cells
            cell_rgb = cells_to_rgb(grids.CELLS)
            im = ax.imshow(cell_rgb, interpolation='nearest', origin='lower')
        elif idx < 5:  # Environmental factors
            data = [grids.O2, grids.G, grids.CO2, grids.MMP, grids.ECM][idx]
            vmin_i = np.min(data) if vmin is None else (vmin[idx] if isinstance(vmin, (list, np.ndarray)) else vmin)
            vmax_i = np.max(data) if vmax is None else (vmax[idx] if isinstance(vmax, (list, np.ndarray)) else vmax)
            
            im = ax.imshow(data, cmap=cmap, interpolation='bilinear', 
                           vmin=vmin_i, vmax=vmax_i, origin='lower')
            plt.colorbar(im, ax=ax, label='Concentration')
        else:  # Vitality or Energy
            if idx == 5:  # Vitality
                data = grids.V
            else:  # Energy
                data = grids.E
            
            vmin_i = np.min(data) if vmin is None else (vmin[idx] if isinstance(vmin, (list, np.ndarray)) else vmin)
            vmax_i = np.max(data) if vmax is None else (vmax[idx] if isinstance(vmax, (list, np.ndarray)) else vmax)
            
            im = ax.imshow(data, cmap=cmap, interpolation='bilinear', 
                           vmin=vmin_i, vmax=vmax_i, origin='lower')
            plt.colorbar(im, ax=ax, label='Value')
        
        ax.set_title(f"{titles[idx]} - Step 0")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ims.append(im)
    
    plt.tight_layout()
    
    for step in range(1, STEPS + 1):
        update_function()
        
        # Update environmental factors
        for idx in range(5):
            data = [grids.O2, grids.G, grids.CO2, grids.MMP, grids.ECM][idx]
            ims[idx].set_data(data)
            axes[idx].set_title(f"{titles[idx]} - Step {step}")
        
        # Update Vitality
        ims[5].set_data(grids.V)
        axes[5].set_title(f'Vitality - Step {step}')
        
        # Update Energy
        ims[6].set_data(grids.E)
        axes[6].set_title(f'Energy - Step {step}')
        
        # Update cells
        cell_rgb = cells_to_rgb(grids.CELLS)
        ims[7].set_data(cell_rgb)
        axes[7].set_title(f'Cells - Step {step}')
        
        plt.pause(0.0001)
    
    plt.ioff()
    plt.show()


