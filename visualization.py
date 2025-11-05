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
        4: np.array([0.545, 0.271, 0.075]), # necrotic tumor - brown
        5: np.array([0.0, 1.0, 0.0])       # vein entry points - green
    }
    
    for cell_type in [0, 5, 1, 2, 3, 4]:
        mask = cells[cell_type] > 0
        rgb[mask] = colors[cell_type]
    
    return rgb

def animate(update_function, cmap='viridis', vmin=None, vmax=None):
    """Animate only the cells in a single, large window.

    Parameters are kept for backward compatibility but are not used.
    """
    plt.ion()

    # Single large square window for better visibility of cells
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Initial render of cells
    cell_rgb = cells_to_rgb(grids.CELLS)
    im = ax.imshow(cell_rgb, interpolation='nearest', origin='lower')
    ax.set_title('Cells - Step 0')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()

    for step in range(1, STEPS + 1):
        update_function()

        # Update cells only
        cell_rgb = cells_to_rgb(grids.CELLS)
        im.set_data(cell_rgb)
        ax.set_title(f'Cells - Step {step}')

        plt.pause(0.0001)

    plt.ioff()
    plt.show()


