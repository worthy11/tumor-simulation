import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from config import *
import grids

def cells_to_rgb(cells):
    """Convert CELLS tensor to RGB image for visualization."""
    rows, cols = cells.shape[1], cells.shape[2]
    rgb = np.ones((rows, cols, 3))
    # Colors
    endothelial_col = np.array([0.0, 0.0, 1.0])      # veins - blue
    necrotic_col = np.array([0.545, 0.271, 0.075])   # necrotic - brown
    entry_col = np.array([0.0, 1.0, 0.0])            # entry points - green

    # Tumor colors differ by subtype (mez vs nie_mez)
    # nie_mez (non-mesenchymal-like): active red
    active_nie_col = np.array([1.0, 0.0, 0.0])
    # Quiescent color is shared between subtypes (user requested only the starting/active color differ)
    quiescent_col = np.array([1.0, 0.647, 0.0])

    # mez (mesenchymal-like): active uses a distinct color
    active_mez_col = np.array([0.8, 0.0, 0.8])   # magenta-ish for active mez

    migrating_col = np.array([1.0, 1.0, 0.0])    # migrating - yellow

    # Draw endothelial cells and entry points first
    rgb[cells[0] > 0] = endothelial_col
    rgb[cells[5] > 0] = entry_col

    # Active tumor cells split by subtype
    active_mask = cells[1] > 0
    if 'TUMOR_SUBTYPE' in globals():
        mez_mask = (TUMOR_SUBTYPE == 1)
    else:
        # fallback: if not available, treat all as nie_mez
        mez_mask = np.zeros((rows, cols), dtype=bool)

    active_mez_mask = active_mask & mez_mask
    active_nie_mask = active_mask & (~mez_mask)
    rgb[active_nie_mask] = active_nie_col
    rgb[active_mez_mask] = active_mez_col

    # Quiescent tumor cells split by subtype
    quiescent_mask = cells[2] > 0
    # Both subtypes share the same quiescent color
    rgb[quiescent_mask] = quiescent_col

    # Migrating and necrotic cells
    rgb[cells[3] > 0] = migrating_col
    rgb[cells[4] > 0] = necrotic_col
    
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


