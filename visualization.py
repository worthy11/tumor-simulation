import matplotlib.pyplot as plt
from config import *
import grids
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

def cells_to_rgb(cells):
    """Convert CELLS tensor to RGB image for visualization.
    Tumor cells are blurred for smooth color transitions, vessels remain pixelated.
    """
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

    # mez (mesenchymal-like): active uses a distinct color - bright purple/magenta
    active_mez_col = np.array([0.9, 0.0, 1.0])   # bright purple for active mez (more distinct)

    migrating_col = np.array([1.0, 1.0, 0.0])    # migrating - yellow

    # Create separate layers for tumor (will be blurred) and vessels (will stay sharp)
    tumor_rgb = np.ones((rows, cols, 3))
    
    # Build tumor layer with smooth color blending
    # Convert cell masks to float for blending
    active_mask = cells[1].astype(np.float32)
    quiescent_mask = cells[2].astype(np.float32)
    necrotic_mask = cells[4].astype(np.float32)
    migrating_mask = cells[3].astype(np.float32)
    
    # Split active cells by subtype (mez vs nie_mez)
    if 'TUMOR_SUBTYPE' in globals():
        mez_mask = (TUMOR_SUBTYPE == 1).astype(np.float32)
    else:
        # fallback: if not available, treat all as nie_mez
        mez_mask = np.zeros((rows, cols), dtype=np.float32)
    
    active_mez_mask = active_mask * mez_mask
    active_nie_mask = active_mask * (1 - mez_mask)
    
    # Normalize masks to sum to 1 where there are cells (for smooth blending)
    total_mask = active_mask + quiescent_mask + necrotic_mask + migrating_mask
    total_mask = np.where(total_mask > 0, total_mask, 1.0)  # Avoid division by zero
    
    # Blend colors based on cell type presence
    active_mez_norm = active_mez_mask / total_mask
    active_nie_norm = active_nie_mask / total_mask
    quiescent_norm = quiescent_mask / total_mask
    necrotic_norm = necrotic_mask / total_mask
    migrating_norm = migrating_mask / total_mask
    
    # Apply colors with blending - mez and nie_mez active cells use different colors
    for c in range(3):
        tumor_rgb[:, :, c] = (
            active_mez_norm * active_mez_col[c] +
            active_nie_norm * active_nie_col[c] +
            quiescent_norm * quiescent_col[c] +
            necrotic_norm * necrotic_col[c] +
            migrating_norm * migrating_col[c]
        )
    
    # Store original tumor cell mask before blur
    tumor_cell_present = (active_mask + quiescent_mask + necrotic_mask + migrating_mask) > 0
    
    # Apply Gaussian blur to tumor layer first (blur everything)
    blur_sigma = 2.0  # Adjust for more/less blur
    for c in range(3):
        tumor_rgb[:, :, c] = gaussian_filter(tumor_rgb[:, :, c], sigma=blur_sigma)
    
    # Then apply the mask to keep background white where no tumor cells
    for c in range(3):
        tumor_rgb[:, :, c] = np.where(tumor_cell_present, tumor_rgb[:, :, c], 1.0)
    
    # Combine: start with tumor layer
    rgb = tumor_rgb.copy()
    
    # Draw vessels and entry points on top (sharp, pixelated)
    vessel_mask = cells[0] > 0
    entry_mask = cells[5] > 0 if cells.shape[0] > 5 else np.zeros((rows, cols), dtype=bool)
    
    rgb[vessel_mask] = endothelial_col
    rgb[entry_mask] = entry_col
    
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
    im = ax.imshow(
        cell_rgb,
        vmin=0,
        vmax=1,
        interpolation='nearest',  # Smooth display for pre-blurred tumor, vessels remain sharp
        origin='lower',
    )
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

    # Calculate and display summary statistics
    print_summary()
    
    plt.ioff()
    plt.show()

def print_summary():
    """Print summary of cell counts and tissue coverage percentages."""
    total_pixels = ROWS * COLS
    
    # Count cells
    active_count = np.sum(grids.CELLS[1])
    quiescent_count = np.sum(grids.CELLS[2])
    necrotic_count = np.sum(grids.CELLS[4])
    migrating_count = np.sum(grids.CELLS[3])
    total_tumor_cells = active_count + quiescent_count + necrotic_count + migrating_count
    
    # Calculate percentages
    active_pct = (active_count / total_pixels) * 100
    quiescent_pct = (quiescent_count / total_pixels) * 100
    necrotic_pct = (necrotic_count / total_pixels) * 100
    migrating_pct = (migrating_count / total_pixels) * 100
    total_tumor_pct = (total_tumor_cells / total_pixels) * 100
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Total grid size: {ROWS} x {COLS} = {total_pixels:,} pixels")
    print(f"Total simulation steps: {STEPS}")
    print("\nCell Counts:")
    print(f"  Active cells:        {active_count:>8,} ({active_pct:>6.2f}% of tissue)")
    print(f"  Quiescent cells:     {quiescent_count:>8,} ({quiescent_pct:>6.2f}% of tissue)")
    print(f"  Necrotic (dead) cells: {necrotic_count:>8,} ({necrotic_pct:>6.2f}% of tissue)")
    print(f"  Migrating cells:     {migrating_count:>8,} ({migrating_pct:>6.2f}% of tissue)")
    print(f"  Total tumor cells:   {total_tumor_cells:>8,} ({total_tumor_pct:>6.2f}% of tissue)")
    print("="*60 + "\n")


