import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from config import *
import grids

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

def cells_to_rgb(cells):
    """Convert CELLS tensor to RGB image for visualization."""
    rows, cols = cells.shape[1], cells.shape[2]
    rgb = np.ones((rows, cols, 3))
    
    colors = {
        0: np.array([1.0, 1.0, 1.0]),      # healthy - white
        1: np.array([0.0, 1.0, 1.0]),      # endothelial - cyan
        2: np.array([1.0, 0.0, 0.0]),      # active - red
        3: np.array([1.0, 0.647, 0.0]),    # quiescent - orange
        4: np.array([1.0, 1.0, 0.0]),      # migrating - yellow
        5: np.array([0.545, 0.271, 0.075]) # necrotic - brown
    }
    
    for cell_type in [0, 1, 4, 3, 2, 5]:
        mask = cells[cell_type] > 0
        rgb[mask] = colors[cell_type]
    
    return rgb

def animate(initial_concentration, update_function, cmap='viridis', vmin=None, vmax=None, cells=None):
    plt.ion()
    num_plots = 4 if cells is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    concentration = initial_concentration.copy()
    titles = ['O2', 'Glucose', 'CO2']
    
    ims = []
    for idx, ax in enumerate(axes[:3]):
        data = concentration[idx]
        vmin_i = np.min(data) if vmin is None else (vmin[idx] if isinstance(vmin, (list, np.ndarray)) else vmin)
        vmax_i = np.max(data) if vmax is None else (vmax[idx] if isinstance(vmax, (list, np.ndarray)) else vmax)
        
        im = ax.imshow(data, cmap=cmap, interpolation='bilinear', 
                       vmin=vmin_i, vmax=vmax_i, origin='lower')
        plt.colorbar(im, ax=ax, label='Concentration (mM)')
        ax.set_title(f"{titles[idx]} - Step 0")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ims.append(im)
    
    if cells is not None:
        cell_rgb = cells_to_rgb(cells)
        cell_im = axes[3].imshow(cell_rgb, interpolation='nearest', origin='lower')
        axes[3].set_title('Cells - Step 0')
        axes[3].set_xlabel('X')
        axes[3].set_ylabel('Y')
    
    plt.tight_layout()
    
    for step in range(1, STEPS + 1):
        update_function()
        for idx, im in enumerate(ims):
            im.set_data(grids.ENV[idx])
            axes[idx].set_title(f"{titles[idx]} - Step {step}")
        
        if cells is not None:
            cell_rgb = cells_to_rgb(cells)
            cell_im.set_data(cell_rgb)
            axes[3].set_title(f'Cells - Step {step}')
            
            # Print cell statistics
            active_mask = cells[2] > 0
            quiescent_mask = cells[3] > 0
            necrotic_mask = cells[5] > 0
            
            active_count = np.sum(active_mask)
            quiescent_count = np.sum(quiescent_mask)
            necrotic_count = np.sum(necrotic_mask)
            
            active_vitality = np.mean(grids.TUMOR[0][active_mask]) if active_count > 0 else 0
            quiescent_vitality = np.mean(grids.TUMOR[0][quiescent_mask]) if quiescent_count > 0 else 0
            necrotic_vitality = np.mean(grids.TUMOR[0][necrotic_mask]) if necrotic_count > 0 else 0
            
            active_energy = np.mean(grids.TUMOR[1][active_mask]) if active_count > 0 else 0
            quiescent_energy = np.mean(grids.TUMOR[1][quiescent_mask]) if quiescent_count > 0 else 0
            necrotic_energy = np.mean(grids.TUMOR[1][necrotic_mask]) if necrotic_count > 0 else 0
            
            print(f"Step {step}:")
            print(f"  Active: {active_count} (V: {active_vitality:.2e}, E: {active_energy:.2e})")
            print(f"  Quiescent: {quiescent_count} (V: {quiescent_vitality:.2e}, E: {quiescent_energy:.2e})")
            print(f"  Necrotic: {necrotic_count} (V: {necrotic_vitality:.2e}, E: {necrotic_energy:.2e})")
        
        plt.pause(DT)
    
    plt.ioff()
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from config import ROWS, COLS, VESSEL_MAP

class VesselDrawingTool:
    def __init__(self, rows=ROWS, cols=COLS):
        self.rows = rows
        self.cols = cols
        self.vessel_map = VESSEL_MAP.copy()
        self.drawing = False
        self.last_point = None
        self.brush_size = 3
        self.mode = 'draw'
        
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = plt.subplot2grid((10, 10), (0, 0), colspan=10, rowspan=9)
        
        self.im = self.ax.imshow(self.vessel_map, cmap='Reds', interpolation='nearest', 
                                origin='lower', vmin=0, vmax=1)
        self.ax.set_title('Draw Blood Vessels\n[Left Click] Draw | [Right Click] Erase | [Mouse Wheel] Brush Size\n[C] Clear | [R] Reset | [Enter] Done', 
                         fontsize=10, pad=10)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # Add colorbar
        self.cbar = plt.colorbar(self.im, ax=self.ax, label='Vessel Density', fraction=0.046, pad=0.04)
        
        # Add preset pattern buttons if available
        self.buttons = []
        self._add_pattern_buttons()
        
        # Add brush size indicator
        self.brush_circle = Circle((0, 0), self.brush_size, fill=False, 
                                  edgecolor='blue', linewidth=2, alpha=0.5)
        self.ax.add_patch(self.brush_circle)
        self.brush_circle.set_visible(False)
        
        # Add text for brush size
        self.brush_text = self.ax.text(0.02, 0.98, f'Brush Size: {self.brush_size}', 
                                       transform=self.ax.transAxes, 
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.done = False
    
    def _add_pattern_buttons(self):
        """Add buttons for loading preset patterns."""
        pattern_names = ['border', 'grid', 'radial', 'tree', 'concentric']
        button_width = 0.12
        button_height = 0.04
        start_x = 0.1
        y_pos = 0.05
        spacing = 0.15
        
        for idx, pattern_name in enumerate(pattern_names):
            x_pos = start_x + idx * spacing
            ax_button = plt.axes([x_pos, y_pos, button_width, button_height])
            button = Button(ax_button, pattern_name.capitalize())
            button.on_clicked(lambda event, name=pattern_name: self.load_pattern(name))
            self.buttons.append(button)
    
    def load_pattern(self, pattern_name):
        """Load a preset vessel pattern."""
        try:
            self.vessel_map = get_pattern(pattern_name, rows=self.rows, cols=self.cols)
            self.update_display()
            print(f"Loaded pattern: {pattern_name}")
        except Exception as e:
            print(f"Error loading pattern {pattern_name}: {e}")
    
    def on_press(self, event):
        """Handle mouse button press."""
        if event.inaxes != self.ax:
            return
        
        self.drawing = True
        
        if event.button == 1:  # Left click - draw
            self.mode = 'draw'
        elif event.button == 3:  # Right click - erase
            self.mode = 'erase'
        
        x, y = int(round(event.xdata)), int(round(event.ydata))
        self.last_point = (x, y)
        self.draw_at_point(x, y)
    
    def on_release(self, event):
        """Handle mouse button release."""
        self.drawing = False
        self.last_point = None
    
    def on_motion(self, event):
        """Handle mouse motion."""
        if event.inaxes != self.ax:
            self.brush_circle.set_visible(False)
            self.fig.canvas.draw_idle()
            return
        
        # Update brush indicator position
        x, y = event.xdata, event.ydata
        self.brush_circle.center = (x, y)
        self.brush_circle.set_radius(self.brush_size)
        self.brush_circle.set_visible(True)
        
        # Draw if mouse is pressed
        if self.drawing and event.xdata is not None and event.ydata is not None:
            x_int, y_int = int(round(event.xdata)), int(round(event.ydata))
            
            # Draw line between last point and current point for smooth drawing
            if self.last_point is not None:
                self.draw_line(self.last_point[0], self.last_point[1], x_int, y_int)
            else:
                self.draw_at_point(x_int, y_int)
            
            self.last_point = (x_int, y_int)
        
        self.fig.canvas.draw_idle()
    
    def on_scroll(self, event):
        """Handle mouse scroll to change brush size."""
        if event.button == 'up':
            self.brush_size = min(20, self.brush_size + 1)
        elif event.button == 'down':
            self.brush_size = max(1, self.brush_size - 1)
        
        self.brush_text.set_text(f'Brush Size: {self.brush_size}')
        self.brush_circle.set_radius(self.brush_size)
        self.fig.canvas.draw_idle()
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'c':
            # Clear all vessels
            self.vessel_map = np.zeros((self.rows, self.cols))
            self.update_display()
        elif event.key == 'r':
            # Reset to original
            self.vessel_map = VESSEL_MAP.copy()
            self.update_display()
        elif event.key == 'enter':
            # Done drawing
            self.done = True
            plt.close(self.fig)
    
    def draw_at_point(self, x, y):
        """Draw or erase at a specific point with current brush size."""
        value = 1.0 if self.mode == 'draw' else 0.0
        
        for i in range(max(0, y - self.brush_size), min(self.rows, y + self.brush_size + 1)):
            for j in range(max(0, x - self.brush_size), min(self.cols, x + self.brush_size + 1)):
                # Check if point is within circular brush
                dist = np.sqrt((i - y)**2 + (j - x)**2)
                if dist <= self.brush_size:
                    self.vessel_map[i, j] = value
        
        self.update_display()
    
    def draw_line(self, x0, y0, x1, y1):
        """Draw a line between two points using Bresenham's algorithm."""
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        points = []
        
        while True:
            points.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        # Draw at each point along the line
        for px, py in points:
            self.draw_at_point(px, py)
    
    def update_display(self):
        """Update the display with current vessel map."""
        self.im.set_data(self.vessel_map)
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Show the drawing tool and return the modified vessel map."""
        plt.tight_layout()
        plt.show()
        return self.vessel_map

def draw_vessels():
    tool = VesselDrawingTool()
    vessel_map = tool.show()
    return vessel_map

"""
Preset vessel patterns for tumor simulation.
These can be used as starting points or loaded directly.
"""

import numpy as np
from config import ROWS, COLS

def create_border_vessels(rows=ROWS, cols=COLS, thickness=3):
    """Create vessels around the border of the grid."""
    vessel_map = np.zeros((rows, cols))
    vessel_map[:thickness, :] = 1  # Top
    vessel_map[-thickness:, :] = 1  # Bottom
    vessel_map[:, :thickness] = 1  # Left
    vessel_map[:, -thickness:] = 1  # Right
    return vessel_map

def create_grid_pattern(rows=ROWS, cols=COLS, spacing=20, thickness=2):
    """Create a grid pattern of vessels."""
    vessel_map = np.zeros((rows, cols))
    
    # Horizontal lines
    for i in range(0, rows, spacing):
        vessel_map[max(0, i-thickness):min(rows, i+thickness), :] = 1
    
    # Vertical lines
    for j in range(0, cols, spacing):
        vessel_map[:, max(0, j-thickness):min(cols, j+thickness)] = 1
    
    return vessel_map

def create_radial_pattern(rows=ROWS, cols=COLS, center=None, num_rays=8, thickness=2):
    """Create a radial pattern of vessels emanating from center."""
    vessel_map = np.zeros((rows, cols))
    
    if center is None:
        center = (rows // 2, cols // 2)
    
    cy, cx = center
    
    for angle_idx in range(num_rays):
        angle = 2 * np.pi * angle_idx / num_rays
        
        # Draw ray from center to edge
        max_dist = max(rows, cols)
        for dist in range(0, max_dist, 1):
            y = int(cy + dist * np.sin(angle))
            x = int(cx + dist * np.cos(angle))
            
            if 0 <= y < rows and 0 <= x < cols:
                # Add thickness to the line
                for dy in range(-thickness, thickness+1):
                    for dx in range(-thickness, thickness+1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            vessel_map[ny, nx] = 1
            else:
                break
    
    return vessel_map

def create_circle_pattern(rows=ROWS, cols=COLS, center=None, radius=50, thickness=3):
    """Create a circular vessel pattern."""
    vessel_map = np.zeros((rows, cols))
    
    if center is None:
        center = (rows // 2, cols // 2)
    
    cy, cx = center
    
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            if abs(dist - radius) <= thickness:
                vessel_map[i, j] = 1
    
    return vessel_map

def create_tree_pattern(rows=ROWS, cols=COLS, start_side='bottom', branching_angle=30, thickness=2):
    """Create a tree-like branching vessel pattern."""
    vessel_map = np.zeros((rows, cols))
    
    if start_side == 'bottom':
        start_x, start_y = cols // 2, rows - 1
        direction = -np.pi / 2  # Up
    elif start_side == 'top':
        start_x, start_y = cols // 2, 0
        direction = np.pi / 2  # Down
    elif start_side == 'left':
        start_x, start_y = 0, rows // 2
        direction = 0  # Right
    else:  # right
        start_x, start_y = cols - 1, rows // 2
        direction = np.pi  # Left
    
    def draw_branch(x, y, angle, length, depth):
        """Recursively draw branches."""
        if depth <= 0 or length < 5:
            return
        
        # Draw the branch
        end_x = int(x + length * np.cos(angle))
        end_y = int(y + length * np.sin(angle))
        
        # Bresenham's line
        points = get_line_points(int(x), int(y), end_x, end_y)
        for px, py in points:
            if 0 <= py < rows and 0 <= px < cols:
                for dy in range(-thickness, thickness+1):
                    for dx in range(-thickness, thickness+1):
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            vessel_map[ny, nx] = 1
        
        # Draw child branches
        branch_angle = np.radians(branching_angle)
        new_length = length * 0.7
        
        draw_branch(end_x, end_y, angle - branch_angle, new_length, depth - 1)
        draw_branch(end_x, end_y, angle + branch_angle, new_length, depth - 1)
    
    def get_line_points(x0, y0, x1, y1):
        """Get points along a line using Bresenham's algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    # Draw the main trunk and branches
    trunk_length = min(rows, cols) // 3
    draw_branch(start_x, start_y, direction, trunk_length, depth=4)
    
    return vessel_map

def create_random_network(rows=ROWS, cols=COLS, density=0.1, cluster_size=5):
    """Create a random vessel network with clustering."""
    vessel_map = np.zeros((rows, cols))
    
    num_seeds = int(rows * cols * density / (cluster_size ** 2))
    
    for _ in range(num_seeds):
        seed_y = np.random.randint(0, rows)
        seed_x = np.random.randint(0, cols)
        
        # Create a cluster around the seed
        for dy in range(-cluster_size, cluster_size + 1):
            for dx in range(-cluster_size, cluster_size + 1):
                y, x = seed_y + dy, seed_x + dx
                if 0 <= y < rows and 0 <= x < cols:
                    dist = np.sqrt(dy**2 + dx**2)
                    if dist <= cluster_size and np.random.rand() > dist / cluster_size:
                        vessel_map[y, x] = 1
    
    return vessel_map

def create_concentric_circles(rows=ROWS, cols=COLS, center=None, num_circles=5, thickness=2):
    """Create concentric circular vessels."""
    vessel_map = np.zeros((rows, cols))
    
    if center is None:
        center = (rows // 2, cols // 2)
    
    cy, cx = center
    max_radius = min(rows, cols) // 2
    
    for circle_idx in range(num_circles):
        radius = max_radius * (circle_idx + 1) / num_circles
        vessel_map += create_circle_pattern(rows, cols, center, radius, thickness)
    
    vessel_map = np.clip(vessel_map, 0, 1)
    return vessel_map


# Dictionary of all available patterns
PATTERNS = {
    'border': create_border_vessels,
    'grid': create_grid_pattern,
    'radial': create_radial_pattern,
    'circle': create_circle_pattern,
    'tree': create_tree_pattern,
    'random': create_random_network,
    'concentric': create_concentric_circles
}


def get_pattern(pattern_name, **kwargs):
    """
    Get a vessel pattern by name.
    
    Args:
        pattern_name: Name of the pattern ('border', 'grid', 'radial', etc.)
        **kwargs: Additional parameters for the pattern function
    
    Returns:
        numpy array representing the vessel map
    """
    if pattern_name not in PATTERNS:
        raise ValueError(f"Unknown pattern '{pattern_name}'. Available: {list(PATTERNS.keys())}")
    
    return PATTERNS[pattern_name](**kwargs)
