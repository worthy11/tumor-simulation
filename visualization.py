
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import numpy as np

cmap = ListedColormap(['white', 'red'])

def animate(grid, update_function, steps):
    plt.ion()
    fig, ax = plt.subplots()
    
    for step in range(steps):
        ax.clear()
        ax.imshow(grid, cmap=cmap, interpolation='none')
        ax.set_title(f"Krok {step}")
        ax.axis('off')
        plt.pause(0.1)
        grid = update_function(grid)
    
    plt.ioff()
    plt.show()


def animate_concentration(initial_concentration, update_function, steps, title="Concentration", 
                          cmap='viridis', vmin=None, vmax=None, pause_time=0.1):
    """
    Animate changes in concentration over time.
    
    Parameters:
    -----------
    initial_concentration : numpy array
        Initial concentration values
    update_function : callable
        Function that updates the concentration array
    steps : int
        Number of time steps to simulate
    title : str
        Base title for the plot
    cmap : str
        Colormap name (e.g., 'viridis', 'plasma', 'hot', 'coolwarm')
    vmin, vmax : float or None
        Min and max values for colorbar. If None, auto-scaled
    pause_time : float
        Pause time between frames in seconds
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initial plot
    concentration = initial_concentration.copy()
    
    # Auto-scale if not provided
    if vmin is None:
        vmin = np.min(concentration)
    if vmax is None:
        vmax = np.max(concentration)
    
    im = ax.imshow(concentration, cmap=cmap, interpolation='bilinear', 
                   vmin=vmin, vmax=vmax, origin='lower')
    cbar = plt.colorbar(im, ax=ax, label='Concentration (mM)')
    ax.set_title(f"{title} - Step 0")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    
    for step in range(1, steps + 1):
        # Update concentration
        concentration = update_function()
        
        # Update plot
        im.set_data(concentration)
        
        # Optionally update color limits based on current data
        # Uncomment these lines if you want dynamic range scaling
        # current_min = np.min(concentration)
        # current_max = np.max(concentration)
        # im.set_clim(current_min, current_max)
        
        ax.set_title(f"{title} - Step {step}")
        
        plt.pause(pause_time)
    
    plt.ioff()
    plt.show()


def save_concentration_animation(initial_concentration, update_function, steps, 
                                 filename="concentration_animation.gif",
                                 title="Concentration", cmap='viridis', 
                                 vmin=None, vmax=None, fps=10):
    """
    Save concentration animation as a file.
    
    Parameters:
    -----------
    filename : str
        Output filename (e.g., 'animation.gif', 'animation.mp4')
    fps : int
        Frames per second
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    concentration = initial_concentration.copy()
    concentrations = [concentration.copy()]
    
    # Generate all frames
    for step in range(steps):
        concentration = update_function()
        concentrations.append(concentration.copy())
    
    # Auto-scale if not provided
    if vmin is None:
        vmin = min(np.min(c) for c in concentrations)
    if vmax is None:
        vmax = max(np.max(c) for c in concentrations)
    
    im = ax.imshow(concentrations[0], cmap=cmap, interpolation='bilinear',
                   vmin=vmin, vmax=vmax, origin='lower')
    cbar = plt.colorbar(im, ax=ax, label='Concentration (mM)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    title_text = ax.set_title(f"{title} - Step 0")
    
    def animate_frame(frame):
        im.set_data(concentrations[frame])
        title_text.set_text(f"{title} - Step {frame}")
        return [im, title_text]
    
    anim = animation.FuncAnimation(fig, animate_frame, frames=len(concentrations),
                                   interval=1000/fps, blit=True)
    
    anim.save(filename, writer='pillow', fps=fps)
    plt.close()
    print(f"Animation saved as {filename}")
