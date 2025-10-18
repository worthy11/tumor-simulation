
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
