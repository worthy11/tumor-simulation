import numpy as np
from grids import set_initial_conditions, update_tme
from visualization import animate
import config

def main():
    set_initial_conditions()
    
    animate(
        update_function=update_tme,
        cmap='plasma',
        vmin=np.array([0, 0, 0]),
        vmax=np.array([np.max(config.O2) * 2, np.max(config.G) * 2, np.max(config.CO2) * 2])
    )

if __name__ == "__main__":
    main()

