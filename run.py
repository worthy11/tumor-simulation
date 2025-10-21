import numpy as np
from grids import ENV, grow_tumor
from visualization import animate, draw_vessels
import config

def main():
    new_vessel_map = draw_vessels()
    config.VESSEL_MAP[:] = new_vessel_map
    
    initial_env = ENV.copy()
    
    grow_tumor()
    
    animate(
        initial_concentration=initial_env,
        update_function=grow_tumor,
        cmap='plasma',
        vmin=np.array([0, 0, 0]),
        vmax=np.array([np.max(initial_env[0]) * 2, np.max(initial_env[1]) * 2, np.max(initial_env[2]) * 2]),
        cells=config.CELLS
    )

if __name__ == "__main__":
    main()

