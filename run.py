import numpy as np
from grids import O2, update_O2_concentration
from visualization import animate, draw_vessels
import config

def main():
    new_vessel_map = draw_vessels()
    config.VESSEL_MAP[:] = new_vessel_map
    
    initial_O2 = O2.copy()
    def update_func():
        return update_O2_concentration()
    
    animate(
        initial_concentration=initial_O2,
        update_function=update_func,
        cmap='plasma',
        vmin=0,
        vmax=np.max(initial_O2) * 2,
    )

if __name__ == "__main__":
    main()

