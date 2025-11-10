import numpy as np
from grids import set_initial_conditions, update_tme
from visualization import animate
import config

def main():
    set_initial_conditions()
    
    animate(
        update_function=update_tme,
        cmap='plasma',
        vmin=np.zeros(6),
        vmax=np.array([
            np.max(config.O2) * 2, 
            np.max(config.RHO_TC) * 2,
            1.5e-9,
            5e-11,
            1.,
            60.
        ])
    )

if __name__ == "__main__":
    main()

