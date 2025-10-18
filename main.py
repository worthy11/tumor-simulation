
from grid import create_grid, update_grid
from visualization import animate
from config import STEPS

grid = create_grid()
animate(grid, update_grid, STEPS)
