
import numpy as np
import random
from config import ROWS, COLS, DIVISION_PROB, DEATH_PROB

def create_grid():
    grid = np.zeros((ROWS, COLS), dtype=int)
    grid[ROWS//2, COLS//2] = 1  
    return grid

def update_grid(grid):
    new_grid = grid.copy()
    rows, cols = grid.shape
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                if random.random() < DEATH_PROB:
                    new_grid[r, c] = 0
                    continue
                if random.random() < DIVISION_PROB:
                    neighbors = [(r-1,c-1),(r-1,c),(r-1,c+1),
                                 (r,c-1),        (r,c+1),
                                 (r+1,c-1),(r+1,c),(r+1,c+1)]
                    random.shuffle(neighbors)
                    for nr, nc in neighbors:
                        if 0 <= nr < rows and 0 <= nc < cols and new_grid[nr, nc] == 0:
                            new_grid[nr, nc] = 1
                            break
    return new_grid

