import random
import taichi as ti
import torch
import numpy as np

from util import print_timing_stats, time_function

WIDTH = 500
HEIGHT = 400
RESOLUTION = 2
ti.init()

@ti.data_oriented
class GameOfLife:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.cell_size = RESOLUTION
        
        self.grid = ti.field(dtype=ti.i32, shape=(self.height, self.width))
        self.next_grid = ti.field(dtype=ti.i32, shape=(self.height, self.width))
        
        self.initialize_random()
        
        self.gui = ti.GUI("Conway's Game of Life", 
                         res=(self.width * self.cell_size, self.height * self.cell_size))
        
        self.cell_positions = ti.Vector.field(2, dtype=ti.f32, shape=(self.height, self.width))
        self.cell_colors = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))
        self.precompute_cell_positions()
        
    def precompute_cell_positions(self):
        cell_width = 1.0 / self.width
        cell_height = 1.0 / self.height
        
        for y in range(self.height):
            for x in range(self.width):
                self.cell_positions[y, x] = [
                    (x + 0.5) * cell_width, 
                    (y + 0.5) * cell_height
                ]
                self.cell_colors[y, x] = [1.0, 1.0, 1.0]

    @time_function
    def initialize_random(self):
        random_grid = np.random.choice([0, 1], size=(self.height, self.width))
        
        self.grid.from_numpy(random_grid)

    @time_function
    def conways_fast(self, board: np.ndarray):
        n_neighbour = sum(np.roll(np.roll(board, i, 0), j, 1) 
                       for i in (-1, 0, 1) 
                       for j in (-1, 0, 1) 
                       if (i != 0 or j != 0))
        
        board_new = board.copy()
        board_new[(n_neighbour < 2) | (n_neighbour > 3)] = 0
        board_new[(n_neighbour == 3)] = 1
        
        return board_new
    
    @ti.kernel
    @time_function
    def update_grid(self, new_grid: ti.types.ndarray()):
        for i, j in self.grid:
            self.grid[i, j] = new_grid[i, j]

    @time_function
    def update(self):
        current_grid = self.grid.to_numpy()
        next_grid = self.conways_fast(current_grid)
        self.update_grid(next_grid)

    @time_function
    def draw_cells(self):
        alive_cells = np.where(self.grid.to_numpy() == 1)
        positions = self.cell_positions.to_numpy()[alive_cells]
        if positions.shape[0] > 0:
            self.gui.circles(positions, color=0xFFFFFF, radius=self.cell_size / 2)
        
    def run(self):
        counter = 0
        while self.gui.running:
            counter += 1
            for e in self.gui.get_events():
                if e.key == ti.GUI.ESCAPE:
                    self.gui.running = False
                elif e.key == ti.GUI.SPACE and e.type == ti.GUI.PRESS:
                    self.initialize_random()
            
            self.update()
            if counter % 1 == 0:
                self.draw_cells()
                counter = 0
            else: 
                counter += 1
            self.gui.show()

if __name__ == "__main__":
    game = GameOfLife()
    game.run()
    print_timing_stats()