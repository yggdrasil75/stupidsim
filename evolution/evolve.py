import random
import taichi as ti
import torch
import numpy as np

from util import print_timing_stats, time_function

WIDTH = 50
HEIGHT = 30
RESOLUTION = 10
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
        cell_width = 1.0 / self.width
        cell_height = 1.0 / self.height
        
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x]:
                    x1 = x * cell_width
                    y1 = y * cell_height
                    x2 = (x + 1) * cell_width
                    y2 = (y + 1) * cell_height
                    
                    self.gui.rect(topleft=(x1, y1), 
                                 bottomright=(x2, y2), 
                                 color=0xFFFFFF)
    
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
            if counter % 5 == 0:
                self.draw_cells()
                counter = 0
            else: counter +=1
            self.gui.show()

if __name__ == "__main__":
    game = GameOfLife()
    game.run()
    print_timing_stats()