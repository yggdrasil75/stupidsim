import random
import taichi as ti
import torch
import numpy as np

from util import print_timing_stats, time_function

WIDTH = 100
HEIGHT = 100
RESOLUTION = 4
ti.init()

@ti.data_oriented
class GameOfLife:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.cell_size = RESOLUTION
        
        # Grid stores both alive status (0 or 1) and red value (0-255)
        self.grid = ti.Vector.field(2, dtype=ti.i32, shape=(self.height, self.width))
        self.next_grid = ti.Vector.field(2, dtype=ti.i32, shape=(self.height, self.width))
        
        self.initialize_random()
        
        self.gui = ti.GUI("Trees Game of Life", 
                         res=(self.width * self.cell_size, self.height * self.cell_size))
        
        self.cell_positions = ti.Vector.field(2, dtype=ti.f32, shape=(self.height, self.width))
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

    @time_function
    def initialize_random(self):
        # Initialize with random alive status and random red values (0-255)
        random_status = np.random.choice([0, 1], size=(self.height, self.width))
        random_red = np.random.randint(0, 256, size=(self.height, self.width))
        
        combined = np.stack([random_status, random_red], axis=-1)
        self.grid.from_numpy(combined)

    @time_function
    def extended_rules(self, board: np.ndarray):
        # Extract status and red values
        status = board[:, :, 0]
        red = board[:, :, 1]
        
        # Calculate number of neighbors (excluding self)
        n_neighbour = sum(np.roll(np.roll(status, i, 0), j, 1) 
                       for i in (-1, 0, 1) 
                       for j in (-1, 0, 1) 
                       if (i != 0 or j != 0))
        
        # Calculate average red value of neighbors for reproduction
        neighbor_red = sum(np.roll(np.roll(red * status, i, 0), j, 1) 
                        for i in (-1, 0, 1) 
                        for j in (-1, 0, 1) 
                        if (i != 0 or j != 0))
        neighbor_count = n_neighbour.copy()
        neighbor_count[neighbor_count == 0] = 1  # avoid division by zero
        avg_red = neighbor_red // neighbor_count
        
        board_new = board.copy()
        
        min_neighbors = np.clip(3 - (red / 255), 1, 4)
        max_neighbors = np.clip(3 + (red / 255), 2, 5)
        
        # Survival rules
        survives = (n_neighbour >= min_neighbors) & (n_neighbour <= max_neighbors)
        board_new[:, :, 0] = status & survives  # Current cells that survive
        
        # Reproduction rules (dead cells with exactly 3 neighbors come alive)
        reproduces = (status == 0) & (n_neighbour == 3)
        board_new[:, :, 0] = board_new[:, :, 0] | reproduces
        
        if np.any(reproduces):
            total_red = np.zeros_like(red)
            total_alive = np.zeros_like(red)
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if i != 0 or j != 0:
                        shifted_red = np.roll(np.roll(red, i, 0), j, 1)
                        shifted_alive = np.roll(np.roll(status, i, 0), j, 1)
                        total_red += shifted_red * shifted_alive
                        total_alive += shifted_alive
            
            new_red = np.where(total_alive > 0, total_red / total_alive, 128) * np.random.uniform(0.99,1.01)
            board_new[:, :, 1] = np.where(reproduces, new_red, board_new[:, :, 1])
        
        return board_new

    @ti.kernel
    #@time_function
    def update_grid(self, new_grid: ti.types.ndarray()):
        for i, j in self.grid:
            self.grid[i, j] = ti.Vector([new_grid[i, j, 0], new_grid[i, j, 1]])

    @time_function
    def update(self):
        current_grid = self.grid.to_numpy()
        next_grid = self.extended_rules(current_grid)
        self.update_grid(next_grid)

    @time_function
    def draw_cells(self, init_shape):
        grid_np = self.grid.to_numpy()
        alive_mask = grid_np[:, :, 0] == 1
        positions = self.cell_positions.to_numpy()[alive_mask]
        red_values = grid_np[:, :, 1][alive_mask]
        if init_shape != self.gui.res:
            cellmult = self.cell_size / 2 * np.divide(init_shape,self.gui.res)
        else:
            cellmult = self.cell_size / 2
        if positions.shape[0] > 0:
            # Convert red values to 0xRRGGBB format (GG=BB=0)
            colors = (red_values.astype(np.uint32) << 16)  # Equivalent to 0xRR0000
            self.gui.circles(positions, radius=cellmult, color=colors)
        #print(self.gui.__dict__)
        
    def run(self):
        counter = 0
        init_shape = self.gui.res
        while self.gui.running:
            
            counter += 1
            for e in self.gui.get_events():
                if e.key == ti.GUI.ESCAPE:
                    self.gui.running = False
                elif e.key == ti.GUI.SPACE and e.type == ti.GUI.PRESS:
                    self.initialize_random()
            
            self.update()
            self.draw_cells(init_shape)
            self.gui.show()

if __name__ == "__main__":
    game = GameOfLife()
    game.run()
    print_timing_stats()