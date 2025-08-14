
import random
import time
import os

def create_grid(rows, cols):
    # Create a grid with rows x cols cells, initially all dead (0).
    # Each cell will have a value of either 0 or 1, where 1 represents an alive cell.
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    return grid

def initialize_grid(grid):
    # Randomly populate the grid with alive cells.
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if random.choice([True, False]):
                grid[row][col] = 1
    return grid

def get_neighbors(grid, row, col):
    # Get the neighbors of a given cell.
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue  # Skip the current cell itself
            new_row = row + i
            new_col = col + j
            if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                neighbors.append(grid[new_row][new_col])
    return neighbors

def clear_screen():
    # Clear the terminal screen using ANSI escape codes.
    os.system('cls' if os.name == 'nt' else 'clear')


def next_generation(grid):
    # Calculate the next generation based on Conway's Game of Life rules.
    next_gen = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            alive_neighbors = sum(get_neighbors(grid, row, col))
            if grid[row][col] == 1:
                if alive_neighbors < 2 or alive_neighbors > 3:
                    next_gen[row][col] = 0  # Dies due to underpopulation or overpopulation
                else:
                    next_gen[row][col] = 1  # Survives
            else:
                if alive_neighbors == 3:
                    next_gen[row][col] = 1  # Becomes alive due to reproduction
    return next_gen

def display_grid(grid):
    # Display the grid state using 'X' for alive and '.' for dead cells.
    for row in grid:
        print(''.join(['X' if cell else '.' for cell in row]))

def main():
    # Main function that sets up the game loop.
    rows, cols = 20, 40
    grid = create_grid(rows, cols)
    grid = initialize_grid(grid)
    while True:
        clear_screen()
        display_grid(grid)
        time.sleep(0.5)
        grid = next_generation(grid)

if __name__ == '__main__':
    main()
