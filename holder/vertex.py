import numpy as np


class Vertex:
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=float)
        self.elevation = 0.0
        self.water = 0.0
        self.plate_id = -1
        self.neighbors = {}
        self.river_size = 0.0

    def normalize(self, radius=1.0):
        norm = np.linalg.norm(self.pos)
        if norm > 1e-9:
            self.pos = (self.pos / norm) * radius

    def __repr__(self):
        return f"Vertex({self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f})"
    
    #def get_nearest_neighbor():
