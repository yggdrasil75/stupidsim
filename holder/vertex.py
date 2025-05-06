from functools import lru_cache
import numpy as np
import torch
from util import time_function


class Vertex:
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=float)
        self.elevation = 0.0
        self.water = 0.0
        self.plate_id = -1
        self.neighbors = {}
        self.river_size = 0.0
        self.water_depth = 0.0
        self.water_volume = 0.0
        self.albedo = 0.06

    @time_function
    def normalize(self, radius=1.0):
        norm = np.linalg.norm(self.pos)
        if norm > 1e-9:
            self.pos = (self.pos / norm) * radius

    def __repr__(self):
        return f"Vertex({self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f})"
    
    @lru_cache
    def distance_to(self, other_vertex, device=None):
        """Calculate distance between this vertex and another vertex."""
        if device is not None:
            v1 = torch.from_numpy(self.pos).to(device)
            v2 = torch.from_numpy(other_vertex.pos).to(device)
            return torch.norm(v1 - v2).item()
        return np.linalg.norm(self.pos - other_vertex.pos)

    @lru_cache
    def direction_to(self, other_vertex, normalize=True, device=None):
        """Get direction vector to another vertex."""
        if device is not None:
            v1 = torch.from_numpy(self.pos).to(device)
            v2 = torch.from_numpy(other_vertex.pos).to(device)
            direction = v2 - v1
            if normalize:
                direction = direction / torch.norm(direction)
            return direction.cpu().numpy()
        
        direction = other_vertex.pos - self.pos
        if normalize:
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                direction = direction / norm
        return direction

    @lru_cache
    def angle_between(self, v1, v2, degrees=True, device=None):
        """
        Calculate the angle between vectors from this vertex to two other vertices.
        v1 and v2 should be Vertex objects.
        """
        vec1 = self.direction_to(v1, normalize=True, device=device)
        vec2 = self.direction_to(v2, normalize=True, device=device)

        if device is not None:
            vec1 = torch.from_numpy(vec1).to(device)
            vec2 = torch.from_numpy(vec2).to(device)
            dot_product = torch.dot(vec1, vec2).clamp(-1.0, 1.0)
            angle = torch.acos(dot_product)
            if degrees:
                angle = torch.rad2deg(angle)
            return angle.item()
        
        dot_product = np.dot(vec1, vec2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        if degrees:
            angle = np.degrees(angle)
        return angle

    def add_neighbor(self, neighbor_vertex, device=None):
        """Add a neighboring vertex with calculated distance."""
        if neighbor_vertex not in self.neighbors:
            distance = self.distance_to(neighbor_vertex, device=device)
            self.neighbors[neighbor_vertex] = distance
            neighbor_vertex.neighbors[self] = distance  # reciprocal

    @lru_cache
    def get_slope(self, neighbor_vertex, device=None):
        """Calculate slope (elevation difference / distance) to a neighbor."""
        if neighbor_vertex not in self.neighbors:
            self.add_neighbor(neighbor_vertex, device=device)
        elevation_diff = neighbor_vertex.elevation - self.elevation
        return elevation_diff / self.neighbors[neighbor_vertex]

    @lru_cache
    def to_tensor(self, device=None):
        """Convert position to PyTorch tensor (if device is specified)."""
        if device is not None:
            return torch.from_numpy(self.pos).to(device)
        return None