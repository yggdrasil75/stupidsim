from dataclasses import dataclass, field
import numpy as np
from typing import List

@dataclass
class Plate:
    """Represents a tectonic plate in the world simulation."""
    ID: int
    vertex_ids: np.ndarray  # Vertex indices belonging to this plate
    speed: np.ndarray  # Overall movement speed of the plate
    angular_velocity: np.ndarray  # Rotation speed (radians per time step)
    linear_velocity: np.ndarray  # 3D vector for linear movement
    plate_type: str  # 'oceanic' or 'continental'
    growth_rate: np.ndarray  # Rate at which the plate grows at boundaries
    base_elevation: np.ndarray  # Base elevation for this plate
    continental_centers: List[int]  # Vertex IDs of continental centers
    continental_border_vertices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # New field
    mass: np.ndarray = field(init=False)  # Will be calculated
    collision_force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))  # Collision force vector
    
    def calculate_mass(self):
        """Calculate plate mass based on vertex count, base elevation and growth rate"""
        vertex_factor = len(self.vertex_ids) / 1000.0  # Normalize vertex count
        elevation_factor = (self.base_elevation - 6367) / 5.0  # Normalize elevation
        growth_factor = self.growth_rate * 10.0
        
        # Combine factors - continental plates are heavier
        if self.plate_type == 'continental':
            self.mass = np.array(1.5 + vertex_factor + elevation_factor + growth_factor, dtype=np.float32)
        else:
            self.mass = np.array(1.0 + vertex_factor * 0.8 + elevation_factor * 0.5 + growth_factor * 0.7, dtype=np.float32)
    
    def update_continental_borders(self, all_vertices: np.ndarray, threshold: float = 0.2):
        """
        Identify continental border vertices - where continental crust meets oceanic.
        Returns vertex IDs that are on continental borders.
        """
        if not self.continental_centers:
            self.continental_border_vertices = np.array([], dtype=np.int64)
            return
            
        # Get positions of our continental vertices
        continental_verts = self.get_continental_vertices()
        if len(continental_verts) == 0:
            self.continental_border_vertices = np.array([], dtype=np.int64)
            return
            
        # Find boundary vertices that are also continental
        all_boundaries = self.get_boundary_vertices(all_vertices, threshold)
        continental_boundaries = np.array(
            [v for v in all_boundaries if v in continental_verts],
            dtype=np.int64
        )
        
        self.continental_border_vertices = continental_boundaries

    @classmethod
    def create_oceanic_plate(cls, ID, vertex_ids: np.ndarray, speed: float = 1.0):
        """Factory method for creating an oceanic plate."""
        if type(vertex_ids) is not np.ndarray:
            vertex_ids = np.array([vertex_ids])
        return cls(
            ID=ID,
            vertex_ids=vertex_ids,
            speed=np.array(speed, dtype=np.float32),
            angular_velocity=np.array(0.01, dtype=np.float32),
            linear_velocity=np.random.normal(0, 0.1, 3).astype(np.float32),
            plate_type="oceanic",
            growth_rate=np.array(0.05, dtype=np.float32),
            base_elevation=np.array(6367, dtype=np.float32),  # Oceanic plates are lower
            continental_centers=[]
        )
    
    @classmethod
    def create_continental_plate(cls, ID, vertex_ids: np.ndarray, speed: float = 0.5):
        """Factory method for creating a continental plate."""
        return cls(
            ID=ID,
            vertex_ids=vertex_ids,
            speed=np.array(speed, dtype=np.float32),
            angular_velocity=np.array(0.005, dtype=np.float32),
            linear_velocity=np.random.normal(0, 0.05, 3).astype(np.float32),
            plate_type="continental",
            growth_rate=np.array(0.02, dtype=np.float32),
            base_elevation=np.array(6372, dtype=np.float32),  # Continental plates are higher
            continental_centers=[]
        )
    
    def add_continental_center(self, vertex_id: int):
        """Add a continental center to this plate."""
        self.continental_centers.append(vertex_id)
    
    def move(self):
        norm = np.linalg.norm(self.linear_velocity)
        if norm > 0:
            self.linear_velocity = (self.linear_velocity / norm) * self.speed
        
        self.angular_velocity = np.clip(
            self.angular_velocity, -0.1, 0.1)
    
    def grow(self, amount: float = 0.0):
        """Grow the plate by adjusting its growth rate."""
        if amount is None:
            amount = self.growth_rate.item()
        self.growth_rate = np.clip(
            self.growth_rate + amount, 0.0, 0.2)
    
    def get_boundary_vertices(self, all_vertices: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Identify boundary vertices of the plate.
        Returns vertex IDs that are on the boundary of the plate.
        """
        # Get positions of our vertices
        our_verts = all_vertices[self.vertex_ids]
        
        # For each vertex, find distance to nearest non-plate vertex
        # Create mask for non-plate vertices
        mask = np.ones(all_vertices.shape[0], dtype=bool)
        mask[self.vertex_ids] = False
        other_verts = all_vertices[mask]
        
        # Calculate distances between our vertices and other vertices
        distances = np.linalg.norm(our_verts[:, np.newaxis] - other_verts, axis=2)
        
        # Find minimum distance to non-plate vertices for each plate vertex
        min_distances = np.min(distances, axis=1)
        
        # Boundary vertices are those close to non-plate vertices
        boundary_mask = min_distances < threshold
        return self.vertex_ids[boundary_mask]
    
    def get_continental_vertices(self) -> np.ndarray:
        """Get all vertices that are part of continental landmasses."""
        if not self.continental_centers:
            return np.array([], dtype=np.int64)
        
        # For each continental center, find nearby vertices
        # (Implementation depends on your world structure)
        # This is a placeholder - you'd want to implement proper continent detection
        continental_verts = []
        for center in self.continental_centers:
            continental_verts.append(center)
        
        return np.array(continental_verts, dtype=np.int64)