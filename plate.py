from dataclasses import dataclass, field
import torch
from typing import List

@dataclass
class Plate:
    """Represents a tectonic plate in the world simulation."""
    ID: int
    vertex_ids: torch.Tensor  # Vertex indices belonging to this plate
    speed: torch.Tensor  # Overall movement speed of the plate
    angular_velocity: torch.Tensor  # Rotation speed (radians per time step)
    linear_velocity: torch.Tensor  # 3D vector for linear movement
    plate_type: str  # 'oceanic' or 'continental'
    growth_rate: torch.Tensor  # Rate at which the plate grows at boundaries
    base_elevation: torch.Tensor  # Base elevation for this plate
    continental_centers: List[int]  # Vertex IDs of continental centers
    continental_border_vertices: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))  # New field
    mass: torch.Tensor = field(init=False)  # Will be calculated
    collision_force: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float32))  # Collision force vector
    
    
    def calculate_mass(self):
        """Calculate plate mass based on vertex count, base elevation and growth rate"""
        vertex_factor = len(self.vertex_ids) / 1000.0  # Normalize vertex count
        elevation_factor = (self.base_elevation - 6367) / 5.0  # Normalize elevation
        growth_factor = self.growth_rate * 10.0
        
        # Combine factors - continental plates are heavier
        if self.plate_type == 'continental':
            self.mass = torch.tensor(1.5 + vertex_factor + elevation_factor + growth_factor, dtype=torch.float32)
        else:
            self.mass = torch.tensor(1.0 + vertex_factor * 0.8 + elevation_factor * 0.5 + growth_factor * 0.7, dtype=torch.float32)
    
    def update_continental_borders(self, all_vertices: torch.Tensor, threshold: float = 0.2):
        """
        Identify continental border vertices - where continental crust meets oceanic.
        Returns vertex IDs that are on continental borders.
        """
        if not self.continental_centers:
            self.continental_border_vertices = torch.tensor([], dtype=torch.long)
            return
            
        # Get positions of our continental vertices
        continental_verts = self.get_continental_vertices()
        if len(continental_verts) == 0:
            self.continental_border_vertices = torch.tensor([], dtype=torch.long)
            return
            
        # Find boundary vertices that are also continental
        all_boundaries = self.get_boundary_vertices(all_vertices, threshold)
        continental_boundaries = torch.tensor(
            [v for v in all_boundaries if v in continental_verts],
            dtype=torch.long
        )
        
        self.continental_border_vertices = continental_boundaries

    @classmethod
    def create_oceanic_plate(cls, ID, vertex_ids: torch.Tensor, speed: float = 1.0):
        """Factory method for creating an oceanic plate."""
        return cls(
            ID = ID,
            vertex_ids=vertex_ids,
            speed=torch.tensor(speed, dtype=torch.float32),
            angular_velocity=torch.tensor(0.01, dtype=torch.float32),
            linear_velocity=torch.randn(3, dtype=torch.float32).normal_(0, 0.1),
            plate_type="oceanic",
            growth_rate=torch.tensor(0.05, dtype=torch.float32),
            base_elevation=torch.tensor(6367, dtype=torch.float32),  # Oceanic plates are lower
            continental_centers=[]
        )
    
    @classmethod
    def create_continental_plate(cls, ID, vertex_ids: torch.Tensor, speed: float = 0.5):
        """Factory method for creating a continental plate."""
        return cls(
            ID = ID,
            vertex_ids=vertex_ids,
            speed=torch.tensor(speed, dtype=torch.float32),
            angular_velocity=torch.tensor(0.005, dtype=torch.float32),
            linear_velocity=torch.randn(3, dtype=torch.float32).normal_(0, 0.05),
            plate_type="continental",
            growth_rate=torch.tensor(0.02, dtype=torch.float32),
            base_elevation=torch.tensor(6372, dtype=torch.float32),  # Continental plates are higher
            continental_centers=[]
        )
    
    def add_continental_center(self, vertex_id: int):
        """Add a continental center to this plate."""
        self.continental_centers.append(vertex_id)
    
    def move(self):
        self.linear_velocity = torch.nn.functional.normalize(
            self.linear_velocity, dim=0) * self.speed
        
        self.angular_velocity = torch.clamp(
            self.angular_velocity, -0.1, 0.1)
    
    def grow(self, amount: float = 0.0):
        """Grow the plate by adjusting its growth rate."""
        if amount is None:
            amount = self.growth_rate.item()
        self.growth_rate = torch.clamp(
            self.growth_rate + amount, 0.0, 0.2)
    
    def get_boundary_vertices(self, all_vertices: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """
        Identify boundary vertices of the plate.
        Returns vertex IDs that are on the boundary of the plate.
        """
        # Get positions of our vertices
        our_verts = all_vertices[self.vertex_ids]
        
        # For each vertex, find distance to nearest non-plate vertex
        # (This is a simplified approach - could be optimized)
        distances = torch.cdist(our_verts, all_vertices)
        
        # Mask to exclude our own vertices
        mask = torch.ones(all_vertices.shape[0], dtype=torch.bool)
        mask[self.vertex_ids] = False
        
        # Find minimum distance to non-plate vertices for each plate vertex
        min_distances = distances[:, mask].min(dim=1).values
        
        # Boundary vertices are those close to non-plate vertices
        boundary_mask = min_distances < threshold
        return self.vertex_ids[boundary_mask]
    
    def get_continental_vertices(self) -> torch.Tensor:
        """Get all vertices that are part of continental landmasses."""
        if not self.continental_centers:
            return torch.tensor([], dtype=torch.long)
        
        # For each continental center, find nearby vertices
        # (Implementation depends on your world structure)
        # This is a placeholder - you'd want to implement proper continent detection
        continental_verts = []
        for center in self.continental_centers:
            continental_verts.append(center)
        
        return torch.tensor(continental_verts, dtype=torch.long)