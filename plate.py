from dataclasses import dataclass
import torch
from typing import List

@dataclass
class Plate:
    """Represents a tectonic plate in the world simulation."""
    vertex_ids: torch.Tensor  # Vertex indices belonging to this plate
    speed: torch.Tensor  # Overall movement speed of the plate
    angular_velocity: torch.Tensor  # Rotation speed (radians per time step)
    linear_velocity: torch.Tensor  # 3D vector for linear movement
    plate_type: str  # 'oceanic' or 'continental'
    growth_rate: torch.Tensor  # Rate at which the plate grows at boundaries
    base_elevation: torch.Tensor  # Base elevation for this plate
    continental_centers: List[int]  # Vertex IDs of continental centers
    
    @classmethod
    def create_oceanic_plate(cls, vertex_ids: torch.Tensor, speed: float = 1.0):
        """Factory method for creating an oceanic plate."""
        return cls(
            vertex_ids=vertex_ids,
            speed=torch.tensor(speed, dtype=torch.float32),
            angular_velocity=torch.tensor(0.01, dtype=torch.float32),
            linear_velocity=torch.randn(3, dtype=torch.float32).normal_(0, 0.1),
            plate_type="oceanic",
            growth_rate=torch.tensor(0.05, dtype=torch.float32),
            base_elevation=torch.tensor(-0.8, dtype=torch.float32),  # Oceanic plates are lower
            continental_centers=[]
        )
    
    @classmethod
    def create_continental_plate(cls, vertex_ids: torch.Tensor, speed: float = 0.5):
        """Factory method for creating a continental plate."""
        return cls(
            vertex_ids=vertex_ids,
            speed=torch.tensor(speed, dtype=torch.float32),
            angular_velocity=torch.tensor(0.005, dtype=torch.float32),
            linear_velocity=torch.randn(3, dtype=torch.float32).normal_(0, 0.05),
            plate_type="continental",
            growth_rate=torch.tensor(0.02, dtype=torch.float32),
            base_elevation=torch.tensor(0.2, dtype=torch.float32),  # Continental plates are higher
            continental_centers=[]
        )
    
    def add_continental_center(self, vertex_id: int):
        """Add a continental center to this plate."""
        self.continental_centers.append(vertex_id)
    
    def move(self):
        """Update plate movement based on velocities."""
        # Update linear velocity (can add more complex physics here)
        self.linear_velocity = torch.nn.functional.normalize(
            self.linear_velocity, dim=0) * self.speed
        
        # Update angular velocity (can add torque effects here)
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