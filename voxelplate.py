import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import random

from holder.voxelmesh import VoxelGrid
from util import normalize

@dataclass
class VoxelPlate:
    id: int
    voxel_grid: VoxelGrid  # Reference to the underlying voxel grid
    plate_type: str = 'continental'  # 'oceanic' or 'continental'
    growth_rate: float = 0.01  # Rate at which the plate grows (voxels per time step)
    base_elevation: float = 0.0  # Base elevation for this plate
    color: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=np.uint8))
    
    # Movement properties
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    # Internal state
    _boundary_voxels: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=int), init=False)
    _continental_centers: List[np.ndarray] = field(default_factory=list, init=False)
    _continental_border_voxels: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=int), init=False)
    
    def __post_init__(self):
        # Assign a random color if not provided
        if np.array_equal(self.color, [0, 0, 0]):
            self.color = np.array([
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            ], dtype=np.uint8)
        
        # Initialize boundary voxels
        self._update_boundary_voxels()
        
        # For continental plates, identify centers and borders
        if self.plate_type == 'continental':
            self._identify_continental_features()
    
    @property
    def mass(self) -> float:
        """Calculate mass based on number of active voxels"""
        active_voxels = np.sum(self.voxel_grid.data)
        voxel_volume = self.voxel_grid.scale ** 3
        
        # Different densities for different plate types
        density = 2.7 if self.plate_type == 'continental' else 3.0  # g/cm³
        
        return active_voxels * voxel_volume * density
    
    @property
    def boundary_voxels(self) -> np.ndarray:
        """Get the boundary voxels of this plate"""
        return self._boundary_voxels
    
    @property
    def continental_centers(self) -> List[np.ndarray]:
        """Get the continental centers (only for continental plates)"""
        return self._continental_centers
    
    @property
    def continental_border_voxels(self) -> np.ndarray:
        """Get the continental border voxels (only for continental plates)"""
        return self._continental_border_voxels
    
    def _update_boundary_voxels(self):
        """Update the boundary voxels of this plate"""
        self._boundary_voxels = self._find_boundary_voxels()
    
    def _find_boundary_voxels(self) -> np.ndarray:
        """Find all boundary voxels (voxels adjacent to empty space)"""
        if not np.any(self.voxel_grid.data):
            return np.zeros((0, 3), dtype=int)
        
        # Create a padded version for neighbor checking
        padded = np.pad(self.voxel_grid.data.astype(bool), 1, mode='constant')
        
        # Prepare the boundary mask
        boundary_mask = np.zeros_like(self.voxel_grid.data, dtype=bool)
        
        # Check all 6 neighbors
        offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        for dx, dy, dz in offsets:
            # Get the appropriately shifted view
            shifted = padded[1+dx:1+dx+self.voxel_grid.resolution[0], 
                            1+dy:1+dy+self.voxel_grid.resolution[1], 
                            1+dz:1+dz+self.voxel_grid.resolution[2]]
            boundary_mask |= (self.voxel_grid.data == 1) & (shifted == 0)
        
        return np.argwhere(boundary_mask)
    
    def _identify_continental_features(self):
        """Identify continental centers and border voxels for continental plates"""
        if self.plate_type != 'continental':
            return
            
        # Simple implementation: find the largest connected component as the main continent
        from scipy.ndimage import label
        labeled, num_features = label(self.voxel_grid.data)
        
        if num_features == 0:
            self._continental_centers = []
            self._continental_border_voxels = np.zeros((0, 3), dtype=int)
            return
        
        # Find the largest connected component
        sizes = np.bincount(labeled.ravel())
        if len(sizes) <= 1:  # Only background (0)
            self._continental_centers = []
            self._continental_border_voxels = np.zeros((0, 3), dtype=int)
            return
        
        main_component = np.argmax(sizes[1:]) + 1
        
        # Get all voxels in the main component
        main_voxels = np.argwhere(labeled == main_component)
        
        # Calculate center of mass
        center = np.mean(main_voxels, axis=0)
        self._continental_centers = [center]
        
        # Find border voxels (voxels in the main component that are on the boundary)
        boundary_mask = np.zeros_like(self.voxel_grid.data, dtype=bool)
        boundary_mask[tuple(main_voxels.T)] = True
        boundary_mask &= (labeled == main_component)
        
        self._continental_border_voxels = np.argwhere(boundary_mask)
    
    def grow(self):
        """Grow the plate by adding voxels at the boundaries"""
        if len(self._boundary_voxels) == 0:
            return
            
        # Determine number of voxels to add based on growth rate
        num_to_add = max(1, int(len(self._boundary_voxels) * self.growth_rate))
        
        # Randomly select boundary voxels to grow from
        selected_indices = np.random.choice(len(self._boundary_voxels), 
                                          size=min(num_to_add, len(self._boundary_voxels)), 
                                          replace=False)
        
        for idx in selected_indices:
            boundary_voxel = self._boundary_voxels[idx]
            
            # Find empty neighbor positions
            empty_neighbors = []
            offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
            
            for dx, dy, dz in offsets:
                neighbor = boundary_voxel + np.array([dx, dy, dz])
                
                # Check if neighbor is within bounds and empty
                if (np.all(neighbor >= 0) and 
                    np.all(neighbor < self.voxel_grid.resolution) and 
                    self.voxel_grid.data[tuple(neighbor)] == 0):
                    empty_neighbors.append(neighbor)
            
            # Add a new voxel to a random empty neighbor
            if empty_neighbors:
                new_voxel = random.choice(empty_neighbors)
                self.voxel_grid.data[tuple(new_voxel)] = 1
                
                # Update color for the new voxel
                if self.voxel_grid.color.ndim == 3:  # Per-voxel colors
                    self.voxel_grid.color[tuple(new_voxel)] = self.color
        
        # Update boundary voxels after growth
        self._update_boundary_voxels()
        
        # Update continental features if this is a continental plate
        if self.plate_type == 'continental':
            self._identify_continental_features()
    
    def move(self, dt: float):
        """Move the plate according to its velocities"""
        # Apply linear velocity
        self.voxel_grid.origin += self.linear_velocity * dt
        
        # Apply angular velocity (simplified rotation around center)
        center = self.voxel_grid.origin + self.voxel_grid.dimensions / 2
        rotation = np.linalg.norm(self.angular_velocity) * dt
        if rotation > 0:
            axis = np.linalg.norm(self.angular_velocity)
            
            # Rotate each voxel position (simplified approach)
            # In a real implementation, you'd want to rotate the whole grid more efficiently
            active_voxels = np.argwhere(self.voxel_grid.data == 1)
            for voxel in active_voxels:
                world_pos = self.voxel_grid.voxel_to_world(voxel + 0.5)  # Center of voxel
                relative_pos = world_pos - center
                
                # Apply rotation
                rotated_pos = rotate_vector(relative_pos, axis, rotation)
                new_world_pos = center + rotated_pos
                
                # Convert back to voxel coordinates (this is simplified)
                new_voxel = self.voxel_grid.world_to_voxel(new_world_pos)
                
                # Check if new position is within bounds
                if np.all(new_voxel >= 0) and np.all(new_voxel < self.voxel_grid.resolution):
                    # Mark old position as empty
                    self.voxel_grid.data[tuple(voxel)] = 0
                    # Mark new position as filled
                    self.voxel_grid.data[tuple(new_voxel)] = 1
                    
                    # Update color if using per-voxel colors
                    if self.voxel_grid.color.ndim == 3:
                        self.voxel_grid.color[tuple(new_voxel)] = self.voxel_grid.color[tuple(voxel)]
                        self.voxel_grid.color[tuple(voxel)] = 0
        
        # Update boundary voxels after movement
        self._update_boundary_voxels()

def rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a vector around an axis by a given angle"""
    # Rodrigues' rotation formula
    axisn = np.linalg.norm(axis)
    axis = axis / axisn
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    return (v * cos_theta + 
            np.cross(axis, v) * sin_theta + 
            axis * np.dot(axis, v) * (1 - cos_theta))