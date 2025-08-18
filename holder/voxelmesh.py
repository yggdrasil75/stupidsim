import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple
from util import norm, normalize, cross, time_function

_voxel_cache = OrderedDict()

@dataclass
class VoxelGrid:
    id: int
    resolution: Tuple[int, int, int]  # (width, height, depth) in voxels
    origin: np.ndarray  # 3D position of grid origin (world coordinates)
    scale: float  # size of each voxel in world units
    data: np.ndarray  # 3D array of occupancy/values (0=empty, 1=solid)
    color: np.ndarray = field(default_factory=lambda: np.array([255, 255, 255], dtype=np.uint8))
    interactive: bool = True
    physics: bool = True
    mass: float = 1.0
    restitution: float = 0.3
    linearVelocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angularVelocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    _needs_octree: bool = field(default=True, init=False)
    _octree: dict = field(default_factory=dict, init=False)
    _lod_grids: List['VoxelGrid'] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.uint8)
        if not np.array_equal(self.data.shape, self.resolution):
            raise ValueError(f"Data shape {self.data.shape} must match resolution {self.resolution}")
        
        # Handle color initialization
        if self.color is None:
            self.color = np.array([255, 255, 255], dtype=np.uint8)
        elif isinstance(self.color, np.ndarray) and self.color.ndim == 1:
            self.color = np.tile(self.color, (*self.resolution, 1))
    
    @property
    def dimensions(self) -> np.ndarray:
        """Return physical dimensions of the voxel grid in world units"""
        return np.array(self.resolution) * self.scale
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min, max) bounding box coordinates"""
        min_bound = self.origin
        max_bound = self.origin + self.dimensions
        return min_bound, max_bound
    
    def world_to_voxel(self, point: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel indices"""
        return ((point - self.origin) / self.scale).astype(int)
    
    def voxel_to_world(self, index: np.ndarray) -> np.ndarray:
        """Convert voxel indices to world coordinates"""
        return self.origin + (index * self.scale)
    
    def get_surface_voxels(self) -> np.ndarray:
        """Get indices of all surface voxels (exterior faces)"""
        if not np.any(self.data):
            return np.zeros((0, 3), dtype=int)
        
        # Create a padded version for neighbor checking
        padded = np.pad(self.data.astype(bool), 1, mode='constant')
        
        # Prepare the surface mask
        surface_mask = np.zeros_like(self.data, dtype=bool)
        
        # Check all 6 neighbors
        offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        for dx, dy, dz in offsets:
            # Get the appropriately shifted view
            shifted = padded[1+dx:1+dx+self.resolution[0], 
                        1+dy:1+dy+self.resolution[1], 
                        1+dz:1+dz+self.resolution[2]]
            surface_mask |= (self.data == 1) & (shifted == 0)
        
        return np.argwhere(surface_mask)
    
    def generate_lods(self, levels=3, factor=2):
        """Generate lower LOD versions by downsampling"""
        self._lod_grids = []
        current_data = self.data
        current_scale = self.scale
        
        for _ in range(levels):
            # Downsample by averaging blocks of factor^3 voxels
            new_shape = (np.array(current_data.shape) // factor).astype(int)
            downsampled = np.zeros(new_shape, dtype=np.uint8)
            
            # Simple majority voting for downsampling
            for x in range(new_shape[0]):
                for y in range(new_shape[1]):
                    for z in range(new_shape[2]):
                        block = current_data[x*factor:(x+1)*factor, 
                                            y*factor:(y+1)*factor, 
                                            z*factor:(z+1)*factor]
                        # Fix: explicitly convert mean to scalar with item()
                        downsampled[x,y,z] = 1 if np.mean(block).item() > 0.5 else 0
            
            # Create new voxel grid with larger scale
            new_scale = current_scale * factor
            lod_grid = VoxelGrid(
                id=self.id * 100 + len(self._lod_grids) + 1,
                resolution=new_shape,
                origin=self.origin,
                scale=new_scale,
                data=downsampled,
                color=self.color,
                interactive=self.interactive,
                physics=self.physics,
                mass=self.mass,
                restitution=self.restitution,
                linearVelocity=self.linearVelocity.copy(),
                angularVelocity=self.angularVelocity.copy()
            )
            
            self._lod_grids.append(lod_grid)
            current_data = downsampled
            current_scale = new_scale
    
    def get_lod(self, level=0):
        """Get a specific LOD level (0 is highest detail)"""
        if level == 0 or not self._lod_grids:
            return self
        if level > len(self._lod_grids):
            return self._lod_grids[-1]
        return self._lod_grids[level-1]
    
    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_dist=100.0) -> Tuple[bool, np.ndarray]:
        """Cast a ray through the voxel grid, return (hit, position)"""
        direction = normalize(direction)
        current_pos = origin.copy()
        step = direction * self.scale * 0.5  # conservative step size
        
        for _ in range(int(max_dist / self.scale * 2)):
            voxel_idx = self.world_to_voxel(current_pos)
            
            # Check bounds
            if np.any(voxel_idx < 0) or np.any(voxel_idx >= self.resolution):
                return False, None
                
            # Check occupancy
            if self.data[tuple(voxel_idx)] == 1:
                return True, current_pos
                
            current_pos += step
            
        return False, None

@time_function
def project_voxels_2d(grids: List[VoxelGrid], eye: np.ndarray, lookat: np.ndarray, up: np.ndarray,
                     fov: float = 90.0, res: Tuple[int,int] = (800,600), 
                     near: float = 1.0, far: float = 1000.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Project voxel grids to 2D screen space"""
    # Compute view and projection matrices
    zAxis = lookat - eye
    zAxis = zAxis / np.linalg.norm(zAxis)
    xAxis = np.linalg.cross(up, zAxis)
    xAxis = xAxis / np.linalg.norm(xAxis)
    yAxis = np.linalg.cross(zAxis, xAxis)

    viewMatrix = np.eye(4, dtype=np.float32)
    viewMatrix[:3, 0] = xAxis
    viewMatrix[:3, 1] = yAxis
    viewMatrix[:3, 2] = -zAxis
    viewMatrix[:3, 3] = eye
    viewMatrix = np.linalg.inv(viewMatrix)

    aspectRatio = res[0] / res[1]
    fovRad = np.radians(fov)
    tanhalf = np.tan(fovRad / 2.0)
    
    projMatrix = np.zeros((4,4), dtype=np.float32)
    projMatrix[0,0] = 1.0 / (aspectRatio * tanhalf)
    projMatrix[1,1] = 1.0 / tanhalf
    projMatrix[2,2] = -(far + near) / (far - near)
    projMatrix[2,3] = -2 * far * near / (far - near)
    projMatrix[3,2] = -1.0

    all_screen_positions = []
    all_colors = []

    for grid in grids:
        surface_voxels = grid.get_surface_voxels()
        if len(surface_voxels) == 0:
            continue
            
        world_positions = grid.origin + (surface_voxels + 0.5) * grid.scale
        
        # Transform to view space
        homog_pos = np.column_stack([world_positions, np.ones(len(world_positions))])
        view_pos = (viewMatrix @ homog_pos.T).T[:, :3]
        
        # Backface culling
        visible_mask = view_pos[:, 2] < 0
        if not np.any(visible_mask):
            continue
            
        visible_pos = view_pos[visible_mask]
        visible_voxels = surface_voxels[visible_mask]
        
        # Project to screen space
        proj_pos = (projMatrix @ np.column_stack([visible_pos, np.ones(len(visible_pos))]).T).T
        proj_pos = proj_pos / proj_pos[:, 3:4]
        
        screen_pos = np.empty((len(proj_pos), 2), dtype=np.float32)
        screen_pos[:, 0] = (proj_pos[:, 0] + 1) * 0.5 * res[0]
        screen_pos[:, 1] = (1 - (proj_pos[:, 1] + 1) * 0.5) * res[1]
        
        # Get colors for visible voxels
        if grid.color.ndim == 1:  # Single color
            colors = np.tile(grid.color, (len(visible_voxels), 1))
        else:  # Per-voxel colors
            colors = grid.color[tuple(visible_voxels.T)]
        
        all_screen_positions.append((screen_pos, grid.scale * 0.5))
        all_colors.append(colors)
    
    return all_screen_positions, all_colors

@time_function
def rasterize_voxels(screen_positions: List[Tuple[np.ndarray, float]], 
                    colors: List[np.ndarray], 
                    width: int, height: int) -> np.ndarray:
    """Rasterize voxels to a 2D image with expanded coverage to prevent gaps"""
    if width <= 0 or height <= 0:
        return np.zeros([], dtype=np.uint8)
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    
    for (positions, size), color in zip(screen_positions, colors):
        if len(positions) == 0:
            continue
            
        # Calculate adaptive size that ensures overlap
        # Start with the base size and expand until we get full coverage
        base_size = max(1, int(size))
        expanded_size = base_size
        
        # If we have enough voxels, calculate the average screen-space distance
        if len(positions) > 1:
            # Compute pairwise distances between adjacent voxels
            dists = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
            avg_dist = np.mean(dists)
            
            # Expand size to cover at least half the average distance
            if avg_dist > 0:
                expanded_size = max(base_size, int(avg_dist * 0.7))
        
        # Alternative approach: fixed expansion factor
        expanded_size = base_size * 2  # Double the size for guaranteed overlap
        
        # Draw each voxel with expanded coverage
        for i, ((x, y), col) in enumerate(zip(positions, color)):
            x, y = int(x), int(y)
            
            # Calculate depth for simple depth testing
            # Using index as pseudo-depth since we don't have real 3D info here
            voxel_depth = i / len(positions)
            
            # Draw expanded square representing the voxel
            x_min = max(0, x - expanded_size)
            x_max = min(width, x + expanded_size + 1)
            y_min = max(0, y - expanded_size)
            y_max = min(height, y + expanded_size + 1)
            
            # Update pixels where this voxel is closer
            region = image[y_min:y_max, x_min:x_max]
            depth_region = depth_buffer[y_min:y_max, x_min:x_max]
            
            mask = voxel_depth < depth_region
            region[mask] = col
            depth_region[mask] = voxel_depth
    
    return image