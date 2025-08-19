import numpy as np
import numba
import util
from typing import List, Tuple, Optional, Union

spec = [
    ('ID', numba.int32),
    ('voxels', numba.float32[:, :]),
    ('colors', numba.uint8[:,:]),
    ('voxel_size', numba.float32),
    ('absorption', numba.float32),
]

@numba.experimental.jitclass(spec)
class VoxelMesh:
    def __init__(self, ID: int, voxels: np.ndarray, colors: Optional[np.ndarray] = None, 
                 voxel_size: float = 1.0, absorption: float = 0.8) -> None:
        self.ID = ID
        self.voxels = voxels  # [voxel[xyz]]
        self.voxel_size = voxel_size
        self.absorption = absorption
        
        if colors is None:
            self.colors = np.full((voxels.shape[0], 4), 255, dtype=np.uint8)  # Default white
        else:
            self.colors = colors
    
    @property
    def voxel_count(self) -> int:
        """Return the number of voxels"""
        return self.voxels.shape[0]
    
    def get_voxel_position(self, index: int) -> np.ndarray:
        """Get the xyz position of a specific voxel"""
        return self.voxels[index]
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the bounding box of all voxels"""
        if self.voxels.shape[0] == 0:
            return np.zeros(3), np.zeros(3)
        
        min_coords = np.array([np.min(self.voxels[:, 0]), 
                              np.min(self.voxels[:, 1]), 
                              np.min(self.voxels[:, 2])])
        max_coords = np.array([np.max(self.voxels[:, 0]), 
                              np.max(self.voxels[:, 1]), 
                              np.max(self.voxels[:, 2])])
        return min_coords, max_coords
    
    def set_voxel_color(self, index: int, color: np.ndarray) -> None:
        self.colors[index] = color

    def get_voxel_color(self, index: int) -> np.ndarray:
        return self.colors[index]

@numba.njit(cache=True)
def compute_view_projection(fov: float, lookat: np.ndarray, eye: np.ndarray, 
                           up: np.ndarray, res0: int, res1: int, 
                           near: float, far: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Matrices needed for viewer - FIXED parameter order'''
    zAxis = lookat - eye
    zAxis = util.normalize(zAxis)
    xAxis = util.cross(up, zAxis)
    xAxis = util.normalize(xAxis)
    yAxis = util.cross(zAxis, xAxis)

    viewMatrix = np.eye(4, dtype=np.float32)
    viewMatrix[:3, 0] = xAxis
    viewMatrix[:3, 1] = yAxis
    viewMatrix[:3, 2] = -zAxis
    viewMatrix[:3, 3] = eye
    viewMatrix = np.linalg.inv(viewMatrix)

    aspectRatio = res0 / res1
    fovRad = np.radians(fov)
    tanhalf = np.tan(fovRad / 2.0)

    projMatrix = np.zeros(shape=(4,4), dtype=np.float32)
    projMatrix[0,0] = 1.0 / (aspectRatio * tanhalf)
    projMatrix[1,1] = 1.0 / tanhalf
    projMatrix[2,2] = -(far + near) / (far - near)  # FIXED: (far - near) not (near - far)
    projMatrix[2,3] = (-2 * far * near) / (far - near)  # FIXED
    projMatrix[3,2] = -1.0
										  
    
    view_dir = util.normalize(lookat - eye)

    return viewMatrix, projMatrix, view_dir

@numba.njit
def transform_point(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    point_h = np.ones(4, dtype=np.float32)
    point_h[:3] = point
    transformed = matrix @ point_h
    return transformed[:3] / transformed[3]

@numba.njit
def is_point_in_voxel(point: np.ndarray, voxel_center: np.ndarray, voxel_size: float) -> bool:
    """Check if a point is inside a voxel"""
    half_size = voxel_size / 2.0
    return (abs(point[0] - voxel_center[0]) <= half_size and
            abs(point[1] - voxel_center[1]) <= half_size and
            abs(point[2] - voxel_center[2]) <= half_size)

@numba.njit
def ray_march(origin: np.ndarray, dir: np.ndarray, mesh: VoxelMesh, far: float, 
              maxsteps: int = 1000, step_size: float = 0.1, threshold: float = 0.05) -> Tuple[int, Optional[np.ndarray]]:
    '''March to find voxels - FIXED implementation'''
    pos = origin.copy()
    total_distance = 0.0
    
    for step in range(maxsteps):
        # Check if we've gone too far
        if total_distance > far:
            return -1, None
        
        # Check all voxels for intersection
        for i in range(mesh.voxels.shape[0]):
            voxel_center = mesh.voxels[i]
            
            # Fast AABB intersection test
            if is_point_in_voxel(pos, voxel_center, mesh.voxel_size):
                return i, pos.copy()
        
        # Move along ray
        pos += dir * step_size
        total_distance += step_size
    
    return -1, None

@numba.njit
def project_2d_voxels(meshes: List[VoxelMesh], eye: np.ndarray, lookat: np.ndarray,
                      up: np.ndarray, fov: float, res: Tuple[int, int],
                      near: float, far: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D voxels to 2D image
    
    Returns:
        Tuple containing (image, depthbuffer)
    """
    width, height = res
    viewMatrix, projMatrix, view_dir = compute_view_projection(
        fov, lookat, eye, up, width, height, near, far
    )
    
    image = np.zeros((height, width, 4), dtype=np.uint8)
    depthbuffer = np.full((height, width), far)  # Initialize with far distance
    
    # Use parallel processing for performance
    for y in numba.prange(height):
        for x in numba.prange(width):
            # Convert pixel coordinates to normalized device coordinates
            ndcx = (2.0 * x / width) - 1.0
            ndcy = 1.0 - (2.0 * y / height)
            
            # Create ray direction in view space
            raydirV = np.array([ndcx, ndcy, -1.0], dtype=np.float32)
            
            # Transform to world space
            raydirW = transform_point(raydirV, projMatrix)
            raydirW = util.normalize(raydirW - eye)
            
            # Ray march for each mesh
            closest_depth = far
            closest_color = np.array([0, 0, 0, 0], dtype=np.uint8)
            
            for mesh in meshes:
                vidx, hitpos = ray_march(eye, raydirW, mesh, far)
                
                if vidx >= 0:
                    depth = util.norm(hitpos - eye)
                    
                    if depth < closest_depth:
                        closest_depth = depth
                        mesh_color = mesh.get_voxel_color(vidx)
                        
                        # Apply absorption based on distance
                        absorption_factor = np.exp(-mesh.absorption * depth)
                        closest_color = (mesh_color.astype(np.float32) * absorption_factor).astype(np.uint8)
            
            if closest_depth < far:
                image[y, x] = closest_color
                depthbuffer[y, x] = closest_depth
    
    return image, depthbuffer