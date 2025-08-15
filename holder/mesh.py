from collections import OrderedDict
from dataclasses import dataclass, field
import heapq
from weakref import WeakKeyDictionary
import numba
import numpy as np
#from globals import DEVICE
from util import cross_2d, time_function, cross, normalize, norm
from numba import njit, float32, types, int64, jit, int32
from numba.typed import typedlist
from scipy.sparse import csr_matrix

_mesh_cache = OrderedDict()

@njit(cache=True)
def triangulate(polys):
    new_tris = []
    
    # If polys is already a 2D array where each row represents a polygon
    if polys.ndim == 2:
        for poly in polys:
            # Remove any padding values (like -1) if present
            valid_verts = poly[poly >= 0]
            n = len(valid_verts)
            
            if n < 3:
                continue  # skip degenerate polygons
            elif n == 3:
                new_tris.append(valid_verts)
            else:
                # Fan triangulation
                v0 = valid_verts[0]
                for i in range(1, n-1):
                    new_tris.append(np.array([v0, valid_verts[i], valid_verts[i+1]]))
    else:
        # Handle case where polys is a flat array with separators
        current_poly = []
        
        for idx in polys:
            if idx == -1:
                if len(current_poly) >= 3:
                    n = len(current_poly)
                    v0 = current_poly[0]
                    for i in range(1, n-1):
                        new_tris.append([v0, current_poly[i], current_poly[i+1]])
                current_poly = []
            else:
                current_poly.append(idx)
        
        if len(current_poly) >= 3:
            n = len(current_poly)
            v0 = current_poly[0]
            for i in range(1, n-1):
                new_tris.append([v0, current_poly[i], current_poly[i+1]])

    return new_tris

def _partition_faces(_vertices, polys):
    _directional_faces = [[] for _ in range(26)]

    # Predefined directions for all 26 groups
    directions = []
    for x in [-1.0, 0.0, 1.0]:
        for y in [-1.0, 0.0, 1.0]:
            for z in [-1.0, 0.0, 1.0]:
                if x == 0 and y == 0 and z == 0:
                    continue
                directions.append([x, y, z])
    directions = np.array(directions, dtype=np.float32)
        
    # Normalize all directions
    directions = [d / norm(d) for d in directions]

    # Calculate face normals and group them
    for face_idx, face in enumerate(polys):
        verts = [v for v in face if v >= 0]
        if len(verts) < 3:
            continue  # Skip degenerate faces
                
        # Get face normal
        v0, v1, v2 = _vertices[verts[0]], _vertices[verts[1]], _vertices[verts[2]]
        normal = cross(v1 - v0, v2 - v0)
        
        # Check for zero-length normal before normalizing
        normal = normalize(normal)
            
        # Find the closest matching direction
        threshold = np.float32(0.5)  # Lower threshold
        matched = False
        for i, dir_vec in enumerate(directions):
            dot = np.dot(normal, dir_vec)
            if dot > threshold:
                _directional_faces[i].append(face_idx)
                matched = True
                
        # If no matches, add to all groups with positive dot product
        if not matched:
            for i, dir_vec in enumerate(directions):
                if np.dot(normal, dir_vec) > 0:
                    _directional_faces[i].append(face_idx)
            
    return _directional_faces

#@njit
def _getfaces(view_dir):
    # Normalize view direction
    view_dir = normalize(-view_dir)
    
    # Generate all 26 directions (6 faces, 8 corners, 12 edges) in one line
    directions = np.array([(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if (x,y,z) != (0,0,0)], dtype=np.float32)
    
    # Normalize all directions
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    #directions = [normalize(d) for d in directions]
    
    # Compute dot products and find visible groups
    visible_groups = np.where(np.dot(directions, view_dir) < 0.3)[0]
    
    # Collect all faces from visible groups
    return visible_groups

@dataclass
class mesh:
    id: int
    _vertices: np.ndarray
    _polys: np.ndarray
    _color: np.ndarray
    _tris: np.ndarray = field(default=None, init=False)  # Stores triangulated version
    interactive: bool = True  # can stuff collide
    physics: bool = True  # does it fall from gravity
    mass: np.ndarray = field(default_factory=lambda: np.array(1.0, dtype=np.float32))
    restitution: np.ndarray = field(default_factory=lambda: np.array(0.3, dtype=np.float32))
    linearVelocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angularVelocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    _neighbor_map: dict = field(default_factory=dict, init=False)  # Stores adjacency information
    _needs_neighbor_update: bool = field(default=True, init=False)  # Flag for when to rebuild neighbor map
    _needs_triangulation: bool = field(default=True, init=False)  # Flag for when to rebuild triangles

    _directional_faces: list = field(default_factory=list, init=False)  # Stores partitioned faces by direction
    _needs_directional_partition: bool = field(default=True, init=False)  # Flag for when to repartition
    

    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)
        if not value.dtype == np.float32:
            value = value.astype(np.float32)
        
        # Maintain color consistency when vertices change
        if len(value) != len(self._vertices):
            # If number of vertices changed, adjust color tensor
            if len(self._color) == len(self._vertices):
                # If we had one color per vertex, we can't maintain that anymore
                # So we'll just keep the first color for all vertices
                self._color = np.tile(self._color[0], (len(value), 1))
            elif len(self._color) == 1:
                # If we had a single color for all vertices, keep it
                self._color = np.tile(self._color, (len(value), 1))
        
        self._vertices = value
        self._needs_neighbor_update = True  # Mark for update when vertices change

    @property
    def polys(self):
        return self._polys

    @polys.setter
    def polys(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.int64)
        
        # Check if polygon indices are within vertex bounds
        if len(self._vertices) > 0 and value.size > 0:
            if np.any(value >= len(self._vertices)):
                raise ValueError("Polygon indices exceed vertex array bounds")
        
        self._polys = value
        self._needs_neighbor_update = True  # Mark for update when polygons change
        self._needs_triangulation = True  # Need to update triangles when polygons change
        #self._tris = None  # Invalidate existing triangles

    @property
    def tris(self):
        """Returns the triangulated version of the mesh. If not already triangulated,
        will perform triangulation and cache the result."""
        if self._needs_triangulation or self._tris is None:
            self._triangulate()
        return self._tris

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.uint8)
        
        # Ensure color has correct shape
        if value.ndim == 1:
            value = value[np.newaxis, :]
        
        if value.shape[0] != 1 and value.shape[0] != len(self._vertices):
             raise ValueError(f"Color array must have 1 or {len(self._vertices)} rows, but got {value.shape[0]}")
        
        self._color = value

    @property
    def is_triangulated(self):
        return not self._needs_triangulation
    
    def __post_init__(self):
        self._update_neighbor_map()
        self._partition_directional_faces()

    def _partition_directional_faces(self):
        if not self._needs_directional_partition:
            return
        
        #print(numba.typeof(self._vertices))
        #print(numba.typeof(self._polys))

        _directional_faces = _partition_faces(self._vertices, self._polys)
        self._directional_faces = _directional_faces
        
        self._needs_directional_partition = False

    def get_potentially_visible_faces(self, view_dir):
        """Return face indices that are potentially visible given a view direction"""
        if self._needs_directional_partition:
            self._partition_directional_faces()
        visible_groups = _getfaces(view_dir)
        # # Normalize view direction
        # view_dir = -view_dir / np.linalg.norm(-view_dir)
        
        # # Generate all 26 directions (6 faces, 8 corners, 12 edges) in one line
        # directions = np.array([(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if (x,y,z) != (0,0,0)])
        
        # # Normalize all directions
        # directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        # # Compute dot products and find visible groups
        # visible_groups = np.where(np.dot(directions, view_dir) < 0.3)[0]
        
        # # Collect all faces from visible groups
        return np.concatenate([self._directional_faces[i] for i in visible_groups]).astype(np.int64)

    def _update_neighbor_map(self):
        """Build or update the neighbor map data structure"""
        if not self._needs_neighbor_update:
            return
        
        self._neighbor_map = {
            'vertex_to_faces': {},  # Maps vertex indices to list of face indices
            'face_to_faces': {},    # Maps face indices to adjacent face indices
            'vertex_to_vertices': {}  # Maps vertex indices to adjacent vertex indices
        }
        
        # Initialize structures
        for v_idx in range(len(self._vertices)):
            self._neighbor_map['vertex_to_faces'][v_idx] = []
            self._neighbor_map['vertex_to_vertices'][v_idx] = set()
        
        # Process each face
        for face_idx, face in enumerate(self._polys):
            # Remove any padding values (like -1) if present
            verts = [v for v in face if v >= 0]
            n_verts = len(verts)
            
            # Skip degenerate faces
            if n_verts < 3:
                continue
                
            # Add this face to each vertex's face list
            for v in verts:
                self._neighbor_map['vertex_to_faces'][v].append(face_idx)
            
            # For each edge in the face, build adjacency information
            for i in range(n_verts):
                v1 = verts[i]
                v2 = verts[(i+1)%n_verts]
                
                # Add to vertex adjacency
                self._neighbor_map['vertex_to_vertices'][v1].add(v2)
                self._neighbor_map['vertex_to_vertices'][v2].add(v1)
        
        # Convert sets to lists for easier use
        for v in self._neighbor_map['vertex_to_vertices']:
            self._neighbor_map['vertex_to_vertices'][v] = list(self._neighbor_map['vertex_to_vertices'][v])
        
        # Now build face-to-face adjacency
        edge_to_faces = {}  # Maps edges (as sorted tuples) to list of face indices
        
        for face_idx, face in enumerate(self._polys):
            verts = [v for v in face if v >= 0]
            n_verts = len(verts)
            
            if n_verts < 3:
                continue
                
            for i in range(n_verts):
                v1, v2 = verts[i], verts[(i+1)%n_verts]
                edge = tuple(sorted((v1, v2)))
                
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
        
        # Now use edge_to_faces to find adjacent faces
        for face_idx in range(len(self._polys)):
            self._neighbor_map['face_to_faces'][face_idx] = set()
        
        for edge, faces in edge_to_faces.items():
            if len(faces) > 1:  # Only edges shared by multiple faces create adjacency
                for i in range(len(faces)):
                    for j in range(i+1, len(faces)):
                        self._neighbor_map['face_to_faces'][faces[i]].add(faces[j])
                        self._neighbor_map['face_to_faces'][faces[j]].add(faces[i])
        
        # Convert sets to lists
        for face_idx in self._neighbor_map['face_to_faces']:
            self._neighbor_map['face_to_faces'][face_idx] = list(self._neighbor_map['face_to_faces'][face_idx])
        
        self._needs_neighbor_update = False

    def get_adjacent_faces(self, face_idx):
        """Get list of face indices adjacent to the given face"""
        self._update_neighbor_map()
        return self._neighbor_map['face_to_faces'].get(face_idx, [])

    def get_faces_for_vertex(self, vertex_idx):
        """Get list of face indices that contain the given vertex"""
        self._update_neighbor_map()
        return self._neighbor_map['vertex_to_faces'].get(vertex_idx, [])

    def get_adjacent_vertices(self, vertex_idx):
        """Get list of vertex indices adjacent to the given vertex"""
        self._update_neighbor_map()
        return self._neighbor_map['vertex_to_vertices'].get(vertex_idx, [])

    def toTri(self):
        """
        Convert all polygons to triangles using a simple fan triangulation.
        Modifies the mesh in-place by replacing the polys with triangles.
        Returns self for method chaining.
        """
        if not self._needs_triangulation:
            return self
            
        # Ensure we have triangles computed
        if self._tris is None or self._needs_triangulation:
            self._triangulate()
            
        # Replace polys with triangles
        self._polys = self._tris
        self._needs_neighbor_update = True
        self._needs_triangulation = False
        return self
    
    def _triangulate(self):
        """Internal method to triangulate the polygons and cache the result"""
        if len(self._polys) == 0:
            self._tris = np.zeros((0, 3), dtype=np.int64)
            self._needs_triangulation = False
            return
        else:
            new_tris = triangulate(self.polys)
        
        self._tris = np.stack(new_tris) if len(new_tris) > 0 else np.zeros((0, 3), dtype=np.int64)
        self._needs_triangulation = False
        return self

@time_function
def get_lod_level(view_verts: np.ndarray, triangles: np.ndarray):
    #TODO: Please implement
    pass

@time_function
def simplify_mesh(vertices: np.ndarray, 
                 triangles: np.ndarray, 
                 lod_level: float):
    #TODO: please implement
    pass

@njit((float32, float32[:], float32[:], float32[:], int64, int64, int64, float32), cache=True)
def comped(fov, lookat, eye, up, res0, res1, far, near):
    zAxis = lookat - eye
    zAxis = zAxis / np.linalg.norm(zAxis)
    xAxis = cross(up, zAxis)
    xAxis = xAxis / np.linalg.norm(xAxis)
    yAxis = cross(zAxis, xAxis)

    viewMatrix = np.eye(4, dtype=np.float32)
    viewMatrix[:3, 0] = xAxis
    viewMatrix[:3, 1] = yAxis
    viewMatrix[:3, 2] = -zAxis
    viewMatrix[:3, 3] = eye
    viewMatrix = np.linalg.inv(viewMatrix)

    aspectRatio = res0 / res1
    fovRad = np.radians(fov)
    tanhalf = np.tan(fovRad / 2.0)
    f = 1.0 / tanhalf

    projMatrix = np.zeros(shape=(4,4), dtype=np.float32)
    projMatrix[0,0] = 1.0 / (aspectRatio * tanhalf)
    projMatrix[1,1] = f
    projMatrix[2,2] = -(far + near) / (near - far)
    projMatrix[2,3] = (-2 * far * near) / (near - far)
    projMatrix[3,2] = -1.0

    view_dir = (lookat - eye)
    view_dir = view_dir / np.linalg.norm(view_dir)

    return viewMatrix, projMatrix, view_dir

@njit((float32[:,:], float32[:,:], float32[:,:], int64, int64), fastmath=True, cache=True)
def compedObj(verts, viewmatrix, projMatrix, res0, res1):
    
    n = verts.shape[0]
    # Create homogeneous coordinates (n x 4)
    homogenous_verts = np.empty((n, 4), dtype=verts.dtype)
    homogenous_verts[:, :3] = verts  # Assuming verts is n x 3
    homogenous_verts[:, 3] = 1.0

    viewmatrix_contig = np.ascontiguousarray(viewmatrix)
    homogenous_verts_contig = np.ascontiguousarray(homogenous_verts)

    # Transform vertices
    view_verts = (viewmatrix_contig @ homogenous_verts_contig.T).T
    proj_verts = (np.ascontiguousarray(projMatrix) @ view_verts.T).T

    # Perspective division
    proj_verts = proj_verts / proj_verts[:, 3:4]
    
    # Convert to screen coordinates
    screen_verts = np.empty_like(proj_verts[:, :2])
    screen_verts[:, 0] = (proj_verts[:, 0] + 1) * 0.5 * res0
    screen_verts[:, 1] = (1 - (proj_verts[:, 1] + 1) * 0.5) * res1
    

    return screen_verts, view_verts

@njit((int32[:,:], int64[:]), fastmath=True, cache=True)
def triface(polys, visible_face_indices):
    # Pre-allocate arrays for better performance
    total_tris = 0
    for i in visible_face_indices:
        face = polys[i]
        valid_verts = face[face >= 0]
        if len(valid_verts) >= 3:
            total_tris += len(valid_verts) - 2
    
    if total_tris == 0:
        return np.empty((0, 3), dtype=np.int32)
    
    result = np.empty((total_tris, 3), dtype=np.int32)
    idx = 0
    
    for face_idx in visible_face_indices:
        face = polys[face_idx]
        valid_verts = face[face >= 0]
        n = len(valid_verts)
        
        if n < 3:
            continue
        if n == 3:
            result[idx] = valid_verts
            idx += 1
        else:
            # Fan triangulation
            v0 = valid_verts[0]
            for i in range(1, n-1):
                result[idx] = np.array([v0, valid_verts[i], valid_verts[i+1]])
                idx += 1
                
    return result

@time_function
def project_2d(meshes: list[mesh], eye: np.ndarray, lookat: np.ndarray, up: np.ndarray, 
               fovfl: float = 90.0, res: tuple[int,int] = (800,600), 
               near: float = 1.0, far: float = 1000) -> tuple[list[np.ndarray], list, list[np.ndarray], list[np.ndarray]]:
    
    viewMatrix, projMatrix, view_dir = comped(fovfl, lookat, eye, up, res[0], res[1], far, near)

    all_screen_verts = []
    all_visible_tris = []
    all_depths = []
    all_colors = []

    eyet = tuple(eye)
    lot = tuple(lookat)
    tup = tuple(up)
    rest = tuple(res)
    for obj in meshes:
        cache_key = (id(obj), eyet, lot, tup, rest)
        if cache_key in _mesh_cache and not obj._needs_neighbor_update:
            screen_verts, visible_tris, depths, colors = _mesh_cache[cache_key]
        else:
            mesh_center = np.mean(obj.vertices, axis=0)
            view_center = np.dot(np.append(mesh_center, 1), viewMatrix.T)[:3]
            
            # Skip if entire mesh is backfacing
            if np.dot(view_center, view_center) < 0:
                continue

            # Get potentially visible faces
            visible_face_indices = obj.get_potentially_visible_faces(view_dir)
            if len(visible_face_indices) == 0:
                continue

            # Process only the potentially visible faces
            screen_verts, view_verts = compedObj(obj.vertices, viewMatrix, projMatrix, res[0], res[1])
            
            # Get triangles from visible faces
            #print(numba.typeof(obj.polys))
            #print(numba.typeof(visible_face_indices))
            triangles = triface(obj.polys, visible_face_indices)
            
            if len(triangles) == 0:
                continue

            triangles = np.array(triangles)
            
            # Backface culling on the potentially visible subset
            tri_verts_view = view_verts[triangles][:, :, :3]
            v0 = tri_verts_view[:, 0]
            v1 = tri_verts_view[:, 1]
            v2 = tri_verts_view[:, 2]
            normals = cross_2d(v1 - v0, v2 - v0)
            view_dir_tri = -tri_verts_view.mean(axis=1)
            dot_prods = np.sum(normals * view_dir_tri, axis=1)
            visible_mask = dot_prods > 0

            visible_tris = triangles[visible_mask].tolist()
            depths = np.mean(tri_verts_view[visible_mask][:, :, 2], axis=1)

            obj_colors = obj.color
            if obj_colors.shape[0] == 1:
                colors = np.tile(obj_colors, (len(visible_tris), 1))
            else:
                tri_vert_colors = obj_colors[visible_tris]  # Shape: (n_tris, 3, color_channels)
                colors = np.mean(tri_vert_colors, axis=1).astype(np.uint8)
            
            _mesh_cache[cache_key] = (screen_verts, visible_tris, depths, colors)
            if len(_mesh_cache) > 100:
                _mesh_cache.popitem(last=False)

        all_screen_verts.append(screen_verts)
        all_visible_tris.append(visible_tris)
        all_depths.append(depths)
        all_colors.append(colors)

    return all_screen_verts, all_visible_tris, all_depths, all_colors

@time_function
def rasterize(vertices, tris, depths, colors, width, height):
    """
    Rasterize triangles onto an image using numpy.
    
    Args:
        vertices: np.ndarray of shape (2, n) containing x,y coordinates
        tris: np.ndarray of shape (m, 3) containing vertex indices for each triangle
        depths: np.ndarray of shape (m,) containing z-depth for each triangle
        colors: np.ndarray of shape (n, 3) containing RGB colors for each vertex
        width: output image width
        height: output image height
    
    Returns:
        Rasterized image as np.ndarray of shape (height, width, 3)
    """
    # Early return for invalid dimensions
    if width <= 0 or height <= 0:
        return np.zeros([], dtype=np.float32)
    
    # Initialize output image and depth buffer
    image = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    
    # Sort triangles by depth (back to front for painter's algorithm)
    sorted_indices = np.argsort(depths).ravel()
    
    # Pre-compute pixel coordinates only once
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    pixel_coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    
    for tri_idx in sorted_indices:
        # Get triangle vertices
        v_idx = tris[tri_idx]
        tri_verts = vertices[:, v_idx].T  # shape (3, 2)
        
        # Compute barycentric coordinates for all pixels
        v0, v1, v2 = tri_verts
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        
        # Skip degenerate triangles
        if np.abs(denom) < 1e-10:
            continue
            
        # Vectorized computation of barycentric coordinates
        w0 = ((v1[1] - v2[1]) * (pixel_coords[:, 0] - v2[0]) + (v2[0] - v1[0]) * (pixel_coords[:, 1] - v2[1])) / denom
        w1 = ((v2[1] - v0[1]) * (pixel_coords[:, 0] - v2[0]) + (v0[0] - v2[0]) * (pixel_coords[:, 1] - v2[1])) / denom
        w2 = 1.0 - w0 - w1
        
        # Find pixels inside the triangle (all barycentric coords >= 0)
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & \
               (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) & \
               (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
        
        if not np.any(mask):
            continue
            
        # Get pixel indices inside the triangle
        pixel_indices = pixel_coords[mask].astype(int)
        
        # Compute barycentric coordinates only for inside pixels
        w0_inside = w0[mask]
        w1_inside = w1[mask]
        w2_inside = w2[mask]
        
        # Get colors for each vertex
        tri_colors = colors[v_idx]  # shape (3, 3)
        
        # Interpolate colors using barycentric coordinates
        interpolated_colors = (
            w0_inside[:, np.newaxis] * tri_colors[0] +
            w1_inside[:, np.newaxis] * tri_colors[1] +
            w2_inside[:, np.newaxis] * tri_colors[2]
        )
        
        # Get current depth for this triangle
        current_depth = depths[tri_idx]
        
        # Update image and depth buffer in a vectorized manner
        rows = pixel_indices[:, 1]
        cols = pixel_indices[:, 0]
        
        # Create mask for pixels where current triangle is closer
        depth_mask = current_depth < depth_buffer[rows, cols]
        
        # Apply the mask and update
        valid_rows = rows[depth_mask]
        valid_cols = cols[depth_mask]
        image[valid_rows, valid_cols] = interpolated_colors[depth_mask]
        depth_buffer[valid_rows, valid_cols] = current_depth
    
    return image.ravel()