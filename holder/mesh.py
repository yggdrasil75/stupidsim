from dataclasses import dataclass, field
import heapq
import numba
import numpy as np
#from globals import DEVICE
from util import cross_2d, time_function, cross, normalize, norm
from numba import njit, float32, types, int64, jit, int32
from numba.typed import typedlist
from scipy.sparse import csr_matrix

_mesh_cache = {}
_DIRECTIONS = np.array([(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) 
                       if (x,y,z) != (0,0,0)], dtype=np.float32)
_DIRECTIONS_NORMALIZED = np.ascontiguousarray(_DIRECTIONS / np.linalg.norm(_DIRECTIONS, axis=1, keepdims=True))

@njit(cache=True)
def triangulate(polys):
    valid_polys = [poly[poly >= 0] for poly in polys]
    valid_polys = [p for p in valid_polys if len(p) >= 3]
    triangles = [p for p in valid_polys if len(p) == 3]
    for p in valid_polys:
        if len(p) > 3:
            v0 = p[0]
            triangles.extend([np.array([v0, p[i], p[i+1]]) for i in range(1, len(p)-1)])

    return triangles

#@njit(cache=True)
def _partition_faces(polys, norms):
    # Generate all 26 directions (3^3 - 1)
    directions = _DIRECTIONS_NORMALIZED
    
    # Calculate dot products between all normals and all directions
    dots = np.dot(norms, directions.T)  # shape: (num_faces, 26)
    
    # Threshold for matching
    threshold = np.float32(0.5)
    
    # Find best matches (above threshold)
    above_threshold = dots > threshold
    
    # For faces with at least one match above threshold
    has_match = np.any(above_threshold, axis=1)
    matched_groups = [np.where(above_threshold[i])[0] for i in range(len(norms)) if has_match[i]]
    
    # For faces with no match above threshold, use all positive dot products
    unmatched_groups = [np.where(dots[i] > 0)[0] for i in range(len(norms)) if not has_match[i]]
    
    # Initialize directional faces
    _directional_faces = [[] for _ in range(26)]
    
    # Process matched faces
    matched_indices = np.where(has_match)[0]
    for i, groups in zip(matched_indices, matched_groups):
        for group in groups:
            _directional_faces[group].append(i)
    
    # Process unmatched faces
    unmatched_indices = np.where(~has_match)[0]
    for i, groups in zip(unmatched_indices, unmatched_groups):
        for group in groups:
            _directional_faces[group].append(i)
    
    return _directional_faces

@njit(cache=True)
def _getfaces(view_dir):
    # Normalize view direction
    view_dir = normalize(-view_dir)
    
    visible_groups = np.where(np.dot(_DIRECTIONS_NORMALIZED, view_dir) < 0.3)[0]
    
    # Collect all faces from visible groups
    return visible_groups

@dataclass
class mesh:
    id: int
    _vertices: np.ndarray
    _polys: np.ndarray
    _color: np.ndarray
    _tris: np.ndarray = field(default=None, init=False)  # Stores triangulated version
    _norms: np.ndarray = field(default=None, init=False)
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

    _center: np.ndarray = field(default=None, init=False)

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
            self._calnorms()
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
    
    @property
    def norms(self):
        if self._needs_triangulation or self._norms is None:
            self._triangulate()
            self._calnorms()
        return self._norms

    @property
    def center(self):
        if self._center is None:
            self._center = np.mean(self.vertices, axis=0)
        return self._center

    def __post_init__(self):
        self._update_neighbor_map()
        self._partition_directional_faces()

    #@time_function
    def _partition_directional_faces(self):
        if not self._needs_directional_partition:
            return
        

        _directional_faces = _partition_faces(self._polys, self.norms)
        self._directional_faces = _directional_faces
        
        self._needs_directional_partition = False

    #@time_function
    def get_potentially_visible_faces(self, view_dir):
        """Return face indices that are potentially visible given a view direction"""
        if self._needs_directional_partition:
            self._partition_directional_faces()
        visible_groups = _getfaces(view_dir)
        return np.concatenate([self._directional_faces[i] for i in visible_groups]).astype(np.int64)

    #@time_function
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

    #@time_function
    def get_adjacent_faces(self, face_idx):
        """Get list of face indices adjacent to the given face"""
        self._update_neighbor_map()
        return self._neighbor_map['face_to_faces'].get(face_idx, [])

    #@time_function
    def get_faces_for_vertex(self, vertex_idx):
        """Get list of face indices that contain the given vertex"""
        self._update_neighbor_map()
        return self._neighbor_map['vertex_to_faces'].get(vertex_idx, [])

    #@time_function
    def get_adjacent_vertices(self, vertex_idx):
        """Get list of vertex indices adjacent to the given vertex"""
        self._update_neighbor_map()
        return self._neighbor_map['vertex_to_vertices'].get(vertex_idx, [])

    #@time_function
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
    
    #@time_function
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
    
    #@time_function
    def _calnorms(self):
        tempnorms = []
        for tri in self._tris:
            v0 = self._vertices[tri[0]]
            v1 = self._vertices[tri[1]]
            v2 = self._vertices[tri[2]]
            normal = cross(v1 - v0, v2 - v0)
            tempnorms.append(normalize(normal))
        self._norms = np.array(tempnorms)

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
def getMats(fov, lookat, eye, up, res0, res1, far, near):
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
    

    return screen_verts

#@time_function
@njit((int32[:,:], int64[:]), fastmath=True, cache=True)
def triface(polys, visible_face_indices):
    visible_faces = polys[visible_face_indices]
    
    valid_verts_masks = visible_faces >= 0
    valid_verts_counts = np.sum(valid_verts_masks, axis=1)
    
    # Calculate number of triangles per face (max(0, n-2))
    tris_per_face = np.maximum(valid_verts_counts - 2, 0)
    total_tris = np.sum(tris_per_face)
    
    if total_tris == 0:
        return np.empty((0, 3), dtype=np.int32)
    
    # Create output array
    result = np.empty((total_tris, 3), dtype=np.int32)
    
    # Initialize indices
    start_idx = 0
    
    for i in range(len(visible_faces)):
        #n = valid_verts_counts[i]

        face = visible_faces[i]
        valid_verts = face[valid_verts_masks[i]]
        
        result[start_idx] = valid_verts
        start_idx += 1

    return result

@time_function
def project_2d(meshes: list[mesh], eye: np.ndarray, lookat: np.ndarray, up: np.ndarray, 
               fovfl: float = 90.0, res: tuple[int,int] = (800,600), 
               near: float = 1.0, far: float = 1000): # -> tuple[list[np.ndarray], list, list[np.ndarray]]:
    
    # 0.000146 seconds per slowest call when logging.
    viewMatrix, projMatrix, view_dir = getMats(fovfl, lookat, eye, up, res[0], res[1], far, near)
        
    all_screen_verts = []
    all_visible_tris = []
    #all_depths = []

    for obj in meshes:
        cache_key = (id(obj), tuple(eye), tuple(lookat), tuple(up))
        if cache_key in _mesh_cache:
            screen_verts, visible_tris = _mesh_cache[cache_key]
        else:

            # Get potentially visible faces
            visible_face_indices = obj.get_potentially_visible_faces(view_dir)

            #  0.000360 per slowest call when logging
            screen_verts = compedObj(obj.vertices, viewMatrix, projMatrix, res[0], res[1])
            
            # Get triangles from visible faces
            # 0.005 per slowest call when logging
            triangles = triface(obj.polys, visible_face_indices)
            visible_tris = triangles.tolist()
            

            _mesh_cache[cache_key] = (screen_verts, visible_tris)

        all_screen_verts.append(screen_verts)
        all_visible_tris.append(visible_tris)

    return all_screen_verts, all_visible_tris
