from dataclasses import dataclass, field
import heapq
import math
import torch
import dearpygui.dearpygui as dpg
from globals import DEVICE
from util import time_function
import numpy as np

_mesh_cache = {}

@dataclass
class mesh:
    torch.set_default_device(DEVICE)
    id: int
    _vertices: torch.Tensor
    _polys: torch.Tensor
    _color: torch.Tensor
    interactive: bool = True # can stuff collide
    physics: bool = True #does it fall from gravity
    mass: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    restitution: torch.Tensor = field(default_factory=lambda: torch.tensor(0.3))
    linearVelocity: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float32, device=DEVICE))
    angularVelocity: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float32, device=DEVICE))
    _neighbor_map: dict = field(default_factory=dict, init=False)  # Stores adjacency information
    _needs_neighbor_update: bool = field(default=True, init=False)  # Flag for when to rebuild neighbor map
    

    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=DEVICE)
        
        # Maintain color consistency when vertices change
        if len(value) != len(self._vertices):
            # If number of vertices changed, adjust color tensor
            if len(self._color) == len(self._vertices):
                # If we had one color per vertex, we can't maintain that anymore
                # So we'll just keep the first color for all vertices
                self._color = self._color[0].unsqueeze(0).expand(len(value), -1)
            elif len(self._color) == 1:
                # If we had a single color for all vertices, keep it
                self._color = self._color.expand(len(value), -1)
        
        self._vertices = value
        self._needs_neighbor_update = True  # Mark for update when vertices change

    @property
    def polys(self):
        return self._polys

    @polys.setter
    def polys(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.long, device=DEVICE)
        
        # Check if polygon indices are within vertex bounds
        if len(self._vertices) > 0 and value.numel() > 0:
            if torch.any(value >= len(self._vertices)):
                raise ValueError("Polygon indices exceed vertex array bounds")
        
        self._polys = value
        self._needs_neighbor_update = True  # Mark for update when polygons change

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.uint8, device=DEVICE)
        
        # Ensure color has correct shape
        if value.ndim == 1:
            value = value.unsqueeze(0)
        
        if value.shape[0] != 1 and value.shape[0] != len(self._vertices):
             raise ValueError(f"Color tensor must have 1 or {len(self._vertices)} rows, but got {value.shape[0]}")
        
        self._color = value

    def __post_init__(self):
        self._update_neighbor_map()
        pass

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
        for face_idx, face in enumerate(self._polys.cpu().numpy()):
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
        
        for face_idx, face in enumerate(self._polys.cpu().numpy()):
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

    @classmethod
    def toTri(cls, self):
        """
        Convert all polygons to triangles using a simple fan triangulation.
        For each polygon with more than 3 vertices, creates a triangle fan.
        Returns a new mesh with only triangular faces.
        """
        if len(self._polys) == 0:
            return self
        
        # First, determine the polygon sizes
        # Assuming _polys is a flat list of vertex indices with -1 as separator
        # or some other way to indicate polygon boundaries
        
        # If polys is already a 2D tensor where each row represents a polygon
        if self._polys.ndim == 2:
            new_tris = []
            for poly in self._polys:
                # Remove any padding values (like -1) if present
                valid_verts = poly[poly >= 0]
                n = len(valid_verts)
                
                if n < 3:
                    continue  # skip degenerate polygons
                elif n == 3:
                    new_tris.append(valid_verts)
                else:
                    # Fan triangulation: create triangles from first vertex to each subsequent pair
                    v0 = valid_verts[0]
                    for i in range(1, n-1):
                        new_tris.append(torch.stack([v0, valid_verts[i], valid_verts[i+1]]))
            
            if len(new_tris) == 0:
                return self
                
            new_polys = torch.stack(new_tris)
            
        else:
            # Handle case where polys is a flat array with separators
            # This implementation assumes -1 is used as a separator
            polys_list = []
            current_poly = []
            
            for idx in self._polys:
                if idx == -1:
                    if len(current_poly) >= 3:
                        polys_list.append(current_poly)
                    current_poly = []
                else:
                    current_poly.append(idx)
            
            if len(current_poly) >= 3:
                polys_list.append(current_poly)
            
            new_tris = []
            for poly in polys_list:
                n = len(poly)
                v0 = poly[0]
                for i in range(1, n-1):
                    new_tris.append([v0, poly[i], poly[i+1]])
            
            if len(new_tris) == 0:
                return self
                
            new_polys = torch.tensor(new_tris, dtype=torch.long, device=DEVICE)
        
        # Create a new mesh with the triangulated polygons
        return mesh(
            id=self.id,
            _vertices=self._vertices.clone(),
            _polys=new_polys,
            _color=self._color.clone(),
            interactive=self.interactive,
            physics=self.physics,
            mass=self.mass.clone(),
            restitution=self.restitution.clone(),
            linearVelocity=self.linearVelocity.clone(),
            angularVelocity=self.angularVelocity.clone()
        )

@time_function
def get_lod_level(view_verts: torch.Tensor, triangles: torch.Tensor):
    #TODO: Please implement
    pass

@time_function
def simplify_mesh(vertices: torch.Tensor, 
                 triangles: torch.Tensor, 
                 lod_level: float):
    #TODO: please implement
    pass

def is_in_frustum(mesh_vertices, view_matrix, proj_matrix):
    # Transform all vertices to clip space
    homogenous_verts = torch.cat([mesh_vertices, torch.ones(len(mesh_vertices), 1, device=DEVICE)], dim=1)
    clip_verts = torch.matmul(homogenous_verts, (view_matrix @ proj_matrix).T)
    
    # Normalize to NDC
    ndc_verts = clip_verts / clip_verts[:, 3].unsqueeze(1)
    
    # Check if any vertex is within the frustum
    in_frustum = ((ndc_verts[:, 0].abs() <= 1.0) & (ndc_verts[:, 1].abs() <= 1.0) & 
                 (ndc_verts[:, 2] >= -1.0) & (ndc_verts[:, 2] <= 1.0))
    
    return torch.any(in_frustum)

@time_function
def project_2d(meshes: list[mesh], eye: torch.Tensor, lookat: torch.Tensor, up: torch.Tensor, fov = 90.0,
                res: tuple[int,int] = (800,600), near: float = 1.0, far: float = 1000) \
        -> tuple[list[torch.Tensor],list,list[torch.Tensor]]:
    global _mesh_cache
    if torch.cuda.is_available():
        eye = eye.half()
        lookat = lookat.half()
        up = up.half()
    fov = torch.tensor(fov, dtype=torch.float32, device=DEVICE)
    zAxis: torch.Tensor = lookat - eye
    zAxis = zAxis / torch.norm(zAxis)
    xAxis: torch.Tensor = torch.linalg.cross(up, zAxis)
    xAxis = xAxis / torch.norm(xAxis)
    yAxis: torch.Tensor = torch.linalg.cross(zAxis, xAxis)

    viewMatrix = torch.eye(4, dtype=torch.float32, device=DEVICE)
    viewMatrix[:3, 0] = xAxis
    viewMatrix[:3, 1] = yAxis
    viewMatrix[:3, 2] = -zAxis
    viewMatrix[:3, 3] = eye
    viewMatrix = torch.inverse(viewMatrix)

    aspectRatio = res[0] / res[1]
    fovRad = torch.deg2rad(fov)
    tanhalf = torch.tan(fovRad / 2.0)
    f = 1.0 / tanhalf

    projMatrix = torch.zeros((4,4), dtype=torch.float32, device=DEVICE)
    projMatrix[0,0] = 1.0 / (aspectRatio * tanhalf)
    projMatrix[1,1] = f
    projMatrix[2,2] = -(far + near) / (near - far)
    projMatrix[2,3] = (-2 * far * near) / (near - far)
    projMatrix[3,2] = -1.0
    
    all_screen_verts = []
    all_visible_tris = []
    all_depths = []

    for obj in meshes:
        # unfortunately when I tried this I kept getting the main object outside frustrum if any part of it was. so its not in use.
        # if not is_in_frustum(obj.vertices, viewMatrix, projMatrix):
        #     all_screen_verts.append(torch.empty(0, 2, device=DEVICE))
        #     all_visible_tris.append([])
        #     all_depths.append(torch.empty(0, device=DEVICE))
        #     continue
        cache_key = (id(obj), tuple(eye.cpu().numpy()), tuple(lookat.cpu().numpy()), tuple(up.cpu().numpy()))
        if cache_key in _mesh_cache and not obj._needs_neighbor_update:
            screen_verts, visible_tris, depths = _mesh_cache[cache_key]
        else:
            mesh_center = torch.mean(obj.vertices, dim=0)
            view_center = torch.matmul(torch.cat([mesh_center, torch.ones(1, device=DEVICE)]), viewMatrix.T)[:3]
            if torch.dot(view_center, view_center) < 0:  # Entire mesh is backfacing
                all_screen_verts.append(torch.empty(0, 2, device=DEVICE))
                all_visible_tris.append([])
                all_depths.append(torch.empty(0, device=DEVICE))
                continue

            homogenous_verts = torch.cat([obj.vertices, torch.ones(obj.vertices.shape[0], 1, device=DEVICE), ], dim=1)

            # Transform to view space
            view_verts = torch.matmul(homogenous_verts, viewMatrix.T)
            
            # Project to clip space
            proj_verts = torch.matmul(view_verts, projMatrix.T)
            proj_verts = proj_verts / proj_verts[:, 3].unsqueeze(1)

            # Convert to screen coordinates
            screen_verts = torch.empty_like(proj_verts[:, :2])
            screen_verts[:, 0] = (proj_verts[:, 0] + 1) * 0.5 * res[0]
            screen_verts[:, 1] = (1 - (proj_verts[:, 1] + 1) * 0.5) * res[1]

            # Get triangles (convert to triangles if needed)
            if obj._polys.shape[1] == 3:
                triangles = obj._polys
            else:
                triangulated = obj.toTri(obj)
                triangles = triangulated._polys

            #if I could figure this out, then I would use this as well. but I cant.
            # lod_level = get_lod_level(view_verts, triangles)
            # if lod_level > 0:
            #     vertices, triangles = simplify_mesh(obj.vertices, triangles, lod_level)
            #     # Recompute homogenous_verts with simplified vertices
            #     homogenous_verts = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=DEVICE)], dim=1)

            # Backface culling
            with torch.no_grad():
                tri_verts_view = view_verts[triangles][:, :, :3]  # Get view space coordinates
                v0, v1, v2 = tri_verts_view[:, 0], tri_verts_view[:, 1], tri_verts_view[:, 2]
                normals = torch.linalg.cross(v1 - v0, v2 - v0)
                dot_prods = torch.sum(normals * (v0), dim=1)  # Eye is at origin in view space
                visible_mask = dot_prods < 0

                visible_tris = triangles[visible_mask].tolist()
                depths = torch.mean(tri_verts_view[visible_mask][:, :, 2], dim=1)

            _mesh_cache[cache_key] = (screen_verts, visible_tris, depths)

        all_screen_verts.append(screen_verts)
        all_visible_tris.append(visible_tris)
        all_depths.append(depths)

    return all_screen_verts, all_visible_tris, all_depths
