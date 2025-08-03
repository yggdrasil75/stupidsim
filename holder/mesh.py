from dataclasses import dataclass, field
import math
import torch
import dearpygui.dearpygui as dpg


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class mesh:
    id: int
    _vertices: torch.Tensor
    _polys: torch.Tensor
    _color: torch.Tensor
    interactive: bool = True # can stuff collide
    physics: bool = True #does it fall from gravity
    mass: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    restitution: torch.Tensor = field(default_factory=lambda: torch.tensor(0.3))
    linearVelocity: torch.Tensor= field(default_factory=lambda: torch.zeros(3, dtype=torch.float32, device=DEVICE))
    angularVelocity: torch.Tensor= field(default_factory=lambda: torch.zeros(3, dtype=torch.float32, device=DEVICE))

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

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=DEVICE)
        
        # Ensure color has correct shape (either 1 color or N colors where N is number of vertices)
        if value.ndim == 1:
            value = value.unsqueeze(0)  # make it 2D with single color
        
        if len(value) not in (1, len(self._vertices)):
            raise ValueError(f"Color must have length 1 or {len(self._vertices)} (number of vertices)")
        
        self._color = value

    def __post_init__(self):
        pass

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


def project_2d(meshes: list[mesh], eye: torch.Tensor, lookat: torch.Tensor, up: torch.Tensor, fov = 90.0,
                res: tuple[int,int] = (800,600), near: float = 1.0, far: float = 1000) \
        -> tuple[list[torch.Tensor],list,list[torch.Tensor]]:
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
        # Convert to homogeneous coordinates
        homogenous_verts = torch.cat([
            obj.vertices, 
            torch.ones(obj.vertices.shape[0], 1, device=DEVICE), 
        ], dim=1)

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

        # Backface culling
        tri_verts_view = view_verts[triangles][:, :, :3]  # Get view space coordinates
        v0, v1, v2 = tri_verts_view[:, 0], tri_verts_view[:, 1], tri_verts_view[:, 2]
        normals = torch.linalg.cross(v1 - v0, v2 - v0)
        dot_prods = torch.sum(normals * (v0), dim=1)  # Eye is at origin in view space
        visible_mask = dot_prods < 0

        visible_tris = triangles[visible_mask].tolist()
        depths = torch.mean(tri_verts_view[visible_mask][:, :, 2], dim=1)

        all_screen_verts.append(screen_verts)
        all_visible_tris.append(visible_tris)
        all_depths.append(depths)

    return all_screen_verts, all_visible_tris, all_depths
