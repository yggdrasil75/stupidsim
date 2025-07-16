from enum import Enum
import dearpygui.dearpygui as dpg
import sys
import numpy as np
import torch
from dataclasses import dataclass, field

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProjectionType(Enum):
    PERSPECTIVE = 0
    ISOMETRIC = 1
    CABINET = 2

@dataclass
class block:
    id: int
    vertices: torch.Tensor
    tris: list[tuple[int,int,int]]
    color: torch.Tensor = field(default_factory=lambda: torch.tensor([200,0,0], dtype=torch.uint8, device=DEVICE))

    @classmethod
    def create_ground(cls, id: int, size: float = sys.float_info.max, height: float = 1):
        vertices = torch.tensor([
            [-size, 0, -size],
            [size, 0, -size],
            [size, 0, size],
            [-size, 0, size],
            [-size, -height, -size],
            [size, -height, -size],
            [size, -height, size],
            [-size, -height, size]
        ], dtype=torch.float32, device=DEVICE)
        tris = [(0,1,2), (0,2,3), (4,6,5),(4,7,6), (3,2,6),(3,6,7), (0,5,1),(0,4,5), (0,3,7),(0,7,4), (1,5,6),(1,6,2)]
        cl = cls(id=id, vertices=vertices, tris=tris)
        cl.color = torch.tensor([0,200,0], dtype=torch.uint8, device=DEVICE)
        return cl

    @classmethod
    def create_sphere(cls, id: int, radius: float = 1.0, subdivisions: int = 3):
        vertices = torch.tensor([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ], dtype=torch.float32, device=DEVICE)
        
        tris = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (2, 6, 7), (2, 7, 3),
            (0, 3, 7), (0, 7, 4),
            (1, 5, 6), (1, 6, 2)
        ]
        
        def subdivide(vertices, tris):
            new_tris = []
            edge_vertices = {}
            
            for tri in tris:
                edge_points = []
                for i in range(3):
                    a, b = tri[i], tri[(i+1)%3]
                    key = tuple(sorted((a, b)))
                    if key not in edge_vertices:
                        # Create new vertex at midpoint
                        new_vertex = (vertices[a] + vertices[b]) / 2
                        edge_vertices[key] = len(vertices)
                        vertices = torch.cat([vertices, new_vertex.unsqueeze(0)], dim=0)
                    edge_points.append(edge_vertices[key])
                
                # Create 4 new triangles
                v0, v1, v2 = tri[0], tri[1], tri[2]
                v3, v4, v5 = edge_points[0], edge_points[1], edge_points[2]
                
                new_tris.extend([
                    (v0, v3, v5),
                    (v3, v1, v4),
                    (v5, v4, v2),
                    (v3, v4, v5)
                ])
            
            return vertices, new_tris
        
        # Subdivide the mesh
        for _ in range(subdivisions):
            vertices, tris = subdivide(vertices, tris)
        
        # Normalize vertices to make them spherical
        norms = torch.norm(vertices, dim=1, keepdim=True)
        vertices = vertices / norms * radius
        
        # Create the block
        return cls(id=id, vertices=vertices, tris=tris)
    
    @classmethod
    def create_sphere_quaddivide(cls, id: int, radius: float = 1.0, subdivisions: int = 3):
        # Start with a cube (8 vertices, 6 quad faces)
        vertices = torch.tensor([
            [-1, -1, -1],  # 0
            [1, -1, -1],   # 1
            [1, 1, -1],    # 2
            [-1, 1, -1],   # 3
            [-1, -1, 1],   # 4
            [1, -1, 1],    # 5
            [1, 1, 1],     # 6
            [-1, 1, 1]     # 7
        ], dtype=torch.float32, device=DEVICE)
        
        # Cube faces (6 quads)
        quads = [
            [0, 1, 2, 3],  # front
            [4, 5, 6, 7],  # back
            [0, 4, 7, 3],  # left
            [1, 5, 6, 2],  # right
            [0, 1, 5, 4],  # bottom
            [3, 2, 6, 7]   # top
        ]
        
        def subdivide_quad(vertices, quads):
            new_quads = []
            edge_vertices = {}
            face_vertices = {}
            
            for quad in quads:
                # Get edge midpoints
                edge_points = []
                for i in range(4):
                    a, b = quad[i], quad[(i+1)%4]
                    key = tuple(sorted((a, b)))
                    if key not in edge_vertices:
                        # Create new vertex at midpoint
                        new_vertex = (vertices[a] + vertices[b]) / 2
                        edge_vertices[key] = len(vertices)
                        vertices = torch.cat([vertices, new_vertex.unsqueeze(0)], dim=0)
                    edge_points.append(edge_vertices[key])
                
                # Get face center
                face_center = torch.mean(vertices[quad], dim=0)
                face_key = tuple(sorted(quad))
                face_vertices[face_key] = len(vertices)
                vertices = torch.cat([vertices, face_center.unsqueeze(0)], dim=0)
                
                # Create 4 new quads
                v0, v1, v2, v3 = quad
                e0, e1, e2, e3 = edge_points
                fc = face_vertices[face_key]
                
                new_quads.extend([
                    [v0, e0, fc, e3],
                    [e0, v1, e1, fc],
                    [fc, e1, v2, e2],
                    [e3, fc, e2, v3]
                ])
            
            return vertices, new_quads
        
        # Subdivide the quads
        for _ in range(subdivisions):
            vertices, quads = subdivide_quad(vertices, quads)
        
        # Convert quads to triangles (2 per quad)
        tris = []
        for quad in quads:
            # First triangle
            tris.append((quad[0], quad[1], quad[2]))
            # Second triangle
            tris.append((quad[0], quad[2], quad[3]))
        
        # Normalize vertices to make them spherical
        norms = torch.norm(vertices, dim=1, keepdim=True)
        vertices = vertices / norms * radius
        
        # Create the block
        return cls(id=id, vertices=vertices, tris=tris)

    def project_2d(self, eye: torch.Tensor, lookat: torch.Tensor, projection_type: ProjectionType = ProjectionType.PERSPECTIVE,
                    fov: float = 90, res: tuple[int,int] = (800,600), near: float = 1.0, far: float = 1000,
                    cabinet_angle: float = 45, cabinet_scale: float = 0.5) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
        zAxis = lookat - eye
        zAxis = zAxis / torch.norm(zAxis)
        xAxis = torch.linalg.cross(up, zAxis)
        xAxis = xAxis / torch.norm(xAxis)
        yAxis = torch.linalg.cross(zAxis, xAxis)
        yAxis = yAxis / torch.norm(yAxis)

        viewMatrix = torch.eye(4, dtype=torch.float32, device=DEVICE)
        viewMatrix[:3, 0] = xAxis
        viewMatrix[:3, 1] = yAxis
        viewMatrix[:3, 2] = zAxis
        #viewMatrix[:3, 3] = torch.matmul(viewMatrix[:3, :3], eye)
        viewMatrix[:3, 3] = eye
        viewMatrix = torch.inverse(viewMatrix)

        aspectRatio = res[0] / res[1]

        if projection_type == ProjectionType.PERSPECTIVE:
            # Perspective projection
            fovRad = np.radians(fov)
            tanhalf = np.tan(fovRad / 2.0)
            f = 1.0 / tanhalf

            projMatrix = torch.zeros((4,4), dtype=torch.float32, device=DEVICE)
            projMatrix[0,0] = 1 / (aspectRatio * tanhalf)
            projMatrix[1,1] = f
            projMatrix[2,2] = (far + near) / (near - far)
            projMatrix[2,3] = (-2 * far * near) / (near - far)
            projMatrix[3,2] = -1.0
            
        elif projection_type == ProjectionType.ISOMETRIC:
            # Isometric projection
            # Remove perspective, equal scale on all axes
            scale = 100  # Adjust this to control the zoom level
            projMatrix = torch.zeros((4,4), dtype=torch.float32, device=DEVICE)
            projMatrix[0,0] = scale / aspectRatio
            projMatrix[1,1] = scale
            projMatrix[2,2] = 0
            projMatrix[3,3] = 1
            
            # Apply isometric rotation (35.264° around X, 45° around Y)
            angle_x = np.radians(35.264)
            angle_y = np.radians(45)
            
            rot_x = torch.eye(4, dtype=torch.float32, device=DEVICE)
            rot_x[1,1] = np.cos(angle_x)
            rot_x[1,2] = -np.sin(angle_x)
            rot_x[2,1] = np.sin(angle_x)
            rot_x[2,2] = np.cos(angle_x)
            
            rot_y = torch.eye(4, dtype=torch.float32, device=DEVICE)
            rot_y[0,0] = np.cos(angle_y)
            rot_y[0,2] = np.sin(angle_y)
            rot_y[2,0] = -np.sin(angle_y)
            rot_y[2,2] = np.cos(angle_y)
            
            projMatrix = torch.matmul(projMatrix, torch.matmul(rot_x, rot_y))
            
        elif projection_type == ProjectionType.CABINET:
            # Cabinet projection (oblique)
            angle_rad = np.radians(cabinet_angle)
            scale = 100
            projMatrix = torch.zeros((4,4), dtype=torch.float32, device=DEVICE)
            projMatrix[0,0] = scale / aspectRatio
            projMatrix[1,1] = scale
            projMatrix[2,2] = 0
            projMatrix[3,3] = 1
            
            # Apply oblique shear
            shear = torch.eye(4, dtype=torch.float32, device=DEVICE)
            shear[0,2] = -cabinet_scale * np.cos(angle_rad)
            shear[1,2] = -cabinet_scale * np.sin(angle_rad)
            
            projMatrix = torch.matmul(projMatrix, shear)
            
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

        homogenousVerts = torch.cat([self.vertices, torch.ones((self.vertices.shape[0], 1), dtype=torch.float32, device=DEVICE)], dim=1)
        viewProjMatrix = torch.matmul(projMatrix, viewMatrix)
        projVerts = torch.matmul(homogenousVerts, viewProjMatrix.T)
        if projection_type == ProjectionType.PERSPECTIVE:
            projVerts = projVerts / projVerts[:, 3].unsqueeze(1)
        screenVerts = torch.empty_like(projVerts[:, :2], dtype=torch.float32, device=DEVICE)
        screenVerts[:, 0] = (projVerts[:, 0] + 1) * 0.5 * res[0]
        screenVerts[:, 1] = (1 - (projVerts[:, 1] + 1) * 0.5) * res[1]

        viewVerts = torch.matmul(homogenousVerts, viewMatrix.T)[:, :3]
        visibleTris: list[tuple[int,int,int]] = []
        for tri in self.tris:
            v0, v1, v2 = viewVerts[tri[0]], viewVerts[tri[1]], viewVerts[tri[2]]
            normal = torch.linalg.cross(v1 - v0, v2 - v0)
            normal = normal / torch.norm(normal)
            if torch.dot(normal, v0 - eye) < 0:
                visibleTris.append(tri)
        return screenVerts, visibleTris


class BlockRenderer:
    def __init__(self):
        self.blocks: list[block] = []
        self.eye = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32, device=DEVICE)
        self.lookat = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)

        self.currentID = 0
        ground = block.create_ground(id=self.idCounter())
        sphere1 = block.create_sphere(id=self.idCounter(), radius=1.0, subdivisions=3)
        sphere1.vertices[:, 1] += 1.0
        sphere1.vertices[:, 0] += 1.5
        sphere1.color = torch.tensor([200, 0, 0], dtype=torch.uint8, device=DEVICE)
        sphere2 = block.create_sphere(id=self.idCounter(), radius=1.0, subdivisions=3)
        sphere2.vertices[:, 1] += 1.0
        sphere2.vertices[:, 0] -= 1.5
        sphere2.color = torch.tensor([0, 0, 200], dtype=torch.uint8, device=DEVICE)

        self.add_block(ground)
        self.add_block(sphere1)
        self.add_block(sphere2)

    def idCounter(self):
        self.currentID += 1
        return self.currentID
        
    def add_block(self, block):
        self.blocks.append(block)
        
    def render(self):
        dpg.create_context()
        dpg.create_viewport(title='3D Blocks', width=800, height=600)
        
        with dpg.window(label="3D View", tag="mainView", width=800, height=600):
            with dpg.drawlist(width=800, height=600, tag="drawlist"):
                pass
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("mainView", True)
        while dpg.is_dearpygui_running():
            self.render_frame()
            dpg.render_dearpygui_frame()
        #dpg.start_dearpygui()
        dpg.destroy_context()
        
    def render_frame(self):
        dpg.delete_item("drawlist", children_only=True)
        with dpg.draw_node(parent="drawlist"):
            for block in self.blocks:
                screen_verts, visible_tris = block.project_2d(
                    eye=self.eye,
                    lookat=self.lookat,
                    projection_type=ProjectionType.PERSPECTIVE,
                    res=(800, 600),
                    near=0.1,
                    far=10000
                )
                
                # Convert to CPU numpy if needed
                screen_verts = screen_verts.cpu().numpy() if screen_verts.is_cuda else screen_verts.numpy()
                color = block.color.cpu().numpy() if block.color.is_cuda else block.color.numpy()
                normalized_color = color.tolist()
                
                # Draw each visible triangle
                for tri in visible_tris:
                    v0 = screen_verts[tri[0]]
                    v1 = screen_verts[tri[1]]
                    v2 = screen_verts[tri[2]]
                    
                    dpg.draw_triangle(
                        [v0[0], v0[1]],
                        [v1[0], v1[1]],
                        [v2[0], v2[1]],
                        color=normalized_color,
                        fill=normalized_color
                    )
        
    def _draw_blocks(self):
        for block in self.blocks:
            # Convert vertices to CPU numpy if they're on GPU
            vertices = block.vertices.cpu().numpy() if block.vertices.is_cuda else block.vertices.numpy()
            color = block.color.cpu().numpy() if block.color.is_cuda else block.color.numpy()
            
            # Normalize color to 0-1 range
            normalized_color = (color / 255.0).tolist()
            
            # Draw each triangle
            for tri in block.tris:
                v0 = vertices[tri[0]]
                v1 = vertices[tri[1]]
                v2 = vertices[tri[2]]
                
                dpg.draw_triangle([v0[0], v0[1], v0[2]],[v1[0], v1[1], v1[2]],[v2[0], v2[1], v2[2]],color=normalized_color,fill=normalized_color)


if __name__ == "__main__":
    renderer = BlockRenderer()
        
    # Modify the render method to use this camera
    
    # Update the renderer to use our new render method
    renderer._draw_blocks = renderer.render_frame
    
    # Render
    renderer.render()