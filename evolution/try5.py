from enum import Enum
import time
import dearpygui.dearpygui as dpg
import sys
import numpy as np
import torch
from dataclasses import dataclass, field

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                        new_vertex = (vertices[a] + vertices[b]) / 2
                        edge_vertices[key] = len(vertices)
                        vertices = torch.cat([vertices, new_vertex.unsqueeze(0)], dim=0)
                    edge_points.append(edge_vertices[key])
                
                v0, v1, v2 = tri[0], tri[1], tri[2]
                v3, v4, v5 = edge_points[0], edge_points[1], edge_points[2]
                
                new_tris.extend([
                    (v0, v3, v5),
                    (v3, v1, v4),
                    (v5, v4, v2),
                    (v3, v4, v5)
                ])
            
            return vertices, new_tris
        
        for _ in range(subdivisions):
            vertices, tris = subdivide(vertices, tris)
        
        norms = torch.norm(vertices, dim=1, keepdim=True)
        vertices = vertices / norms * radius
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

    def project_2d(self, eye: torch.Tensor, lookat: torch.Tensor, up: torch.Tensor, fov: float = 90,
                    res: tuple[int,int] = (800,600), near: float = 1.0, far: float = 1000) \
            -> tuple[torch.Tensor, list[tuple[int, int, int]], torch.Tensor]:
        
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
        fovRad = np.radians(fov)
        tanhalf = np.tan(fovRad / 2.0)
        f = 1.0 / tanhalf

        projMatrix = torch.zeros((4,4), dtype=torch.float32, device=DEVICE)
        projMatrix[0,0] = 1.0 / (aspectRatio * tanhalf)
        projMatrix[1,1] = f
        projMatrix[2,2] = -(far + near) / (near - far)
        projMatrix[2,3] = (-2 * far * near) / (near - far)
        projMatrix[3,2] = -1.0
        
        homogenousVerts = torch.cat([self.vertices, torch.ones((self.vertices.shape[0], 1), dtype=torch.float32, device=DEVICE)], dim=1)
        viewProjMatrix = torch.matmul(projMatrix, viewMatrix)
        projVerts = torch.matmul(homogenousVerts, viewProjMatrix.T)
        projVerts = projVerts / projVerts[:, 3].unsqueeze(1)
        
        screenVerts = torch.empty_like(projVerts[:, :2], dtype=torch.float32, device=DEVICE)
        screenVerts[:, 0] = (projVerts[:, 0] + 1) * 0.5 * res[0]
        screenVerts[:, 1] = (1 - (projVerts[:, 1] + 1) * 0.5) * res[1]

        # Back-face culling and depth calculation
        viewVerts = torch.matmul(homogenousVerts, viewMatrix.T)[:, :3]
        visibleTris: list[tuple[int,int,int]] = []
        visible_depths = []
        
        for tri in self.tris:
            v0, v1, v2 = self.vertices[tri[0]], self.vertices[tri[1]], self.vertices[tri[2]]
            normal = torch.linalg.cross(v1 - v0, v2 - v0)
            if torch.dot(normal, v0 - eye) < 0:
                visibleTris.append(tri)
                avg_depth = torch.mean(viewVerts[list(tri), 2])
                visible_depths.append(avg_depth)

        depths = torch.tensor(visible_depths, dtype=torch.float32, device=DEVICE)
        
        return screenVerts, visibleTris, depths

class BlockRenderer:
    def __init__(self):
        self.blocks: list[block] = []
        self.lookat = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
        self.currentResolution: tuple[int,int] = (800,600)
        self.currentID = 0

        self.eye = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32, device=DEVICE)
        self.up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
        self.orbitSpeed = 0.5
        self.orbitRadius = 8.0
        self.orbitAngles = [np.pi/4, np.pi/4, 0.0]
        self.orbitEnable = [False, False, False]
        
        ground = block.create_ground(id=self.idCounter())
        ground.color = torch.tensor([0, 200, 0], dtype=torch.uint8, device=DEVICE)
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
        
        self.target_fps = 60
        self.frame_time_target = 1.0 / self.target_fps
        self.last_frame_time = time.time()

    def idCounter(self):
        self.currentID += 1
        return self.currentID
        
    def add_block(self, block):
        self.blocks.append(block)
        
    def render(self):
        dpg.create_context()
        dpg.create_viewport(title='3D Blocks', width=800, height=600, resizable=True)
        
        with dpg.window(label="3D View", tag="mainView"):
            with dpg.drawlist(width=-1, height=-1, tag="drawlist"):
                pass
        
        with dpg.window(label="cameraControls", width=300,height=200):
            dpg.add_text('Orbit Controls')
            dpg.add_checkbox(label='Orbit X (Azimuth)', tag="orbit_x", callback=lambda: self.toggle_orbit(0))
            dpg.add_checkbox(label='Orbit Y (Elevation)', tag="orbit_y", callback=lambda: self.toggle_orbit(1))
            dpg.add_checkbox(label='Orbit Z (Roll)', tag="orbit_z", callback=lambda: self.toggle_orbit(2))
            dpg.add_slider_float(label="Orbit Speed", tag="orbitspeed", min_value=0.1, max_value=2.0, default_value=self.orbitSpeed, callback=self.update_orbit_speed)
            dpg.add_slider_float(label="Orbit Radius", tag="orbitradius", min_value=1.0, max_value=20.0, default_value=self.orbitRadius, callback=self.update_orbit_radius)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("mainView", True)
        dpg.set_viewport_resize_callback(callback=self._on_resize)
        while dpg.is_dearpygui_running():
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            self.update_camera_position()
            if elapsed >= self.frame_time_target:
                self.update_camera_position()
                self.render_frame()
                dpg.render_dearpygui_frame()
                frame_time = time.time() - current_time
                sleep_time = max(0, self.frame_time_target - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.last_frame_time = current_time
            else:
                dpg.render_dearpygui_frame()
        #dpg.start_dearpygui()
        dpg.destroy_context()
        
    def toggle_orbit(self, axis):
        self.orbitEnable[axis] = not self.orbitEnable[axis]

    def update_orbit_speed(self):
        self.orbitSpeed = dpg.get_value("orbitspeed")

    def update_orbit_radius(self):
        self.orbitRadius = dpg.get_value("orbitradius")

    def update_camera_position(self):
        angle_step = self.orbitSpeed / 60.0

        for i in range(3):
            if self.orbitEnable[i]:
                self.orbitAngles[i] += angle_step
        
        xAngle, yAngle, zAngle = self.orbitAngles

        yAngle = np.clip(yAngle, -np.pi / 2.0 + 1e-6, np.pi / 2.0 - 1e-6)
        self.orbitAngles[1] = yAngle

        eye_x = self.orbitRadius * np.cos(yAngle) * np.sin(xAngle)
        eye_y = self.orbitRadius * np.sin(yAngle)
        eye_z = self.orbitRadius * np.cos(yAngle) * np.cos(xAngle)
        self.eye = torch.tensor([eye_x, eye_y, eye_z], dtype=torch.float32, device=DEVICE) + self.lookat

        forward = self.lookat - self.eye
        forward = forward / torch.norm(forward)

        up_x_dir = -np.sin(yAngle) * np.sin(xAngle)
        up_y_dir = np.cos(yAngle)
        up_z_dir = -np.sin(yAngle) * np.cos(xAngle)
        up_unrolled = torch.tensor([up_x_dir, up_y_dir, up_z_dir], dtype=torch.float32, device=DEVICE)

        right = torch.linalg.cross(forward, up_unrolled)
        right = right / torch.norm(right)
        
        true_up = torch.linalg.cross(right, forward)

        cos_roll = np.cos(zAngle)
        sin_roll = np.sin(zAngle)
        
        self.up = true_up * cos_roll + right * sin_roll


    def _on_resize(self):
        windowWidth = dpg.get_item_width("mainView")
        windowHeight = dpg.get_item_height("mainView")
        dpg.configure_item("drawlist", width=windowWidth, height=windowHeight)
        self.currentResolution = (windowWidth, windowHeight)

    def render_frame(self):
        dpg.delete_item("drawlist", children_only=True)
        triangles_to_draw = []

        for block in self.blocks:
            screen_verts, visible_tris, depths = block.project_2d(
                eye=self.eye,
                lookat=self.lookat,
                up=self.up,
                res=self.currentResolution,
                near=0.1,
                far=10000
            )
            
            screen_verts_np = screen_verts.cpu().numpy()
            color_np = block.color.cpu().numpy().tolist()
            depths_np = depths.cpu().numpy()
            
            for i, tri in enumerate(visible_tris):
                v0 = screen_verts_np[tri[0]]
                v1 = screen_verts_np[tri[1]]
                v2 = screen_verts_np[tri[2]]
                
                triangles_to_draw.append({
                    'points': [(v0[0], v0[1]), (v1[0], v1[1]), (v2[0], v2[1])],
                    'color': color_np,
                    'depth': depths_np[i]
                })
        triangles_to_draw.sort(key=lambda x: x['depth'], reverse=False)
  
        # Draw all triangles in sorted order
        with dpg.draw_node(parent="drawlist"):
            for triangle in triangles_to_draw:
                dpg.draw_triangle(
                    *triangle['points'],
                    color=triangle['color'],
                    fill=triangle['color']
                )
        
if __name__ == "__main__":
    renderer = BlockRenderer()
    renderer.render()