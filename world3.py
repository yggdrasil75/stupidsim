from dataclasses import dataclass, field
import torch
from holder.mesh import mesh, project_2d
from globals import DEVICE
from shapes.sphere import create_sphere_mesh
import dearpygui.dearpygui as dpg


@dataclass
class World:
    sphere_mesh: mesh = field(default_factory=lambda: create_sphere_mesh(segments=64, rings=64))
    sea_level: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0, dtype=torch.float32))
    min_height: torch.Tensor = field(default_factory=lambda: torch.tensor(-1.0, dtype=torch.float32))
    max_height: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.float32))
    plate_count: torch.Tensor = field(default_factory=lambda: torch.tensor(10, dtype=torch.int32))
    rainfall_rate: torch.Tensor = field(default_factory=lambda: torch.tensor(0.1, dtype=torch.float32))
    evaporation_rate: torch.Tensor = field(default_factory=lambda: torch.tensor(0.05, dtype=torch.float32))
    water_flow_max: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.float32))
    water_flow_min: torch.Tensor = field(default_factory=lambda: torch.tensor(0.01, dtype=torch.float32))
    
    # Additional fields for simulation state
    heightmap: torch.Tensor = field(init=False)
    water_content: torch.Tensor = field(init=False)
    plate_ids: torch.Tensor = field(init=False)
    
    def __post_init__(self):
        # Initialize simulation state arrays based on the sphere mesh vertices
        num_vertices = len(self.sphere_mesh.vertices)
        self.heightmap = torch.zeros(num_vertices, dtype=torch.float32)
        self.water_content = torch.zeros(num_vertices, dtype=torch.float32)
        
        # Initialize heightmap with random noise
        self.heightmap = torch.rand(num_vertices, dtype=torch.float32) * \
                        (self.max_height - self.min_height) + self.min_height
    
    def update_vertices_based_on_heightmap(self):
        """Update the mesh vertices based on the current heightmap"""
        vertices = self.sphere_mesh.vertices.clone()
        normals = torch.nn.functional.normalize(vertices, dim=1)
        scaled_vertices = normals * (1.0 + self.heightmap.unsqueeze(1))
        self.sphere_mesh.vertices = scaled_vertices
    
    def simulate_erosion(self, steps: int = 1):
        """Simple erosion simulation step"""
        for _ in range(steps):
            # Simple water accumulation and erosion simulation
            self.water_content += self.rainfall_rate
            water_diff = torch.roll(self.water_content, 1, 0) - self.water_content
            self.water_content += water_diff * self.water_flow_min
            
            # Erosion proportional to water content and slope
            height_diff = torch.roll(self.heightmap, 1, 0) - self.heightmap
            erosion_amount = self.water_content * torch.sigmoid(height_diff * 10.0) * 0.01
            self.heightmap -= erosion_amount
            self.water_content *= (1.0 - self.evaporation_rate)
            
            # Ensure height stays within bounds
            self.heightmap = torch.clamp(self.heightmap, self.min_height.item(), self.max_height.item())
        
        # Update the mesh vertices
        self.update_vertices_based_on_heightmap()



def render_world():
    world = World()
    # Camera parameters
    eye = torch.tensor([3.0, 2.0, 3.0], dtype=torch.float32, device=DEVICE)
    lookat = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
    
    # Create DPG context
    dpg.create_context()
    dpg.create_viewport(title='3D Sphere', width=800, height=600)
    
    with dpg.window(label="3D View",tag="primary", width=800, height=600):
        with dpg.drawlist(width=800, height=600):
            # Project 3D mesh to 2D
            screen_verts, visible_tris, _ = project_2d(
                [world.sphere_mesh], eye, lookat, up, fov=60.0, res=(800, 600)
                )
            
            # Draw each triangle
            for tri in visible_tris[0]:
                points = screen_verts[0][tri].cpu().numpy()
                dpg.draw_triangle(
                    points[0], points[1], points[2],
                    color=(200, 50, 50),
                    thickness=1
                )
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    render_world()