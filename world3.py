from dataclasses import dataclass, field
import torch
from holder.mesh import mesh, project_2d
from globals import DEVICE
from shapes.sphere import create_sphere_mesh
import dearpygui.dearpygui as dpg
import numpy as np
from plate import Plate

@dataclass
class World:
    torch.set_default_device(DEVICE)
    sphere_mesh: mesh = field(default_factory=lambda: create_sphere_mesh(segments=64, rings=64))
    sea_level: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0, dtype=torch.float32))
    min_height: torch.Tensor = field(default_factory=lambda: torch.tensor(-1.0, dtype=torch.float32))
    max_height: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.float32))
    plate_count: torch.Tensor = field(default_factory=lambda: torch.tensor(15, dtype=torch.int32))
    rainfall_rate: torch.Tensor = field(default_factory=lambda: torch.tensor(0.1, dtype=torch.float32))
    evaporation_rate: torch.Tensor = field(default_factory=lambda: torch.tensor(0.05, dtype=torch.float32))
    water_flow_max: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.float32))
    water_flow_min: torch.Tensor = field(default_factory=lambda: torch.tensor(0.01, dtype=torch.float32))
    
    # Additional fields for simulation state
    heightmap: torch.Tensor = field(init=False)
    water_content: torch.Tensor = field(init=False)
    plate_ids: torch.Tensor = field(init=False)
    colormap_mode: str = field(default="water")  # Added colormap mode
    
    def __post_init__(self):
        # Initialize simulation state arrays based on the sphere mesh vertices
        num_vertices = len(self.sphere_mesh.vertices)
        self.heightmap = torch.zeros(num_vertices, dtype=torch.float32)
        self.water_content = torch.zeros(num_vertices, dtype=torch.float32)
        self.plate_ids = torch.zeros(num_vertices, dtype=torch.int32)
        
        # Initialize heightmap with random noise
        self.heightmap = torch.rand(num_vertices, dtype=torch.float32) * \
                        (self.max_height - self.min_height) + self.min_height
        

        self.plates = []
        for i in range(self.plate_count):
            # Randomly assign vertices to plates (simplified)
            plate_verts = torch.randperm(len(self.sphere_mesh.vertices))
            if torch.rand(1) < 0.3:  # 30% chance of continental plate
                plate = Plate.create_continental_plate(plate_verts)
                plate.add_continental_center(plate_verts[0].item())  # Add first vertex as continent center
            else:
                plate = Plate.create_oceanic_plate(plate_verts)
            self.plates.append(plate)
            self.plate_ids[plate_verts] = i 
    
    def update_vertices_based_on_heightmap(self):
        """Update the mesh vertices based on the current heightmap"""
        vertices = self.sphere_mesh.vertices.clone()
        normals = torch.nn.functional.normalize(vertices, dim=1)
        scaled_vertices = normals * (1.0 + self.heightmap.unsqueeze(1))
        self.sphere_mesh.vertices = scaled_vertices
    
    def update_colors(self):
        """Update mesh colors based on current colormap mode"""
        if self.colormap_mode == "water":
            # Water visualization - blue based on water content
            water_normalized = (self.water_content - self.water_content.min()) / \
                             (self.water_content.max() - self.water_content.min() + 1e-6)
            blue = torch.clamp(water_normalized * 255, 0, 255)
            colors = torch.stack([
                torch.zeros_like(blue),
                torch.zeros_like(blue),
                blue,
            ], dim=1)
            self.sphere_mesh.color = colors / 255  # Normalize to 0-1
            
        elif self.colormap_mode == "terrain":
            # TODO: Implement terrain colormap
            pass
        elif self.colormap_mode == "temperature":
            # TODO: Implement temperature colormap
            pass
        elif self.colormap_mode == "plates":
            # TODO: Implement plate colormap
            pass
    
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
        
        # Update the mesh vertices and colors
        self.update_vertices_based_on_heightmap()
        self.update_colors()

def render_world():
    world = World()
    # Camera parameters
    eye = torch.tensor([3.0, 2.0, 3.0], dtype=torch.float32, device=DEVICE)
    lookat = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
    
    # Create DPG context
    dpg.create_context()
    dpg.create_viewport(title='3D Sphere', width=800, height=600)
    
    with dpg.window(label="3D View", tag="primary", width=800, height=600):
        with dpg.group(horizontal=True):
            dpg.add_text("Colormap:")
            dpg.add_radio_button(
                items=["water", "terrain", "temperature", "plates"],
                default_value="water",
                callback=lambda sender, data: setattr(world, 'colormap_mode', data),
                tag="colormap_selector"
            )
        
        with dpg.drawlist(width=800, height=600):
            # Project 3D mesh to 2D
            screen_verts, visible_tris, _ = project_2d(
                [world.sphere_mesh], eye, lookat, up, fov=60.0, res=(800, 600)
                )
            
            # Draw each triangle
            for tri in visible_tris[0]:
                points = screen_verts[0][tri].cpu().numpy()
                # Get color from the first vertex of the triangle
                color = world.sphere_mesh.color[tri[0]].cpu().numpy() * 255
                dpg.draw_triangle(
                    points[0], points[1], points[2],
                    color=(color[0], color[1], color[2]),
                    fill=color[0],
                    thickness=1
                )
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    
    # Main simulation loop
    while dpg.is_dearpygui_running():
        world.simulate_erosion()
        dpg.render_dearpygui_frame()
    
    dpg.destroy_context()

if __name__ == "__main__":
    render_world()