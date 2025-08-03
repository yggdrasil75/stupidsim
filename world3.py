from dataclasses import dataclass, field
import torch
from holder.mesh import mesh, project_2d
from globals import DEVICE
from shapes.sphere import create_sphere_mesh
import dearpygui.dearpygui as dpg
import numpy as np
from plate import Plate
import math
from util import time_function, print_timing_stats

@dataclass
class World:
    torch.set_default_device(DEVICE)
    sphere_mesh: mesh = field(default_factory=lambda: create_sphere_mesh(segments=128, rings=128))
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
    plates: list[Plate] = field(default_factory=list, init=False)
    plate_colors: torch.Tensor = field(init=False)
    colormap_mode: str = field(default="null")  # Changed default to show new feature
    
    @time_function
    def __post_init__(self):
        # Initialize simulation state arrays based on the sphere mesh vertices
        num_vertices = len(self.sphere_mesh.vertices)
        self.heightmap = torch.zeros(num_vertices, dtype=torch.float32)
        self.water_content = torch.zeros(num_vertices, dtype=torch.float32)
        self.plate_ids = torch.full((num_vertices,), -1, dtype=torch.long, device=DEVICE)
        
        # Initialize heightmap with random noise
        self.heightmap = torch.rand(num_vertices, dtype=torch.float32) * \
                        (self.max_height - self.min_height) + self.min_height

        print("Initializing plates...")
        self._initialize_plates()
        print("Growing plates...")
        self._grow_plates()
        print("Validating plates...")
        self._validate_and_reindex_plates()
        print(f"Final plate count: {len(self.plates)}")

        # Generate colors for the final set of plates
        self.plate_colors = torch.randint(0, 256, (len(self.plates), 4), dtype=torch.uint8, device=DEVICE)
        self.plate_colors[:, 3] = 255 # Full alpha
        print(f'total vertices: {len(self.sphere_mesh.vertices)}')

    @time_function
    def _initialize_plates(self):
        """Selects initial plate centers and creates Plate objects."""
        num_vertices = len(self.sphere_mesh.vertices)
        center_indices = torch.randperm(num_vertices, device=DEVICE)[:self.plate_count]

        for i in range(self.plate_count):
            center_idx: int = int(center_indices[i].item())
            plate_verts = torch.tensor([center_idx], dtype=torch.long, device=DEVICE)

            if torch.rand(1) < 0.3:  # 30% chance of continental plate
                plate = Plate.create_continental_plate(plate_verts)
                plate.add_continental_center(center_idx)
            else:
                plate = Plate.create_oceanic_plate(plate_verts)
            self.plates.append(plate)
            self.plate_ids[center_idx] = i

    @time_function
    def _grow_plates(self):
        """Optimized plate growth using tensor operations and batched processing."""
        vertex_neighbors = self.sphere_mesh._neighbor_map['vertex_to_vertices']
        num_vertices = len(self.sphere_mesh.vertices)
        
        # Convert neighbor map to tensor format for faster access
        max_neighbors = max(len(v) for v in vertex_neighbors.values())
        neighbor_tensor = torch.full((num_vertices, max_neighbors), -1, dtype=torch.long, device=DEVICE)
        for v_idx, neighbors in vertex_neighbors.items():
            neighbor_tensor[v_idx, :len(neighbors)] = torch.tensor(neighbors, dtype=torch.long, device=DEVICE)
        
        # Precompute all vertex positions as a tensor
        vertex_positions = self.sphere_mesh.vertices
        
        # Initialize plate centers
        plate_centers = torch.zeros(len(self.plates), 3, device=DEVICE)
        for i, plate in enumerate(self.plates):
            plate_centers[i] = vertex_positions[plate.vertex_ids[0]]
        
        # Create a mask for unassigned vertices
        unassigned = self.plate_ids == -1
        
        while torch.any(unassigned):
            # Find all frontier vertices (unassigned vertices adjacent to assigned ones)
            assigned_neighbors = neighbor_tensor[self.plate_ids != -1]
            frontier_mask = torch.isin(neighbor_tensor, assigned_neighbors) & (self.plate_ids == -1).unsqueeze(1)
            frontier_verts = torch.unique(torch.where(frontier_mask)[0])
            
            if len(frontier_verts) == 0:
                break
                
            # Get positions of frontier vertices
            frontier_pos = vertex_positions[frontier_verts]
            
            # Find all possible plate claims (vectorized)
            # Distance from each frontier vertex to each plate center
            dists = torch.cdist(frontier_pos, plate_centers)
            
            # Get growth rates for all plates
            growth_rates = torch.tensor([p.growth_rate for p in self.plates], device=DEVICE)
            
            # Calculate scores (vectorized)
            rand_factors = 1.0 + (torch.rand(len(frontier_verts), device=DEVICE) * 0.5)
            scores = (growth_rates / (dists + 1e-6)) * rand_factors.unsqueeze(1)
            
            # Find best plate for each frontier vertex
            best_plate_ids = torch.argmax(scores, dim=1)
            
            # Update plate assignments
            self.plate_ids[frontier_verts] = best_plate_ids
            
            # Update unassigned mask
            unassigned = self.plate_ids == -1
        
        # Update Plate objects with their final vertex sets
        for i, plate in enumerate(self.plates):
            plate.vertex_ids = torch.where(self.plate_ids == i)[0]

    @time_function
    def _validate_and_reindex_plates(self):
        """Checks for disjointed plates, splits or merges them, and re-indexes all plates."""
        vertex_neighbors = self.sphere_mesh._neighbor_map['vertex_to_vertices']
        plates_to_process = self.plates.copy()
        final_plates = []
        
        processed_plate_ids = set()

        while plates_to_process:
            plate = plates_to_process.pop(0)
            plate_id = self.plate_ids[plate.vertex_ids[0]].item()

            if plate_id in processed_plate_ids:
                continue
            processed_plate_ids.add(plate_id)
            
            all_plate_verts = set(torch.where(self.plate_ids == plate_id)[0].tolist())
            visited = set()
            components = []

            # Use BFS to find all contiguous components for the current plate ID
            while all_plate_verts:
                q = [all_plate_verts.pop()]
                component = {q[0]}
                visited.add(q[0])
                head = 0
                while head < len(q):
                    curr = q[head]
                    head += 1
                    for neighbor in vertex_neighbors.get(curr, []):
                        if neighbor in all_plate_verts and neighbor not in visited:
                            visited.add(neighbor)
                            component.add(neighbor)
                            q.append(neighbor)
                            all_plate_verts.remove(neighbor)
                components.append(list(component))

            if not components: continue
            
            # Sort components by size, largest first
            components.sort(key=len, reverse=True)

            # The largest component becomes the main plate
            main_component_verts = torch.tensor(components[0], dtype=torch.long, device=DEVICE)
            new_plate = plate # Reuse original plate object
            new_plate.vertex_ids = main_component_verts
            final_plates.append(new_plate)
            
            # Handle other, smaller "fragment" components
            for fragment_verts_list in components[1:]:
                fragment_verts = torch.tensor(fragment_verts_list, dtype=torch.long, device=DEVICE)
                
                # If fragment is large, create a new plate for it
                if len(fragment_verts) > 50: # Threshold for a new plate
                    new_fragment_plate = Plate.create_oceanic_plate(fragment_verts)
                    final_plates.append(new_fragment_plate)
                else: # If fragment is small, merge it with the best neighbor
                    border_counts = {}
                    for vert_idx in fragment_verts_list:
                        for neighbor in vertex_neighbors.get(vert_idx, []):
                            neighbor_plate_id = self.plate_ids[neighbor].item()
                            if self.plate_ids[vert_idx] != neighbor_plate_id:
                                border_counts[neighbor_plate_id] = border_counts.get(neighbor_plate_id, 0) + 1
                    
                    if border_counts:
                        best_neighbor_id = max(border_counts, key=border_counts.get)
                        self.plate_ids[fragment_verts] = best_neighbor_id

        # Final re-indexing step
        self.plates = final_plates
        new_plate_ids_tensor = torch.full_like(self.plate_ids, -1)
        for new_id, plate in enumerate(self.plates):
            # Update the main plate_ids map with the new, consolidated IDs
            new_plate_ids_tensor[plate.vertex_ids] = new_id
        
        self.plate_ids = new_plate_ids_tensor
        self.plate_count = torch.tensor(len(self.plates))

    @time_function
    def update_vertices_based_on_heightmap(self):
        """Update the mesh vertices based on the current heightmap"""

        ### THIS IS BROKEN! UPDATE HEIGHMAP AS COLOR INSTEAD OF USING THIS
        vertices = self.sphere_mesh.vertices.clone()
        #normals = torch.nn.functional.normalize(vertices, dim=1)
        #scaled_vertices = normals * (1.0 + self.heightmap.unsqueeze(1))
        #self.sphere_mesh.vertices = scaled_vertices
    
    @time_function
    def update_colors(self):
        """Update mesh colors based on current colormap mode"""
        num_vertices = len(self.sphere_mesh.vertices)
        if self.colormap_mode == "water":
            water_normalized = (self.water_content - self.water_content.min()) / \
                             (self.water_content.max() - self.water_content.min() + 1e-6)
            blue = torch.clamp(water_normalized * 255, 0, 255)
            colors = torch.stack([
                torch.zeros_like(blue), torch.zeros_like(blue), blue, torch.full_like(blue, 255)
            ], dim=1)
            self.sphere_mesh.color = colors.to(torch.uint8)
            
        elif self.colormap_mode == "plates":
            full_colors = torch.full((num_vertices, 4), 128, dtype=torch.uint8, device=DEVICE)
            full_colors[:, 3] = 255 # Set alpha
            
            assigned_mask = self.plate_ids != -1
            if torch.any(assigned_mask):
                assigned_ids = self.plate_ids[assigned_mask]
                if assigned_ids.max() < len(self.plate_colors):
                    full_colors[assigned_mask] = self.plate_colors[assigned_ids]
            
            self.sphere_mesh.color = full_colors

        elif self.colormap_mode == "terrain":
            # TODO: Implement terrain colormap
            self.sphere_mesh.color = torch.full((num_vertices, 4), 128, dtype=torch.uint8, device=DEVICE)
        elif self.colormap_mode == "temperature":
            # TODO: Implement temperature colormap
            self.sphere_mesh.color = torch.full((num_vertices, 4), 128, dtype=torch.uint8, device=DEVICE)
    
    @time_function
    def simulate_erosion(self, steps: int = 1):
        """Simple erosion simulation step"""
        for _ in range(steps):
            self.water_content += self.rainfall_rate
            water_diff = torch.roll(self.water_content, 1, 0) - self.water_content
            self.water_content += water_diff * self.water_flow_min
            
            height_diff = torch.roll(self.heightmap, 1, 0) - self.heightmap
            erosion_amount = self.water_content * torch.sigmoid(height_diff * 10.0) * 0.01
            self.heightmap -= erosion_amount
            self.water_content *= (1.0 - self.evaporation_rate)
            
            self.heightmap = torch.clamp(self.heightmap, self.min_height.item(), self.max_height.item())
        
        if steps > 0:
            self.update_vertices_based_on_heightmap()
            self.update_colors()




def render_world():
    world = World()
    
    # Initialize camera parameters
    eye = torch.tensor([3.0, 2.0, 3.0], dtype=torch.float32, device=DEVICE)
    lookat = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
    
    # Camera control parameters
    camera_distance = torch.norm(eye).item()
    azimuth = np.atan2(eye[2].item(), eye[0].item())  # Angle around Y axis
    elevation = np.asin(eye[1].item() / camera_distance)  # Angle above/below horizon
    
    dpg.create_context()
    dpg.create_viewport(title='Procedural World', width=1000, height=700)

    #@time_function
    def update_colormap_callback(sender, app_data):
        oldmap = world.colormap_mode
        world.colormap_mode = app_data
        if world.colormap_mode != oldmap:
            dpg.configure_item("legend_window", show=(app_data == "plates"))
            if app_data == "plates":
                # Rebuild legend
                dpg.delete_item("legend_window", children_only=True)
                with dpg.group(parent="legend_window"):
                    for i, plate in enumerate(world.plates):
                        if i < len(world.plate_colors):
                            color = world.plate_colors[i].cpu().numpy().tolist()
                            with dpg.group(horizontal=True):
                                # Use a drawlist to create a colored square
                                with dpg.drawlist(width=20, height=20):
                                    dpg.draw_rectangle((0, 0), (20, 20), color=color, fill=color)
                                dpg.add_text(f"Plate {i} ({plate.plate_type[:4]})")
            world.update_colors()

    @time_function
    def update_camera_position():
        nonlocal eye
        # Convert spherical coordinates (distance, azimuth, elevation) to Cartesian
        x = camera_distance * np.cos(elevation) * np.cos(azimuth)
        y = camera_distance * np.sin(elevation)
        z = camera_distance * np.cos(elevation) * np.sin(azimuth)
        eye = torch.tensor([x, y, z], dtype=torch.float32, device=DEVICE)
        
        # Update camera info text
        dpg.set_value("camera_info", 
                     f"Camera: Dist={camera_distance:.2f}, Azim={np.degrees(azimuth):.1f}°, Elev={np.degrees(elevation):.1f}°")

    def camera_control_callback(sender, app_data):
        nonlocal camera_distance, azimuth, elevation
        if sender == "camera_distance":
            camera_distance = app_data
        elif sender == "camera_azimuth":
            azimuth = np.radians(app_data)
        elif sender == "camera_elevation":
            # Limit elevation to prevent flipping
            elevation = np.radians(max(-89, min(89, app_data)))
        
        update_camera_position()

    def mouse_drag_callback(sender, app_data):
        if dpg.is_mouse_button_dragging(dpg.mvMouseButton_Left, 1.0):
            drag_delta = dpg.get_mouse_drag_delta()
            dpg.is_mouse_button_released(dpg.mvMouseButton_Left)
            
            nonlocal azimuth, elevation
            if isinstance(drag_delta, float):
                azimuth -= drag_delta * 0.01
                elevation += drag_delta * 0.01
            # if isinstance(drag_delta, list):
            #     azimuth -= drag_delta[0] * 0.01
            #     elevation += drag_delta[1] * 0.01

            elevation = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, elevation))
            
            update_camera_position()

    def mouse_wheel_callback(sender, app_data):
        nonlocal camera_distance
        camera_distance *= 0.9 if app_data > 0 else 1.1
        camera_distance = max(0.1, min(20.0, camera_distance))
        update_camera_position()
        dpg.set_value("camera_distance", camera_distance)

    with dpg.window(label="Legend", tag="legend_window", show=True, width=200, pos=(800, 0)):
        pass

    with dpg.window(label="3D View", tag="primary", width=800, height=600):
        with dpg.group(horizontal=True):
            dpg.add_text("Colormap:")
            dpg.add_radio_button(
                items=["plates", "water", "terrain", "temperature"],
                default_value=world.colormap_mode,
                callback=update_colormap_callback,
                tag="colormap_selector",
                horizontal=True
            )
        
        with dpg.drawlist(width=800, height=600, tag="draw_area"):
            pass
    
    with dpg.window(label='Camera Controls'):
        # Add camera controls
        with dpg.collapsing_header(label="Camera Controls", default_open=True):
            dpg.add_text("Camera Position:", tag="camera_info")
            dpg.add_slider_float(
                label="Distance", 
                tag="camera_distance",
                min_value=1.0, 
                max_value=15.0, 
                default_value=camera_distance,
                callback=camera_control_callback
            )
            dpg.add_slider_float(
                label="Azimuth (degrees)", 
                tag="camera_azimuth",
                min_value=-180, 
                max_value=180, 
                default_value=math.degrees(azimuth),
                callback=camera_control_callback
            )
            dpg.add_slider_float(
                label="Elevation (degrees)", 
                tag="camera_elevation",
                min_value=-180, 
                max_value=180, 
                default_value=math.degrees(elevation),
                callback=camera_control_callback
            )

    # Set up mouse controls
    with dpg.handler_registry():
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=mouse_drag_callback)
        dpg.add_mouse_wheel_handler(callback=mouse_wheel_callback)

    # Initial setup
    update_colormap_callback(None, "plates")
    update_camera_position()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    
    while dpg.is_dearpygui_running():
        print_timing_stats()
        world.simulate_erosion(steps=0) # Run simulation logic, but don't advance time yet
        #world.update_colors() # ensure colors are correct

        dpg.delete_item("draw_area", children_only=True)
        
        screen_verts, visible_tris, _ = project_2d(
            [world.sphere_mesh], eye, lookat, up, fov=60.0, res=(800, 600)
            )
        
        if screen_verts and visible_tris[0]:
            points_np = screen_verts[0].cpu().numpy()
            colors_np = world.sphere_mesh.color.cpu().numpy()
            for tri in visible_tris[0]:
                p1, p2, p3 = points_np[tri[0]], points_np[tri[1]], points_np[tri[2]]
                # Use color from the first vertex of the triangle
                color = colors_np[tri[0]].tolist()
                dpg.draw_triangle(p1, p2, p3, color=color, fill=color, parent="draw_area")
    
        dpg.render_dearpygui_frame()
    
    dpg.destroy_context()

if __name__ == "__main__":
    render_world()