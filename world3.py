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
    colormap_mode: str = field(default="plates")  # Changed default to show new feature
    
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

    def _grow_plates(self):
        """Grows plates from centers until all vertices are assigned using a weighted frontier expansion."""
        vertex_neighbors = self.sphere_mesh._neighbor_map['vertex_to_vertices']
        
        while torch.any(self.plate_ids == -1):
            # Find all unassigned vertices adjacent to any assigned vertex (the global frontier)
            assigned_mask = self.plate_ids != -1
            assigned_indices = torch.where(assigned_mask)[0]
            
            global_frontier = set()
            for v_idx in assigned_indices.tolist():
                plate_id = self.plate_ids[v_idx].item()
                for neighbor in vertex_neighbors[v_idx]:
                    if self.plate_ids[neighbor] == -1:
                        global_frontier.add((neighbor, plate_id))

            if not global_frontier:
                # Should not happen if graph is connected, but as a safeguard
                break

            # Score each potential claim based on distance, growth rate, and randomness
            claims = []
            for frontier_vert_idx, claiming_plate_id in global_frontier:
                plate = self.plates[claiming_plate_id]
                center_pos = self.sphere_mesh.vertices[plate.vertex_ids[0]]
                frontier_vert_pos = self.sphere_mesh.vertices[frontier_vert_idx]
                
                dist = torch.norm(center_pos - frontier_vert_pos)
                rand_factor = 1.0 + (torch.rand(1).item() * 0.5) # Jagged edges
                
                score = (plate.growth_rate / (dist + 1e-6)) * rand_factor
                claims.append((score.item(), frontier_vert_idx, claiming_plate_id))

            # Sort claims by score, highest first
            claims.sort(key=lambda x: x[0], reverse=True)
            
            # Process claims, ensuring each vertex is claimed only once per round by the highest bidder
            claimed_this_round = set()
            for _, vert_idx, plate_id in claims:
                if vert_idx not in claimed_this_round:
                    self.plate_ids[vert_idx] = plate_id
                    claimed_this_round.add(vert_idx)
        
        # After loop, update the Plate objects with their final vertex sets
        for i, plate in enumerate(self.plates):
            plate.vertex_ids = torch.where(self.plate_ids == i)[0]

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


    def update_vertices_based_on_heightmap(self):
        """Update the mesh vertices based on the current heightmap"""

        ### THIS IS BROKEN! UPDATE HEIGHMAP AS COLOR INSTEAD OF USING THIS
        vertices = self.sphere_mesh.vertices.clone()
        #normals = torch.nn.functional.normalize(vertices, dim=1)
        #scaled_vertices = normals * (1.0 + self.heightmap.unsqueeze(1))
        #self.sphere_mesh.vertices = scaled_vertices
    
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
        
        self.update_vertices_based_on_heightmap()
        self.update_colors()

def render_world():
    world = World()
    eye = torch.tensor([3.0, 2.0, 3.0], dtype=torch.float32, device=DEVICE)
    lookat = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
    
    dpg.create_context()
    dpg.create_viewport(title='Procedural World', width=1000, height=700)

    def update_colormap_callback(sender, app_data):
        world.colormap_mode = app_data
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
    
    with dpg.window(label="Legend", tag="legend_window", show=True, width=200, pos=(800, 0)):
        pass # To be populated by callback

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

    # Initial setup
    update_colormap_callback(None, world.colormap_mode)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    
    while dpg.is_dearpygui_running():
        world.simulate_erosion(steps=0) # Run simulation logic, but don't advance time yet
        world.update_colors() # ensure colors are correct

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