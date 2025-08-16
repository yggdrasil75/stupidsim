import array
import copy
from dataclasses import dataclass, field
import random
import numba
import numpy as np
from holder.mesh import mesh, project_2d, rasterize
from shapes.sphere import create_sphere_mesh
import dearpygui.dearpygui as dpg
from numba import njit, prange, int64, float32
from plate import Plate
import math
from util import norm, spherical_distance, time_function, print_timing_stats, make_2D_array

#@njit(cache=True)
def _update_elevations_a(plates, _heightmap):
    # Reset heightmap to base elevations
    for plate in plates:
        if len(plate.vertex_ids) > 0:
            _heightmap[plate.vertex_ids] = plate.base_elevation
            
    # Apply continental effects
    for plate in plates:
        if plate.plate_type == 'continental' and len(plate.continental_centers) > 0:
            continental_verts = plate.get_continental_vertices()
            if len(continental_verts) > 0:
                _heightmap[plate.continental_centers] += 200.0
                if len(plate.continental_border_vertices) > 0:
                    _heightmap[plate.continental_border_vertices] += 100.0
    
    return _heightmap
    
#@njit(cache=True)
def _update_elevations_b(plates, _heightmap, max_height, min_height):
    # Apply plate movement effects
    for plate in plates:
        if len(plate.vertex_ids) > 0:
            movement_factor = norm(plate.linear_velocity) * 0.1
            noise = (np.random.rand(len(plate.vertex_ids)) * movement_factor)
            _heightmap[plate.vertex_ids] += noise
        
            if norm(plate.collision_force) > 0:
                force_factor = norm(plate.collision_force) * 0.05
                _heightmap[plate.vertex_ids] += np.random.rand(len(plate.vertex_ids)) * force_factor

    # Normalize heightmap
    _heightmap = (_heightmap - _heightmap.min()) / \
                    (_heightmap.max() - _heightmap.min() + 1e-6) * \
                    (max_height - min_height) + min_height

    return _heightmap

@dataclass
class World:
    sphere_mesh: mesh = field(default_factory=lambda: create_sphere_mesh(segments=64, rings=64))
    sea_level: np.ndarray = field(default_factory=lambda: np.array(0.0, dtype=np.float32))
    plate_count: int = 15
    rainfall_rate: np.ndarray = field(default_factory=lambda: np.array(0.1, dtype=np.float32))
    evaporation_rate: np.ndarray = field(default_factory=lambda: np.array(0.05, dtype=np.float32))
    water_flow_max: np.ndarray = field(default_factory=lambda: np.array(1.0, dtype=np.float32))
    water_flow_min: np.ndarray = field(default_factory=lambda: np.array(0.01, dtype=np.float32))
    
    # Additional fields for simulation state
    heightmap: np.ndarray = field(init=False)
    water_content: np.ndarray = field(init=False)
    plate_ids: np.ndarray = field(init=False)
    plates: list[Plate] = field(default_factory=list, init=False)
    plate_colors: np.ndarray = field(init=False)
    colormap_mode: str = field(default="null")
    min_height_value: float = field(default=6357.0)
    max_height_value: float = field(default=6378.0)
    
    @property
    def min_height(self) -> np.ndarray:
        """Returns the minimum height from the heightmap"""
        return np.min(self.heightmap) if len(self.heightmap) > 0 else np.array(-1.0)
    
    @property
    def max_height(self) -> np.ndarray:
        """Returns the maximum height from the heightmap"""
        return np.max(self.heightmap) if len(self.heightmap) > 0 else np.array(1.0)

    @property
    def radius(self):
        return (self.min_height_value + self.max_height_value) / 4

    @time_function
    def __post_init__(self):
        # Initialize simulation state arrays based on the sphere mesh vertices
        num_vertices = len(self.sphere_mesh.vertices)
        self.heightmap = np.zeros(num_vertices, dtype=np.float32)
        self.water_content = np.zeros(num_vertices, dtype=np.float32)
        self.plate_ids = np.full((num_vertices,), -1, dtype=np.int64)
        
        # Initialize heightmap with random noise
        self.heightmap = np.random.rand(num_vertices).astype(np.float32) * \
                        (self.max_height - self.min_height) + self.min_height

        print("Initializing plates...")
        self.plates = self.gen_plates()
        print(f"Final plate count: {len(self.plates)}")
        for plate in self.plates:
            plate.calculate_mass()
        print("Calculating plate elevations...")
        self.update_elevations()

        print("Initial world generation complete")
        self.plate_colors = np.random.randint(0, 256, (len(self.plates), 4), dtype=np.uint8)
        self.plate_colors[:, 3] = 255
        print(f'total vertices: {len(self.sphere_mesh.vertices)}')

    def assign_origins(self, vertices: np.ndarray, num_plates: int, radius: float):
        random_indices = np.random.choice(len(vertices), num_plates, replace=False)
        plate_origins: np.ndarray = vertices[random_indices]
        # Check distances between all pairs of plate origins
        need_reassignment: bool = True
        while need_reassignment:
            need_reassignment = False
            for i in range(len(plate_origins)):
                for j in range(i+1, len(plate_origins)):
                    # Calculate spherical distance between two plate origins
                    dist = spherical_distance(vertices[random_indices[i]], vertices[random_indices[j]], radius)
                    
                    # If too close, replace one of them with a new random vertex
                    if dist < (radius / 10):
                        # Get all vertex indices not currently used as origins
                        all_indices = set(range(len(vertices)))
                        used_indices = set(random_indices)
                        available_vertices = list(all_indices - used_indices)
                        #available_vertices = [v for v in vertices if v not in plate_origins]
                        if available_vertices:  # Ensure there are vertices left to choose from
                            ni = np.random.choice(len(available_vertices))
                            random_indices[j] = ni
                            plate_origins[j] = vertices[ni]
                            need_reassignment = True  # Need to check all pairs again
                        else:
                            raise ValueError("Not enough vertices to maintain minimum distance")
        
        return random_indices

    def expand_plate(self, obj: mesh, vertices: np.ndarray, plates: list[Plate]):
        nv = len(vertices)
        mask = np.zeros(nv, dtype=bool)
        
        for plate in plates:
            #for vid in plate.vertex_ids:
            mask[plate.vertex_ids] = True
        
        while True:
            # Track if we assigned any vertices in this iteration
            assigned_any = False
            
            for plate in plates:
                # Get all adjacent vertices to this plate's current vertices
                adjacent_vertices = np.zeros(nv, dtype=bool)

                vids = np.atleast_1d(plate.vertex_ids)
                for vid in vids:
                    asj = obj.get_adjacent_vertices(vid)
                    adjacent_vertices[asj] = True
                
                # Find unassigned adjacent vertices
                candidates = np.where(adjacent_vertices & ~mask)[0]
                
                if len(candidates) > 0:
                    # Randomly select one to add to this plate
                    new_vertex = np.random.choice(candidates)
                    plate.vertex_ids = np.append(plate.vertex_ids, new_vertex)
                    mask[new_vertex] = True
                    assigned_any = True
            
            # If no plates could expand, break to avoid infinite loop
            if not assigned_any or np.all(mask):
                break
        
        return plates

    def gen_plates(self):
        obj: mesh = self.sphere_mesh
        radius: float = self.radius
        
        print("generating plates: assigning origins")
        plate_origins = self.assign_origins(obj.vertices, self.plate_count + 5, radius)
        plates: list[Plate] = []
        for i, plate in enumerate(plate_origins):
            arandomnumber = np.random.rand()
            if arandomnumber > 0.4:
                a = Plate.create_oceanic_plate(ID=i, vertex_ids=plate)
            else:
                a = Plate.create_continental_plate(ID=i, vertex_ids=plate)
                a.add_continental_center(plate)
            plates.append(a)
        print("expanding plates")
        plates = self.expand_plate(obj, obj.vertices, plates)
        
        def get_plate_for_vertex(vid):
            for plate in plates:
                if vid in plate.vertex_ids:
                    return plate
            return None
        
        for plate in plates:
            plate.calculate_mass()

        print("moving erratic vertices")
        for plate in plates:
            to_remove = []
            for vid in plate.vertex_ids:
                # Get adjacent vertices
                adj_vids = set(obj.get_adjacent_vertices(vid))
                
                # Count how many are in the same plate
                same_plate_count = sum(1 for adj_vid in adj_vids if adj_vid in plate.vertex_ids)
                
                # If only 1 or less, consider moving to adjacent plate
                if same_plate_count <= 1:
                    # Find adjacent plates
                    adjacent_plates = set()
                    for adj_vid in adj_vids:
                        adj_plate = get_plate_for_vertex(adj_vid)
                        if adj_plate and adj_plate != plate:
                            adjacent_plates.add(adj_plate)
                    
                    # Move to a random adjacent plate if any exist
                    if adjacent_plates:
                        new_plate = np.random.choice(list(adjacent_plates))
                        new_plate.vertex_ids = np.append(new_plate.vertex_ids, vid)
                        to_remove.append(vid)
            
            # Remove from current plate
            plate.vertex_ids = np.array([v for v in plate.vertex_ids if v not in to_remove])
        
        print("find locked plates and merge to parent")
        i = 0
        while i < len(plates):
            plate = plates[i]
            # Get all adjacent vertices to this plate
            plate_boundary = set()
            for vid in plate.vertex_ids:
                adj_vids = obj.get_adjacent_vertices(vid)
                for adj_vid in adj_vids:
                    if adj_vid not in plate.vertex_ids:
                        plate_boundary.add(adj_vid)
            
            # Find plates that contain all boundary vertices
            surrounding_plates = []
            for other_plate in plates:
                if other_plate == plate:
                    continue
                if all(boundary_vid in other_plate.vertex_ids for boundary_vid in plate_boundary):
                    surrounding_plates.append(other_plate)
            
            # If found, merge into one of them
            if surrounding_plates:
                # Choose the largest surrounding plate
                largest_surrounder = max(surrounding_plates, key=lambda p: len(p.vertex_ids))
                largest_surrounder.vertex_ids.extend(plate.vertex_ids)
                plates.pop(i)
                # Don't increment i since we removed an element
                continue
            i += 1
        
        print("if too few plates after removing locked, split a massive one")
        while len(plates) < self.plate_count and len(plates) > 0:
            # Find largest plate
            largest_plate = max(plates, key=lambda p: len(p.vertex_ids))
            
            # Split into two roughly equal parts using BFS
            if len(largest_plate.vertex_ids) >= 2:
                # Start BFS from two distant points
                start1 = largest_plate.vertex_ids[0]
                visited = set()
                queue = [start1]
                part1 = set()
                
                target_size = len(largest_plate.vertex_ids) // 2
                
                while queue and len(part1) < target_size:
                    current = queue.pop(0)
                    if current in part1:
                        continue
                    part1.add(current)
                    
                    # Add adjacent vertices in same plate
                    for neighbor in obj.get_adjacent_vertices(current):
                        if neighbor in largest_plate.vertex_ids and neighbor not in part1:
                            queue.append(neighbor)
                
                # Create new plate with part1
                new_plate = Plate.create_oceanic_plate(len(plates), vertex_ids=part1)
                plates.append(new_plate)
                
                # Update original plate
                largest_plate.vertex_ids = np.array([v for v in largest_plate.vertex_ids if v not in part1])
            else:
                break  # Can't split further
        
        print("if too many plates after locking, merge some tiny ones.")
        while len(plates) > self.plate_count and len(plates) > 1:
            # Find all pairs of adjacent plates
            adjacent_pairs = []
            for i in range(len(plates)):
                for j in range(i+1, len(plates)):
                    plate1 = plates[i]
                    plate2 = plates[j]
                    
                    # Check if plates are adjacent
                    for vid in plate1.vertex_ids:
                        adj_vids = obj.get_adjacent_vertices(vid)
                        if any(adj_vid in plate2.vertex_ids for adj_vid in adj_vids):
                            adjacent_pairs.append((i, j))
                            break
            
            if adjacent_pairs:
                # Randomly select a pair to merge
                i, j = adjacent_pairs[np.random.randint(len(adjacent_pairs))]
                #plates[i].vertex_ids.extend(plates[j].vertex_ids)
                plates[i].vertex_ids = np.concatenate((plates[i].vertex_ids, plates[j].vertex_ids))
                plates.pop(j)
            else:
                break  # No adjacent plates left to merge
        
        for plate in plates:
            plate.get_boundary_vertices(self.sphere_mesh, self.sphere_mesh.vertices)
            print(f"assigning {plate.ID} to {plate.vertex_ids}")
            for vid in plate.vertex_ids:
                self.plate_ids[vid] = plate.ID

        return plates

    @time_function
    def calculate_plate_collisions(self):
        """Calculate collisions between plates and adjust elevations accordingly"""
        vertex_positions = self.sphere_mesh.vertices
        vertex_normals = vertex_positions / np.linalg.norm(vertex_positions, axis=1, keepdims=True)
        
        # First reset collision forces
        for plate in self.plates:
            plate.collision_force = np.zeros(3, dtype=np.float32)
        
        # Check all plate pairs for collisions
        for i, plate1 in enumerate(self.plates):
            for j, plate2 in enumerate(self.plates[i+1:], i+1):
                if len(plate1.vertex_ids) == 0 or len(plate2.vertex_ids) == 0:
                    continue
                    
                # Find boundary vertices that are close to each other
                bound1 = plate1.get_boundary_vertices(self.sphere_mesh, vertex_positions)
                bound2 = plate2.get_boundary_vertices(self.sphere_mesh, vertex_positions)
                
                if len(bound1) == 0 or len(bound2) == 0:
                    continue
                    
                # Get positions of boundary vertices
                pos1 = vertex_positions[bound1]
                pos2 = vertex_positions[bound2]
                
                # Calculate distances between boundaries
                distances = np.linalg.norm(pos1[:, np.newaxis] - pos2, axis=2)
                close_pairs = np.where(distances < 0.15)
                
                if len(close_pairs[0]) == 0:
                    continue
                    
                # Calculate collision force based on mass and velocity difference
                velocity_diff = plate1.linear_velocity - plate2.linear_velocity
                force_magnitude = np.linalg.norm(velocity_diff) * (plate1.mass + plate2.mass) * 0.1
                force_direction = np.mean(vertex_normals[bound1[close_pairs[0]]] - 
                                    vertex_normals[bound2[close_pairs[1]]], axis=0)
                force_direction = force_direction / np.linalg.norm(force_direction)
                
                collision_force = force_direction * force_magnitude
                
                # Apply forces to plates (opposite directions)
                plate1.collision_force += collision_force
                plate2.collision_force -= collision_force
                
                # Adjust elevations at collision points
                self._adjust_elevation_at_collision(
                    bound1[close_pairs[0]], 
                    bound2[close_pairs[1]], 
                    plate1, 
                    plate2,
                    force_magnitude
                )

    @time_function
    def _adjust_elevation_at_collision(self, verts1, verts2, plate1, plate2, force):
        """Adjust elevation for colliding vertices based on plate types and force"""
        if plate1.plate_type == 'oceanic' and plate2.plate_type == 'continental':
            subducting, overriding = plate1, plate2
            subduct_verts, override_verts = verts1, verts2
        elif plate1.plate_type == 'continental' and plate2.plate_type == 'oceanic':
            subducting, overriding = plate2, plate1
            subduct_verts, override_verts = verts2, verts1
        else:
            subducting, overriding = None, None
            
        if subducting:
            self.heightmap[subduct_verts] -= 0.1 * force
            self.heightmap[override_verts] += 0.15 * force
        else:
            uplift = 0.08 * force
            self.heightmap[verts1] += uplift
            self.heightmap[verts2] += uplift

    @time_function
    def update_elevations(self):
        """Update elevations based on plate properties and collisions"""
        _heightmap = _update_elevations_a(self.plates, self.heightmap)
        self.heightmap = _heightmap
        # Calculate plate collisions
        self.calculate_plate_collisions()

        _heightmap = _update_elevations_b(self.plates, _heightmap, self.max_height, self.min_height)

        # Normalize heightmap
        self.heightmap = _heightmap

    @time_function
    def update_vertices_based_on_heightmap(self):
        """Update the mesh vertices based on the current heightmap"""
        pass

    @time_function
    def update_colors(self):
        """Update mesh colors based on current colormap mode"""
        num_vertices = len(self.sphere_mesh.vertices)
        if self.colormap_mode == "water":
            water_normalized = (self.water_content - self.water_content.min()) / \
                            (self.water_content.max() - self.water_content.min() + 1e-6)
            blue = np.clip(water_normalized * 255, 0, 255)
            colors = np.stack([
                np.zeros_like(blue), np.zeros_like(blue), blue, np.full_like(blue, 255)
            ], axis=1)
            self.sphere_mesh.color = colors.astype(np.uint8)
            
        elif self.colormap_mode == "plates":
            full_colors = np.full((num_vertices, 4), 128, dtype=np.uint8)
            full_colors[:, 3] = 255
            

            for plate in self.plates:
                if plate.ID < len(self.plate_colors):
                    full_colors[plate.vertex_ids] = self.plate_colors[plate.ID]
            # assigned_mask = self.plate_ids != -1
            # if np.any(assigned_mask):
            #     assigned_ids = self.plate_ids[assigned_mask]
            #     if assigned_ids.max() < len(self.plate_colors):
            #         full_colors[assigned_mask] = self.plate_colors[assigned_ids]
            
            self.sphere_mesh.color = full_colors

        elif self.colormap_mode == "terrain":
            normalized_height = (self.heightmap - self.min_height) / \
                            (self.max_height - self.min_height + 1e-6)
            
            colors = np.zeros((num_vertices, 4), dtype=np.uint8)
            
            deep_ocean = np.array([5, 10, 80, 255], dtype=np.float32)
            shallow_water = np.array([50, 150, 255, 255], dtype=np.float32)
            beach = np.array([240, 240, 180, 255], dtype=np.float32)
            grass = np.array([50, 180, 50, 255], dtype=np.float32)
            forest = np.array([100, 150, 50, 255], dtype=np.float32)
            mountain = np.array([80, 100, 40, 255], dtype=np.float32)
            rock = np.array([120, 120, 120, 255], dtype=np.float32)
            snow = np.array([200, 200, 200, 255], dtype=np.float32)
            snow_cap = np.array([255, 255, 255, 255], dtype=np.float32)
            
            deep_ocean_mask = normalized_height < self.sea_level * 0.3
            colors[deep_ocean_mask] = deep_ocean.astype(np.uint8)
            
            shallow_mask = (normalized_height >= self.sea_level * 0.3) & (normalized_height < self.sea_level)
            shallow_factor = ((normalized_height[shallow_mask] - self.sea_level * 0.3) / (self.sea_level * 0.7))[:, None]
            colors[shallow_mask] = (deep_ocean * (1 - shallow_factor) + shallow_water * shallow_factor).astype(np.uint8)
            
            beach_mask = (normalized_height >= self.sea_level) & (normalized_height < self.sea_level + 0.05)
            colors[beach_mask] = beach.astype(np.uint8)
            
            lowland_mask = (normalized_height >= self.sea_level + 0.05) & (normalized_height < self.sea_level + 0.3)
            lowland_factor = ((normalized_height[lowland_mask] - (self.sea_level + 0.05)) / 0.25)
            lowland_factor = lowland_factor[:, None]
            colors[lowland_mask] = (grass * (1 - lowland_factor) + forest * lowland_factor).astype(np.uint8)
            
            highland_mask = (normalized_height >= self.sea_level + 0.3) & (normalized_height < self.sea_level + 0.6)
            highland_factor = ((normalized_height[highland_mask] - (self.sea_level + 0.3)) / 0.3)
            highland_factor = highland_factor[:, None]
            colors[highland_mask] = (forest * (1 - highland_factor) + mountain * highland_factor).astype(np.uint8)
            
            mountain_mask = (normalized_height >= self.sea_level + 0.6) & (normalized_height < self.sea_level + 0.9)
            mountain_factor = ((normalized_height[mountain_mask] - (self.sea_level + 0.6)) / 0.3)
            mountain_factor = mountain_factor[:, None]
            colors[mountain_mask] = (rock * (1 - mountain_factor) + snow * mountain_factor).astype(np.uint8)
            
            snow_mask = normalized_height >= self.sea_level + 0.9
            colors[snow_mask] = snow_cap.astype(np.uint8)
            
            self.sphere_mesh.color = colors
        elif self.colormap_mode == "temperature":
            self.sphere_mesh.color = np.full((num_vertices, 4), 128, dtype=np.uint8)
    
    @time_function
    def simulate_erosion(self, steps: int = 1):
        """Simple erosion simulation step"""
        for _ in range(steps):
            self.water_content += self.rainfall_rate
            water_diff = np.roll(self.water_content, 1, 0) - self.water_content
            self.water_content += water_diff * self.water_flow_min
            
            height_diff = np.roll(self.heightmap, 1, 0) - self.heightmap
            erosion_amount = self.water_content * (1 / (1 + np.exp(-height_diff * 10.0))) * 0.01
            self.heightmap -= erosion_amount
            self.water_content *= (1.0 - self.evaporation_rate)
            
            self.heightmap = np.clip(self.heightmap, self.min_height.item(), self.max_height.item())
        
        if steps > 0:
            self.update_vertices_based_on_heightmap()
            self.update_colors()

def render_world(resolution=128, min_height=6357, max_height=6378, plate_count=15, viewport = None, context = None):

    world = World(
        sphere_mesh=create_sphere_mesh(
            radius=(min_height+max_height) / 2, 
            #radius=1,
            segments=resolution, 
            rings=resolution, 
            deformable=False),
        plate_count=plate_count,
        min_height_value=min_height,
        max_height_value=max_height
    )
    
    eye = np.array([3.0, 2.0, 3.0], dtype=np.float32)
    lookat = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    camera_distance = np.linalg.norm(eye)
    azimuth = np.arctan2(eye[2], eye[0])
    elevation = np.arcsin(eye[1] / camera_distance)
    
    
    if context is None: dpgContext = dpg.create_context()
    else: dpgContext = context
    if viewport is None: viewport = dpg.create_viewport(title='Procedural World', width=800, height=600)
    

    def update_colormap_callback(sender, app_data):
        oldmap = world.colormap_mode
        world.colormap_mode = app_data
        if world.colormap_mode != oldmap:
            if app_data == "plates":
                dpg.delete_item("legend_window", children_only=True)
                with dpg.group(parent="legend_window"):
                    for i, plate in enumerate(world.plates):
                        if i < len(world.plate_colors):
                            color = world.plate_colors[i].tolist()
                            with dpg.group(horizontal=True):
                                with dpg.drawlist(width=20, height=20):
                                    dpg.draw_rectangle((0, 0), (20, 20), color=color, fill=color)
                                dpg.add_text(f"Plate {i} ({plate.plate_type[:4]})")
            
            elif app_data == "terrain":
                dpg.delete_item("legend_window", children_only=True)
                heightmap_np = world.heightmap
                percentiles = [0, 1, 10, 25, 50, 75, 90, 99, 100]
                percentile_values = np.percentile(heightmap_np, percentiles)
                normalized_heights = (percentile_values - world.min_height) / \
                                (world.max_height - world.min_height + 1e-6)
                color_stops = {
                    0: [0, 0, 0],
                    1: [5, 10, 80],
                    10: [50, 150, 255],
                    25: [240, 240, 180],
                    50: [50, 180, 50],
                    75: [100, 150, 50],
                    90: [80, 100, 40],
                    99: [200, 200, 200],
                    100: [255, 255, 255]
                }
                
                with dpg.group(parent="legend_window"):
                    dpg.add_text("Terrain Height Percentiles:")
                    for pct, height in zip(percentiles, percentile_values):
                        color = color_stops[pct]
                        with dpg.group(horizontal=True):
                            with dpg.drawlist(width=20, height=20):
                                dpg.draw_rectangle((0, 0), (20, 20), color=color, fill=color)
                            dpg.add_text(f"{pct}%: {height:.3f}")
            world.update_colors()

    def update_camera_position():
        nonlocal eye
        x = (camera_distance * np.cos(elevation) * np.cos(azimuth)) * min_height
        y = (camera_distance * np.sin(elevation)) * min_height
        z = (camera_distance * np.cos(elevation) * np.sin(azimuth)) * min_height
        eye = np.array([x, y, z], dtype=np.float32)
        
        dpg.set_value("camera_info", 
                     f"Camera: Dist={camera_distance:.2f}, Azim={np.degrees(azimuth):.1f}°, Elev={np.degrees(elevation):.1f}°")

    def camera_control_callback(sender, app_data):
        nonlocal camera_distance, azimuth, elevation
        if sender == "camera_distance":
            camera_distance = app_data
        elif sender == "camera_azimuth":
            azimuth = np.radians(app_data)
        elif sender == "camera_elevation":
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
        
        with dpg.drawlist(width=-1, height=-1, tag="draw_area"):
            pass
    
    with dpg.window(label='Camera Controls', pos=(0, 50)):
        with dpg.collapsing_header(label="Camera Controls", default_open=True):
            dpg.add_text("Camera Position:", tag="camera_info")
            dpg.add_slider_float(
                label="Distance", 
                tag="camera_distance",
                min_value=1.0, 
                max_value=15.0, 
                default_value=float(camera_distance),
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
                min_value=-90, 
                max_value=90, 
                default_value=math.degrees(elevation),
                callback=camera_control_callback
            )

    with dpg.handler_registry():
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=mouse_drag_callback)
        dpg.add_mouse_wheel_handler(callback=mouse_wheel_callback)

    update_colormap_callback(None, "plates")
    update_camera_position()
    dpg.setup_dearpygui(viewport=viewport)
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)

    while dpg.is_dearpygui_running():
        res=(int(dpg.get_item_width('primary') or 1), int(dpg.get_item_height('primary') or 1))
        dpg.set_item_width("draw_area", res[0])
        dpg.set_item_height("draw_area", res[1])
        #print_timing_stats()
        world.simulate_erosion(steps=0)

        dpg.delete_item("draw_area", children_only=True)
        
        screen_verts, visible_tris, depths, colors = project_2d(
            [world.sphere_mesh], eye, lookat, up, fovfl=60.0, res=res
            )
        #rasterize(screen_verts, visible_tris, depths, colors, res[0], res[1])
        
        if screen_verts and visible_tris[0]:
            points_np = screen_verts[0]
            
            colors_np = colors[0]
            for i, tri in enumerate(visible_tris[0]):
                p1 = points_np[tri[0]].tolist()
                p2 = points_np[tri[1]].tolist()
                p3 = points_np[tri[2]].tolist()
                #print(colors_np)
                color = colors_np[i].tolist()
                dpg.draw_triangle(p1, p2, p3, color=color, fill=color, 
                                parent="draw_area")
        dpg.render_dearpygui_frame()
    
    dpg.destroy_context()

if __name__ == "__main__":
    render_world()