import copy
from dataclasses import dataclass, field
import random
import numba
import numpy as np
from holder.mesh import mesh, project_2d
from globals import DEVICE
from shapes.sphere import create_sphere_mesh
import dearpygui.dearpygui as dpg
from numba import njit, prange, int64, float32
from plate import Plate
import math
from util import norm, time_function, print_timing_stats, make_2D_array

@njit((int64[:], float32[:,:], float32[:,:], float32[:,:], float32[:], int64), cache=True)
def _numba_grow_plates(plate_ids, neighbor_ndarrays, vertex_positions,
                       plate_centers, growth_rates, num_vertices):    
    assigned_mask = plate_ids != -1
    assigned_indices = np.where(assigned_mask)[0]

    if len(assigned_indices) == 0:
        return plate_ids
        
    # Find frontier vertices in batches
    frontier_list = []
    batch_size = 1024  # Adjust based on memory constraints
    
    # First pass to calculate total size needed
    total_neighbors = 0
    for i in range(0, len(assigned_indices), batch_size):
        batch_indices = assigned_indices[i:i+batch_size]
        for idx in batch_indices:
            total_neighbors += len(neighbor_ndarrays[idx])
    
    # Pre-allocate array for all neighbors
    all_neighbors = np.empty(total_neighbors, dtype=np.int64)
    pos = 0
    
    for i in range(0, len(assigned_indices), batch_size):
        batch_indices = assigned_indices[i:i+batch_size]
        
        # Fill pre-allocated array with neighbors
        for idx in batch_indices:
            neighbors = neighbor_ndarrays[idx]
            all_neighbors[pos:pos+len(neighbors)] = neighbors
            pos += len(neighbors)
        
        # Find unassigned neighbors in this segment
        segment = all_neighbors[:pos]  # Only the filled portion
        unassigned = segment[plate_ids[segment] == -1]
        if len(unassigned) > 0:
            frontier_list.append(unassigned)
        
    if len(frontier_list) == 0:
        return plate_ids
        
    # Second pass to concatenate frontier vertices (now with known sizes)
    frontier_total = 0
    for arr in frontier_list:
        frontier_total += len(arr)
    
    frontier_verts = np.empty(frontier_total, dtype=np.int64)
    pos = 0
    for arr in frontier_list:
        frontier_verts[pos:pos+len(arr)] = arr
        pos += len(arr)
    
    frontier_verts = np.unique(frontier_verts)

    # For each frontier vertex, find all plates that could claim it
    frontier_pos = vertex_positions[frontier_verts]
    dists = np.empty((len(frontier_verts), len(plate_centers)), dtype=np.float32)
    for i in range(len(frontier_verts)):
        for j in range(len(plate_centers)):
            # Add some noise to the distance calculation
            noise = 1.0 + (np.random.rand() * 0.2 - 0.1)  # ±10% noise
            dists[i,j] = np.sqrt(np.sum((frontier_pos[i] - plate_centers[j])**2)) * noise

    # Modified scoring - distance is less important, randomness more important
    rand_factors = 0.8 + (np.random.rand(len(frontier_verts)) * 0.4)  # 0.8-1.2 range
    distance_weight = 0.3  # Reduced from implicit 1.0 in original
    scores = (growth_rates * rand_factors.reshape(-1, 1)) / (dists**distance_weight + 1e-6)

    # Get all possible claims (plate, vertex pairs)
    potential_plates = np.argmax(scores, axis=1)
    potential_claims = np.empty((len(frontier_verts), 2), dtype=np.int64)
    for i in range(len(frontier_verts)):
        potential_claims[i,0] = frontier_verts[i]
        potential_claims[i,1] = potential_plates[i]

    # Sort claims by score to maintain realistic growth priority
    max_scores = np.empty(len(frontier_verts), dtype=np.float32)
    for i in range(len(frontier_verts)):
        max_scores[i] = scores[i, potential_plates[i]]
    sorted_indices = np.argsort(max_scores)[::-1]
    sorted_claims = potential_claims[sorted_indices]

    # Process claims in order, tracking which vertices get claimed
    claimed = np.zeros(num_vertices, dtype=np.bool_)
    for i in range(len(sorted_claims)):
        vert_idx = sorted_claims[i,0]
        plate_id = sorted_claims[i,1]
        if not claimed[vert_idx]:
            plate_ids[vert_idx] = plate_id
            claimed[vert_idx] = True

    return plate_ids

@njit((int64[:], int64[:], float32[:,:]), cache=True)
def _numba_check_containment(inner_verts, outer_verts_set, 
                           neighbor_map) -> bool:
    """
    Numba-accelerated helper function to check plate containment.
    """    
    for vert in inner_verts:
        for neighbor in neighbor_map[vert]:
            if (neighbor not in outer_verts_set) and (neighbor not in inner_verts):
                return False
    return True

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
    plate_count: np.ndarray = field(default_factory=lambda: np.array(20, dtype=np.int32))
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
        self._initialize_plates()
        print("Validating plates...")
        self._validate_and_reindex_plates()
        print(f"Final plate count: {len(self.plates)}")
        for plate in self.plates:
            plate.calculate_mass()
        print("Calculating plate elevations...")
        self.update_elevations()
        for plate in self.plates:
            plate.update_continental_borders(self.sphere_mesh.vertices)

        unassigned = np.sum(self.plate_ids == -1)
        if unassigned > 0:
            print(f"Warning: {unassigned} vertices remain unassigned after plate validation")
            self._assign_remaining_vertices()

        print("Initial world generation complete")
        self.plate_colors = np.random.randint(0, 256, (len(self.plates), 4), dtype=np.uint8)
        self.plate_colors[:, 3] = 255
        print(f'total vertices: {len(self.sphere_mesh.vertices)}')

    @time_function
    def _assign_remaining_vertices(self):
        """Assign any remaining unassigned vertices to the nearest plate"""
        unassigned = np.where(self.plate_ids == -1)[0]
        if len(unassigned) == 0:
            return
            
        vertex_positions = self.sphere_mesh.vertices
        plate_centers = np.stack([vertex_positions[plate.vertex_ids[0]] 
                                for plate in self.plates])
        
        for vert_idx in unassigned:
            pos = vertex_positions[vert_idx]
            distances = np.linalg.norm(plate_centers - pos, axis=1)
            nearest_plate = np.argmin(distances)
            self.plate_ids[vert_idx] = nearest_plate
            self.plates[nearest_plate].vertex_ids = np.concatenate([self.plates[nearest_plate].vertex_ids, np.array([vert_idx], dtype=np.int64)])

    @time_function
    def _initialize_plates(self):
        """Selects initial plate centers and creates Plate objects."""
        num_vertices = len(self.sphere_mesh.vertices)
        center_indices = np.random.permutation(num_vertices)[:self.plate_count]

        for i in range(self.plate_count):
            center_idx = int(center_indices[i])
            plate_verts = np.array([center_idx], dtype=np.int64)

            if np.random.rand() < 0.4:
                plate = Plate.create_continental_plate(ID=i, vertex_ids=plate_verts)
                plate.add_continental_center(center_idx)
            else:
                plate = Plate.create_oceanic_plate(ID=i, vertex_ids=plate_verts)
            self.plates.append(plate)
            self.plate_ids[center_idx] = i
        print("Growing plates...")
        self._grow_plates()

        for i in range(self.plate_count):
            for j in range(self.plate_count + i):
                if self.is_plate_contained(i, j):
                    self.merge_plates(i, j)
        
    @time_function
    def _grow_plates(self):
        """Optimized but still realistic plate growth using batched frontier processing."""
        vertex_neighbors = self.sphere_mesh._neighbor_map['vertex_to_vertices']
        num_vertices = len(self.sphere_mesh.vertices)
        vertex_positions = self.sphere_mesh.vertices.astype(np.float32)
        
        # Precompute plate centers and growth rates
        plate_centers = np.stack([vertex_positions[plate.vertex_ids[0]] 
                                for plate in self.plates], dtype=np.float32)
        growth_rates = np.array([plate.growth_rate for plate in self.plates], dtype=np.float32)
        
        # Convert neighbor map to list of arrays for faster access
        neighbor_arrays = [np.array(neighbors, dtype=np.int64) 
                         for neighbors in vertex_neighbors.values()]
        
        iteration = 0
        while np.any(self.plate_ids == -1):
            unassigned_count = np.sum(self.plate_ids == -1)
            print(f"Iteration {iteration}: {unassigned_count} unassigned vertices remaining")
            self.plate_ids = _numba_grow_plates(
                self.plate_ids, 
#                neighbor_arrays,
                make_2D_array(neighbor_arrays)[0],

                vertex_positions, 
                plate_centers,
                growth_rates, 
                num_vertices
            )
            iteration += 1
        
        # Update Plate objects with their final vertex sets
        for i, plate in enumerate(self.plates):
            plate.vertex_ids = np.where(self.plate_ids == i)[0]

    @time_function
    def is_plate_contained(self, inner_plate_id: int, outer_plate_id: int) -> bool:
        """
        Check if one plate is wholly contained within another plate's territory.
        """
        if inner_plate_id == outer_plate_id:
            return False
            
        if inner_plate_id >= len(self.plates) or outer_plate_id >= len(self.plates):
            return False
            
        inner_plate = self.plates[inner_plate_id]
        outer_plate = self.plates[outer_plate_id]
        
        if len(inner_plate.vertex_ids) == 0 or len(outer_plate.vertex_ids) == 0:
            return False
        
        inner_verts_np = inner_plate.vertex_ids
        outer_verts_set = outer_plate.vertex_ids
        
        neighbor_list = []
        for neighbors in self.sphere_mesh._neighbor_map['vertex_to_vertices'].values():
            neighbor_list.append(neighbors)
        
        neighbor_list, _ = make_2D_array(neighbor_list)
        return _numba_check_containment(inner_verts_np, outer_verts_set, neighbor_list)

    @time_function
    def merge_plates(self, plate_a_id: int, plate_b_id: int):
        """
        Merge plate B into plate A, transferring all vertices and properties.
        """
        if plate_a_id == plate_b_id:
            return
        
        if plate_a_id >= len(self.plates) or plate_b_id >= len(self.plates):
            raise ValueError("Invalid plate ID")
        
        plate_a = self.plates[plate_a_id]
        plate_b = self.plates[plate_b_id]
        
        plate_a.vertex_ids = np.concatenate([plate_a.vertex_ids, plate_b.vertex_ids])
        self.plate_ids[plate_b.vertex_ids] = plate_a_id
        
        if hasattr(plate_b, 'continental_centers'):
            if not hasattr(plate_a, 'continental_centers'):
                plate_a.continental_centers = []
            plate_a.continental_centers.extend(plate_b.continental_centers)
        
        if plate_b.plate_type == "continental":
            plate_a.plate_type = "continental"
            plate_a.growth_rate = np.maximum(plate_a.growth_rate, plate_b.growth_rate)
        else:
            if plate_a.plate_type == "continental":
                pass
            else:
                plate_a.growth_rate = (plate_a.growth_rate + plate_b.growth_rate) / 2
        
        plate_b.vertex_ids = np.array([], dtype=np.int64)

    @time_function
    def _validate_and_reindex_plates(self):
        """Checks for disjointed plates, splits or merges them, and re-indexes all plates."""
        vertex_neighbors = self.sphere_mesh._neighbor_map['vertex_to_vertices']
        plates_to_process = self.plates.copy()
        final_plates = []
        
        processed_plate_ids = set()

        # First handle any completely unassigned vertices
        unassigned_verts = np.where(self.plate_ids == -1)[0].tolist()
        if unassigned_verts:
            print(f"Found {len(unassigned_verts)} unassigned vertices - assigning to random neighbors")
            
            changed = True
            while changed and unassigned_verts:
                changed = False
                remaining_unassigned = []
                
                for vert_idx in unassigned_verts:
                    neighbor_plates = set()
                    for neighbor in vertex_neighbors.get(vert_idx, []):
                        plate_id = self.plate_ids[neighbor]
                        if plate_id != -1:
                            neighbor_plates.add(plate_id)
                    
                    if neighbor_plates:
                        chosen_plate = random.choice(list(neighbor_plates))
                        self.plate_ids[vert_idx] = chosen_plate
                        self.plates[chosen_plate].vertex_ids = np.concatenate([
                            self.plates[chosen_plate].vertex_ids,
                            np.array([vert_idx], dtype=np.int64)
                        ])
                        changed = True
                    else:
                        remaining_unassigned.append(vert_idx)
                
                unassigned_verts = remaining_unassigned
                print(f"Assigned some vertices, {len(unassigned_verts)} remaining unassigned")
            
            if unassigned_verts:
                print(f"Warning: {len(unassigned_verts)} vertices could not be assigned (no plate neighbors)")

        while plates_to_process:
            plate = plates_to_process.pop(0)
            if len(plate.vertex_ids) == 0: continue
            plate_id = self.plate_ids[plate.vertex_ids[0]]

            if plate_id in processed_plate_ids:
                continue
            processed_plate_ids.add(plate_id)
            
            all_plate_verts = set(np.where(self.plate_ids == plate_id)[0].tolist())
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
            main_component_verts = np.array(components[0], dtype=np.int64)
            new_plate = plate
            new_plate.vertex_ids = main_component_verts
            final_plates.append(new_plate)
            
            # Handle other, smaller "fragment" components
            for fragment_verts_list in components[1:]:
                fragment_verts = np.array(fragment_verts_list, dtype=np.int64)
                
                if len(fragment_verts) > 50:
                    new_fragment_plate = Plate.create_oceanic_plate(fragment_verts, vertex_ids=fragment_verts)
                    final_plates.append(new_fragment_plate)
                else:
                    border_counts = {}
                    for vert_idx in fragment_verts_list:
                        for neighbor in vertex_neighbors.get(vert_idx, []):
                            neighbor_plate_id = self.plate_ids[neighbor]
                            if self.plate_ids[vert_idx] != neighbor_plate_id:
                                border_counts[neighbor_plate_id] = border_counts.get(neighbor_plate_id, 0) + 1
                    
                    if border_counts:
                        best_neighbor_id = max(border_counts, key=border_counts.get)
                        self.plate_ids[fragment_verts] = best_neighbor_id

        # Final re-indexing step
        self.plates = final_plates
        new_plate_ids_array = np.full_like(self.plate_ids, -1)
        for new_id, plate in enumerate(self.plates):
            new_plate_ids_array[plate.vertex_ids] = new_id
        
        self.plate_ids = new_plate_ids_array
        self.plate_count = np.array(len(self.plates))
        
        # One final check for any remaining unassigned vertices
        unassigned_verts = np.where(self.plate_ids == -1)[0]
        if len(unassigned_verts) > 0:
            print(f"Warning: {len(unassigned_verts)} vertices remain unassigned after plate validation")
            
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
                bound1 = plate1.get_boundary_vertices(vertex_positions)
                bound2 = plate2.get_boundary_vertices(vertex_positions)
                
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
            
            assigned_mask = self.plate_ids != -1
            if np.any(assigned_mask):
                assigned_ids = self.plate_ids[assigned_mask]
                if assigned_ids.max() < len(self.plate_colors):
                    full_colors[assigned_mask] = self.plate_colors[assigned_ids]
            
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

def render_world(resolution=64, min_height=6357, max_height=6378, plate_count=20):
    world = World(
        sphere_mesh=create_sphere_mesh(radius=(min_height+max_height) / 2, segments=resolution, rings=resolution, deformable=False),
        plate_count=np.array(plate_count, dtype=np.int32),
        min_height_value=min_height,
        max_height_value=max_height
    )
    
    eye = np.array([3.0, 2.0, 3.0], dtype=np.float32)
    lookat = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    camera_distance = np.linalg.norm(eye)
    azimuth = np.arctan2(eye[2], eye[0])
    elevation = np.arcsin(eye[1] / camera_distance)
    
    dpg.create_context()
    dpg.create_viewport(title='Procedural World', width=1000, height=700)

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
        x = camera_distance * np.cos(elevation) * np.cos(azimuth)
        y = camera_distance * np.sin(elevation)
        z = camera_distance * np.cos(elevation) * np.sin(azimuth)
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
        
        with dpg.drawlist(width=800, height=600, tag="draw_area"):
            pass
    
    with dpg.window(label='Camera Controls'):
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
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    
    while dpg.is_dearpygui_running():
        print_timing_stats()
        world.simulate_erosion(steps=0)

        dpg.delete_item("draw_area", children_only=True)
        
        screen_verts, visible_tris, depths = project_2d(
            [world.sphere_mesh], eye, lookat, up, fovfl=60.0, res=(800, 600)
            )
        
        if screen_verts and visible_tris[0]:
            points_np = screen_verts[0]
            colors_np = world.sphere_mesh.color
            #depths_np = depths[0]
            for tri in visible_tris[0]:
                p1 = points_np[tri[0]].tolist()
                p2 = points_np[tri[1]].tolist()
                p3 = points_np[tri[2]].tolist()
                color = colors_np[tri[0]].tolist()
                dpg.draw_triangle(p1, p2, p3, color=color, fill=color, 
                                parent="draw_area")
    
        dpg.render_dearpygui_frame()
    
    dpg.destroy_context()

if __name__ == "__main__":
    render_world()