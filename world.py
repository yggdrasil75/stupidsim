import copy
from dataclasses import dataclass, field
import random
import numba
import torch
from holder.mesh import mesh, project_2d
from globals import DEVICE
from shapes.sphere import create_sphere_mesh
import dearpygui.dearpygui as dpg
import numpy as np
from numba import njit, prange, int64, float32
from plate import Plate
import math
from util import time_function, print_timing_stats

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

def make_2D_array(lis):
    """Funciton to get 2D array from a list of lists
    """
    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = np.max(lengths)
    arr = np.zeros((n, max_len), np.float32)

    for i in range(n):
        arr[i, :lengths[i]] = lis[i]
    return arr, lengths

@dataclass
class World:
    torch.set_default_device(DEVICE)
    sphere_mesh: mesh = field(default_factory=lambda: create_sphere_mesh(segments=256, rings=256))
    sea_level: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0, dtype=torch.float32))
    #min_height: torch.Tensor = field(default_factory=lambda: torch.tensor(-1.0, dtype=torch.float32))
    #max_height: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.float32))
    plate_count: torch.Tensor = field(default_factory=lambda: torch.tensor(20, dtype=torch.int32))
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
    
    @property
    def min_height(self) -> torch.Tensor:
        """Returns the minimum height from the heightmap"""
        return torch.min(self.heightmap) if len(self.heightmap) > 0 else torch.tensor(-1.0)
    
    @property
    def max_height(self) -> torch.Tensor:
        """Returns the maximum height from the heightmap"""
        return torch.max(self.heightmap) if len(self.heightmap) > 0 else torch.tensor(1.0)


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
        # print("Growing plates...")
        # self._grow_plates()
        print("Validating plates...")
        self._validate_and_reindex_plates()
        print(f"Final plate count: {len(self.plates)}")
        for plate in self.plates:
            plate.calculate_mass()
        #     plate.update_continental_borders(self.sphere_mesh.vertices, 0.2)
        print("Calculating plate elevations...")
        self.update_elevations()
        for plate in self.plates:
            plate.update_continental_borders(self.sphere_mesh.vertices)

        unassigned = torch.sum(self.plate_ids == -1).item()
        if unassigned > 0:
            print(f"Warning: {unassigned} vertices remain unassigned after plate validation")
            # Assign remaining vertices to nearest plates
            self._assign_remaining_vertices()

        print("Initial world generation complete")
        # Generate colors for the final set of plates
        self.plate_colors = torch.randint(0, 256, (len(self.plates), 4), dtype=torch.uint8, device=DEVICE)
        self.plate_colors[:, 3] = 255 # Full alpha
        print(f'total vertices: {len(self.sphere_mesh.vertices)}')

    @time_function
    def _assign_remaining_vertices(self):
        """Assign any remaining unassigned vertices to the nearest plate"""
        unassigned = torch.where(self.plate_ids == -1)[0]
        if len(unassigned) == 0:
            return
            
        # For each unassigned vertex, find the nearest plate center
        vertex_positions = self.sphere_mesh.vertices
        plate_centers = torch.stack([vertex_positions[plate.vertex_ids[0]] 
                                    for plate in self.plates])
        
        for vert_idx in unassigned:
            pos = vertex_positions[vert_idx]
            distances = torch.norm(plate_centers - pos, dim=1)
            nearest_plate = torch.argmin(distances).item()
            self.plate_ids[vert_idx] = nearest_plate
            self.plates[nearest_plate].vertex_ids = torch.cat([
                self.plates[nearest_plate].vertex_ids,
                torch.tensor([vert_idx], dtype=torch.long, device=DEVICE)
            ])

    @time_function
    def _initialize_plates(self):
        """Selects initial plate centers and creates Plate objects."""
        num_vertices = len(self.sphere_mesh.vertices)
        center_indices = torch.randperm(num_vertices, device=DEVICE)[:self.plate_count]

        for i in range(self.plate_count):
            center_idx: int = int(center_indices[i].item())
            plate_verts = torch.tensor([center_idx], dtype=torch.long, device=DEVICE)

            if torch.rand(1) < 0.4:
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
        vertex_positions = self.sphere_mesh.vertices
        
        # Precompute plate centers and growth rates
        plate_centers = torch.stack([vertex_positions[plate.vertex_ids[0]] 
                                    for plate in self.plates])
        growth_rates = torch.tensor([plate.growth_rate for plate in self.plates], 
                                device=DEVICE)
        
        # Convert neighbor map to list of tensors for faster access
        neighbor_tensors = [torch.tensor(neighbors, device=DEVICE) 
                            for neighbors in vertex_neighbors.values()]
        
        pidsnp = self.plate_ids.cpu().numpy()
        neinp = [neighbor_tensor.cpu().numpy() for neighbor_tensor in neighbor_tensors]
        neinp, lens = make_2D_array(neinp)
        vertposnp = vertex_positions.cpu().numpy()
        plaecennp = plate_centers.cpu().numpy()
        grownp = growth_rates.cpu().numpy()
        
        iteration = 0
        while np.any(pidsnp == -1):
            unassigned_count = np.sum(pidsnp == -1)
            print(f"Iteration {iteration}: {unassigned_count} unassigned vertices remaining")
            pidsnp = _numba_grow_plates(pidsnp, neinp,
                                                vertposnp, plaecennp,
                                                grownp, num_vertices)
            iteration += 1
        self.plate_ids = torch.tensor(pidsnp)
        # Update Plate objects with their final vertex sets
        for i, plate in enumerate(self.plates):
            plate.vertex_ids = torch.where(self.plate_ids == i)[0]

    @time_function
    def is_plate_contained(self, inner_plate_id: int, outer_plate_id: int) -> bool:
        """
        Check if one plate is wholly contained within another plate's territory.
        Optimized version using numba.
        """
        if inner_plate_id == outer_plate_id:
            return False
            
        if inner_plate_id >= len(self.plates) or outer_plate_id >= len(self.plates):
            return False
            
        inner_plate = self.plates[inner_plate_id]
        outer_plate = self.plates[outer_plate_id]
        
        # Check if either plate is empty
        if len(inner_plate.vertex_ids) == 0 or len(outer_plate.vertex_ids) == 0:
            return False
        
        # Convert to numpy arrays and sets for numba
        inner_verts_np = inner_plate.vertex_ids.cpu().numpy()
        outer_verts_set = outer_plate.vertex_ids.cpu().numpy()
        
        # Convert neighbor map to numba-compatible format
        neighbor_list = []
        for nten in self.sphere_mesh._neighbor_map['vertex_to_vertices'].values():
            #nten, _ = make_2D_array(nten)
            neighbor_list.append(nten)
        
        neighbor_list, _ = make_2D_array(neighbor_list)
        #outer_verts_list, _ = make_2D_array(outer_verts_set)
        # print("numba containment check")
        # print(numba.typeof(inner_verts_np))
        # print(numba.typeof(outer_verts_set))
        # print(numba.typeof(neighbor_list))
        return _numba_check_containment(inner_verts_np, outer_verts_set, neighbor_list)

    @time_function
    def merge_plates(self, plate_a_id: int, plate_b_id: int):
        """
        Merge plate B into plate A, transferring all vertices and properties.
        
        Args:
            plate_a_id: ID of the plate that will absorb the other plate
            plate_b_id: ID of the plate that will be absorbed
        """
        if plate_a_id == plate_b_id:
            return  # Can't merge a plate with itself
        
        if plate_a_id >= len(self.plates) or plate_b_id >= len(self.plates):
            raise ValueError("Invalid plate ID")
        
        plate_a = self.plates[plate_a_id]
        plate_b = self.plates[plate_b_id]
        
        # Transfer all vertices from plate B to plate A
        plate_a.vertex_ids = torch.cat([plate_a.vertex_ids, plate_b.vertex_ids])
        self.plate_ids[plate_b.vertex_ids] = plate_a_id
        
        # Transfer continental centers if they exist
        if hasattr(plate_b, 'continental_centers'):
            if not hasattr(plate_a, 'continental_centers'):
                plate_a.continental_centers = []
            plate_a.continental_centers.extend(plate_b.continental_centers)
        
        # Update plate properties based on what's being merged
        if plate_b.plate_type == "continental":
            plate_a.plate_type = "continental"
            plate_a.growth_rate = torch.max(plate_a.growth_rate, plate_b.growth_rate)
        else:
            # If merging oceanic into continental, keep continental properties
            if plate_a.plate_type == "continental":
                pass  # Keep continental properties
            else:
                # Both are oceanic - average their properties
                plate_a.growth_rate = (plate_a.growth_rate + plate_b.growth_rate) / 2
        
        
        # Mark plate B as inactive (we'll clean it up later)
        plate_b.vertex_ids = torch.tensor([], dtype=torch.long, device=DEVICE)
        
        # After merging, we should reindex plates to remove the now-empty plate B
        # This could be done immediately or during the next validation step
        #self._validate_and_reindex_plates()

    @time_function
    def _validate_and_reindex_plates(self):
        """Checks for disjointed plates, splits or merges them, and re-indexes all plates."""
        vertex_neighbors = self.sphere_mesh._neighbor_map['vertex_to_vertices']
        plates_to_process = self.plates.copy()
        final_plates = []
        
        processed_plate_ids = set()

        # First handle any completely unassigned vertices
        unassigned_verts = torch.where(self.plate_ids == -1)[0].tolist()
        if unassigned_verts:
            print(f"Found {len(unassigned_verts)} unassigned vertices - assigning to random neighbors")
            
            changed = True
            while changed and unassigned_verts:
                changed = False
                remaining_unassigned = []
                
                for vert_idx in unassigned_verts:
                    # Find all neighboring plates
                    neighbor_plates = set()
                    for neighbor in vertex_neighbors.get(vert_idx, []):
                        plate_id = self.plate_ids[neighbor].item()
                        if plate_id != -1:
                            neighbor_plates.add(plate_id)
                    
                    if neighbor_plates:
                        # Assign to a random neighboring plate
                        chosen_plate = random.choice(list(neighbor_plates))
                        self.plate_ids[vert_idx] = chosen_plate
                        self.plates[chosen_plate].vertex_ids = torch.cat([
                            self.plates[chosen_plate].vertex_ids,
                            torch.tensor([vert_idx], dtype=torch.long, device=DEVICE)
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
                    new_fragment_plate = Plate.create_oceanic_plate(fragment_verts, vertex_ids=fragment_verts)
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
        
        # One final check for any remaining unassigned vertices
        unassigned_verts = torch.where(self.plate_ids == -1)[0]
        if len(unassigned_verts) > 0:
            print(f"Warning: {len(unassigned_verts)} vertices remain unassigned after plate validation")
            
    @time_function
    def calculate_plate_collisions(self):
        """Calculate collisions between plates and adjust elevations accordingly"""
        vertex_positions = self.sphere_mesh.vertices
        vertex_normals = torch.nn.functional.normalize(vertex_positions, dim=1)
        
        # First reset collision forces
        for plate in self.plates:
            plate.collision_force = torch.zeros(3, dtype=torch.float32)
        
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
                distances = torch.cdist(pos1, pos2)
                close_pairs = torch.where(distances < 0.15)  # Collision threshold
                
                if len(close_pairs[0]) == 0:
                    continue
                    
                # Calculate collision force based on mass and velocity difference
                velocity_diff = plate1.linear_velocity - plate2.linear_velocity
                force_magnitude = torch.norm(velocity_diff) * (plate1.mass + plate2.mass) * 0.1
                force_direction = torch.mean(vertex_normals[bound1[close_pairs[0]]] - 
                                        vertex_normals[bound2[close_pairs[1]]], dim=0)
                force_direction = torch.nn.functional.normalize(force_direction, dim=0)
                
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
        # Determine which plate is subducting (oceanic plates subduct under continental)
        if plate1.plate_type == 'oceanic' and plate2.plate_type == 'continental':
            subducting, overriding = plate1, plate2
            subduct_verts, override_verts = verts1, verts2
        elif plate1.plate_type == 'continental' and plate2.plate_type == 'oceanic':
            subducting, overriding = plate2, plate1
            subduct_verts, override_verts = verts2, verts1
        else:
            # Continental-continental collision or oceanic-oceanic
            subducting, overriding = None, None
            
        if subducting:
            # Oceanic plate subducts - create trench and mountains
            self.heightmap[subduct_verts] -= 0.1 * force  # Trench
            self.heightmap[override_verts] += 0.15 * force  # Mountain range
        else:
            # Continental collision or oceanic-oceanic - both get uplifted
            uplift = 0.08 * force
            self.heightmap[verts1] += uplift
            self.heightmap[verts2] += uplift

    @time_function
    def update_elevations(self):
        """Update elevations based on plate properties and collisions"""
        vertex_positions = self.sphere_mesh.vertices
        
        # Reset heightmap to base elevations
        for plate in self.plates:
            if len(plate.vertex_ids) > 0:
                self.heightmap[plate.vertex_ids] = plate.base_elevation
                
        # Apply continental effects
        for plate in self.plates:
            if plate.plate_type == 'continental' and len(plate.continental_centers) > 0:
                continental_verts = plate.get_continental_vertices()
                if len(continental_verts) > 0:
                    # Continental centers are higher
                    self.heightmap[plate.continental_centers] += 200.0
                    # Continental borders have moderate elevation
                    if len(plate.continental_border_vertices) > 0:
                        self.heightmap[plate.continental_border_vertices] += 100.0
        
        # Calculate plate collisions
        self.calculate_plate_collisions()
        
        # Apply plate movement effects
        for plate in self.plates:
            if len(plate.vertex_ids) > 0:
                # Add some noise based on movement
                movement_factor = torch.norm(plate.linear_velocity) * 0.1
                noise = (torch.rand(len(plate.vertex_ids)) * movement_factor)
                self.heightmap[plate.vertex_ids] += noise
            
                # Apply collision forces
                if torch.norm(plate.collision_force) > 0:
                    force_factor = torch.norm(plate.collision_force) * 0.05
                    self.heightmap[plate.vertex_ids] += torch.rand(len(plate.vertex_ids)) * force_factor
    
        # Normalize heightmap
        self.heightmap = (self.heightmap - self.heightmap.min()) / \
                        (self.heightmap.max() - self.heightmap.min() + 1e-6) * \
                        (self.max_height - self.min_height) + self.min_height

    @time_function
    def update_vertices_based_on_heightmap(self):
        """Update the mesh vertices based on the current heightmap"""
        pass
        ### THIS IS BROKEN! UPDATE HEIGHMAP AS COLOR INSTEAD OF USING THIS
        #vertices = self.sphere_mesh.vertices.clone()
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
            # Normalize heightmap to 0-1 range
            normalized_height = (self.heightmap - self.min_height) / \
                            (self.max_height - self.min_height + 1e-6)
            
            # Create terrain color gradient
            colors = torch.zeros((num_vertices, 4), dtype=torch.uint8, device=DEVICE)
            
            # Convert color values to float for lerp
            deep_ocean = torch.tensor([5, 10, 80, 255], dtype=torch.float32, device=DEVICE)
            shallow_water = torch.tensor([50, 150, 255, 255], dtype=torch.float32, device=DEVICE)
            beach = torch.tensor([240, 240, 180, 255], dtype=torch.float32, device=DEVICE)
            grass = torch.tensor([50, 180, 50, 255], dtype=torch.float32, device=DEVICE)
            forest = torch.tensor([100, 150, 50, 255], dtype=torch.float32, device=DEVICE)
            mountain = torch.tensor([80, 100, 40, 255], dtype=torch.float32, device=DEVICE)
            rock = torch.tensor([120, 120, 120, 255], dtype=torch.float32, device=DEVICE)
            snow = torch.tensor([200, 200, 200, 255], dtype=torch.float32, device=DEVICE)
            snow_cap = torch.tensor([255, 255, 255, 255], dtype=torch.float32, device=DEVICE)
            
            # Deep ocean (below sea level)
            deep_ocean_mask = normalized_height < self.sea_level * 0.3
            colors[deep_ocean_mask] = deep_ocean.to(torch.uint8)
            
            # Shallow water
            shallow_mask = (normalized_height >= self.sea_level * 0.3) & (normalized_height < self.sea_level)
            shallow_factor = ((normalized_height[shallow_mask] - self.sea_level * 0.3) / (self.sea_level * 0.7)).unsqueeze(1)
            colors[shallow_mask] = torch.lerp(
                deep_ocean, 
                shallow_water, 
                shallow_factor
            ).to(torch.uint8)
            
            # Beach/sand (just above sea level)
            beach_mask = (normalized_height >= self.sea_level) & (normalized_height < self.sea_level + 0.05)
            colors[beach_mask] = beach.to(torch.uint8)
            
            # Lowlands/grass
            lowland_mask = (normalized_height >= self.sea_level + 0.05) & (normalized_height < self.sea_level + 0.3)
            lowland_factor = ((normalized_height[lowland_mask] - (self.sea_level + 0.05)) / 0.25)
            lowland_factor = lowland_factor.unsqueeze(1)
            colors[lowland_mask] = torch.lerp(
                grass,
                forest,
                lowland_factor
            ).to(torch.uint8)
            
            # Highlands/forest
            highland_mask = (normalized_height >= self.sea_level + 0.3) & (normalized_height < self.sea_level + 0.6)
            highland_factor = ((normalized_height[highland_mask] - (self.sea_level + 0.3)) / 0.3)
            highland_factor = highland_factor.unsqueeze(1)
            colors[highland_mask] = torch.lerp(
                forest,
                mountain,
                highland_factor
            ).to(torch.uint8)
            
            # Mountains
            mountain_mask = (normalized_height >= self.sea_level + 0.6) & (normalized_height < self.sea_level + 0.9)
            mountain_factor = ((normalized_height[mountain_mask] - (self.sea_level + 0.6)) / 0.3)
            mountain_factor = mountain_factor.unsqueeze(1)
            colors[mountain_mask] = torch.lerp(
                rock,
                snow,
                mountain_factor
            ).to(torch.uint8)
            
            # Snow caps
            snow_mask = normalized_height >= self.sea_level + 0.9
            colors[snow_mask] = snow_cap.to(torch.uint8)
            
            self.sphere_mesh.color = colors
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
            #dpg.configure_item("legend_window", show=(app_data == "plates"))
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
            
            elif app_data == "terrain":
                dpg.delete_item("legend_window", children_only=True)
                # Calculate height percentiles
                heightmap_np = world.heightmap.cpu().numpy()
                percentiles = [0, 1, 10, 25, 50, 75, 90, 99, 100]
                percentile_values = np.percentile(heightmap_np, percentiles)
                # Get terrain colors for each percentile
                normalized_heights = (percentile_values - world.min_height.cpu().numpy()) / \
                                (world.max_height.cpu().numpy() - world.min_height.cpu().numpy() + 1e-6)
                # Define terrain color stops (same as in terrain colormap)
                color_stops = {
                    0: [0, 0, 0],       # Deep ocean
                    1: [5, 10, 80],       # Deep ocean
                    10: [50, 150, 255],    # Shallow water
                    25: [240, 240, 180],   # Beach
                    50: [50, 180, 50],     # Grass
                    75: [100, 150, 50],    # Forest
                    90: [80, 100, 40],     # Mountain
                    #99: [120, 120, 120],  # Rock
                    99: [200, 200, 200],  # Snow
                    100: [255, 255, 255]
                    #(1.0, [255, 255, 255])   # Snow cap
                }
                
                # Create legend
                with dpg.group(parent="legend_window"):
                    dpg.add_text("Terrain Height Percentiles:")
                    for pct, height in zip(percentiles, percentile_values):
                        # Find color for this height
                        color = color_stops[pct]
                        # for i in range(1, len(color_stops)):
                        #     if normalized_heights[i-1] <= color_stops[i][0]:
                        #         # t = (normalized_heights[i-1] - color_stops[i-1][0]) / \
                        #         #     (color_stops[i][0] - color_stops[i-1][0])
                        #         # color = [
                        #         #     int(color_stops[i-1][1][0] + t * (color_stops[i][1][0] - color_stops[i-1][1][0])),
                        #         #     int(color_stops[i-1][1][1] + t * (color_stops[i][1][1] - color_stops[i-1][1][1])),
                        #         #     int(color_stops[i-1][1][2] + t * (color_stops[i][1][2] - color_stops[i-1][1][2]))
                        #         # ]
                        #         break
                        
                        with dpg.group(horizontal=True):
                            with dpg.drawlist(width=20, height=20):
                                dpg.draw_rectangle((0, 0), (20, 20), color=color, fill=color)
                            dpg.add_text(f"{pct}%: {height:.3f}")
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
                min_value=-90, 
                max_value=90, 
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
        
        screen_verts, visible_tris, depths = project_2d(
            [world.sphere_mesh], eye, lookat, up, fovfl=60.0, res=(800, 600)
            )
        
        if screen_verts and visible_tris[0]:
            points_np = screen_verts[0].cpu().numpy()
            colors_np = world.sphere_mesh.color.cpu().numpy()
            depths_np = depths[0].cpu().numpy()
            for i, tri in enumerate(visible_tris[0]):
                #depth = depths_np[i]
                p1 = points_np[tri[0]].tolist()
                p2 = points_np[tri[1]].tolist()
                p3 = points_np[tri[2]].tolist()
                # Use color from the first vertex of the triangle
                color = colors_np[tri[0]].tolist()
                #print(f'depth issue: {type(depth)}, {depth}')
                dpg.draw_triangle(p1, p2, p3, color=color, fill=color, 
                                #depth=depth,
                                parent="draw_area")
    
        dpg.render_dearpygui_frame()
    
    dpg.destroy_context()

if __name__ == "__main__":
    render_world()