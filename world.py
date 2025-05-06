
from collections import deque
import ctypes
from datetime import time
import multiprocessing
import queue
import random
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from holder.face import Face
from holder.globals import CONTINENTAL_CRUST_THICKNESS, OCEANIC_CRUST_THICKNESS, PLATE_TYPE_CONTINENTAL, PLATE_TYPE_OCEANIC
from holder.vertex import Vertex
from plate import Plate
from util import batch_calculate_areas, plate_grow_worker_process, time_function


class World:
    @time_function
    def __init__(self):
        self.vertices: list[Vertex] = []
        self.faces: list[Face] = []
        self.plates: dict[int, Plate] = {}
        self._neighbor_map_initialized: bool = False
        self.sea_level: np.float64 = 0.0
        self.minheight: np.float64 = -15000
        self.maxheight: np.float64  = 15000
        self.platecount: np.int8 = 15
        self.radius: np.float64 = 1.0 # in earth radii
        self.rainfall_rate: float = 0.002  # 2mm per iteration
        self.evaporation_rate: float = 0.001  # 1mm per iteration
        self.max_water_flow: float = 0.5  # Max 50cm flow per iteration
        self.min_river_flow: float = 1.0  # Minimum flow to be considered a river
        self.min_lake_volume: float = 3.0  # Minimum water volume to form a lake
        self.vertex_areas = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @time_function
    def add_vertex(self, vertex):
        self.vertices.append(vertex)
        return len(self.vertices) - 1

    @time_function
    def add_face(self, face):
        self.faces.append(face)

    @time_function
    def create_world(self, radius=1.0, plates = 15, min_height = -15000, max_height = 15000):
        self.platecount = plates
        self.elevationMin = min_height
        self.elevationMax = max_height
        self._create_world(radius=1.0)

    @time_function
    def _create_world(self, radius = 1.0):
        raise NotImplementedError("Subclasses must implement _create_world()")
    
    @time_function
    def _subdivide_selected(self, faces_to_subdivide, radius=1.0, level=3):
        raise NotImplementedError("Subclasses must implement _subdivide_selected(faces_to_subdivide, radius, level)")

    @time_function
    def subdivide(self, radius=1.0, levels=3):
        self.subdivisions = levels
        for level in range(levels):
            print(f"Starting Subdivision level {level+1}...")
            if level == 2:
                self._create_plates()
                self._assign_tectonic_plates()
            self._subdivide()
            print(f"Subdivision level {level+1} complete. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")
            if level >= 2:
                self._neighbor_map_initialized = False
                self._fix_non_contiguous_vertices()
        #self.fixPlates()

    @time_function
    def _subdivide(self):
        raise NotImplementedError("Subclasses must implement _subdivide()")

    #### plates and elevation

    @time_function
    def fixPlates(self):
        self._neighbor_map_initialized = False
        for plate in self.plates.values():
            plate.resetVertices()
        
        # First pass: assign plate IDs from immediate neighbors
        changed = True
        while changed:  # Keep iterating until no more changes occur
            changed = False
            for vidx, vertex in enumerate(self.vertices):
                if vertex.plate_id == -1:
                    for vidx2 in self._find_vertex_neighbors(vidx):
                        neighbor_pid = self.vertices[vidx2].plate_id
                        if neighbor_pid != -1:
                            vertex.plate_id = neighbor_pid
                            changed = True
                            #print(f'Assigned plate {neighbor_pid} to vertex {vidx} from neighbor {vidx2}')
                            break  # Assign from first valid neighbor found
        
        # Second pass: handle remaining unassigned vertices by expanding search radius
        remaining_unassigned = [vidx for vidx, v in enumerate(self.vertices) if v.plate_id == -1]
        search_radius = 2  # Start by looking 2 steps away
        
        while remaining_unassigned and search_radius < 5:  # Limit how far we search
            new_unassigned = []
            for vidx in remaining_unassigned:
                # Find all vertices within search_radius steps
                nearby = self._find_vertices_within_radius(vidx, search_radius)
                for n_vidx in nearby:
                    if self.vertices[n_vidx].plate_id != -1:
                        self.vertices[vidx].plate_id = self.vertices[n_vidx].plate_id
                        print(f'Assigned plate {self.vertices[n_vidx].plate_id} to vertex {vidx} from vertex {n_vidx} (radius {search_radius})')
                        break
                else:
                    new_unassigned.append(vidx)
            
            remaining_unassigned = new_unassigned
            search_radius += 1
        
        # Final assignment for any remaining vertices (assign to largest plate or create new plate)
        if remaining_unassigned:
            print(f'Warning: {len(remaining_unassigned)} vertices still unassigned after expansion')
            # Option 1: Assign to largest existing plate
            largest_plate = max(self.plates.values(), key=lambda p: len(p.vertices))
            for vidx in remaining_unassigned:
                self.vertices[vidx].plate_id = largest_plate.plate_id
                print(f'Assigned remaining vertex {vidx} to largest plate {largest_plate.plate_id}')
            
            # Option 2: Alternatively, could create a new plate for these
        
        # Update plate vertex assignments
        for vidx, vertex in enumerate(self.vertices):
            self.plates[vertex.plate_id].add_vertex(vidx, self.vertices)
        
        self._fix_non_contiguous_vertices()

    @time_function
    def genElevations(self, num_plates=7, min_height=0.0, max_height=1.0):
        if not self.vertices:
            return
            
        #self._create_plates(num_plates)
        #self._assign_tectonic_plates(num_plates)
        self._assign_base_elevations()
        self.minheight = min_height
        self.maxheight = max_height
        self._calculate_boundary_elevations()
        #self._smooth_elevations(iterations=3)
        #self._add_variations()
    
    @time_function
    def _create_plates(self):
        self.plates = {}
        
        total_plates = self.platecount
        minor_plate_count = max(1, int(total_plates * random.uniform(0.2, 0.3)))
        minor_plate_ids = set(random.sample(range(total_plates), minor_plate_count))
        plate_types = []
        for i in range(self.platecount):
            if len(plate_types) > 0 and plate_types[-1] == PLATE_TYPE_CONTINENTAL:
                next_type = PLATE_TYPE_OCEANIC
            elif len(plate_types) > 0 and plate_types[-1] == PLATE_TYPE_OCEANIC:
                next_type = PLATE_TYPE_CONTINENTAL
            else:
                next_type = random.choice([PLATE_TYPE_CONTINENTAL, PLATE_TYPE_OCEANIC])
            plate_types.append(next_type)
            
        for i in range(self.platecount // 3):
            if random.random() < 0.5:
                plate_types[i] = 1 - plate_types[i]

        for plate_id in range(self.platecount):
            print(plate_types[plate_id])
            self.plates[plate_id] = Plate(plate_id, plate_types[plate_id])
            print(self.plates[plate_id].type)
            #self.plates[plate_id].type = plate_types[plate_id]
            
    @time_function
    def _plate_grow_worker(self, plate_id, plate_queue, vertices, unassigned, plates, assignment_lock):
        my_plate = plates[plate_id]

        while True:
            current_assigned_idx = -1
            try:
                current_assigned_idx = plate_queue.popleft()
            except IndexError: break

            neighbors = self._find_vertex_neighbors(current_assigned_idx)

            potential_unassigned_neighbors_indices = []
            for n_idx in neighbors:
                if vertices[n_idx].plate_id == -1:
                    potential_unassigned_neighbors_indices.append(n_idx)

            if not potential_unassigned_neighbors_indices:
                continue 

            # --- Weighted random selection ---
            valid_neighbors_for_assignment = []
            weights = []
            for n_idx in potential_unassigned_neighbors_indices:
                neighbor_plate_ids = [self.vertices[nn_idx].plate_id for nn_idx in self._find_vertex_neighbors(n_idx) if self.vertices[nn_idx].plate_id != -1]
                opposite_type_count = 0
                for pid in neighbor_plate_ids:
                    if pid in self.plates and self.plates[pid].type != self.plates[plate_id].type:
                        opposite_type_count += 1

                n_neighbors = self._find_vertex_neighbors(n_idx)

                count = sum(1 for nn_idx in n_neighbors
                            if nn_idx != current_assigned_idx and vertices[nn_idx].plate_id == plate_id)

                weight = (opposite_type_count + 1) * count + 1.0 + random.uniform(0, 0.5)
                weights.append(weight)
                valid_neighbors_for_assignment.append(n_idx)

            if not valid_neighbors_for_assignment:
                continue

            # --- Select one neighbor based on weights ---
            selected_neighbor_idx = -1
            weights = np.array(weights, dtype=float)
            total_weight = weights.sum()

            if total_weight > 1e-9 and len(valid_neighbors_for_assignment) > 0:
                try:
                    weights /= total_weight 
                    selected_neighbor_idx = np.random.choice(valid_neighbors_for_assignment, p=weights)
                except ValueError as e:
                    selected_neighbor_idx = random.choice(valid_neighbors_for_assignment)
            elif valid_neighbors_for_assignment:
                    selected_neighbor_idx = random.choice(valid_neighbors_for_assignment)
            else:
                    continue


            if selected_neighbor_idx == -1:
                continue 

            with assignment_lock:
                if selected_neighbor_idx in unassigned:
                    vertices[selected_neighbor_idx].plate_id = plate_id
                    my_plate.add_vertex(selected_neighbor_idx)
                    unassigned.remove(selected_neighbor_idx)

    @time_function
    def _assign_tectonic_plates(self, parallel_threshold_percent=10.0, max_iterations=100):
        num_vertices = len(self.vertices)
        print("Initializing plate assignment...")

        # --- Initialize Shared State ---
        plate_id_array = multiprocessing.Array(ctypes.c_int, num_vertices)
        for i in range(num_vertices):
            plate_id_array[i] = -1 

        unassigned = set(range(num_vertices))
        assignment_lock = multiprocessing.Lock()
        manager = multiprocessing.Manager()
        results_queue = manager.Queue()
        #self.plates = {plate_id: Plate(plate_id) for plate_id in range(self.platecount)}

        # --- Assign Starting Points ---
        plate_starts = np.random.choice(list(unassigned), size=self.platecount, replace=False)
        with assignment_lock:
            for plate_id, start_idx in enumerate(plate_starts):
                if start_idx in unassigned:
                    plate_id_array[start_idx] = plate_id
                    self.plates[plate_id].add_vertex(start_idx, self.vertices)
                    unassigned.remove(start_idx)

        print(f"Assigned {self.platecount} starting points. Remaining unassigned: {len(unassigned)}")

        # --- Iterative Multi-Processing Growth ---
        iteration = 0
        parallel_threshold_count = int(num_vertices * (parallel_threshold_percent / 100.0))

        neighbor_getter = self._find_vertex_neighbors

        while len(unassigned) > parallel_threshold_count and iteration < max_iterations:
            iteration += 1
            unassigned_count_start_iter = len(unassigned)

            # 1. Identify Frontier
            plate_frontiers = {plate_id: [] for plate_id in range(self.platecount)}
            current_unassigned_list = list(unassigned)

            plate_ids_vals = plate_id_array.get_obj()
            for u_idx in current_unassigned_list:
                if plate_ids_vals[u_idx] != -1:
                    if u_idx in unassigned: unassigned.remove(u_idx)
                    continue

                neighbors = self._find_vertex_neighbors(u_idx)
                for n_idx in neighbors:
                    assigned_plate_id = plate_ids_vals[n_idx]
                    if assigned_plate_id != -1:
                        plate_frontiers[assigned_plate_id].append(n_idx)

            # Deduplicate frontier points per plate
            work_items = {}
            active_plate_count = 0
            total_frontier_size = 0
            for plate_id, frontier_indices in plate_frontiers.items():
                unique_indices = list(set(frontier_indices))
                if unique_indices:
                    work_items[plate_id] = unique_indices
                    active_plate_count += 1
                    total_frontier_size += len(unique_indices)

            if active_plate_count == 0:
                #print(f"Iteration {iteration}: No active plates found. Stopping parallel phase.")
                break

            # 2. Start Worker Processes
            processes = []
            for plate_id, indices_to_process in work_items.items():
                if indices_to_process:
                    p = multiprocessing.Process(
                        target=plate_grow_worker_process,
                        args=(
                            plate_id,
                            indices_to_process,
                            plate_id_array,
                            results_queue,
                            assignment_lock,
                            neighbor_getter
                        )
                    )
                    processes.append(p)
                    p.start()

            # 3. Wait for Processes to finish & Collect Results
            for p in processes:
                p.join()

            assigned_in_iter = 0
            while not results_queue.empty():
                try:
                    res_plate_id, assigned_indices = results_queue.get_nowait()
                    newly_assigned_count = 0
                    for v_idx in assigned_indices:
                        if v_idx in unassigned:
                            unassigned.remove(v_idx)
                            self.plates[res_plate_id].add_vertex(v_idx, self.vertices)
                            newly_assigned_count += 1
                    assigned_in_iter += newly_assigned_count
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Error processing results queue: {e}")

            unassigned_count_end_iter = len(unassigned)
            print(f"Iteration {iteration}: Finished | Assigned in iter: {assigned_in_iter} | Remaining Unassigned: {unassigned_count_end_iter}")

            if assigned_in_iter == 0 and unassigned_count_start_iter > 0:
                print(f"Iteration {iteration}: Stalled. Stopping parallel phase.")
                break

        print(f"\nParallel growth phase finished after {iteration} iterations.")
        print(f"Remaining unassigned vertices (local set): {len(unassigned)}")

        # --- Sync vertex objects with final plate_id_array state ---
        final_plate_ids = plate_id_array.get_obj()
        for i, v in enumerate(self.vertices):
            v.plate_id = final_plate_ids[i]

        # --- Fallback Assignment ---
        remaining_indices_final = [i for i, pid in enumerate(final_plate_ids) if pid == -1]
        if remaining_indices_final:
            print(f'Passing {len(remaining_indices_final)} remaining vertices to sequential fallback.')
            self._assign_remaining_vertices_safe(remaining_indices_final)
        else:
            print("All vertices assigned during parallel phase.")

        # --- Handle Non-Contiguous Vertices ---
        self._fix_non_contiguous_vertices()

    @time_function
    def _fix_non_contiguous_vertices(self, min_plate_size=5, minor_plate_ratio=0.2):
        for plate in self.plates.values():
            plate.resetVertices
        for i, vertex in enumerate(self.vertices):
            self.plates[vertex.plate_id].add_vertex(i, self.vertices)
        
        return

    @time_function
    def _find_all_contiguous_regions(self, plate_id):
        """
        Find all contiguous regions in a plate using BFS.
        Returns a list of sets, each containing vertex indices for a region.
        """
        visited = set()
        regions = []
        plate_vertices = self.plates[plate_id].vertices.copy()
        
        while plate_vertices:
            start_idx = plate_vertices.pop()
            if start_idx in visited:
                continue
                
            # Start new region
            queue = deque([start_idx])
            current_region = set()
            
            while queue:
                v_idx = queue.popleft()
                if v_idx in visited:
                    continue
                    
                visited.add(v_idx)
                current_region.add(v_idx)
                
                neighbors = self._find_vertex_neighbors(v_idx)
                for n_idx in neighbors:
                    if self.vertices[n_idx].plate_id == plate_id and n_idx not in visited:
                        queue.append(n_idx)
            
            regions.append(current_region)
            # Remove found vertices from working set
            plate_vertices -= current_region
        
        return regions

    @time_function
    def _find_best_neighbor_plate(self, region_vertices):
        """
        Find the best neighboring plate for a region based on:
        1. Most common neighboring plate
        2. Plate type compatibility
        3. Random choice if ties
        """
        neighbor_counts = {}
        type_matches = {}
        
        for v_idx in region_vertices:
            neighbors = self._find_vertex_neighbors(v_idx)
            for n_idx in neighbors:
                n_plate = self.vertices[n_idx].plate_id
                if n_plate != -1 and n_plate != self.vertices[v_idx].plate_id:
                    neighbor_counts[n_plate] = neighbor_counts.get(n_plate, 0) + 1
                    
                    # Check type compatibility
                    if n_plate in self.plates:
                        if self.plates[n_plate].type == self.plates[self.vertices[v_idx].plate_id].type:
                            type_matches[n_plate] = type_matches.get(n_plate, 0) + 1
        
        if not neighbor_counts:
            return None
            
        # Find plates with max neighbor count
        max_count = max(neighbor_counts.values())
        candidates = [p for p, cnt in neighbor_counts.items() if cnt == max_count]
        
        if len(candidates) == 1:
            return candidates[0]
            
        # Break ties by type matches
        if type_matches:
            max_type_matches = max(type_matches.get(p, 0) for p in candidates)
            candidates = [p for p in candidates if type_matches.get(p, 0) == max_type_matches]
            if len(candidates) == 1:
                return candidates[0]
        
        # Still tied - random choice
        return random.choice(candidates)
    
    @time_function
    def _find_largest_contiguous_region(self, plate_id):
        """
        Find the largest contiguous region of vertices in a plate using BFS.
        Returns a set of vertex indices.
        """
        visited = set()
        largest_region = set()
        plate_vertices = self.plates[plate_id].vertices.copy()

        while plate_vertices:
            start_idx = plate_vertices.pop()
            if start_idx in visited:
                continue

            queue = deque([start_idx])
            current_region = set()

            while queue:
                v_idx = queue.popleft()
                if v_idx in visited:
                    continue

                visited.add(v_idx)
                current_region.add(v_idx)

                neighbors = self._find_vertex_neighbors(v_idx)
                for n_idx in neighbors:
                    if self.vertices[n_idx].plate_id == plate_id and n_idx not in visited:
                        queue.append(n_idx)

            if len(current_region) > len(largest_region):
                largest_region = current_region

        return largest_region
    
    @time_function
    def _assign_remaining_vertices_safe(self, remaining_unassigned_indices):
        print(f"Assigning {len(remaining_unassigned_indices)} remaining vertices sequentially in a safer manner...")
        indices_to_process = list(remaining_unassigned_indices)
        assigned_in_fallback = 0

        pass_num = 0
        while indices_to_process:
            pass_num += 1
            print(f"Fallback Pass {pass_num}: Processing {len(indices_to_process)} indices...")
            processed_indices = set()

            for v_idx in indices_to_process:
                if v_idx in processed_indices:
                    continue

                queue = deque([v_idx])
                visited = {v_idx}
                found_plate = -1
                while queue and found_plate == -1:
                    current = queue.popleft()
                    for n_idx in self._find_vertex_neighbors(current):
                        n_plate = self.vertices[n_idx].plate_id
                        if n_plate != -1:
                            found_plate = n_plate
                            break
                    if n_idx not in visited:
                        visited.add(n_idx)
                        queue.append(n_idx)

                if found_plate != -1:
                    for vidx in visited:
                        if vidx in indices_to_process:
                            self.plates[found_plate].add_vertex(vidx, self.vertices)
                            indices_to_process.remove(vidx)
                            assigned_in_fallback += 1
                    processed_indices.update(visited)

    @time_function
    def _assign_base_elevations(self):
        for plate in self.plates.values():
            for v_idx in plate.vertices:
                variation = np.random.uniform(-0.1, 0.1)
                self.vertices[v_idx].elevation = plate.base_elevation + (variation * plate.base_elevation)

    @time_function
    def _calculate_boundary_elevations(self):
        """Calculate elevations based on plate interactions at boundaries."""
        boundary_vertices = []
        boundary_info = {}  # Store boundary information for each vertex
        
        # First pass: identify boundary vertices and calculate interactions
        for plate in self.plates.values():
            for v_idx in plate.get_boundary_vertices(self):
                vertex = self.vertices[v_idx]
                neighbors = self._find_vertex_neighbors(v_idx)
                
                # Find all adjacent plates
                adjacent_plates = set()
                for n in neighbors:
                    adjacent_plates.add(self.vertices[n].plate_id)
                adjacent_plates.add(vertex.plate_id)
                
                # Calculate relative movement vectors
                movement_vectors = []
                current_plate = self.plates[vertex.plate_id]
                
                for plate_id in adjacent_plates:
                    if plate_id == vertex.plate_id:
                        continue
                        
                    other_plate = self.plates[plate_id]
                    rel_velocity = current_plate.velocity - other_plate.velocity
                    
                    # Project onto vertex normal
                    normal = vertex.pos / np.linalg.norm(vertex.pos)
                    movement = np.dot(rel_velocity, normal)
                    movement_vectors.append(movement)
                
                boundary_info[v_idx] = {
                    'movements': movement_vectors,
                    'adjacent_plates': adjacent_plates,
                    'plate_type': current_plate.type
                }
                boundary_vertices.append(v_idx)

        # Second pass: assign elevations based on boundary interactions
        for v_idx in boundary_vertices:
            vertex = self.vertices[v_idx]
            info = boundary_info[v_idx]
            movements = info['movements']
            plate_type = info['plate_type']
            
            if not movements:
                continue
                
            avg_movement = np.mean(movements)
            
            # Different elevation responses based on plate type interactions
            if plate_type == PLATE_TYPE_CONTINENTAL:
                # Continental plates create mountains when colliding
                elevation = CONTINENTAL_CRUST_THICKNESS + (self.maxheight * (avg_movement + 1)/2)
            else:
                # Oceanic plates create trenches or islands
                if avg_movement > 0:  # Converging
                    elevation = OCEANIC_CRUST_THICKNESS - (OCEANIC_CRUST_THICKNESS - self.minheight) * avg_movement
                else:  # Diverging (mid-ocean ridges)
                    elevation = OCEANIC_CRUST_THICKNESS + (0.5 - OCEANIC_CRUST_THICKNESS) * abs(avg_movement)
            
            vertex.elevation = elevation
        print('boundary effects calculated')

    @time_function
    def _add_variations(self, iterations=5):
            """
            Add natural elevation variations using PyTorch for GPU acceleration.
            Allows for both increases (hills/peaks) and decreases (valleys).
            """
            print("--- Starting PyTorch Elevation Variation (with Valley Formation) ---")
            start_total_time = time.time()

            print(f"Using device: {self.device}")

            num_vertices = len(self.vertices)
            if num_vertices == 0:
                print("No vertices to process.")
                return

            # --- 1. Data Preparation (CPU -> GPU Tensors) ---
            print("Preparing data for GPU...")
            prep_start_time = time.time()

            # Basic vertex data
            positions = torch.tensor([v.pos for v in self.vertices], dtype=torch.float32, device=self.device)
            elevations = torch.tensor([v.elevation for v in self.vertices], dtype=torch.float32, device=self.device)
            plate_ids = torch.tensor([v.plate_id for v in self.vertices], dtype=torch.long, device=self.device)

            # --- Apply initial 5% noise to all elevations ---
            noise_scale = 0.05  # 5% noise
            noise = (torch.rand_like(elevations) * 2.0 - 1.0) * noise_scale * (self.maxheight - self.minheight) #elevations
            elevations += noise
            
            # Identify continental vertices
            is_continental = torch.zeros(num_vertices, dtype=torch.bool, device=self.device)
            continental_plate_ids = set()
            for plate_id, plate in self.plates.items():
                if plate.type == PLATE_TYPE_CONTINENTAL:
                    continental_plate_ids.add(plate_id)
            # Vectorized check for continental plates
            continental_mask_per_plate = torch.zeros_like(plate_ids, dtype=torch.bool)
            for p_id in continental_plate_ids:
                continental_mask_per_plate |= (plate_ids == p_id)
            is_continental = continental_mask_per_plate
            continental_indices = torch.where(is_continental)[0]
            num_continental_vertices = len(continental_indices)

            if num_continental_vertices == 0:
                print("No continental vertices found. Skipping variation.")
                return

            # Precompute neighbors
            neighbor_indices, neighbor_mask, max_neighbors = self._build_neighbor_tensors()

            # Identify high points (peaks)
            # (Peak identification logic remains largely the same as before)
            all_peaks_indices = []
            plate_peak_elevs = {}
            for plate_id in continental_plate_ids:
                plate = self.plates[plate_id]
                if not hasattr(plate, 'vertices') or not plate.vertices: continue # Skip if no vertices attr or empty
                plate_v_indices_list = list(plate.vertices)
                if not plate_v_indices_list: continue

                # Filter out invalid indices before using them
                valid_plate_v_indices = [idx for idx in plate_v_indices_list if 0 <= idx < num_vertices]
                if not valid_plate_v_indices: continue

                plate_v_indices = torch.tensor(valid_plate_v_indices, dtype=torch.long, device=self.device)

                # Check if plate_v_indices is empty after filtering
                if plate_v_indices.numel() == 0: continue

                plate_elevs_tensor = elevations[plate_v_indices]
                num_peaks = max(1, int(len(valid_plate_v_indices) * 0.05))
                num_peaks = min(num_peaks, len(valid_plate_v_indices)) # Cannot request more peaks than vertices

                if num_peaks > 0:
                    _, top_indices_in_plate = torch.topk(plate_elevs_tensor, k=num_peaks)
                    plate_peaks_indices = plate_v_indices[top_indices_in_plate].tolist() # Use filtered indices tensor
                    all_peaks_indices.extend(plate_peaks_indices)
                    if plate_peaks_indices:
                        plate_peak_elevs[plate_id] = torch.max(elevations[plate_peaks_indices]).item()
                    else:
                        plate_peak_elevs[plate_id] = self.maxheight
                else:
                    plate_peak_elevs[plate_id] = self.maxheight # Default if no peaks calculated


            if not all_peaks_indices:
                print("Warning: No peaks found on any continental plate.")
                has_peaks = False
                peaks_pos = torch.empty((0, 3), dtype=torch.float32, device=self.device) # Ensure it's defined
            else:
                peaks_indices_tensor = torch.tensor(list(set(all_peaks_indices)), dtype=torch.long, device=self.device)
                # Final check for valid indices in peaks_indices_tensor
                valid_peak_indices = peaks_indices_tensor[(peaks_indices_tensor >= 0) & (peaks_indices_tensor < num_vertices)]
                if len(valid_peak_indices) < len(peaks_indices_tensor):
                    print(f"Warning: Removed {len(peaks_indices_tensor) - len(valid_peak_indices)} invalid peak indices.")
                if len(valid_peak_indices) == 0:
                    print("Warning: All peak indices were invalid.")
                    has_peaks = False
                    peaks_pos = torch.empty((0, 3), dtype=torch.float32, device=self.device)
                else:
                    peaks_pos = positions[valid_peak_indices]
                    # peaks_plate_id = plate_ids[valid_peak_indices] # We don't use this directly later
                    has_peaks = True


            # Map plate_id to its max peak elevation
            max_peak_elev_map = torch.full_like(elevations, self.maxheight)
            for i in range(num_vertices):
                p_id = plate_ids[i].item()
                if p_id in plate_peak_elevs:
                    max_peak_elev_map[i] = plate_peak_elevs[p_id]


            # Simulation parameters
            # Use subdivisions parameter if it exists, otherwise default to 1
            #effective_subdivisions = getattr(self, 'subdivisions', 1) if subdivisions is None else subdivisions
            
            total_iterations = self.subdivisions * iterations # Match original logic if subdivisions exist

            elev_range = self.maxheight - self.minheight if self.maxheight > self.minheight else 1.0
            significant_change_threshold = 0.001 * elev_range

            # --- Tunable Parameters ---
            peak_force_scale = 0.5    # Weight of the upward push from peaks
            neighbor_push_scale = 0.1  # Max random factor for pushing towards higher neighbors
            neighbor_pull_scale = 0.05 # Max random factor for pulling away from slightly lower neighbors
            slump_scale = 0.15        # Factor for downward force when higher than average neighbor (NEW)
            noise_scale = 0.3         # Increased weight for random up/down noise
            damping_factor = 0.5      # Overall damping rate per iteration

            print(f"Data preparation took {time.time() - prep_start_time:.2f}s")
            print(f"Running {total_iterations} iterations. Min/Max Elev: {self.minheight:.2f}/{self.maxheight:.2f}")
            print(f"Tunable Factors: Peak={peak_force_scale}, NeighborPush={neighbor_push_scale}, NeighborPull={neighbor_pull_scale}, Slump={slump_scale}, Noise={noise_scale}, Damp={damping_factor}")


            # --- 2. Simulation Loop (on GPU) ---
            for iteration in range(total_iterations):
                iter_start_time = time.time()
                #if continental_indices.numel() == 0: break # Exit if no continental vertices left

                # Only process continental vertices
                current_elev = elevations[continental_indices]
                current_pos = positions[continental_indices]
                current_plate_ids = plate_ids[continental_indices]

                # --- Neighbor Calculations ---
                cont_neighbor_indices = neighbor_indices[continental_indices]
                cont_neighbor_mask = neighbor_mask[continental_indices]

                # Gather neighbor data using full tensors
                neighbor_elevs = elevations[cont_neighbor_indices]
                neighbor_pos = positions[cont_neighbor_indices]
                neighbor_plate_ids = plate_ids[cont_neighbor_indices]

                # Mask out invalid neighbors and neighbors from different plates
                same_plate_mask = (neighbor_plate_ids == current_plate_ids.unsqueeze(1))
                valid_neighbor_mask = cont_neighbor_mask & same_plate_mask

                # Calculate elevation difference and distance for valid neighbors
                elev_diff = neighbor_elevs - current_elev.unsqueeze(1)
                dist = torch.norm(current_pos.unsqueeze(1) - neighbor_pos, dim=2)
                inv_dist = 1.0 / (dist + 0.1) # Add epsilon

                # Calculate original neighbor force components (push/pull)
                force_from_higher = torch.rand_like(elev_diff) * neighbor_push_scale * elev_diff * inv_dist
                force_from_lower = -torch.rand_like(elev_diff) * neighbor_pull_scale * inv_dist
                neighbor_force_contribution = torch.where(
                    elev_diff > 0,
                    force_from_higher,
                    torch.where(
                        torch.abs(elev_diff) < 0.1 * elev_range, # Condition for small downward push
                        force_from_lower,
                        torch.zeros_like(elev_diff)
                    )
                )
                # Apply mask and sum original forces
                masked_neighbor_force = neighbor_force_contribution * valid_neighbor_mask.float()
                total_original_neighbor_force = torch.sum(masked_neighbor_force, dim=1)

                # --- NEW: Slumping/Erosion Force based on difference from average ---
                # Calculate average elevation of *valid* neighbors
                masked_neighbor_elevs = neighbor_elevs * valid_neighbor_mask.float()
                num_valid_neighbors = torch.sum(valid_neighbor_mask.float(), dim=1).clamp(min=1) # Avoid div by zero
                avg_local_elev = torch.sum(masked_neighbor_elevs, dim=1) / num_valid_neighbors

                # Calculate difference: positive if current vertex is higher than average
                elev_diff_from_avg = current_elev - avg_local_elev

                # Calculate slump force: proportional to how much higher it is, applies downward force
                # Use relu to only apply slump when elev_diff_from_avg is positive
                slump_force = -torch.relu(elev_diff_from_avg) * slump_scale * torch.rand_like(current_elev) # Random factor per vertex

                # Combine Neighbor Forces (Original push/pull + Slumping)
                # Adjust relative weighting if needed, e.g., 0.5 for original, 0.5 for slump
                total_neighbor_force = (total_original_neighbor_force * 0.7 + slump_force * 0.3)


                # --- Peak Force Calculation ---
                peak_force = torch.zeros_like(current_elev)
                if has_peaks and peaks_pos.numel() > 0: # Check if peaks_pos is not empty
                    dist_to_peaks_sq = torch.sum((current_pos.unsqueeze(1) - peaks_pos.unsqueeze(0))**2, dim=2)
                    min_dist_to_peak_sq, _ = torch.min(dist_to_peaks_sq, dim=1)
                    min_dist_to_peak = torch.sqrt(min_dist_to_peak_sq + 1e-9)

                    peak_influence = 1.0 / (1.0 + min_dist_to_peak * 5.0)

                    current_max_peak_elev = max_peak_elev_map[continental_indices]

                    # Calculate base peak force (can be negative if point is above peak)
                    raw_peak_force = (torch.rand_like(current_elev) * 0.2 * # Base random factor
                                    peak_influence *
                                    (current_max_peak_elev - current_elev))

                    # Keep the clamp: peaks primarily cause uplift in this model.
                    # Valleys form from slumping and noise mainly.
                    peak_force = torch.clamp(raw_peak_force, min=0)


                # --- Random Noise ---
                current_damping = damping_factor * (1.0 - iteration / total_iterations) # Per-iteration damping
                noise = (torch.rand_like(current_elev) * 2.0 - 1.0) # Noise in range [-1, 1]
                noise_force = noise * noise_scale * (1.0 - iteration / total_iterations) # Noise decreases over time

                # --- Combine Forces ---
                # Adjust weights as needed based on experimentation
                total_force = (peak_force * peak_force_scale +
                            total_neighbor_force * (1.0 - peak_force_scale - noise_scale) + # Neighbor force takes remaining weight
                            noise_force * noise_scale) # Apply noise scale here
                total_force *= current_damping # Apply damping to the combined force

                # --- Update Elevation ---
                new_elev = current_elev + total_force
                new_elev_clamped = torch.clamp(new_elev, min=self.minheight, max=self.maxheight)

                changes_mask = torch.abs(new_elev_clamped - current_elev) > significant_change_threshold
                num_changes = torch.sum(changes_mask).item()

                elevations[continental_indices] = new_elev_clamped

                print(f"\rIteration {iteration + 1}/{total_iterations} | Changes: {num_changes} | Time: {time.time() - iter_start_time:.3f}s | Elev Range: {torch.min(elevations[continental_indices]):.2f}-{torch.max(elevations[continental_indices]):.2f}", end="")

            print(f"\nCompleted {total_iterations} PyTorch variation iterations.")

            # --- 3. Data Synchronization (GPU -> CPU) ---
            print("Synchronizing data back to CPU objects...")
            sync_start_time = time.time()

            final_elevations_cpu = elevations.cpu().numpy()
            for i in range(num_vertices):
                # Check if vertex index is valid before assignment
                if 0 <= i < len(self.vertices):
                    self.vertices[i].elevation = final_elevations_cpu[i]
                else:
                    print(f"Warning: Skipping synchronization for invalid vertex index {i}")


            print(f"Data synchronization took {time.time() - sync_start_time:.2f}s")
            print(f"--- PyTorch Elevation Variation Finished (Total Time: {time.time() - start_total_time:.2f}s) ---")

    @time_function
    def _smooth_elevations(self, iterations=2):
        """Smooth elevation values with plate-aware smoothing."""
        for _ in range(iterations):
            new_elevations = []
            for i, vertex in enumerate(self.vertices):
                neighbors = self._find_vertex_neighbors(i)
                if not neighbors:
                    new_elevations.append(vertex.elevation)
                    continue
                    
                # Get neighbors from the same plate
                same_plate_neighbors = [n for n in neighbors 
                                       if self.vertices[n].plate_id == vertex.plate_id]
                
                # Use all neighbors if no same-plate neighbors found
                smoothing_neighbors = same_plate_neighbors if same_plate_neighbors else neighbors
                
                neighbor_elevations = [self.vertices[n].elevation for n in smoothing_neighbors]
                avg = np.mean([vertex.elevation] + neighbor_elevations)
                new_elevations.append(avg)
                
            for i, elevation in enumerate(new_elevations):
                self.vertices[i].elevation = elevation
        print('smoothed elevations')

    #### fluids

    @time_function
    def simulate_water(self, iterations: int = 5):
        print("Starting water simulation...")
        
        # Convert parameters to tensors only when needed
        sea_level = torch.tensor(self.sea_level, device=self.device)
        rainfall_rate = torch.tensor(self.rainfall_rate, device=self.device)
        evaporation_rate = torch.tensor(self.evaporation_rate, device=self.device)
        max_water_flow = torch.tensor(self.max_water_flow, device=self.device)
        min_river_flow = torch.tensor(self.min_river_flow, device=self.device)
        min_lake_volume = torch.tensor(self.min_lake_volume, device=self.device)
        
        # Rest of the implementation remains the same, using these tensor versions
        num_vertices = len(self.vertices)
            
        # Get neighbor information
        neighbor_indices, neighbor_mask, max_neighbors = self._build_neighbor_tensors()
        
        # Initialize water tensors - ensure all are on the same device
        elevation = torch.tensor([v.elevation for v in self.vertices], device=self.device)
        water_volume = torch.zeros(num_vertices, device=self.device)
        water_depth = torch.zeros(num_vertices, device=self.device, dtype=torch.double)
        
        # Precompute face areas for each vertex and move to device
        vertex_areas = torch.tensor(self._compute_vertex_areas(), device=self.device)
        
        print("Initializing water distribution...")
        # Initial water distribution - oceans get water up to sea level
        ocean_mask = elevation <= sea_level
        sea_level = sea_level.double()
        water_depth[ocean_mask] = torch.maximum(
            sea_level - elevation[ocean_mask], 
            torch.tensor(0.0, device=self.device)
        ).double()
        water_volume = water_depth * vertex_areas
        
        # Main simulation loop
        print("Running hydrological cycle...")
        for iter in range(iterations):
            print(f"Iteration {iter + 1}/{iterations}")
            new_water = torch.zeros_like(water_volume, device=self.device)
            
            # 1. Precipitation (rainfall on land)
            land_mask = elevation > sea_level
            rainfall = torch.where(
                land_mask,
                torch.rand(num_vertices, device=self.device) * rainfall_rate * vertex_areas,
                torch.tensor(0.0, device=self.device)
            )
            new_water += rainfall
            
            # 2. Evaporation (from all water surfaces)
            evaporation = torch.minimum(
                water_volume,
                evaporation_rate * vertex_areas
            )
            new_water -= evaporation
            
            # 3. Flow between vertices
            current_water_depth = water_volume / vertex_areas
            effective_elevation = elevation + current_water_depth
            
            # Find downhill flow for each vertex
            for i in range(num_vertices):
                if not land_mask[i]:
                    continue  # Skip ocean cells
                    
                neighbors = neighbor_indices[i][neighbor_mask[i]]
                if len(neighbors) == 0:
                    continue
                    
                # Find lowest neighbor
                neighbor_eff_elev = effective_elevation[neighbors]
                min_elev_idx = torch.argmin(neighbor_eff_elev)
                lowest_neighbor = neighbors[min_elev_idx]
                
                if effective_elevation[lowest_neighbor] >= effective_elevation[i]:
                    continue  # No downhill flow
                    
                # Calculate flow amount based on gradient
                elev_diff = effective_elevation[i] - effective_elevation[lowest_neighbor]
                gradient = elev_diff / torch.norm(
                    torch.tensor(self.vertices[i].pos, device=self.device) - 
                    torch.tensor(self.vertices[lowest_neighbor].pos, device=self.device)
                )
                
                max_possible_flow = min(
                    water_volume[i] * 0.2,  # Max 20% of current water can flow
                    max_water_flow * vertex_areas[i]  # Absolute max flow
                )
                
                flow_amount = max_possible_flow * torch.sigmoid(
                    torch.tensor(5.0, device=self.device) * gradient
                )
                
                # Adjust flow based on areas
                area_ratio = vertex_areas[i] / vertex_areas[lowest_neighbor]
                adjusted_flow = flow_amount * area_ratio
                
                new_water[i] -= flow_amount
                new_water[lowest_neighbor] += adjusted_flow
            
            # Update water volumes
            water_volume = torch.maximum(water_volume + new_water, torch.tensor(0.0, device=self.device))
        
        # Update vertex data
        print("Updating vertex data...")
        water_depth = water_volume / vertex_areas
        water_depth_cpu = water_depth.cpu().numpy()
        water_volume_cpu = water_volume.cpu().numpy()
        
        for i in range(num_vertices):
            self.vertices[i].water_volume = float(water_volume_cpu[i])
            self.vertices[i].water_depth = float(water_depth_cpu[i])
        
        # Generate rivers and lakes
        print("Generating water features...")
        self._generate_rivers(min_river_flow)
        self._find_lakes(min_lake_volume)
        
        print("Water simulation complete")

    @time_function
    def _generate_rivers(self, min_flow=1.0):
        """Identify and mark rivers based on accumulated water flow.
        min_flow: minimum flow rate in TL/iteration to be considered a river."""
        print("Generating rivers...")
        
        # Reset river flags
        for vertex in self.vertices:
            vertex.is_river = False
        
        # Track flow accumulation for each vertex
        flow_accumulation = [0.0] * len(self.vertices)
        
        # Calculate flow accumulation (simplified approach)
        print('flow calculation')
        for i, vertex in enumerate(self.vertices):
            if vertex.elevation > self.sea_level:
                neighbors = self._find_vertex_neighbors(i)
                if neighbors:
                    # Count how many cells flow into this one
                    for neighbor in neighbors:
                        if self.vertices[neighbor].elevation > vertex.elevation:
                            # Neighbor is higher and would flow to this vertex
                            flow_accumulation[i] += self.vertices[neighbor].water
        
        # Mark rivers based on flow accumulation
        print('marking rivers')
        for i, flow in enumerate(flow_accumulation):
            if flow >= min_flow:
                self.vertices[i].is_river = True
                # Scale river size based on flow (could be used for rendering)
                self.vertices[i].river_size = min(3, int(flow / min_flow))
        
        # Connect river segments to form continuous rivers
        print('connecting segments')
        self._connect_river_segments()
        
        print(f"Generated {sum(1 for v in self.vertices if v.is_river)} river segments")

    @time_function
    def _connect_river_segments(self):
        """Ensure rivers form continuous paths from source to ocean/lake."""
        # This is a simplified approach - more sophisticated methods could be used
        for i, vertex in enumerate(self.vertices):
            if vertex.is_river:
                # Find the downstream path
                current = i
                path = [current]
                while True:
                    neighbors = self._find_vertex_neighbors(current)
                    if not neighbors:
                        break
                    
                    # Find the lowest neighbor (natural flow direction)
                    lowest_neighbor = min(neighbors,
                                        key=lambda n: self.vertices[n].elevation)
                    
                    # Stop if we've reached sea level or a lake
                    if (self.vertices[lowest_neighbor].elevation <= self.sea_level or
                        self.vertices[lowest_neighbor].water > 5.0):  # 5 TL threshold for lakes
                        break
                    
                    # If we're going uphill, stop (shouldn't happen)
                    if self.vertices[lowest_neighbor].elevation >= self.vertices[current].elevation:
                        break
                    
                    # Mark the neighbor as river and continue
                    self.vertices[lowest_neighbor].is_river = True
                    self.vertices[lowest_neighbor].river_size = max(
                        self.vertices[lowest_neighbor].river_size,
                        self.vertices[current].river_size - 0.5
                    )
                    current = lowest_neighbor
                    path.append(current)
                    
                    # Prevent infinite loops
                    if current in path[:-1]:
                        break

    @time_function
    def _find_lakes(self, min_size=3.0):
        lakes = []
        visited = set()
        
        for i, vertex in enumerate(self.vertices):
            if (vertex.elevation > self.sea_level and 
                vertex.water >= min_size and i not in visited):
                
                # Flood fill to find connected lake cells
                lake_cells = []
                queue = [i]
                while queue:
                    current = queue.pop()
                    if current not in visited:
                        visited.add(current)
                        lake_cells.append(current)
                        
                        # Add unvisited neighbors with sufficient water
                        neighbors = self._find_vertex_neighbors(current)
                        for neighbor in neighbors:
                            if (self.vertices[neighbor].water >= min_size and
                                neighbor not in visited):
                                queue.append(neighbor)
                
                if len(lake_cells) >= 3:  # Minimum 3 cells to form a lake
                    total_volume = sum(self.vertices[idx].water for idx in lake_cells)
                    lakes.append({
                        'cells': lake_cells,
                        'volume': total_volume,
                        'average_depth': total_volume / len(lake_cells)  # Simplified
                    })
        
        return lakes

    #### utility functions

    @time_function
    def _compute_vertex_areas(self) -> torch.Tensor:
        """Compute vertex areas by distributing adjacent face areas to vertices.
        Each vertex gets an equal share of each adjacent face's area."""
        if not self._neighbor_map_initialized:
            self._initialize_neighbor_map()
            
        # Calculate all face areas
        self.calculate_all_face_areas()
        
        # Initialize vertex areas
        vertex_areas = torch.zeros(len(self.vertices), device=self.device)
        
        # Distribute each face's area equally to its vertices
        for face_idx, face in enumerate(self.faces):
            face_area = face.calculate_area(self.vertices)
            num_vertices_in_face = len(face.v_indices)
            vertex_share = face_area / num_vertices_in_face
            
            for v_idx in face.v_indices:
                vertex_areas[v_idx] = vertex_areas[v_idx] + vertex_share
                
        # Handle any vertices with zero area (shouldn't happen with proper meshes)
        vertex_areas[vertex_areas == 0] = 1.0  # Assign default area
        self.vertex_areas = (vertex_areas).cpu()
        return vertex_areas

    @time_function
    def calculate_all_face_areas(self):
            
        areas = batch_calculate_areas(self.faces, self.vertices)
        
        # Update the face objects with their areas
            
        return areas

    @time_function
    def _get_midpoint_vertex(self, v1_idx, v2_idx, midpoint_cache, next_level_vertices):
        """
        Helper: Calculates or retrieves the midpoint vertex between two vertices.
        Adds the new vertex if it doesn't exist and normalizes it.
        Returns the index of the vertex in next_level_vertices.
        """
        key = tuple(sorted((v1_idx, v2_idx)))
        if key in midpoint_cache:
            # Map cached index (relative to additions) back to next_level_vertices index
            return midpoint_cache[key]

        v1 = self.vertices[v1_idx]
        v2 = self.vertices[v2_idx]
        mid_pos = (v1.pos + v2.pos) / 2.0
        new_v = Vertex(mid_pos[0], mid_pos[1], mid_pos[2])
        new_v.normalize()
        new_v.plate_id = v1.plate_id
        # Add to the *accumulating* vertex list for the next level
        new_idx = len(next_level_vertices)
        next_level_vertices.append(new_v)
        midpoint_cache[key] = new_idx # Cache the index in the accumulating list
        return new_idx

    @time_function
    def _get_face_center_vertex(self, face, center_cache, next_level_vertices, radius):
        """
        Helper: Calculates or retrieves the center vertex of a face.
        Adds the new vertex if it doesn't exist and normalizes it.
        Returns the index of the vertex in next_level_vertices.
        """
        key = tuple(sorted(face.v_indices)) # Use sorted indices as cache key
        if key in center_cache:
             return center_cache[key]

        # Calculate geometric center
        center_pos = np.mean([self.vertices[i].pos for i in face.v_indices], axis=0)
        
        new_v = Vertex(center_pos[0], center_pos[1], center_pos[2])
        new_v.normalize(radius)

        # Add to the *accumulating* vertex list for the next level
        new_idx = len(next_level_vertices)
        next_level_vertices.append(new_v)
        center_cache[key] = new_idx # Cache the index in the accumulating list
        return new_idx

    @time_function
    def _compute_face_centroid(self, v_indices, vertices, radius):
        face_vertex_positions = np.array([vertices[idx].pos for idx in v_indices], dtype=float)
        centroid_pos = np.mean(face_vertex_positions, axis=0)
        norm = np.linalg.norm(centroid_pos)
        normalized_centroid_pos = centroid_pos * (radius / norm)
        return Vertex(*normalized_centroid_pos)
    
    @time_function
    def _initialize_neighbor_map(self) -> None:
        if self._neighbor_map_initialized:
            return
            
        print("Initializing neighbor map...")
        
        vertex_neighbor_sets = [set() for _ in range(len(self.vertices))]
        vertex_face_map = [[] for _ in range(len(self.vertices))]
        face_neighbor_sets = [set() for _ in range(len(self.faces))]
        all_positions = np.array([v.pos for v in self.vertices])
        edge_face_map = {}
        for fid, face in enumerate(self.faces):
            for vid in face.v_indices:
                vertex_face_map[vid].append(fid)

            for i in range(len(face.v_indices)):
                v1 = face.v_indices[i]
                v2 = face.v_indices[(i+1) % len(face.v_indices)]    
                vertex_neighbor_sets[v1].add(v2)
                vertex_neighbor_sets[v2].add(v1)
    
            # Record edge-face relationships
            edge = tuple(sorted((v1, v2)))
            if edge in edge_face_map:
                # This edge is shared with another face - mark as neighbors
                other_face = edge_face_map[edge]
                face_neighbor_sets[fid].add(other_face)
                face_neighbor_sets[other_face].add(fid)
            else:
                edge_face_map[edge] = fid

        for v_idx, neighbors in enumerate(vertex_neighbor_sets):
            neighbor_indices = np.array(list(neighbors))
            dists = np.linalg.norm(all_positions[v_idx] - all_positions[neighbor_indices], axis=1)
            weights = np.divide(1.0, dists, where=dists!=0, out=dists)
            for n_idx in neighbor_indices:
                self.vertices[v_idx].neighbors[n_idx] = weights

        
        self._face_neighbors = [list(s) for s in face_neighbor_sets]
        self._vertex_face_map = vertex_face_map
        self._neighbor_map_initialized = True
        print("Neighbor map initialized")


    @time_function
    def _find_vertex_neighbors(self, vertex_idx):
        self._initialize_neighbor_map()
        return self.vertices[vertex_idx].neighbors
    
    @time_function
    def _get_adjacent_faces(self, vertex_idx):
        """Get all faces adjacent to a vertex"""
        if not self._neighbor_map_initialized:
            self._initialize_neighbor_maps()
        return [self.faces[i] for i in self._vertex_face_map[vertex_idx]]

    @time_function
    def _get_face_neighbors(self, face_idx):
        """Get faces that share an edge with the given face"""
        if not self._neighbor_map_initialized:
            self._initialize_neighbor_maps()
        return [self.faces[i] for i in self._face_neighbors[face_idx]]

    @time_function
    def _build_neighbor_tensors(self):
        num_vertices = len(self.vertices)
        neighbor_lists = [list(self._find_vertex_neighbors(i)) for i in range(num_vertices)]
        max_neighbors = max(len(neighbors) for neighbors in neighbor_lists) if neighbor_lists else 0
        if max_neighbors == 0 and num_vertices > 0 : # Handle case with isolated vertices
             print("Warning: Some vertices may have no neighbors.")
             max_neighbors = 1 # Avoid zero-sized tensor dim if possible

        neighbor_indices = torch.full((num_vertices, max_neighbors), 0,
                                      dtype=torch.long, device=self.device)
        neighbor_mask = torch.zeros((num_vertices, max_neighbors),
                                    dtype=torch.bool, device=self.device)

        for i, neighbors in enumerate(neighbor_lists):
            if neighbors:
                num_n = len(neighbors)
                # Ensure indices are within bounds if padding with 0
                safe_neighbors = [n for n in neighbors if 0 <= n < num_vertices]
                if len(safe_neighbors) < num_n:
                    print(f"Warning: Vertex {i} had invalid neighbor indices removed.")

                num_n = len(safe_neighbors) # Update count
                if num_n > 0:
                    neighbor_indices[i, :num_n] = torch.tensor(safe_neighbors, dtype=torch.long, device=self.device)
                    neighbor_mask[i, :num_n] = True

        return neighbor_indices, neighbor_mask, max_neighbors
  
    @time_function
    def _find_vertices_within_radius(self, start_vidx, radius):
        """
        Find all vertices within a given graph distance (number of edges) from a starting vertex.
        
        Args:
            start_vidx (int): Index of the starting vertex
            radius (int): Maximum graph distance to search (number of edges)
            
        Returns:
            list: List of vertex indices within the specified radius
        """
        if not self._neighbor_map_initialized:
            self._initialize_neighbor_map()
            
        if start_vidx < 0 or start_vidx >= len(self.vertices):
            return []
            
        if radius < 1:
            return []
            
        visited = set()
        queue = deque([(start_vidx, 0)])  # (vertex_index, current_distance)
        vertices_in_radius = []
        
        while queue:
            current_vidx, current_dist = queue.popleft()
            
            if current_vidx in visited:
                continue
                
            visited.add(current_vidx)
            
            if current_dist > 0:  # Don't include the starting vertex unless radius is 0
                vertices_in_radius.append(current_vidx)
                
            if current_dist < radius:
                # Add neighbors to queue with incremented distance
                neighbors = self._find_vertex_neighbors(current_vidx)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, current_dist + 1))
                        
        return vertices_in_radius

    #### display

    @time_function
    def plot(self, fig=None, ax=None, cmap='terrain', edge_color=None, alpha=1):
        """Plots the shape with interactive radio toggle for elevation/plate/water visualization."""
        if fig is None or ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Create radio button axes
        rax = plt.axes([0.05, 0.7, 0.15, 0.15])
        radio = RadioButtons(rax, ('Elevation', 'Plates', 'Water'))
        
        # Store data needed for visualization modes
        plot_data = {
            'fig': fig,
            'ax': ax,
            'cmap': cmap,
            'edge_color': edge_color,
            'alpha': alpha,
            'polygons': [],
            'face_elevations': [],
            'face_plates': [],
            'face_water': [],
            'cbar_ax': None
        }
        
        # Precompute face data
        for face in self.faces:
            face_verts = face.get_vertices_pos(self.vertices)
            if len(face_verts) >= 3:
                plot_data['polygons'].append(face_verts)
                # Elevation data
                elevs = face.get_vertices_elevation(self.vertices)
                plot_data['face_elevations'].append(np.mean(elevs))
                # Plate data
                plates = [self.vertices[i].plate_id for i in face.v_indices]
                plot_data['face_plates'].append(max(set(plates), key=plates.count))
                # Water data
                water = face.get_vertices_water(self.vertices)
                plot_data['face_water'].append(np.mean(water))
        
        # Initial plot (elevation)
        collection = self._create_collection(plot_data, mode='elevation')
        plot_data['collection'] = collection
        ax.add_collection3d(collection)
        
        # Set axes limits
        all_positions = np.array([v.pos for v in self.vertices])
        max_val = np.max(np.abs(all_positions)) * 1.1 if len(all_positions) > 0 else 1.1
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_aspect('equal')
        
        # Add colorbar
        plot_data['cbar'] = self._add_colorbar(plot_data, ax, mode='elevation')
        
        # Radio button callback
        def on_radio_change(label):
            # Remove old elements
            plot_data['collection'].remove()
            if plot_data['cbar'] is not None:
                plot_data['cbar'].remove()
            if plot_data['cbar_ax'] is not None:
                plot_data['cbar_ax'].remove()
            
            new_collection = self._create_collection(plot_data, mode=label.lower())
            plot_data['collection'] = new_collection
            plot_data['ax'].add_collection3d(new_collection)
            
            # Add new colorbar
            plot_data['cbar'] = self._add_colorbar(plot_data, ax, mode=label.lower())
            
            fig.canvas.draw_idle()
        
        radio.on_clicked(on_radio_change)
        
        return fig, ax, radio

    @time_function
    def _create_collection(self, plot_data, mode='elevation'):
        """Create the appropriate Poly3DCollection based on visualization mode."""
        if mode == 'elevation':
            norm = plt.Normalize(
                vmin=min(plot_data['face_elevations']),
                vmax=max(plot_data['face_elevations'])
            )
            cmap = plt.get_cmap(plot_data['cmap'])
            face_colors = cmap(norm(plot_data['face_elevations']))
        elif mode == 'plates':
            unique_plates = list(set(plot_data['face_plates']))
            plate_cmap = plt.get_cmap('tab20')
            norm = plt.Normalize(vmin=min(unique_plates), vmax=len(unique_plates))
            #face_colors = plate_cmap(norm([unique_plates.index(p) for p in plot_data['face_plates']]))
            face_colors = plate_cmap(norm(plot_data['face_plates']))
        else:
            water_cmap = plt.get_cmap('Blues')
            norm = plt.Normalize(
                vmin=0,
                vmax=max(plot_data['face_water'])
            )
            face_colors = water_cmap(norm(plot_data['face_water']))
        
        return Poly3DCollection(
            plot_data['polygons'],
            facecolors=face_colors,
            linewidths=0.3 if plot_data['edge_color'] else 0,
            edgecolors=plot_data['edge_color'],
            alpha=plot_data['alpha']
        )
    
    @time_function
    def _add_colorbar(self, plot_data, ax, mode='elevation'):
        fig = plot_data['fig']
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        if mode == 'elevation':
            norm = plt.Normalize(
                vmin=min(plot_data['face_elevations']),
                vmax=max(plot_data['face_elevations'])
            )
            cmap = plt.get_cmap(plot_data['cmap'])
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(plot_data['face_elevations'])
            cbar = fig.colorbar(mappable, cax=cbar_ax, label='Elevation')
        elif mode == 'plates':
            unique_plates = sorted(list(set(plot_data['face_plates'])))
            plate_cmap = plt.get_cmap('tab20')
            norm = plt.Normalize(vmin=0, vmax=len(unique_plates))
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=plate_cmap)
            #mappable.set_array([unique_plates.index(p) for p in plot_data['face_plates']])
            mappable.set_array(plot_data['face_plates'])
            cbar = fig.colorbar(mappable, cax=cbar_ax, label='Plate ID')
            cbar.set_ticks(range(len(unique_plates)))
            cbar.set_ticklabels(unique_plates)
        else:  # water
            water_cmap = plt.get_cmap('Blues')
            norm = plt.Normalize(
                vmin=0,
                vmax=max(plot_data['face_water'])
            )
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=water_cmap)
            mappable.set_array(plot_data['face_water'])
            cbar = fig.colorbar(mappable, cax=cbar_ax, label='Water (Teraliters)')
        
        return cbar

    @time_function
    def export_to_obj(self, filename, include_attributes=True):
        """Export the world to an OBJ file format.
        
        Args:
            filename (str): Path to save the OBJ file
            include_attributes (bool): Whether to include plate/elevation/water data as vertex colors
        """
        with open(filename, 'w') as f:
            # Write header
            f.write("# Generated by World Simulator\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            f.write(f"# Faces: {len(self.faces)}\n")
            f.write(f"# Plates: {len(self.plates)}\n\n")
            
            # Write vertices
            for v in self.vertices:
                if include_attributes:
                    # Normalize elevation to 0-1 range for vertex color
                    normalized_elev = (v.elevation - self.minheight) / (self.maxheight - self.minheight)
                    # Add plate ID as another color channel (scaled 0-1)
                    plate_color = (v.plate_id % 20) / 20.0  # Using modulo to keep within reasonable range
                    # Add water as third color channel
                    water_color = min(1.0, v.water_depth / 10.0)  # Scale water depth
                    f.write(f"v {v.pos[0]:.6f} {v.pos[1]:.6f} {v.pos[2]:.6f} {normalized_elev:.3f} {plate_color:.3f} {water_color:.3f}\n")
                else:
                    f.write(f"v {v.pos[0]:.6f} {v.pos[1]:.6f} {v.pos[2]:.6f}\n")
            
            # Write vertex normals (calculated from adjacent faces)
            self._calculate_vertex_normals()
            for v in self.vertices:
                f.write(f"vn {v.normal[0]:.6f} {v.normal[1]:.6f} {v.normal[2]:.6f}\n")
            
            # Write faces (1-based indexing)
            for face in self.faces:
                # OBJ format uses 1-based indices, and can include normals
                face_str = "f"
                for v_idx in face.v_indices:
                    face_str += f" {v_idx+1}//{v_idx+1}"  # vertex index//normal index
                f.write(face_str + "\n")
            
            # Write material information if including attributes
            if include_attributes:
                f.write("\nmtllib world_materials.mtl\n")
                f.write("usemtl world_material\n")

        # Create companion MTL file if including attributes
        if include_attributes:
            self._write_mtl_file(filename.replace('.obj', '.mtl'))

    @time_function
    def _calculate_vertex_normals(self):
        """Calculate vertex normals by averaging adjacent face normals."""
        if not self._neighbor_map_initialized:
            self._initialize_neighbor_map()
        
        # Reset all normals
        for v in self.vertices:
            v.normal = np.zeros(3)
        
        # Calculate face normals and accumulate to vertices
        for face in self.faces:
            if len(face.v_indices) >= 3:
                # Get three vertices to calculate face normal
                v0 = self.vertices[face.v_indices[0]].pos
                v1 = self.vertices[face.v_indices[1]].pos
                v2 = self.vertices[face.v_indices[2]].pos
                
                # Calculate face normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal_length = np.linalg.norm(normal)
                if normal_length > 0:
                    normal /= normal_length
                
                # Accumulate to each vertex
                for v_idx in face.v_indices:
                    self.vertices[v_idx].normal += normal
        
        # Normalize all vertex normals
        for v in self.vertices:
            normal_length = np.linalg.norm(v.normal)
            if normal_length > 0:
                v.normal /= normal_length

    @time_function
    def _write_mtl_file(self, mtl_filename):
        """Write a companion material file that uses vertex colors."""
        with open(mtl_filename, 'w') as f:
            f.write("# Material file for world export\n")
            f.write("newmtl world_material\n")
            f.write("Ka 1.0 1.0 1.0\n")  # Ambient color
            f.write("Kd 1.0 1.0 1.0\n")  # Diffuse color
            f.write("Ks 0.2 0.2 0.2\n")  # Specular color
            f.write("Ns 10.0\n")         # Specular exponent
            f.write("illum 2\n")         # Illumination model
            f.write("d 1.0\n")           # Dissolve (opacity)
            f.write("map_Kd vertex_color.png\n")  # Use vertex colors