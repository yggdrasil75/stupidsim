import ctypes
import queue
import random
#import threading
import time
#from matplotlib.colorbar import Colorbar
from matplotlib.widgets import RadioButtons
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
from collections import deque
#from itertools import permutations
import multiprocessing
import torch


OCEANIC_CRUST_THICKNESS = -5000
CONTINENTAL_CRUST_THICKNESS = 4000
PLATE_TYPE_OCEANIC = 0
PLATE_TYPE_CONTINENTAL = 1


# --- Data Holder Classes (Vertex, Face - unchanged) ---

class Vertex:
    """Represents a 3D vertex."""
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=float)
        self.elevation = 0.0
        self.water = 0.0
        self.plate_id = -1
        self.neighbors = {}
        self.river_size = 0.0

    def normalize(self, radius=1.0):
        """Normalizes the vertex position to be on a sphere of given radius."""
        norm = np.linalg.norm(self.pos)
        if norm > 1e-9: # Avoid division by zero
            self.pos = (self.pos / norm) * radius

    def __repr__(self):
        return f"Vertex({self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f})"

class Face:
    """Represents a face defined by vertex indices."""
    def __init__(self, v_indices):
        if len(v_indices) < 3:
            raise ValueError("Face must have at least 3 vertices")
        self.v_indices = list(v_indices) # Store indices referencing the World's vertex list

    def get_vertices_pos(self, vertex_list):
        """Returns the 3D coordinates of the vertices forming this face."""
        return [vertex_list[i].pos for i in self.v_indices]

    def get_vertices_elevation(self, vertex_list):
        """Returns the elevation values of the vertices forming this face."""
        return [vertex_list[i].elevation for i in self.v_indices]

    def get_vertices_water(self, vertex_list):
        """Returns the water values of the vertices forming this face."""
        return [vertex_list[i].water for i in self.v_indices]
    
    def __repr__(self):
        return f"Face({self.v_indices})"

class Plate:
    """Represents a tectonic plate with velocity and associated vertices."""
    def __init__(self, plate_id):
        self.plate_id = plate_id
        self.velocity = np.random.uniform(-1, 1, 3)  # Angular velocity vector
        self.vertices = set()  # Indices of vertices belonging to this plate
        self.type = random.choice([PLATE_TYPE_OCEANIC, PLATE_TYPE_CONTINENTAL])
        self.base_elevation = (OCEANIC_CRUST_THICKNESS if self.type == PLATE_TYPE_OCEANIC 
                              else CONTINENTAL_CRUST_THICKNESS)
        
    def add_vertex(self, vertex_idx):
        """Add a vertex to this plate."""
        self.vertices.add(vertex_idx)
        
    def get_boundary_vertices(self, world):
        """Return vertices on the boundary of this plate."""
        boundary = []
        for v_idx in self.vertices:
            neighbors = world._find_vertex_neighbors(v_idx)
            neighbor_plates = {world.vertices[n].plate_id for n in neighbors}
            if len(neighbor_plates) > 1:  # Boundary vertex
                boundary.append(v_idx)
        return boundary

# --- Multiprocessing functions ---

def plate_grow_worker_process(plate_id, work_queue_indices, plate_id_array, unassigned_indices_queue, assignment_lock, vertices_neighbors_getter):
    """
    Worker function designed for multiprocessing.
    Reads shared plate_id_array, attempts assignments, puts successes
    into a result queue.
    """
    assigned_in_this_run = [] # Track (vertex_idx, plate_id)

    # In a real scenario, efficiently getting neighbors for many vertices might
    # require passing more precomputed data or using a shared structure if feasible.
    # Here, using a passed-in getter function/object.
    find_neighbors = vertices_neighbors_getter

    # Process items from the input queue for this iteration
    local_queue = deque(work_queue_indices) # Process local copy

    while True:
        try:
            current_assigned_idx = local_queue.popleft()
        except IndexError:
            break # Local queue for this iteration empty

        # --- Perform neighbor finding and weighted selection ---
        # (Simplified - Adapt your detailed logic here)
        # Read plate IDs from shared array (no lock needed for read usually)
        # Needs access to neighbor info (passed via vertices_neighbors_getter)

        neighbors = find_neighbors(current_assigned_idx)
        potential_unassigned_neighbors = []

        # Check shared array for unassigned status
        plate_ids_vals = plate_id_array.get_obj() # Get access to array values
        for n_idx in neighbors:
             # LOCKLESS READ - might be slightly stale but checked authoritatively later
             if plate_ids_vals[n_idx] == -1:
                 potential_unassigned_neighbors.append(n_idx)

        if not potential_unassigned_neighbors:
            continue

        # --- Weighted selection (simplified example) ---
        if potential_unassigned_neighbors:
            # Replace with your actual weighted logic
            selected_neighbor_idx = random.choice(potential_unassigned_neighbors)
        else:
            continue

        # --- Critical Section: Attempt Assignment ---
        if selected_neighbor_idx != -1:
            with assignment_lock:
                # *** Authoritative Check & Assignment ***
                # Re-check using locked access to the shared array
                current_plate_id_val = plate_id_array[selected_neighbor_idx]
                if current_plate_id_val == -1:
                    plate_id_array[selected_neighbor_idx] = plate_id
                    # Signal success by putting index onto result queue
                    # We don't directly modify the global 'unassigned' set here
                    assigned_in_this_run.append(selected_neighbor_idx)

    # Put all successful assignments from this worker onto the shared result queue
    if assigned_in_this_run:
        unassigned_indices_queue.put((plate_id, assigned_in_this_run))

def get_neighbors_func(idx, precomputed_neighbors):
    # Example: if neighbors are precomputed in a dict/list
    return precomputed_neighbors[idx]


# --- World Hierarchy ---

class World:
    """Base class for holding 3D shape data."""
    def __init__(self):
        self.vertices = [] # List of Vertex objects
        self.faces = []    # List of Face objects
        self.plates = {}
        self._neighbor_map_initialized = False
        self.sea_level = 0.0

    def add_vertex(self, vertex):
        """Adds a vertex and returns its index."""
        self.vertices.append(vertex)
        return len(self.vertices) - 1

    def add_face(self, face):
        """Adds a face."""
        self.faces.append(face)

    def create_world(self, radius=1.0):
        """Abstract method to populate vertices and faces."""
        raise NotImplementedError("Subclasses must implement create_world()")

    def subdivide(self, radius=1.0):
        """Abstract method to subdivide the faces."""
        raise NotImplementedError("Subclasses must implement subdivide()")

    def _initialize_neighbor_map(self):
        """Initialize the neighbor map with weights based on vertex distances."""
        if self._neighbor_map_initialized:
            return
            
        print("Initializing neighbor map...")
        
        # First pass: find all direct neighbors through shared faces
        neighbor_sets = [set() for _ in range(len(self.vertices))]
        for face in self.faces:
            for i in range(len(face.v_indices)):
                v1 = face.v_indices[i]
                for j in range(i+1, len(face.v_indices)):
                    v2 = face.v_indices[j]
                    neighbor_sets[v1].add(v2)
                    neighbor_sets[v2].add(v1)
        
        # Second pass: calculate weights based on distances
        for v_idx, neighbors in enumerate(neighbor_sets):
            for neighbor_idx in neighbors:
                # Calculate distance between vertices
                dist = np.linalg.norm(self.vertices[v_idx].pos - self.vertices[neighbor_idx].pos)
                # Weight is inverse of distance (closer neighbors have higher weight)
                weight = 1.0 / (dist + 1e-9)  # Small epsilon to avoid division by zero
                self.vertices[v_idx].neighbors[neighbor_idx] = weight
        
        self._neighbor_map_initialized = True
        print("Neighbor map initialized")

    def _find_vertex_neighbors(self, vertex_idx):
        """Get neighbors with weights for a vertex."""
        if not self._neighbor_map_initialized:
            self._initialize_neighbor_map()
        return self.vertices[vertex_idx].neighbors

    def genElevations(self, num_plates=7, min_height=0.0, max_height=1.0):
        """Generate elevation values with plate tectonics and continental/oceanic plates."""
        if not self.vertices:
            return
            
        # Reset all elevations to zero first
        for vertex in self.vertices:
            vertex.elevation = 0.0
            vertex.plate_id = -1
            
        self._create_plates(num_plates)
        self._assign_base_elevations()  # Set base elevations based on plate type
        self.minheight = min_height
        self.maxheight = max_height
        self._calculate_boundary_elevations(min_height, max_height)
        self._smooth_elevations(iterations=3)
        self._add_variations()
        self._simulate_water() 

    #### plates and elevation

    def _assign_base_elevations(self):
        """Assign base elevations based on plate type."""
        for plate in self.plates.values():
            for v_idx in plate.vertices:
                # Add some random variation to the base elevation
                variation = np.random.uniform(-0.1, 0.1)
                self.vertices[v_idx].elevation = plate.base_elevation + variation
    
    def _create_plates(self, num_plates):
        """Create plates and assign vertices to them."""
        self.plates = {}
        
        plate_types = []
        # Create plate objects
        for _ in range(num_plates):
            # Bias towards alternating types
            if len(plate_types) > 0 and plate_types[-1] == PLATE_TYPE_CONTINENTAL:
                next_type = PLATE_TYPE_OCEANIC
            elif len(plate_types) > 0 and plate_types[-1] == PLATE_TYPE_OCEANIC:
                next_type = PLATE_TYPE_CONTINENTAL
            else:
                next_type = random.choice([PLATE_TYPE_CONTINENTAL, PLATE_TYPE_OCEANIC])
            plate_types.append(next_type)
            
        for i in range(num_plates // 3):  # Adjust fraction to control randomness
            if random.random() < 0.5:  # 50% chance to swap a plate's type
                plate_types[i] = 1 - plate_types[i]  # Flips the type

        # Create plates with assigned types
        for plate_id in range(num_plates):
            self.plates[plate_id] = Plate(plate_id)
            self.plates[plate_id].type = plate_types[plate_id]
            
        # Assign vertices to plates
        self._assign_tectonic_plates(num_plates)

    def _get_neighbors_for_worker(self, idx):
        # Assuming self.vertices or some neighbor structure exists
        return self._find_vertex_neighbors(idx)
    
    def _plate_grow_worker(self, plate_id, plate_queue, vertices, unassigned, plates, assignment_lock):
        """
        Worker function for a single plate's growth *in one iteration*.
        Processes its assigned queue until empty.
        """
        my_plate = plates[plate_id]
        # num_vertices_total = len(vertices) # Not used currently

        while True:
            current_assigned_idx = -1
            try:
                # Get an *assigned* vertex from this plate's expansion frontier
                current_assigned_idx = plate_queue.popleft()
            except IndexError:
                # Queue for this iteration is empty for this plate
                break # Exit the loop for this thread for this iteration

            # Find neighbors (read-only operation)
            neighbors = self._find_vertex_neighbors(current_assigned_idx)

            # --- Find potentially unassigned neighbors ---
            # Fast check outside lock (might be slightly stale)
            potential_unassigned_neighbors_indices = []
            # Read unassigned (potentially needs lock if not thread-safe read)
            # Assuming standard Python set reads are generally safe enough here
            for n_idx in neighbors:
                # Check vertex plate_id first (atomic read, less likely to be stale than set)
                if vertices[n_idx].plate_id == -1:
                     # Optional: Double check against the shared 'unassigned' set
                     # if n_idx in unassigned: # This read might need the lock
                     potential_unassigned_neighbors_indices.append(n_idx)

            if not potential_unassigned_neighbors_indices:
                continue # No potential neighbors found from this vertex

            # --- Weighted random selection ---
            # Filter further and calculate weights
            valid_neighbors_for_assignment = []
            weights = []
            for n_idx in potential_unassigned_neighbors_indices:
                neighbor_plate_ids = [self.vertices[nn_idx].plate_id for nn_idx in self._find_vertex_neighbors(n_idx) if self.vertices[nn_idx].plate_id != -1]
                opposite_type_count = 0
                for pid in neighbor_plate_ids:
                    if pid in self.plates and self.plates[pid].type != self.plates[plate_id].type:
                        opposite_type_count += 1

                # The critical check happens later under lock, but we calculate
                # weights based on the state we see *now*.
                n_neighbors = self._find_vertex_neighbors(n_idx)

                # Weight: count neighbors already in *this* plate + bias
                count = sum(1 for nn_idx in n_neighbors
                            if nn_idx != current_assigned_idx and vertices[nn_idx].plate_id == plate_id)

                # Add base weight (e.g., 1) + count + small random factor (optional)
                # Ensure non-zero weight to allow assignment even with count=0
                weight = (opposite_type_count + 1) * count + 1.0 + random.uniform(0, 0.5) # Example weighting
                weights.append(weight)
                valid_neighbors_for_assignment.append(n_idx)

            if not valid_neighbors_for_assignment:
                continue

            # --- Select one neighbor based on weights ---
            selected_neighbor_idx = -1
            weights = np.array(weights, dtype=float)
            total_weight = weights.sum()

            if total_weight > 1e-9 and len(valid_neighbors_for_assignment) > 0: # Check > 0 and length
                try:
                    weights /= total_weight # Normalize
                    selected_neighbor_idx = np.random.choice(valid_neighbors_for_assignment, p=weights)
                except ValueError as e:
                    # print(f"Plate {plate_id}: Weight error {e}. Weights={weights}, Sum={total_weight}. Falling back to random.")
                    # Fallback to uniform random choice among valid neighbors
                    selected_neighbor_idx = random.choice(valid_neighbors_for_assignment)
            elif valid_neighbors_for_assignment:
                 # Fallback if weights sum to zero (or close) but neighbors exist
                 selected_neighbor_idx = random.choice(valid_neighbors_for_assignment)
            else:
                 # Should not happen based on earlier checks, but safety first
                 continue


            if selected_neighbor_idx == -1:
                 continue # No neighbor selected

            # --- Critical Section: Attempt Assignment ---
            # assigned_successfully = False # Not needed with new logic
            with assignment_lock:
                # *** Authoritative Check ***
                # Is the selected neighbor STILL unassigned? Check the definitive source.
                if selected_neighbor_idx in unassigned:
                    # Assign the selected neighbor
                    vertices[selected_neighbor_idx].plate_id = plate_id
                    my_plate.add_vertex(selected_neighbor_idx) # Assumes Plate.add_vertex is thread-safe or trivial
                    unassigned.remove(selected_neighbor_idx)

                    # NOTE: We DO NOT add the newly assigned vertex back to the queue *within the worker*.
                    # The queue for this worker is only for processing the initial frontier
                    # assigned to it *for this iteration*. The main loop will repopulate
                    # queues for the *next* iteration based on the new global state.
                    # assigned_successfully = True # Not needed
                    # print(f"Plate {plate_id} assigned {selected_neighbor_idx}. Remaining: {len(unassigned)}")

            # No need to put current_assigned_idx back in the queue if assignment failed.
            # Just move on to the next item in this worker's queue for this iteration.
            # The iterative approach handles conflicts implicitly in the next round.

    def _assign_tectonic_plates(self, num_plates, parallel_threshold_percent=10.0, max_iterations=100):
        num_vertices = len(self.vertices)
        print("Initializing plate assignment (Multiprocessing)...")

        # --- Initialize Shared State ---
        plate_id_array = multiprocessing.Array(ctypes.c_int, num_vertices)
        for i in range(num_vertices):
            plate_id_array[i] = -1  # Initialize all as unassigned

        # Master 'unassigned' set (managed only by main process)
        unassigned = set(range(num_vertices))

        # Lock for synchronizing writes to plate_id_array
        assignment_lock = multiprocessing.Lock()

        # Queue for workers to report successful assignments back to main process
        manager = multiprocessing.Manager()
        results_queue = manager.Queue()

        # Plates dictionary (managed only by main process)
        self.plates = {plate_id: Plate(plate_id) for plate_id in range(num_plates)}

        # --- Assign Starting Points ---
        plate_starts = np.random.choice(list(unassigned), size=num_plates, replace=False)
        with assignment_lock:
            for plate_id, start_idx in enumerate(plate_starts):
                if start_idx in unassigned:
                    plate_id_array[start_idx] = plate_id
                    self.plates[plate_id].add_vertex(start_idx)
                    unassigned.remove(start_idx)

        print(f"Assigned {num_plates} starting points. Remaining unassigned: {len(unassigned)}")

        # --- Iterative Multi-Processing Growth ---
        iteration = 0
        parallel_threshold_count = int(num_vertices * (parallel_threshold_percent / 100.0))

        neighbor_getter = self._get_neighbors_for_worker

        while len(unassigned) > parallel_threshold_count and iteration < max_iterations:
            iteration += 1
            unassigned_count_start_iter = len(unassigned)
            #print(f"\n--- Iteration {iteration} --- Starting | Unassigned: {unassigned_count_start_iter}/{num_vertices}")

            # 1. Identify Frontier
            plate_frontiers = {plate_id: [] for plate_id in range(num_plates)}
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

            #print(f"Iteration {iteration}: Found {total_frontier_size} frontier points across {active_plate_count} active plates.")

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
                            self.plates[res_plate_id].add_vertex(v_idx)
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

    def _fix_non_contiguous_vertices(self):
        """
        Identify and reassign vertices that are not contiguous with their plate's main body.
        """
        print("Checking for non-contiguous vertices...")
        for plate_id, plate in self.plates.items():
            if not plate.vertices:
                continue

            # Find the largest contiguous region in the plate
            main_region = self._find_largest_contiguous_region(plate_id)
            all_vertices = plate.vertices.copy()

            # Identify orphaned vertices (not in the main region)
            orphaned_vertices = all_vertices - main_region

            if orphaned_vertices:
                print(f"Plate {plate_id} has {len(orphaned_vertices)} non-contiguous vertices. Reassigning...")

                for v_idx in orphaned_vertices:
                    # Find the most common plate among neighbors
                    neighbor_plates = []
                    neighbors = self._find_vertex_neighbors(v_idx)
                    for n_idx in neighbors:
                        neighbor_plate = self.vertices[n_idx].plate_id
                        if neighbor_plate != -1 and neighbor_plate != plate_id:
                            neighbor_plates.append(neighbor_plate)

                    if neighbor_plates:
                        # Assign to the most common neighboring plate
                        new_plate_id = max(set(neighbor_plates), key=neighbor_plates.count)
                        self.vertices[v_idx].plate_id = new_plate_id
                        self.plates[new_plate_id].add_vertex(v_idx)
                        plate.vertices.remove(v_idx)
                    else:
                        # No neighboring plates found, assign randomly
                        new_plate_id = random.choice(list(self.plates.keys()))
                        self.vertices[v_idx].plate_id = new_plate_id
                        self.plates[new_plate_id].add_vertex(v_idx)
                        plate.vertices.remove(v_idx)

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
    
    def _assign_remaining_vertices_safe(self, remaining_unassigned_indices):
        """
        Assign remaining unassigned vertices sequentially (BFS from orphan).
        Operates on the provided list and modifies self.vertices/self.plates.
        Relies on vertex.plate_id, not the global `unassigned` set.
        """
        print(f"Assigning {len(remaining_unassigned_indices)} remaining vertices sequentially using BFS...")
        assigned_count_in_fallback = 0
        still_unassigned_after_bfs = []
        processed_indices = set() # Track which of the input list we've handled

        # Make multiple passes if necessary, as assigning one orphan might
        # make a previously isolated orphan reachable.
        pass_num = 0
        indices_to_process = list(remaining_unassigned_indices)

        while indices_to_process:
             pass_num += 1
             print(f"Fallback Pass {pass_num}: Processing {len(indices_to_process)} indices...")
             next_indices_to_process = []
             assigned_in_pass = 0

             queue = deque()
             visited_bfs = set()

             for v_idx in indices_to_process:
                 # Check if already processed or assigned by another orphan's BFS in this pass
                 if v_idx in processed_indices or self.vertices[v_idx].plate_id != -1:
                      continue

                 processed_indices.add(v_idx) # Mark as processed for this fallback run

                 # Start BFS from this orphan
                 queue.clear()
                 visited_bfs.clear()
                 queue.append(v_idx)
                 visited_bfs.add(v_idx)
                 found_plate = -1
                 closest_assigned_neighbor_idx = -1

                 while queue:
                      current_bfs = queue.popleft()
                      neighbors = self._find_vertex_neighbors(current_bfs)

                      # Check neighbors first for an assigned plate
                      for n_idx in neighbors:
                           neighbor_plate = self.vertices[n_idx].plate_id
                           if neighbor_plate != -1:
                                found_plate = neighbor_plate
                                closest_assigned_neighbor_idx = n_idx # Record who we found it from
                                # print(f"Orphan {v_idx} found plate {found_plate} via neighbor {n_idx}")
                                break # Found the nearest plate

                      if found_plate != -1:
                           break # Exit BFS loop for this orphan

                      # If no assigned neighbor found yet, expand BFS to unassigned neighbors
                      for n_idx in neighbors:
                           if n_idx not in visited_bfs and self.vertices[n_idx].plate_id == -1:
                                visited_bfs.add(n_idx)
                                queue.append(n_idx)

                 # Assign the original orphan v_idx if a plate was found
                 if found_plate != -1:
                      self.vertices[v_idx].plate_id = found_plate
                      if found_plate in self.plates:
                           self.plates[found_plate].add_vertex(v_idx)
                      else:
                           print(f"Warning: Found plate {found_plate} for orphan {v_idx} but plate not in self.plates dict!")
                      assigned_count_in_fallback += 1
                      assigned_in_pass += 1
                 else:
                      # BFS completed without finding an assigned neighbor this pass
                      next_indices_to_process.append(v_idx)
                      # print(f"Vertex {v_idx} remains unassigned after BFS pass {pass_num}.")

             print(f"Fallback Pass {pass_num}: Assigned {assigned_in_pass} vertices.")
             if not next_indices_to_process:
                 print(f"Fallback Pass {pass_num}: No remaining unassigned after BFS.")
                 break # All processed or assigned
             if not assigned_in_pass and next_indices_to_process:
                 # If a pass assigned nothing but vertices remain, they are truly isolated
                 print(f"Fallback Pass {pass_num}: Stalled. Moving to random assignment for remaining.")
                 still_unassigned_after_bfs = next_indices_to_process
                 break

             indices_to_process = next_indices_to_process # Prepare for next pass

             if pass_num > 10: # Safety break for infinite loops
                 print("Warning: Exceeded fallback pass limit.")
                 still_unassigned_after_bfs = indices_to_process
                 break


        # Force-assign any vertices that are still unassigned (truly isolated components)
        if still_unassigned_after_bfs:
             print(f"Warning: {len(still_unassigned_after_bfs)} vertices remain unassigned after BFS. Force assigning randomly.")
             available_plate_ids = list(self.plates.keys())
             if not available_plate_ids:
                  print("Error: No plates available to assign remaining vertices!")
             else:
                  assigned_randomly = 0
                  for v_idx in still_unassigned_after_bfs:
                      # Double check it wasn't assigned somehow
                      if self.vertices[v_idx].plate_id == -1:
                           chosen_plate_id = random.choice(available_plate_ids)
                           self.vertices[v_idx].plate_id = chosen_plate_id
                           self.plates[chosen_plate_id].add_vertex(v_idx)
                           assigned_count_in_fallback += 1
                           assigned_randomly += 1
                  print(f"Force assigned {assigned_randomly} vertices randomly.")


        print(f"Fallback phase finished. Total vertices assigned in fallback: {assigned_count_in_fallback}")

    def _calculate_boundary_elevations(self, min_height, max_height):
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
                elevation = CONTINENTAL_CRUST_THICKNESS + (max_height - CONTINENTAL_CRUST_THICKNESS) * (avg_movement + 1)/2
            else:
                # Oceanic plates create trenches or islands
                if avg_movement > 0:  # Converging
                    elevation = OCEANIC_CRUST_THICKNESS - (OCEANIC_CRUST_THICKNESS - min_height) * avg_movement
                else:  # Diverging (mid-ocean ridges)
                    elevation = OCEANIC_CRUST_THICKNESS + (0.5 - OCEANIC_CRUST_THICKNESS) * abs(avg_movement)
            
            vertex.elevation = elevation
        print('boundary effects calculated')

    def _build_neighbor_tensors(self, device):
        """
        Precomputes neighbor information in a tensor format suitable for PyTorch.
        (Implementation is the same as before)
        """
        num_vertices = len(self.vertices)
        neighbor_lists = [list(self._find_vertex_neighbors(i)) for i in range(num_vertices)]
        max_neighbors = max(len(neighbors) for neighbors in neighbor_lists) if neighbor_lists else 0
        if max_neighbors == 0 and num_vertices > 0 : # Handle case with isolated vertices
             print("Warning: Some vertices may have no neighbors.")
             max_neighbors = 1 # Avoid zero-sized tensor dim if possible

        neighbor_indices = torch.full((num_vertices, max_neighbors), 0, # Pad with 0, mask handles it
                                      dtype=torch.long, device=device)
        neighbor_mask = torch.zeros((num_vertices, max_neighbors),
                                    dtype=torch.bool, device=device)

        for i, neighbors in enumerate(neighbor_lists):
            if neighbors:
                num_n = len(neighbors)
                # Ensure indices are within bounds if padding with 0
                safe_neighbors = [n for n in neighbors if 0 <= n < num_vertices]
                if len(safe_neighbors) < num_n:
                    print(f"Warning: Vertex {i} had invalid neighbor indices removed.")

                num_n = len(safe_neighbors) # Update count
                if num_n > 0:
                    neighbor_indices[i, :num_n] = torch.tensor(safe_neighbors, dtype=torch.long, device=device)
                    neighbor_mask[i, :num_n] = True

        return neighbor_indices, neighbor_mask, max_neighbors
  
    def _add_variations(self, iterations=5, device=None, subdivisions=1): # Added subdivisions param if used
            """
            Add natural elevation variations using PyTorch for GPU acceleration.
            Allows for both increases (hills/peaks) and decreases (valleys).
            """
            print("--- Starting PyTorch Elevation Variation (with Valley Formation) ---")
            start_total_time = time.time()

            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            num_vertices = len(self.vertices)
            if num_vertices == 0:
                print("No vertices to process.")
                return

            # --- 1. Data Preparation (CPU -> GPU Tensors) ---
            print("Preparing data for GPU...")
            prep_start_time = time.time()

            # Basic vertex data
            positions = torch.tensor([v.pos for v in self.vertices], dtype=torch.float32, device=device)
            elevations = torch.tensor([v.elevation for v in self.vertices], dtype=torch.float32, device=device)
            plate_ids = torch.tensor([v.plate_id for v in self.vertices], dtype=torch.long, device=device)

            # --- Apply initial 5% noise to all elevations ---
            noise_scale = 0.05  # 5% noise
            noise = (torch.rand_like(elevations) * 2.0 - 1.0) * noise_scale * elevations
            elevations += noise
            
            # Identify continental vertices
            is_continental = torch.zeros(num_vertices, dtype=torch.bool, device=device)
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
            neighbor_indices, neighbor_mask, max_neighbors = self._build_neighbor_tensors(device)

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

                plate_v_indices = torch.tensor(valid_plate_v_indices, dtype=torch.long, device=device)

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
                        plate_peak_elevs[plate_id] = CONTINENTAL_CRUST_THICKNESS
                else:
                    plate_peak_elevs[plate_id] = CONTINENTAL_CRUST_THICKNESS # Default if no peaks calculated


            if not all_peaks_indices:
                print("Warning: No peaks found on any continental plate.")
                has_peaks = False
                peaks_pos = torch.empty((0, 3), dtype=torch.float32, device=device) # Ensure it's defined
            else:
                peaks_indices_tensor = torch.tensor(list(set(all_peaks_indices)), dtype=torch.long, device=device)
                # Final check for valid indices in peaks_indices_tensor
                valid_peak_indices = peaks_indices_tensor[(peaks_indices_tensor >= 0) & (peaks_indices_tensor < num_vertices)]
                if len(valid_peak_indices) < len(peaks_indices_tensor):
                    print(f"Warning: Removed {len(peaks_indices_tensor) - len(valid_peak_indices)} invalid peak indices.")
                if len(valid_peak_indices) == 0:
                    print("Warning: All peak indices were invalid.")
                    has_peaks = False
                    peaks_pos = torch.empty((0, 3), dtype=torch.float32, device=device)
                else:
                    peaks_pos = positions[valid_peak_indices]
                    # peaks_plate_id = plate_ids[valid_peak_indices] # We don't use this directly later
                    has_peaks = True


            # Map plate_id to its max peak elevation
            max_peak_elev_map = torch.full_like(elevations, CONTINENTAL_CRUST_THICKNESS)
            for i in range(num_vertices):
                p_id = plate_ids[i].item()
                if p_id in plate_peak_elevs:
                    max_peak_elev_map[i] = plate_peak_elevs[p_id]


            # Simulation parameters
            # Use subdivisions parameter if it exists, otherwise default to 1
            #effective_subdivisions = getattr(self, 'subdivisions', 1) if subdivisions is None else subdivisions
            
            total_iterations = self.subdivisions * iterations # Match original logic if subdivisions exist

            # Ensure minheight/maxheight are available
            min_elev_val = getattr(self, 'minheight', torch.min(elevations).item() - 1.0) # Provide fallback
            max_elev_val = getattr(self, 'maxheight', torch.max(elevations).item() + 1.0) # Provide fallback
            elev_range = max_elev_val - min_elev_val if max_elev_val > min_elev_val else 1.0
            significant_change_threshold = 0.001 * elev_range

            # --- Tunable Parameters ---
            peak_force_scale = 0.5    # Weight of the upward push from peaks
            neighbor_push_scale = 0.1  # Max random factor for pushing towards higher neighbors
            neighbor_pull_scale = 0.05 # Max random factor for pulling away from slightly lower neighbors
            slump_scale = 0.15        # Factor for downward force when higher than average neighbor (NEW)
            noise_scale = 0.3         # Increased weight for random up/down noise
            damping_factor = 0.5      # Overall damping rate per iteration

            print(f"Data preparation took {time.time() - prep_start_time:.2f}s")
            print(f"Running {total_iterations} iterations. Min/Max Elev: {min_elev_val:.2f}/{max_elev_val:.2f}")
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
                new_elev_clamped = torch.clamp(new_elev, min=min_elev_val, max=max_elev_val)

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

    def _simulate_water(self, iterations=5):
        """Simulate water distribution based on elevation.
        Water is measured in teraliters (TL) for easier tracking of lakes and rivers."""
        print("Simulating water distribution...")
        
        # First pass: identify ocean basins and initial water placement
        for vertex in self.vertices:
            if vertex.elevation <= self.sea_level:
                # Ocean gets full water (simplified)
                vertex.water = (self.sea_level - vertex.elevation) * 10.0  # In TL
            else:
                vertex.water = 0.0
                
        # Second pass: simulate rainfall and river flow
        for _ in range(iterations):
            new_water = [0.0] * len(self.vertices)
            
            for i, vertex in enumerate(self.vertices):
                if vertex.elevation > self.sea_level:
                    # Land can receive rainfall (0.01-0.05 TL per iteration)
                    rainfall = random.uniform(0.01, 0.05)
                    new_water[i] += rainfall
                    
                    # Find lowest neighbor for runoff
                    neighbors = self._find_vertex_neighbors(i)
                    if neighbors:
                        lowest_neighbor = min(neighbors, 
                                            key=lambda n: self.vertices[n].elevation)
                        if self.vertices[lowest_neighbor].elevation < vertex.elevation:
                            # Move some water downhill (20% of current water or max 0.5 TL)
                            flow_amount = min(vertex.water * 0.2, 0.5)
                            new_water[i] -= flow_amount
                            new_water[lowest_neighbor] += flow_amount
        
            # Apply changes
            for i in range(len(self.vertices)):
                self.vertices[i].water = max(0.0, self.vertices[i].water + new_water[i])
        
        # Generate rivers based on water flow accumulation
        self._generate_rivers()
        
        print("Water simulation complete")

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
        for i, flow in enumerate(flow_accumulation):
            if flow >= min_flow:
                self.vertices[i].is_river = True
                # Scale river size based on flow (could be used for rendering)
                self.vertices[i].river_size = min(3, int(flow / min_flow))
        
        # Connect river segments to form continuous rivers
        self._connect_river_segments()
        
        print(f"Generated {sum(1 for v in self.vertices if v.is_river)} river segments")

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

    def _find_lakes(self, min_size=3.0):
        """Identify lakes (standing water bodies above sea level).
        min_size: minimum water volume in TL to be considered a lake."""
        lakes = []
        visited = set()
        
        for i, vertex in enumerate(self.vertices):
            if (vertex.elevation > self.sea_level and 
                vertex.water >= min_size and 
                i not in visited):
                
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

    def _get_midpoint_vertex(self, v1_idx, v2_idx, midpoint_cache, next_level_vertices, radius):
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
        new_v.normalize(radius)

        # Add to the *accumulating* vertex list for the next level
        new_idx = len(next_level_vertices)
        next_level_vertices.append(new_v)
        midpoint_cache[key] = new_idx # Cache the index in the accumulating list
        return new_idx
        
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

    def _compute_face_centroid(self, v_indices, vertices, radius):
        face_vertex_positions = np.array([vertices[idx].pos for idx in v_indices], dtype=float)
        centroid_pos = np.mean(face_vertex_positions, axis=0)
        norm = np.linalg.norm(centroid_pos)
        normalized_centroid_pos = centroid_pos * (radius / norm)
        return Vertex(*normalized_centroid_pos)
    
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
            norm = plt.Normalize(vmin=0, vmax=len(unique_plates))
            face_colors = plate_cmap(norm([unique_plates.index(p)] for p in plot_data['face_plates']))
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
            norm = plt.Normalize(vmin=0, vmax=len(unique_plates)-1)
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=plate_cmap)
            mappable.set_array([unique_plates.index(p) for p in plot_data['face_plates']])
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

class Icosahedron(World):
    """Represents an Icosahedron."""
    def __init__(self):
        super().__init__()

    def create_world(self, radius=1.0):
        """Creates the vertices and faces for a unit icosahedron."""
        self.vertices = []
        self.faces = []
        phi = (1 + math.sqrt(5)) / 2
        
        # Note: Using slightly different vertex definition for easier face indexing
        v_data = [
            (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
            (0, -1,  phi), (0,  1,  phi), (0, -1, -phi), (0,  1, -phi),
            ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1)
        ]
        
        for p in v_data:
            v = Vertex(*p)
            v.normalize(radius)
            self.add_vertex(v)

        # Add faces (triangles)
        f_indices = [
            # 5 faces around point 0
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            # 5 adjacent faces
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            # 5 faces around point 3
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            # 5 adjacent faces
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
        ]
        for f in f_indices:
            self.add_face(Face(f))

    def subdivide(self, radius=1.0, level = 3):
        self.subdivisions = level
        for li in range(level):
            print(f"Subdividing Icosahedron with {len(self.faces)} faces...")
            midpoint_cache = {}
            new_faces = []
            next_level_vertices = list(self.vertices) # Start with existing vertices

            for face in self.faces:
                if len(face.v_indices) != 3:
                    print(f"Warning: Skipping subdivision of non-triangular face: {face}")
                    new_faces.append(face)
                    continue

                v1_idx, v2_idx, v3_idx = face.v_indices

                # Get or create midpoint vertices for each edge, adding to next_level_vertices
                m12_idx = self._get_midpoint_vertex(v1_idx, v2_idx, midpoint_cache, next_level_vertices, radius)
                m23_idx = self._get_midpoint_vertex(v2_idx, v3_idx, midpoint_cache, next_level_vertices, radius)
                m31_idx = self._get_midpoint_vertex(v3_idx, v1_idx, midpoint_cache, next_level_vertices, radius)

                # Create the four new faces using original and midpoint vertex indices
                # Indices must reference the combined list (original + new midpoints)
                new_faces.append(Face((v1_idx, m12_idx, m31_idx)))
                new_faces.append(Face((v2_idx, m23_idx, m12_idx)))
                new_faces.append(Face((v3_idx, m31_idx, m23_idx)))
                new_faces.append(Face((m12_idx, m23_idx, m31_idx))) # Center face

            self.vertices = next_level_vertices
            self.faces = new_faces
            print(f"Icosahedron subdivision complete. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")

class TruncatedIcosahedron(World):
    """Represents a Truncated Icosahedron (soccer ball)."""
    def __init__(self):
        super().__init__()

    def create_world(self, radius=1.0):
        """Creates the vertices and faces for a unit truncated icosahedron."""

        C0 = (1 + np.sqrt(5)) / 4
        C1 =  (1 + np.sqrt(5)) / 2
        C2 = (5 + np.sqrt(5)) / 4
        C3 = (2 + np.sqrt(5)) / 2
        C4 =  3 * (1 + np.sqrt(5)) / 4
        
        truncated_v_data = [
            ( 0.5,  0.0,   C4),
            ( 0.5,  0.0,  -C4),
            (-0.5,  0.0,   C4),
            (-0.5,  0.0,  -C4),
            (  C4,  0.5,  0.0),
            (  C4, -0.5,  0.0),
            ( -C4,  0.5,  0.0),
            ( -C4, -0.5,  0.0),
            ( 0.0,   C4,  0.5),
            ( 0.0,   C4, -0.5),
            ( 0.0,  -C4,  0.5),
            ( 0.0,  -C4, -0.5),
            ( 1.0,   C0,   C3),
            ( 1.0,   C0,  -C3),
            ( 1.0,  -C0,   C3),
            ( 1.0,  -C0,  -C3),
            (-1.0,   C0,   C3),
            (-1.0,   C0,  -C3),
            (-1.0,  -C0,   C3),
            (-1.0,  -C0,  -C3),
            (  C3,  1.0,   C0),
            (  C3,  1.0,  -C0),
            (  C3, -1.0,   C0),
            (  C3, -1.0,  -C0),
            ( -C3,  1.0,   C0),
            ( -C3,  1.0,  -C0),
            ( -C3, -1.0,   C0),
            ( -C3, -1.0,  -C0),
            (  C0,   C3,  1.0),
            (  C0,   C3, -1.0),
            (  C0,  -C3,  1.0),
            (  C0,  -C3, -1.0),
            ( -C0,   C3,  1.0),
            ( -C0,   C3, -1.0),
            ( -C0,  -C3,  1.0),
            ( -C0,  -C3, -1.0),
            ( 0.5,   C1,   C2),
            ( 0.5,   C1,  -C2),
            ( 0.5,  -C1,   C2),
            ( 0.5,  -C1,  -C2),
            (-0.5,   C1,   C2),
            (-0.5,   C1,  -C2),
            (-0.5,  -C1,   C2),
            (-0.5,  -C1,  -C2),
            (  C2,  0.5,   C1),
            (  C2,  0.5,  -C1),
            (  C2, -0.5,   C1),
            (  C2, -0.5,  -C1),
            ( -C2,  0.5,   C1),
            ( -C2,  0.5,  -C1),
            ( -C2, -0.5,   C1),
            ( -C2, -0.5,  -C1),
            (  C1,   C2,  0.5),
            (  C1,   C2, -0.5),
            (  C1,  -C2,  0.5),
            (  C1,  -C2, -0.5),
            ( -C1,   C2,  0.5),
            ( -C1,   C2, -0.5),
            ( -C1,  -C2,  0.5),
            ( -C1,  -C2, -0.5)
        ]

        for p in truncated_v_data:
            v = Vertex(*p)
            v.normalize(radius)
            self.add_vertex(v)

        faces = [
            (  0,  2, 18, 42, 38, 14 ),
            (  1,  3, 17, 41, 37, 13 ),
            (  2,  0, 12, 36, 40, 16 ),
            (  3,  1, 15, 39, 43, 19 ),
            (  4,  5, 23, 47, 45, 21 ),
            (  5,  4, 20, 44, 46, 22 ),
            (  6,  7, 26, 50, 48, 24 ),
            (  7,  6, 25, 49, 51, 27 ),
            (  8,  9, 33, 57, 56, 32 ),
            (  9,  8, 28, 52, 53, 29 ),
            ( 10, 11, 31, 55, 54, 30 ),
            ( 11, 10, 34, 58, 59, 35 ),
            ( 12, 44, 20, 52, 28, 36 ),
            ( 13, 37, 29, 53, 21, 45 ),
            ( 14, 38, 30, 54, 22, 46 ),
            ( 15, 47, 23, 55, 31, 39 ),
            ( 16, 40, 32, 56, 24, 48 ),
            ( 17, 49, 25, 57, 33, 41 ),
            ( 18, 50, 26, 58, 34, 42 ),
            ( 19, 43, 35, 59, 27, 51 ),
            (  0, 14, 46, 44, 12 ),
            (  1, 13, 45, 47, 15 ),
            (  2, 16, 48, 50, 18 ),
            (  3, 19, 51, 49, 17 ),
            (  4, 21, 53, 52, 20 ),
            (  5, 22, 54, 55, 23 ),
            (  6, 24, 56, 57, 25 ),
            (  7, 27, 59, 58, 26 ),
            (  8, 32, 40, 36, 28 ),
            (  9, 29, 37, 41, 33 ),
            ( 10, 30, 38, 42, 34 ),
            ( 11, 35, 43, 39, 31 )

        ]

        for f_idx_tuple in faces:
            # Ensure indices are integers if they aren't already
            int_indices = tuple(int(i) for i in f_idx_tuple)
            self.add_face(Face(int_indices))

    def subdivide(self, radius=1.0, level=3):
        self.subdivisions = level
        for li in range(level):
            midpoint_cache = {}
            new_faces = []
            next_vertices = list(self.vertices)
            
            for face in self.faces:
                v_indices = face.v_indices
                n = len(v_indices)
                
                # Skip subdivision for certain levels based on face type
                if (n == 3 and li % 2 != 0) or (n == 5 and li % 4 != 0) or (n == 6 and not li):
                    new_faces.append(face)
                    continue
                    
                # Get all midpoints for this face
                mid_indices = []
                for i in range(n):
                    v1 = v_indices[i]
                    v2 = v_indices[(i+1)%n]
                    mid_idx = self._get_midpoint_vertex(v1, v2, midpoint_cache, next_vertices, radius)
                    mid_indices.append(mid_idx)
                
                # Create central face
                central_face = Face(tuple(mid_indices))
                new_faces.append(central_face)
                
                # Create surrounding faces
                for i in range(n):
                    v_idx = v_indices[i]
                    prev_mid = mid_indices[i-1] if i > 0 else mid_indices[-1]
                    curr_mid = mid_indices[i]
                    
                    if n == 3:  # Triangle becomes 3 quads (but your original creates 4 tris)
                        new_faces.append(Face((v_idx, curr_mid, prev_mid)))
                    else:  # N-gon becomes N triangles + 1 central N-gon
                        new_faces.append(Face((v_idx, curr_mid, prev_mid)))
            
            self.vertices = next_vertices
            self.faces = new_faces
            print(f"Subdivision level {li+1} complete. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")

class truncatedTetrahedron(World):
    def __init__(self):
        super().__init__()

    def create_world(self, radius=1):
        c0 = np.sqrt(2)/4
        c1 = 3 * np.sqrt(2) / 4

        v_data = [
            (c0, -c0, c1),
            (c0, c0, -c1),
            (-c0, c0, c1),
            (-c0, -c0, -c1),
            (c1, -c0, c0),
            (c1, c0, -c0),
            (-c1, c0, c0),
            (-c1, -c0, c0),
            (c0, -c1, c0),
            (c0, c1, -c0),
            (-c0, c1, c0),
            (-c0, -c1, -c0)            
        ]

        for p in v_data:
            v = Vertex(*p)
            v.normalize(radius)
            self.add_vertex(v)

        faces = [
            (0, 4, 5, 9, 10, 2),
            (1, 5, 4, 8, 11, 3),
            (2, 6, 7, 11, 8, 0),
            (3, 7, 6, 10, 9, 1),
            (0, 8, 4),
            (1, 9, 5),
            (2, 10, 6),
            (3, 11, 7)
        ]

        for f_idx_tuple in faces:
            # Ensure indices are integers if they aren't already
            int_indices = tuple(int(i) for i in f_idx_tuple)
            self.add_face(Face(int_indices))


    def subdivide(self, radius=1.0, level=3):
        self.subdivisions = level
        for li in range(level):
            midpoint_cache = {}
            new_faces = []
            next_vertices = list(self.vertices)
            
            for face in self.faces:
                v_indices = face.v_indices
                n = len(v_indices)
                
                # # Skip subdivision for squares every other level
                # if n == 4 and li % 2 != 0:
                #     new_faces.append(face)
                #     continue
                    
                # Get all edge midpoints for this face
                mid_indices = []
                for i in range(n):
                    v1 = v_indices[i]
                    v2 = v_indices[(i+1)%n]
                    mid_idx = self._get_midpoint_vertex(v1, v2, midpoint_cache, next_vertices, radius)
                    mid_indices.append(mid_idx)
                
                # Get face centroid (for Catmull-Clark-like subdivision)
                centroid = self._compute_face_centroid(v_indices, next_vertices, radius)
                centroid_idx = len(next_vertices)
                next_vertices.append(centroid)
                
                # Create new faces
                for i in range(n):
                    v_idx = v_indices[i]
                    curr_mid = mid_indices[i]
                    prev_mid = mid_indices[i-1] if i > 0 else mid_indices[-1]
                    
                    if n == 4:
                        # For squares: Create a new quad (maintains shape)
                        next_mid = mid_indices[(i+1)%n]
                        new_faces.append(Face((v_idx, curr_mid, centroid_idx, prev_mid)))
                    else:
                        # For decagons/hexagons: Create a new n-gon face
                        new_faces.append(Face((v_idx, curr_mid, centroid_idx, prev_mid)))
            
            self.vertices = next_vertices
            self.faces = new_faces
            print(f"Subdivision level {li+1}: {len(self.vertices)} vertices, {len(self.faces)} faces")

class TruncatedIcosidodecahedron(World):
    def __init__(self):
        super().__init__()

    def create_world(self, radius=1):
        C0 = (3 + np.sqrt(5)) / 4
        C1 = (1 + np.sqrt(5)) / 2
        C2 = (5 + np.sqrt(5)) / 4
        C3 = (2 + np.sqrt(5)) / 2
        C4 = 3 * (1 + np.sqrt(5)) / 4
        C5 = (3 + np.sqrt(5)) / 2
        C6 = (5 + 3 * np.sqrt(5)) / 4
        C7 = (4 + np.sqrt(5)) / 2
        C8 = (7 + 3 * np.sqrt(5)) / 4
        C9 = (3 + 2 * np.sqrt(5)) / 2

        v_data = [
            ( 0.5,  0.5,   C9),
            ( 0.5,  0.5,  -C9),
            ( 0.5, -0.5,   C9),
            ( 0.5, -0.5,  -C9),
            (-0.5,  0.5,   C9),
            (-0.5,  0.5,  -C9),
            (-0.5, -0.5,   C9),
            (-0.5, -0.5,  -C9),
            (  C9,  0.5,  0.5),
            (  C9,  0.5, -0.5),
            (  C9, -0.5,  0.5),
            (  C9, -0.5, -0.5),
            ( -C9,  0.5,  0.5),
            ( -C9,  0.5, -0.5),
            ( -C9, -0.5,  0.5),
            ( -C9, -0.5, -0.5),
            ( 0.5,   C9,  0.5),
            ( 0.5,   C9, -0.5),
            ( 0.5,  -C9,  0.5),
            ( 0.5,  -C9, -0.5),
            (-0.5,   C9,  0.5),
            (-0.5,   C9, -0.5),
            (-0.5,  -C9,  0.5),
            (-0.5,  -C9, -0.5),
            ( 1.0,   C0,   C8),
            ( 1.0,   C0,  -C8),
            ( 1.0,  -C0,   C8),
            ( 1.0,  -C0,  -C8),
            (-1.0,   C0,   C8),
            (-1.0,   C0,  -C8),
            (-1.0,  -C0,   C8),
            (-1.0,  -C0,  -C8),
            (  C8,  1.0,   C0),
            (  C8,  1.0,  -C0),
            (  C8, -1.0,   C0),
            (  C8, -1.0,  -C0),
            ( -C8,  1.0,   C0),
            ( -C8,  1.0,  -C0),
            ( -C8, -1.0,   C0),
            ( -C8, -1.0,  -C0),
            (  C0,   C8,  1.0),
            (  C0,   C8, -1.0),
            (  C0,  -C8,  1.0),
            (  C0,  -C8, -1.0),
            ( -C0,   C8,  1.0),
            ( -C0,   C8, -1.0),
            ( -C0,  -C8,  1.0),
            ( -C0,  -C8, -1.0),
            ( 0.5,   C3,   C7),
            ( 0.5,   C3,  -C7),
            ( 0.5,  -C3,   C7),
            ( 0.5,  -C3,  -C7),
            (-0.5,   C3,   C7),
            (-0.5,   C3,  -C7),
            (-0.5,  -C3,   C7),
            (-0.5,  -C3,  -C7),
            (  C7,  0.5,   C3),
            (  C7,  0.5,  -C3),
            (  C7, -0.5,   C3),
            (  C7, -0.5,  -C3),
            ( -C7,  0.5,   C3),
            ( -C7,  0.5,  -C3),
            ( -C7, -0.5,   C3),
            ( -C7, -0.5,  -C3),
            (  C3,   C7,  0.5),
            (  C3,   C7, -0.5),
            (  C3,  -C7,  0.5),
            (  C3,  -C7, -0.5),
            ( -C3,   C7,  0.5),
            ( -C3,   C7, -0.5),
            ( -C3,  -C7,  0.5),
            ( -C3,  -C7, -0.5),
            (  C2,   C1,   C6),
            (  C2,   C1,  -C6),
            (  C2,  -C1,   C6),
            (  C2,  -C1,  -C6),
            ( -C2,   C1,   C6),
            ( -C2,   C1,  -C6),
            ( -C2,  -C1,   C6),
            ( -C2,  -C1,  -C6),
            (  C6,   C2,   C1),
            (  C6,   C2,  -C1),
            (  C6,  -C2,   C1),
            (  C6,  -C2,  -C1),
            ( -C6,   C2,   C1),
            ( -C6,   C2,  -C1),
            ( -C6,  -C2,   C1),
            ( -C6,  -C2,  -C1),
            (  C1,   C6,   C2),
            (  C1,   C6,  -C2),
            (  C1,  -C6,   C2),
            (  C1,  -C6,  -C2),
            ( -C1,   C6,   C2),
            ( -C1,   C6,  -C2),
            ( -C1,  -C6,   C2),
            ( -C1,  -C6,  -C2),
            (  C0,   C4,   C5),
            (  C0,   C4,  -C5),
            (  C0,  -C4,   C5),
            (  C0,  -C4,  -C5),
            ( -C0,   C4,   C5),
            ( -C0,   C4,  -C5),
            ( -C0,  -C4,   C5),
            ( -C0,  -C4,  -C5),
            (  C5,   C0,   C4),
            (  C5,   C0,  -C4),
            (  C5,  -C0,   C4),
            (  C5,  -C0,  -C4),
            ( -C5,   C0,   C4),
            ( -C5,   C0,  -C4),
            ( -C5,  -C0,   C4),
            ( -C5,  -C0,  -C4),
            (  C4,   C5,   C0),
            (  C4,   C5,  -C0),
            (  C4,  -C5,   C0),
            (  C4,  -C5,  -C0),
            ( -C4,   C5,   C0),
            ( -C4,   C5,  -C0),
            ( -C4,  -C5,   C0),
            ( -C4,  -C5,  -C0)
        ]

        for p in v_data:
            v = Vertex(*p)
            v.normalize(radius)
            self.add_vertex(v)

        faces = [
            (   0,   2,  26,  74, 106,  58,  56, 104,  72,  24 ),
            (   1,  25,  73, 105,  57,  59, 107,  75,  27,   3 ),
            (   4,  28,  76, 108,  60,  62, 110,  78,  30,   6 ),
            (   5,   7,  31,  79, 111,  63,  61, 109,  77,  29 ),
            (   8,   9,  33,  81, 113,  65,  64, 112,  80,  32 ),
            (  10,  34,  82, 114,  66,  67, 115,  83,  35,  11 ),
            (  12,  36,  84, 116,  68,  69, 117,  85,  37,  13 ),
            (  14,  15,  39,  87, 119,  71,  70, 118,  86,  38 ),
            (  16,  20,  44,  92, 100,  52,  48,  96,  88,  40 ),
            (  17,  41,  89,  97,  49,  53, 101,  93,  45,  21 ),
            (  18,  42,  90,  98,  50,  54, 102,  94,  46,  22 ),
            (  19,  23,  47,  95, 103,  55,  51,  99,  91,  43 ),
            (   0,  24,  48,  52,  28,   4 ),
            (   1,   5,  29,  53,  49,  25 ),
            (   2,   6,  30,  54,  50,  26 ),
            (   3,  27,  51,  55,  31,   7 ),
            (   8,  32,  56,  58,  34,  10 ),
            (   9,  11,  35,  59,  57,  33 ),
            (  12,  14,  38,  62,  60,  36 ),
            (  13,  37,  61,  63,  39,  15 ),
            (  16,  40,  64,  65,  41,  17 ),
            (  18,  19,  43,  67,  66,  42 ),
            (  20,  21,  45,  69,  68,  44 ),
            (  22,  46,  70,  71,  47,  23 ),
            (  72, 104,  80, 112,  88,  96 ),
            (  73,  97,  89, 113,  81, 105 ),
            (  74,  98,  90, 114,  82, 106 ),
            (  75, 107,  83, 115,  91,  99 ),
            (  76, 100,  92, 116,  84, 108 ),
            (  77, 109,  85, 117,  93, 101 ),
            (  78, 110,  86, 118,  94, 102 ),
            (  79, 103,  95, 119,  87, 111 ),
            (   0,   4,   6,   2 ),
            (   1,   3,   7,   5 ),
            (   8,  10,  11,   9 ),
            (  12,  13,  15,  14 ),
            (  16,  17,  21,  20 ),
            (  18,  22,  23,  19 ),
            (  24,  72,  96,  48 ),
            (  25,  49,  97,  73 ),
            (  26,  50,  98,  74 ),
            (  27,  75,  99,  51 ),
            (  28,  52, 100,  76 ),
            (  29,  77, 101,  53 ),
            (  30,  78, 102,  54 ),
            (  31,  55, 103,  79 ),
            (  32,  80, 104,  56 ),
            (  33,  57, 105,  81 ),
            (  34,  58, 106,  82 ),
            (  35,  83, 107,  59 ),
            (  36,  60, 108,  84 ),
            (  37,  85, 109,  61 ),
            (  38,  86, 110,  62 ),
            (  39,  63, 111,  87 ),
            (  40,  88, 112,  64 ),
            (  41,  65, 113,  89 ),
            (  42,  66, 114,  90 ),
            (  43,  91, 115,  67 ),
            (  44,  68, 116,  92 ),
            (  45,  93, 117,  69 ),
            (  46,  94, 118,  70 ),
            (  47,  71, 119,  95 )
        ]

        for f_idx_tuple in faces:
            # Ensure indices are integers if they aren't already
            int_indices = tuple(int(i) for i in f_idx_tuple)
            self.add_face(Face(int_indices))
    
    def subdivide(self, radius=1.0, level=3):
        self.subdivisions = level
        for li in range(level):
            midpoint_cache = {}
            new_faces = []
            next_vertices = list(self.vertices)
            
            for face in self.faces:
                v_indices = face.v_indices
                n = len(v_indices)
                
                # # Skip subdivision for squares every other level
                # if n == 4 and li % 2 != 0:
                #     new_faces.append(face)
                #     continue
                    
                # Get all edge midpoints for this face
                mid_indices = []
                for i in range(n):
                    v1 = v_indices[i]
                    v2 = v_indices[(i+1)%n]
                    mid_idx = self._get_midpoint_vertex(v1, v2, midpoint_cache, next_vertices, radius)
                    mid_indices.append(mid_idx)
                
                # Get face centroid (for Catmull-Clark-like subdivision)
                centroid = self._compute_face_centroid(v_indices, next_vertices, radius)
                centroid_idx = len(next_vertices)
                next_vertices.append(centroid)
                
                # Create new faces
                for i in range(n):
                    v_idx = v_indices[i]
                    curr_mid = mid_indices[i]
                    prev_mid = mid_indices[i-1] if i > 0 else mid_indices[-1]
                    
                    if n == 4:
                        # For squares: Create a new quad (maintains shape)
                        next_mid = mid_indices[(i+1)%n]
                        new_faces.append(Face((v_idx, curr_mid, centroid_idx, prev_mid)))
                    else:
                        # For decagons/hexagons: Create a new n-gon face
                        new_faces.append(Face((v_idx, curr_mid, centroid_idx, prev_mid)))
            
            self.vertices = next_vertices
            self.faces = new_faces
            print(f"Subdivision level {li+1}: {len(self.vertices)} vertices, {len(self.faces)} faces")

class Cube(World):
    """Represents a Cube."""
    def __init__(self):
        super().__init__()

    def create_world(self, radius=1.0):
        """Creates the vertices and faces for a unit cube."""
        self.vertices = []
        self.faces = []
        
        # Create the 8 vertices of a cube
        v_data = [
            (-1, -1, -1),  # 0
            ( 1, -1, -1),  # 1
            ( 1,  1, -1),  # 2
            (-1,  1, -1),  # 3
            (-1, -1,  1),  # 4
            ( 1, -1,  1),  # 5
            ( 1,  1,  1),  # 6
            (-1,  1,  1)   # 7
        ]
        
        for p in v_data:
            v = Vertex(*p)
            v.normalize(radius)  # Normalize to make it a spherical cube
            self.add_vertex(v)

        # Add the 6 faces (quadrilaterals)
        f_indices = [
            (0, 1, 2, 3),  # Bottom face
            (4, 5, 6, 7),  # Top face
            (0, 1, 5, 4),  # Front face
            (1, 2, 6, 5),  # Right face
            (2, 3, 7, 6),  # Back face
            (3, 0, 4, 7)   # Left face
        ]
        
        for f in f_indices:
            self.add_face(Face(f))

    def subdivide(self, radius=1.0, level=3):
        self.subdivisions = level
        """Subdivides faces into quadrilaterals using a Catmull-Clark like approach."""
        
        # Initial normalization check (if create_world didn't normalize)
        # for v in self.vertices:
        #     v.normalize(radius)
            
        for li in range(level):
            print(f"Starting Subdivision level {li+1}...")
            midpoint_cache = {} # Cache edge midpoints for this level
            new_faces = []
            # Important: Make a *copy* of the vertex list at the start of the level
            # because we will be appending new vertices to it during processing.
            current_level_vertices = list(self.vertices)
            
            # We will build the next vertex list incrementally
            next_vertices = list(current_level_vertices) 

            processed_faces = 0
            for face in self.faces:
                v_indices = face.v_indices
                n = len(v_indices)

                # We only handle quads in this specific subdivision logic
                # If you start with triangles or other shapes, you'd need different rules
                if n != 4:
                    # Keep non-quad faces as they are (or implement different subdivision)
                    print(f"Warning: Skipping non-quad face: {face}")
                    new_faces.append(face)
                    continue
                    
                # --- 1. Calculate Face Point ---
                face_center_pos = np.zeros(3, dtype=float)
                for v_idx in v_indices:
                    face_center_pos += current_level_vertices[v_idx].pos
                face_center_pos /= n
                
                center_v = Vertex(*face_center_pos)
                center_v.normalize(radius)
                
                # Add face point vertex to the list for the *next* level
                center_idx = len(next_vertices)
                next_vertices.append(center_v)

                # --- 2. Calculate Edge Midpoints ---
                mid_indices = []
                for i in range(n):
                    v1_idx = v_indices[i]
                    v2_idx = v_indices[(i + 1) % n] # Handle wrap-around
                    # Use the _get_midpoint_vertex helper which uses the cache
                    # Pass next_vertices so new midpoints are added correctly
                    mid_idx = self._get_midpoint_vertex(v1_idx, v2_idx, midpoint_cache, next_vertices, radius)
                    mid_indices.append(mid_idx)
                
                # --- 3. Create New Quadrilateral Faces ---
                # Connect original vertex -> edge midpoint -> face center -> previous edge midpoint
                for i in range(n):
                    v_orig_idx = v_indices[i] # Original vertex index
                    mid_curr_idx = mid_indices[i] # Midpoint of edge starting at v_orig_idx
                    # Midpoint of edge ending at v_orig_idx (handle wrap-around)
                    mid_prev_idx = mid_indices[i - 1] # Python's negative indexing handles wrap-around nicely
                    
                    # Create the new quad face
                    # Order: Original Corner -> Edge Midpoint -> Face Center -> Previous Edge Midpoint
                    new_quad = Face((v_orig_idx, mid_curr_idx, center_idx, mid_prev_idx))
                    new_faces.append(new_quad)

                processed_faces += 1


            # Update the world state for the next iteration or final result
            self.vertices = next_vertices
            self.faces = new_faces
            print(f"Subdivision level {li+1} complete. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")

# --- Main Execution ---
if __name__ == "__main__":
    shape_type = "cube" # Choose "icosahedron" or "truncated"
    num_subdivisions = 7     # Adjust level of detail
    sphere_radius = 1.0
    plates = 15
    elevationmin = -15000
    elevationmax = 15000


    if shape_type == "icosahedron":
        shape = Icosahedron()
        shape.create_world(radius=sphere_radius)
        print(f"\nInitial Icosahedron created.")
    elif shape_type == "truncated":
        shape = TruncatedIcosahedron()
        shape.create_world(radius=sphere_radius)
        print(f"\nInitial Truncated Icosahedron created.")
    elif shape_type == "cube":
        shape = Cube()
        shape.create_world(radius=sphere_radius)
        print(f"\nInitial cube created.")
    elif shape_type == "tetrahedron":
        shape = truncatedTetrahedron()
        shape.create_world(radius=sphere_radius)
        print(f"\nInitial Truncated Icosahedron created.")
    elif shape_type == "TruncatedIcosidodecahedron":
        shape = TruncatedIcosidodecahedron()
        shape.create_world(radius=sphere_radius)
        print(f"\nInitial Truncated Icosadodecahedron created.")
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


    # Subdivide the shape
    shape.subdivide(sphere_radius, num_subdivisions) # Call the correct subdivide

    shape.genElevations(plates, elevationmin, elevationmax)

    # # --- Plotting ---
    # fig = plt.figure(figsize=(9, 9))
    # ax = fig.add_subplot(111, projection='3d')

    # Plot the final shape
    #shape.plot(ax, cmap='terrain', edge_color='darkgreen', alpha=0.9)
    fig, ax, radio = shape.plot()

    ax.set_title(f'Sphere Approx. ({shape_type.capitalize()} Subdivided {num_subdivisions} Times)')
    plt.show()
    