import ctypes
import queue
import random
import copy
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

from sympy import acos, asin, atan2, cos, pi, sin, sqrt
import torch
from functools import lru_cache


OCEANIC_CRUST_THICKNESS = -5000
CONTINENTAL_CRUST_THICKNESS = 4000
PLATE_TYPE_OCEANIC = 0
PLATE_TYPE_CONTINENTAL = 1
SHAPE = "cube"
SUBDIVISIONS = 4
SPHERE_SIZE = 1.0
PLATES = 15
ELEVATION_MIN = -15000
ELEVATION_MAX = 11000
EQUATORIAL_RADIUS = 6378137.0
POLAR_RADIUS = 6356752.3 #probably not gonna use this for a while.

# --- generic utilities ---

def cart_to_sphere(p):
    x, y, z = p
    lon = atan2(y, x)
    lat = math.asin(z)
    return lat, lon

# --- Data Holder Classes (Vertex, Face - unchanged) ---

class Vertex:
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=float)
        self.elevation = 0.0
        self.water = 0.0
        self.plate_id = -1
        self.neighbors = {}
        self.river_size = 0.0

    def normalize(self, radius=1.0):
        norm = np.linalg.norm(self.pos)
        if norm > 1e-9:
            self.pos = (self.pos / norm) * radius

    def __repr__(self):
        return f"Vertex({self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f})"
    
    #def get_nearest_neighbor():

class Face:
    def __init__(self, v_indices):
        self.area = -1
        if len(v_indices) < 3:
            raise ValueError("Face must have at least 3 vertices")
        self.v_indices = list(v_indices)

    def get_vertices_pos(self, vertex_list) -> list:
        return [vertex_list[i].pos for i in self.v_indices]

    def get_vertices_elevation(self, vertex_list) -> list:
        return [vertex_list[i].elevation for i in self.v_indices]

    def get_vertices_water(self, vertex_list) -> list:
        return [vertex_list[i].water for i in self.v_indices]
    
    def __repr__(self) -> str:
        return f"Face({self.v_indices})"

    def calculate_area(self, vertex_list) -> float:
        if self.area > 0:
            return self.area
            
        points = [vertex_list[i].pos for i in self.v_indices]
        
        if len(points) == 4:  # Quad case
            # Split quad into two triangles and sum their areas
            tri1 = [points[0], points[1], points[2]]
            tri2 = [points[0], points[2], points[3]]
            
            def spherical_triangle_area(tri):
                # Convert to unit vectors
                a, b, c = [p/np.linalg.norm(p) for p in tri]
                
                # Calculate angles using dot products
                alpha = acos(np.dot(np.cross(a, b), np.cross(a, c)) / 
                    (np.linalg.norm(np.cross(a, b)) * np.linalg.norm(np.cross(a, c))))
                beta = acos(np.dot(np.cross(b, a), np.cross(b, c)) / 
                    (np.linalg.norm(np.cross(b, a)) * np.linalg.norm(np.cross(b, c))))
                gamma = acos(np.dot(np.cross(c, a), np.cross(c, b)) / 
                        (np.linalg.norm(np.cross(c, a)) * np.linalg.norm(np.cross(c, b))))
                
                # Spherical excess
                excess = alpha + beta + gamma - pi
                return excess * EQUATORIAL_RADIUS**2
                
        self.area = spherical_triangle_area(tri1) + spherical_triangle_area(tri2)
        return self.area

    
class Plate:
    def __init__(self, plate_id, is_minor=False):
        self.plate_id = plate_id
        self.velocity = np.random.uniform(low=-1, high=1, size=3)
        self.vertices = set()
        self.type = random.choice([PLATE_TYPE_OCEANIC, PLATE_TYPE_CONTINENTAL])
        self.is_minor = is_minor
        self.growth_rate = 0.5 if is_minor else 1.0
        self.base_elevation = (OCEANIC_CRUST_THICKNESS if self.type == PLATE_TYPE_OCEANIC 
                              else CONTINENTAL_CRUST_THICKNESS)
        
    def resetVertices(self):
        self.vertices.clear()

    def add_vertex(self, vertex_idx, vertices):
        vertices[vertex_idx].plate_id = self.plate_id
        self.vertices.add(vertex_idx)
        
    def get_boundary_vertices(self, world) -> list:
        boundary = []
        for v_idx in self.vertices:
            neighbors = world._find_vertex_neighbors(v_idx)
            neighbor_plates = {world.vertices[n].plate_id for n in neighbors}
            if len(neighbor_plates) > 1: 
                boundary.append(v_idx)
        return boundary

# --- Multiprocessing functions ---

def plate_grow_worker_process(plate_id, work_queue_indices, plate_id_array, unassigned_indices_queue, assignment_lock, vertices_neighbors_getter):
    assigned_in_this_run = [] 

    find_neighbors = vertices_neighbors_getter

    local_queue = deque(work_queue_indices)

    while True:
        try:
            current_assigned_idx = local_queue.popleft()
        except IndexError:
            break 

        neighbors = find_neighbors(current_assigned_idx)
        potential_unassigned_neighbors = []

        plate_ids_vals = plate_id_array.get_obj() 
        for n_idx in neighbors:
             if plate_ids_vals[n_idx] == -1:
                 potential_unassigned_neighbors.append(n_idx)

        if not potential_unassigned_neighbors:
            continue

        if potential_unassigned_neighbors:
            selected_neighbor_idx = random.choice(potential_unassigned_neighbors)
        else:
            continue

        if selected_neighbor_idx != -1:
            with assignment_lock:
                current_plate_id_val = plate_id_array[selected_neighbor_idx]
                if current_plate_id_val == -1:
                    plate_id_array[selected_neighbor_idx] = plate_id
                    assigned_in_this_run.append(selected_neighbor_idx)

    if assigned_in_this_run:
        unassigned_indices_queue.put((plate_id, assigned_in_this_run))

@staticmethod
def batch_calculate_areas(faces, vertex_list, device='cuda'):
    """
    Calculate areas for all faces in parallel using PyTorch on GPU
    
    Args:
        faces: List of Face objects
        vertex_list: List of vertices with pos attributes
        device: 'cuda' or 'cpu'
        
    Returns:
        Tensor of areas for all faces
    """
    # Extract all vertex positions
    all_pos = torch.tensor([v.pos for v in vertex_list], dtype=torch.float32, device=device)
    
    # Prepare face data
    max_verts = max(len(face.v_indices) for face in faces)
    face_indices = []
    is_tri = []
    
    for face in faces:
        idxs = face.v_indices
        # For quads, we'll process as two triangles (0,1,2 and 0,2,3)
        if len(idxs) == 4:
            face_indices.append(idxs[:3])  # First triangle
            face_indices.append([idxs[0], idxs[2], idxs[3]])  # Second triangle
            is_tri.extend([True, True])
        else:
            face_indices.append(idxs)
            is_tri.append(True)
    
    # Convert to tensor
    face_indices = torch.tensor(face_indices, dtype=torch.long, device=device)
    
    # Get vertex positions for all faces
    face_verts = all_pos[face_indices]  # Shape: (n_faces, 3, 3)
    
    # Normalize to unit vectors
    face_verts = face_verts / torch.norm(face_verts, dim=2, keepdim=True)
    
    # Calculate cross products
    a, b, c = face_verts[:,0], face_verts[:,1], face_verts[:,2]
    cross_ab = torch.cross(a, b)
    cross_ac = torch.cross(a, c)
    cross_bc = torch.cross(b, c)
    
    # Calculate angles
    alpha = torch.acos(torch.sum(cross_ab * cross_ac, dim=1) / 
            (torch.norm(cross_ab, dim=1) * torch.norm(cross_ac, dim=1)))
    beta = torch.acos(torch.sum(-cross_ab * cross_bc, dim=1) / 
            (torch.norm(cross_ab, dim=1) * torch.norm(cross_bc, dim=1)))
    gamma = torch.acos(torch.sum(cross_ac * cross_bc, dim=1) / 
            (torch.norm(cross_ac, dim=1) * torch.norm(cross_bc, dim=1)))
    
    # Spherical excess
    excess = alpha + beta + gamma - pi
    areas = excess * (EQUATORIAL_RADIUS ** 2)
    
    # Combine areas for quads (first two entries belong to first quad, etc.)
    is_tri = torch.tensor(is_tri, device=device)
    quad_areas = torch.zeros(len(faces), device=device)
    tri_areas = torch.where(is_tri, areas, torch.zeros_like(areas))
    
    # This part handles combining triangle areas back into quads
    # We need to track which triangles belong to which original face
    face_idx = 0
    out_idx = 0
    combined_areas = []
    
    for face in faces:
        if face.is_triangle:
            combined_areas.append(areas[out_idx])
            out_idx += 1
        else:
            combined_areas.append(areas[out_idx] + areas[out_idx+1])
            out_idx += 2
    
    return torch.tensor(combined_areas, device=device)
    
# --- World Hierarchy ---

class World:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.plates: dict[int, Plate] = {}
        self._neighbor_map_initialized = False
        self.sea_level = 0.0
        self.minheight = -15000
        self.maxheight  = 15000
        self.platecount = 15

    def add_vertex(self, vertex):
        self.vertices.append(vertex)
        return len(self.vertices) - 1

    def add_face(self, face):
        self.faces.append(face)

    def create_world(self, radius=1.0, plates = 15, min_height = -15000, max_height = 15000):
        self.platecount = plates
        self.elevationMin = min_height
        self.elevationMax = max_height
        self._create_world(radius=1.0)

    def _create_world(self, radius = 1.0):
        raise NotImplementedError("Subclasses must implement _create_world()")
    
    def _subdivide_selected(self, faces_to_subdivide, radius=1.0, level=3):
        raise NotImplementedError("Subclasses must implement _subdivide_selected(faces_to_subdivide, radius, level)")

    def subdivide(self, radius=1.0, levels=3):
        self.subdivisions = levels
        for level in range(levels):
            print(f"Starting Subdivision level {level+1}...")
            if np.floor(levels / 2) == level:
                self._create_plates()
                self._assign_tectonic_plates()
            self._subdivide()
            print(f"Subdivision level {level+1} complete. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")
            self._neighbor_map_initialized = False
            self._fix_non_contiguous_vertices()
        self.fixPlates()

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

    def _subdivide(self):
        raise NotImplementedError("Subclasses must implement _subdivide()")

    #### plates and elevation

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
    
    def _create_plates(self):
        self.plates = {}
        
        total_plates = self.platecount
        minor_plate_count = max(1, int(total_plates * random.uniform(0.2, 0.3)))
        minor_plate_ids = set(random.sample(range(total_plates), minor_plate_count))
        plate_types = []
        for _ in range(self.platecount):
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
            self.plates[plate_id] = Plate(plate_id)
            self.plates[plate_id].type = plate_types[plate_id]
            
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
        self.plates = {plate_id: Plate(plate_id) for plate_id in range(self.platecount)}

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

    def _fix_non_contiguous_vertices(self, min_plate_size=5, minor_plate_ratio=0.2):
        min_plate_size *= min_plate_size * self.subdivisions
        """
        Identify and handle non-contiguous vertices, potentially creating minor plates.
        
        Args:
            min_plate_size: Minimum number of vertices to consider creating a new minor plate
            minor_plate_ratio: Ratio of main plate size to consider creating a minor plate
        """
        print("Checking for non-contiguous vertices...")
        
        # First pass: Identify all contiguous regions in all plates
        all_regions = []
        plate_regions = {}  # {plate_id: [region1, region2, ...]}
        
        for plate_id, plate in self.plates.items():
            if not plate.vertices:
                continue
                
            regions = self._find_all_contiguous_regions(plate_id)
            plate_regions[plate_id] = regions
            all_regions.extend([(plate_id, region) for region in regions])
        
        # Second pass: Process regions and potentially create minor plates
        new_plates = {}
        next_plate_id = max(self.plates.keys()) + 1 if self.plates else 0
        
        for plate_id, regions in plate_regions.items():
            if len(regions) <= 1:
                continue  # Only one region - nothing to fix
                
            # Sort regions by size (descending)
            regions.sort(key=lambda r: len(r), reverse=True)
            main_region = regions[0]
            main_region_size = len(main_region)
            
            for region in regions[1:]:
                region_size = len(region)
                
                # Decision: Merge to neighbor or create new minor plate?
                if (region_size >= min_plate_size and 
                    region_size >= main_region_size * minor_plate_ratio):
                    # Significant region - make it a new minor plate
                    new_plate_id = next_plate_id
                    next_plate_id += 1
                    
                    # Determine plate type (same as parent or opposite)
                    parent_type = self.plates[plate_id].type
                    if random.random() < 0.7:  # 70% chance same type
                        new_type = parent_type
                    else:
                        new_type = 1 - parent_type
                    
                    # Create new minor plate
                    new_plate = Plate(new_plate_id)
                    new_plate.type = new_type
                    new_plate.is_minor = True
                    
                    # Assign vertices
                    for v_idx in region:
                        self.vertices[v_idx].plate_id = new_plate_id
                        new_plate.add_vertex(v_idx, self.vertices)
                        self.plates[plate_id].vertices.discard(v_idx)
                    
                    new_plates[new_plate_id] = new_plate
                    print(f"Created minor plate {new_plate_id} (type {'continental' if new_type == PLATE_TYPE_CONTINENTAL else 'oceanic'}) with {region_size} vertices")
                    
                else:
                    # Small region - merge to neighboring plate
                    neighbor_plate = self._find_best_neighbor_plate(region)
                    if neighbor_plate is not None and neighbor_plate != plate_id:
                        # Merge to neighbor
                        for v_idx in region:
                            self.vertices[v_idx].plate_id = neighbor_plate
                            self.plates[neighbor_plate].add_vertex(v_idx, self.vertices)
                            self.plates[plate_id].vertices.discard(v_idx)
                        #print(f"Merged {len(region)} vertices from plate {plate_id} to plate {neighbor_plate}")
                    else:
                        # No better neighbor found, leave with original plate
                        pass
        
        # Add any new minor plates to main plates dictionary
        self.plates.update(new_plates)
        
        # Update plate count if we created new plates
        if new_plates:
            self.platecount = len(self.plates)
            print(f"Total plates now: {self.platecount} ({len(new_plates)} new minor plates)")

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
                           self.plates[found_plate].add_vertex(v_idx, self.vertices)
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
                           self.plates[chosen_plate_id].add_vertex(v_idx, self.vertices)
                           assigned_count_in_fallback += 1
                           assigned_randomly += 1
                  print(f"Force assigned {assigned_randomly} vertices randomly.")


        print(f"Fallback phase finished. Total vertices assigned in fallback: {assigned_count_in_fallback}")

    def _assign_base_elevations(self):
        for plate in self.plates.values():
            for v_idx in plate.vertices:
                variation = np.random.uniform(-0.1, 0.1)
                self.vertices[v_idx].elevation = plate.base_elevation + (variation * plate.base_elevation)

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

    def _add_variations(self, iterations=5, device=None):
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
            noise = (torch.rand_like(elevations) * 2.0 - 1.0) * noise_scale * (self.maxheight - self.minheight) #elevations
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
                        plate_peak_elevs[plate_id] = self.maxheight
                else:
                    plate_peak_elevs[plate_id] = self.maxheight # Default if no peaks calculated


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

    def simulate_water(self, iterations=5):
        """Simulate water distribution based on elevation.
        Uses water depth in meters and accounts for face areas.
        Water depth affects effective elevation."""
        print("Simulating water distribution...")
        
        # First pass: identify ocean basins and initial water placement
        for vidx, vertex in enumerate(self.vertices):
            # Get the average area of adjacent faces for this vertex
            adjacent_faces = self._get_adjacent_faces(vidx)
            if adjacent_faces:
                avg_area = sum(face.calculate_area(self.vertices) for face in adjacent_faces) / len(adjacent_faces)
            else:
                avg_area = 1.0  # Default if no faces found (shouldn't happen)
                
            if vertex.elevation <= self.sea_level:
                # Ocean gets water up to sea level
                # Convert water depth to volume for storage
                water_depth = max(0, self.sea_level - vertex.elevation)
                vertex.water_volume = water_depth * avg_area  # m³
                vertex.water_depth = water_depth
            else:
                vertex.water_volume = 0.0
                vertex.water_depth = 0.0
                    
        # Second pass: simulate rainfall and river flow
        for _ in range(iterations):
            new_water_volume = [0.0] * len(self.vertices)
            
            for i, vertex in enumerate(self.vertices):
                adjacent_faces = self._get_adjacent_faces(i)
                avg_area = sum(face.calculate_area(self.vertices) for face in adjacent_faces) / len(adjacent_faces) if adjacent_faces else 1.0
                
                effective_elevation = vertex.elevation + vertex.water_depth
                
                if effective_elevation > self.sea_level:
                    # Land receives rainfall (1-5 mm per iteration)
                    rainfall_depth = random.uniform(0.001, 0.005)  # 1-5 mm in meters
                    rainfall_volume = rainfall_depth * avg_area
                    new_water_volume[i] += rainfall_volume
                    
                    # Find lowest neighbor considering effective elevation
                    neighbors = self._find_vertex_neighbors(i)
                    if neighbors:
                        # Get neighbor data with effective elevations
                        neighbor_data = []
                        for n in neighbors:
                            n_faces = self._get_adjacent_faces(n)
                            n_area = sum(f.calculate_area(self.vertices) for f in n_faces) / len(n_faces) if n_faces else 1.0
                            n_depth = self.vertices[n].water_volume / n_area if n_area > 0 else 0
                            neighbor_data.append({
                                'index': n,
                                'effective_elev': self.vertices[n].elevation + n_depth,
                                'area': n_area
                            })
                        
                        # Find the lowest effective elevation neighbor
                        lowest_neighbor = min(neighbor_data, key=lambda x: x['effective_elev'])
                        
                        if lowest_neighbor['effective_elev'] < effective_elevation:
                            # Calculate potential energy difference
                            elev_diff = effective_elevation - lowest_neighbor['effective_elev']
                            
                            # Move water based on gradient (more flow with steeper gradient)
                            max_flow_depth = min(vertex.water_depth * 0.2, 0.5)  # Max 50 cm flow
                            flow_depth = max_flow_depth * min(1.0, elev_diff)  # Scale by gradient
                            
                            # Convert to volume
                            flow_volume = flow_depth * avg_area
                            new_water_volume[i] -= flow_volume
                            new_water_volume[lowest_neighbor['index']] += flow_volume * (avg_area / lowest_neighbor['area'])
            
            # Apply changes and update water depths
            for i in range(len(self.vertices)):
                adjacent_faces = self._get_adjacent_faces(i)
                avg_area = sum(face.calculate_area(self.vertices) for face in adjacent_faces) / len(adjacent_faces) if adjacent_faces else 1.0
                
                self.vertices[i].water_volume = max(0.0, self.vertices[i].water_volume + new_water_volume[i])
                self.vertices[i].water_depth = self.vertices[i].water_volume / avg_area if avg_area > 0 else 0
        
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

    #### utility functions

    def calculate_all_face_areas(self, device='cuda'):
        if not self.faces:
            return torch.tensor([], device=device)
            
        areas = Face.batch_calculate_areas(self.faces, self.vertices, device)
        
        # Update the face objects with their areas
        areas_cpu = areas.cpu().numpy()
        for i, face in enumerate(self.faces):
            face.area = areas_cpu[i]
            
        return areas

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
    
    def _initialize_neighbor_map(self) -> None:
        if self._neighbor_map_initialized:
            return
            
        print("Initializing neighbor map...")
        
        vertex_neighbor_sets = [set() for _ in range(len(self.vertices))]
        vertex_face_map = [[] for _ in range(len(self.vertices))]
        face_neighbor_sets = [set() for _ in range(len(self.faces))]
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
            for neighbor_idx in neighbors:
                dist = np.linalg.norm(self.vertices[v_idx].pos - self.vertices[neighbor_idx].pos)
                try:
                    weight = 1.0 / dist
                except ZeroDivisionError:
                    weight = 1.0 / (dist + 1e9)
                self.vertices[v_idx].neighbors[neighbor_idx] = weight
        
        self._face_neighbors = [list(s) for s in face_neighbor_sets]
        self._vertex_face_map = vertex_face_map
        self._neighbor_map_initialized = True
        print("Neighbor map initialized")

    def _find_vertex_neighbors(self, vertex_idx):
        if not self._neighbor_map_initialized:
            self._initialize_neighbor_map()
        return self.vertices[vertex_idx].neighbors
    
    def _get_adjacent_faces(self, vertex_idx):
        """Get all faces adjacent to a vertex"""
        if not self._neighbor_map_initialized:
            self._initialize_neighbor_maps()
        return [self.faces[i] for i in self._vertex_face_map[vertex_idx]]

    def _get_face_neighbors(self, face_idx):
        """Get faces that share an edge with the given face"""
        if not self._neighbor_map_initialized:
            self._initialize_neighbor_maps()
        return [self.faces[i] for i in self._face_neighbors[face_idx]]

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
  
    #### display

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
    
# --- Shapes ---

class Icosahedron(World):
    """Represents an Icosahedron."""
    def __init__(self):
        super().__init__()

    def _create_world(self, radius=1.0):
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

    def _create_world(self, radius=1.0):
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

    def _create_world(self, radius=1):
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

    def _create_world(self, radius=1):
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

    def _create_world(self, radius=1.0):
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

    def _subdivide(self, radius=1.0, level=3):
        midpoint_cache = {}
        new_faces = []
        #current_level_vertices = list(self.vertices)
        current_level_vertices = [copy.deepcopy(v) for v in self.vertices]
        
        next_vertices = list(current_level_vertices) 

        processed_faces = 0
        for face in self.faces:
            v_indices = face.v_indices
            n = len(v_indices)

            if n != 4:
                print(f"Warning: Skipping non-quad face: {face}")
                new_faces.append(face)
                continue
                
            face_center_pos = np.zeros(3, dtype=float)
            face_plates = []
            for v_idx in v_indices:
                face_center_pos += current_level_vertices[v_idx].pos
                if current_level_vertices[v_idx].plate_id != -1:
                    face_plates.append(current_level_vertices[v_idx].plate_id)
            face_center_pos /= n
            
            center_v = Vertex(*face_center_pos)
            try:
                center_v.plate_id = random.choice(face_plates)
            except:
                center_v.plate_id = -1
            center_v.normalize(radius)
            
            center_idx = len(next_vertices)
            next_vertices.append(center_v)

            mid_indices = []
            for i in range(n):
                v1_idx = v_indices[i]
                v2_idx = v_indices[(i + 1) % n] 
                mid_idx = self._get_midpoint_vertex(v1_idx, v2_idx, midpoint_cache, next_vertices, radius)
                mid_indices.append(mid_idx)
            
            for i in range(n):
                v_orig_idx = v_indices[i]
                mid_curr_idx = mid_indices[i]
                mid_prev_idx = mid_indices[i - 1]
                
                new_quad = Face((v_orig_idx, mid_curr_idx, center_idx, mid_prev_idx))
                new_faces.append(new_quad)

            processed_faces += 1

        self.vertices = next_vertices
        self.faces = new_faces

    def _subdivide_selected(self, faces_to_subdivide, radius=1.0, level=3):
        midpoint_cache = {}
        new_faces = []
        current_level_vertices = [copy.deepcopy(v) for v in self.vertices]
        next_vertices = list(current_level_vertices) 
        
        # Convert single face to list if needed
        if not isinstance(faces_to_subdivide, list):
            faces_to_subdivide = [faces_to_subdivide]
        
        # Create a set for faster lookups of faces to subdivide
        faces_to_subdivide_set = set(faces_to_subdivide)
        
        processed_faces = 0
        for face in self.faces:
            # Skip faces not in our subdivision list
            if face not in faces_to_subdivide_set:
                new_faces.append(face)
                continue
                
            v_indices = face.v_indices
            n = len(v_indices)

            if n != 4:
                print(f"Warning: Skipping non-quad face: {face}")
                new_faces.append(face)
                continue
                
            face_center_pos = np.zeros(3, dtype=float)
            face_plates = []
            for v_idx in v_indices:
                face_center_pos += current_level_vertices[v_idx].pos
                if current_level_vertices[v_idx].plate_id != -1:
                    face_plates.append(current_level_vertices[v_idx].plate_id)
            face_center_pos /= n
            
            center_v = Vertex(*face_center_pos)
            try:
                center_v.plate_id = random.choice(face_plates)
            except:
                center_v.plate_id = -1
            center_v.normalize(radius)
            
            center_idx = len(next_vertices)
            next_vertices.append(center_v)

            mid_indices = []
            for i in range(n):
                v1_idx = v_indices[i]
                v2_idx = v_indices[(i + 1) % n] 
                mid_idx = self._get_midpoint_vertex(v1_idx, v2_idx, midpoint_cache, next_vertices, radius)
                mid_indices.append(mid_idx)
            
            for i in range(n):
                v_orig_idx = v_indices[i]
                mid_curr_idx = mid_indices[i]
                mid_prev_idx = mid_indices[i - 1]
                
                new_quad = Face((v_orig_idx, mid_curr_idx, center_idx, mid_prev_idx))
                new_faces.append(new_quad)

            processed_faces += 1

        self.vertices = next_vertices
        self.faces = new_faces

# --- Main Execution ---
if __name__ == "__main__":
    shape_type = SHAPE
    num_subdivisions = SUBDIVISIONS
    sphere_radius = SPHERE_SIZE
    plates = PLATES
    elevationmin = ELEVATION_MIN
    elevationmax = ELEVATION_MAX

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

    shape.subdivide(sphere_radius, num_subdivisions)

    shape.genElevations(plates, elevationmin, elevationmax)
    shape.simulate_water()

    # print(f'plates have the following vertex count: ')
    # for plate in shape.plates.values():
    #     print(f'{plate.plate_id} has {len(plate.vertices)}')
    
    fig, ax, radio = shape.plot()

    ax.set_title(f'Sphere Approx. ({shape_type.capitalize()} Subdivided {num_subdivisions} Times)')
    plt.show()
    
