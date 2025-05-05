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
from util import batch_calculate_areas, plate_grow_worker_process

class world:
    def __init__(self):
        self.vertices: list[Vertex] = []
        self.faces: list[Face] = []
        self.plates: list[Plate] = {}
        self._neighbor_map_initialized: bool = False
        self.sea_level: np.float64 = 0.0
        self.minheight: np.float64 = -15000
        self.maxheight: np.float64  = 15000
        self.platecount: np.int8 = 15
        self.rainfall_rate: float = 0.002
        self.evaporation_rate: float = 0.001
        self.max_water_flow: float = 0.5
        self.min_river_flow: float = 1.0
        self.min_lake_volume: float = 3.0
        self.vertex_areas = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    def subdivide(self, levels: int):
        self.subdivisions = levels
        for level in range(levels):
            print(f'subdividing level {level+1}…')
            if level == 2:
                self._create_plates()
                self._assign_tectonic_plates()
            self._subdivide()
            if level >= 2:
                self._neighbor_map_initialized = False
                #self._fix_non_contiguous_vertices()
        for plate in self.plates:
            plate.set_continental_centers(self.vertices)
    
    def _subdivide(self):
        return NotImplementedError("subclass implements _subdivide()")

    def _subdivide_selected(self, faces_to_subdivide, radius=1.0, level=3):
        raise NotImplementedError("Subclasses must implement _subdivide_selected(faces_to_subdivide, radius, level)")

    def _find_vertex_neighbors(self, vertex_idx):
        self._initialize_neighbor_map()
        return self.vertices[vertex_idx].neighbors
    
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
    
            edge = tuple(sorted((v1, v2)))
            if edge in edge_face_map:
                other_face = edge_face_map[edge]
                face_neighbor_sets[fid].add(other_face)
                face_neighbor_sets[other_face].add(fid)
            else:
                edge_face_map[edge] = fid

        for v_idx, neighbors in enumerate(vertex_neighbor_sets):
            neighbor_indices = np.array(list(neighbors))
            dists = np.linalg.norm(all_positions[v_idx] - all_positions[neighbor_indices], axis=1)
            weights = np.divide(1.0, dists, where=dists!=0, out=np.full_like(dists, 1.0/1e9))
            self.vertices[v_idx].neighbors.update(zip(neighbor_indices, weights))

        self._face_neighbors = [list(s) for s in face_neighbor_sets]
        self._vertex_face_map = vertex_face_map
        self._neighbor_map_initialized = True
        print("Neighbor map initialized")

    ########## PLATES ###########

    def _create_plates(self):
        self.plates = {}
        minor_plates_count = max(1, int(self.platecount * random.uniform(0.4, 0.6)))
        major_plates_ids = set(random.sample(range(self.platecount), minor_plates_count))
        plate_types = []
        for _ in range(self.platecount):
            if len(plate_types) > 0:
                if plate_types[-1] == PLATE_TYPE_CONTINENTAL: next_type = PLATE_TYPE_CONTINENTAL
                else: next_type = PLATE_TYPE_OCEANIC
            else:
                next_type = random.choice([PLATE_TYPE_CONTINENTAL, PLATE_TYPE_OCEANIC])
            plate_types.append(next_type)
        
        for i in range(self.platecount // 3):
            if random.random() < 0.3:
                plate_types[i] = 1 - plate_types[i]
        for plate_id in range(self.platecount):
            self.plates[plate_id] = Plate(plate_id, plate_types[i], plate_id in major_plates_ids)

    def _assign_tectonic_plates(self, max_iterations=100):
        num_vertices = len(self.vertices)
        print("Initializing plate assignment...")
        plate_ids = [-1] * num_vertices
        unassigned = set(range(num_vertices))
        
        plate_starts = np.random.choice(list(unassigned), size=self.platecount, replace=False)
        for plate_id, start_idx in enumerate(plate_starts):
            if start_idx in unassigned:
                plate_ids[start_idx] = plate_id
                self.plates[plate_id].add_vertex(start_idx, self.vertices)
                unassigned.remove(start_idx)
        
        print(f"Assigned {self.platecount} starting points. Remaining unassigned: {len(unassigned)}")
        iteration = 0

        while len(unassigned) > 0 and iteration < max_iterations:
            iteration += 1
            assigned_in_iter = 0
            current_unassigned = list(unassigned)

            for u_idx in current_unassigned:
                if plate_ids[u_idx] != -1:
                    unassigned.remove(u_idx)
                    continue

                neighbor_plates = set()
                neighbors = self._find_vertex_neighbors(u_idx)
                for n_idx in neighbors:
                    plate_id = plate_ids[n_idx]
                    if plate_id != -1:
                        neighbor_plates.add(plate_id)
                
                if neighbor_plates:
                    chosen_plate = np.random.choice(list(neighbor_plates))
                    plate_ids[u_idx] = chosen_plate
                    plate = self.plates[chosen_plate]
                    if plate.is_minor:
                        if random.random() < 0.5:
                            self.plates[chosen_plate].add_vertex(u_idx, self.vertices)
                    else:
                        self.plates[chosen_plate].add_vertex(u_idx, self.vertices)
                    unassigned.remove(u_idx)
                    assigned_in_iter += 1
            print(f"Iteration {iteration}: Finished | Assigned in iter: {assigned_in_iter} | Remaining Unassigned: {len(unassigned)}")

            if assigned_in_iter == 0:
                print(f"Iteration {iteration}: Stalled. Stopping parallel phase.")
                break
        
        print(f"\nGrowth phase finished after {iteration} iterations.")
        print(f"Remaining unassigned vertices: {len(unassigned)}")


        remaining_indices = [i for i, pid in enumerate(plate_ids) if pid == -1]
        if remaining_indices:
            self._assign_remaining_vertices_safe(remaining_indices)
        #self._fix_non_contiguous_vertices()
    
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

    def _gen_elevations(self):
        self._assign_base_elevations()
        self._calculate_boundary_elevations()

    def _assign_base_elevations(self):
        for plate in self.plates:
            for v_idx in plate.vertices:
                variation = np.random.uniform(-0.1, 0.1)
                self.vertices[v_idx].elevation = plate.base_elevation + (variation * plate.base_elevation)