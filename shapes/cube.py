
import copy
import random

import numpy as np
from holder.face import Face
from holder.vertex import Vertex
from util import time_function
from world import World
import torch

class Cube(World):
    """Represents a Cube."""
    def __init__(self):
        super().__init__()

    @time_function
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

    @time_function
    def _subdivide_bad(self):
        new_faces = []
        new_vertices = []
        original_face = list(self.faces)
        self.faces.clear()

        for face in original_face:
            _new_faces, _new_vertices = face.subdivide_face(self.vertices, self.faces)
            new_faces.extend(_new_faces)
            new_vertices.extend(_new_vertices)


        self.faces = new_faces
        self.vertices.extend(new_vertices)
        self._neighbor_map_initialized = False
        #return super()._subdivide()

    @time_function
    def _get_midpoint_vertex_torch(self, v1_idx, v2_idx, midpoint_cache, vertices, plate_ids, radius):
        key = tuple(sorted((v1_idx, v2_idx)))
        if key in midpoint_cache:
            return midpoint_cache[key]
        
        # Compute midpoint
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        mid_pos = (v1 + v2) / 2
        
        # Normalize
        mid_pos = mid_pos / torch.norm(mid_pos) * radius
        
        # Determine plate ID
        plate1 = plate_ids[v1_idx]
        plate2 = plate_ids[v2_idx]
        mid_plate = plate1 if plate1 == plate2 else -1
        
        # Add to vertices
        mid_idx = len(vertices)
        vertices = torch.cat([vertices, mid_pos.unsqueeze(0)])
        plate_ids = torch.cat([plate_ids, torch.tensor([mid_plate], device=self.device)])
        
        midpoint_cache[key] = mid_idx
        return mid_idx
    
    @time_function
    def _subdivide(self):
        midpoint_cache = {}
        new_faces = []
        current_level_vertices = [copy.deepcopy(v) for v in self.vertices]
        next_vertices = list(current_level_vertices) 

        for face in self.faces:
            plate = face.get_plate(vertex_list=self.vertices)
            v_indices = face.v_indices
            n = len(v_indices)

            if n != 4:
                print(f"Warning: Skipping non-quad face: {face}")
                new_faces.append(face)
                continue
                
            face_center_pos = np.zeros(3, dtype=float)
            for v_idx in v_indices:
                face_center_pos += current_level_vertices[v_idx].pos
            face_center_pos /= n
            
            center_v = Vertex(*face_center_pos)
            center_v.normalize()
            center_idx = len(next_vertices)
            center_v.plate_id = plate
            next_vertices.append(center_v)

            mid_indices = []
            for i in range(n):
                v1_idx = v_indices[i]
                v2_idx = v_indices[(i + 1) % n] 
                mid_idx = self._get_midpoint_vertex(v1_idx, v2_idx, midpoint_cache, next_vertices)

                mid_indices.append(mid_idx)

            for i in range(n):
                v_orig_idx = v_indices[i]
                mid_curr_idx = mid_indices[i]
                mid_prev_idx = mid_indices[i - 1]
                
                new_quad = Face((v_orig_idx, mid_curr_idx, center_idx, mid_prev_idx))
                new_faces.append(new_quad)
            
            # for i, v in enumerate(next_vertices):
            #     v.plate_id = plate

        self.vertices = next_vertices
        self.faces = new_faces

    @time_function
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
