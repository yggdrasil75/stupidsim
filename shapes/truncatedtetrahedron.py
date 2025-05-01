
import copy
import random

import numpy as np
from holder.face import Face
from holder.vertex import Vertex
from world import World

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
