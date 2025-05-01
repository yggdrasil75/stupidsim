
import copy
import random

import numpy as np
from holder.face import Face
from holder.vertex import Vertex
from world import World

class Icosahedron(World):
    """Represents an Icosahedron."""
    def __init__(self):
        super().__init__()

    def _create_world(self, radius=1.0):
        """Creates the vertices and faces for a unit icosahedron."""
        self.vertices = []
        self.faces = []
        phi = (1 + np.sqrt(5)) / 2
        
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
