
import copy
import random

import numpy as np
from holder.face import Face
from holder.vertex import Vertex
from world import World

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
