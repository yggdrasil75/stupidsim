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
