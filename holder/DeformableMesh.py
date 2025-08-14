import numpy as np
from dataclasses import field, dataclass
from holder.mesh import mesh

@dataclass
class DeformableMesh(mesh):
    original_vertices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    original_polys: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    deformation_resistance: np.ndarray = field(default_factory=lambda: np.array(1.0, dtype=np.float32))
    max_subdivisions: int = 3  # Maximum number of times a triangle can be subdivided
    subdivision_threshold: float = 0.5  # Deformation strength needed to trigger subdivision
    subdivided: dict = field(default_factory=dict)  # Track which polys have been subdivided

    def __post_init__(self):
        super().__post_init__()
        self.original_vertices = self._vertices.copy()
        self.original_polys = self._polys.copy()
        # Initialize subdivision tracking
        self.subdivided = {i: 0 for i in range(len(self._polys))}

    def reset_deformation(self):
        """Reset the terrain to its original shape and detail level"""
        self._vertices = self.original_vertices.copy()
        self._polys = self.original_polys.copy()
        self.subdivided = {i: 0 for i in range(len(self._polys))}
        self._color = self._color[:len(self._vertices)]  # Adjust colors if needed

    def subdivide_poly(self, poly_index):
        """Subdivide a polygon into smaller polygons"""
        if self.subdivided[poly_index] >= self.max_subdivisions:
            return False

        poly = self._polys[poly_index]
        if len(poly) != 3:  # We'll only subdivide triangles for simplicity
            poly = self.toTri(self)._polys[poly_index]

        # Get the vertices of the polygon
        v0, v1, v2 = self._vertices[poly[0]], self._vertices[poly[1]], self._vertices[poly[2]]

        # Create new vertices at midpoints
        new_v0 = (v0 + v1) / 2
        new_v1 = (v1 + v2) / 2
        new_v2 = (v2 + v0) / 2

        # Add new vertices
        new_vertex_indices = []
        for new_v in [new_v0, new_v1, new_v2]:
            # Check if this midpoint already exists (to prevent duplicate vertices)
            distances = np.linalg.norm(self._vertices - new_v, axis=1)
            existing = np.argmin(distances)
            if distances[existing] < 1e-5:  # Threshold for considering it the same vertex
                new_vertex_indices.append(existing)
            else:
                new_vertex_indices.append(len(self._vertices))
                self._vertices = np.vstack([self._vertices, new_v], dtype=np.float32)
                # Add color for new vertex (average of parent vertices)
                if len(self._color) > 1:
                    parent_colors = np.stack([self._color[poly[0]], self._color[poly[1]], 
                                           self._color[poly[2]]], dtype=np.float32)
                    new_color = parent_colors.mean(axis=0, dtype=np.float32)
                    self._color = np.vstack([self._color, new_color], dtype=np.float32)
                else:
                    # If single color, no need to add more
                    pass

        # Create new polygons (4 smaller triangles)
        new_polys = [
            [poly[0], new_vertex_indices[0], new_vertex_indices[2]],  # Top triangle
            [new_vertex_indices[0], poly[1], new_vertex_indices[1]],  # Right triangle
            [new_vertex_indices[2], new_vertex_indices[1], poly[2]],  # Left triangle
            [new_vertex_indices[0], new_vertex_indices[1], new_vertex_indices[2]]  # Center triangle
        ]

        # Replace the original polygon with new ones
        self._polys = np.vstack([
            self._polys[:poly_index],
            np.array(new_polys, dtype=np.int64),
            self._polys[poly_index+1:]
        ], dtype=np.int64)

        # Update subdivision tracking
        del self.subdivided[poly_index]
        for i in range(4):
            self.subdivided[len(self._polys) - 4 + i] = self.subdivided.get(poly_index, 0) + 1

        return True

    def apply_deformation(self, position, radius, strength):
        """
        Deform the terrain with adaptive detail refinement
        Args:
            position: 3D position where deformation occurs (array)
            radius: radius of deformation effect
            strength: how much to deform (positive for digging, negative for adding material)
        """
        # First pass: identify polygons that need subdivision
        if abs(strength) > self.subdivision_threshold:
            # Find polygons within deformation radius
            poly_centers = np.mean(self._vertices[self._polys], axis=1, dtype=np.float32)
            distances = np.linalg.norm(poly_centers - position, axis=1)
            nearby_poly_indices = np.where(distances < radius)[0]

            # Subdivide polygons that are close enough and not already max subdivided
            for poly_idx in nearby_poly_indices:
                if self.subdivided.get(poly_idx.item(), 0) < self.max_subdivisions:
                    self.subdivide_poly(poly_idx.item())

        # Second pass: apply deformation
        distances = np.linalg.norm(self._vertices - position, axis=1)
        
        # Spherical deformation
        influence = np.clip(1 - (distances / radius)**2, 0, 1)
        deformation = strength * influence
        
        # Only deform vertices within radius
        mask = distances < radius
        deformation_vector = (self._vertices[mask] - position)
        deformation_vector = deformation_vector / (np.linalg.norm(deformation_vector, axis=1, keepdims=True) + 1e-6)
        
        self._vertices[mask] += deformation_vector * deformation[mask][:, np.newaxis]