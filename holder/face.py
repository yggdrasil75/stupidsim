import numpy as np
from holder.globals import EQUATORIAL_RADIUS


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

    def spherical_triangle_area(self, tri):
        """Calculate area of a spherical triangle given three points"""
        # Convert to unit vectors
        a, b, c = [p/np.linalg.norm(p) for p in tri]
        
        # Calculate angles using dot products
        alpha = np.acos(np.dot(np.cross(a, b), np.cross(a, c)) / 
                (np.linalg.norm(np.cross(a, b)) * np.linalg.norm(np.cross(a, c))))
        beta = np.acos(np.dot(np.cross(b, a), np.cross(b, c)) / 
               (np.linalg.norm(np.cross(b, a)) * np.linalg.norm(np.cross(b, c))))
        gamma = np.acos(np.dot(np.cross(c, a), np.cross(c, b)) / 
                (np.linalg.norm(np.cross(c, a)) * np.linalg.norm(np.cross(c, b))))
        
        # Spherical excess
        excess = alpha + beta + gamma - np.pi
        return excess * EQUATORIAL_RADIUS**2

    def calculate_area(self, vertex_list) -> float:
        if self.area > 0:
            return self.area
            
        points = [vertex_list[i].pos for i in self.v_indices]
        n = len(points)
        
        if n == 3:
            self.area = self.spherical_triangle_area(points)
        else:
            # Decompose polygon into triangles using a fan triangulation
            # (works for convex polygons on a sphere)
            total_area = 0.0
            center_point = points[0]  # Use first point as center
            for i in range(1, n-1):
                triangle = [center_point, points[i], points[i+1]]
                total_area += self.spherical_triangle_area(triangle)
            self.area = total_area
            
        return self.area
