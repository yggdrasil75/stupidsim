from collections import Counter
import random
from typing import List, Tuple
import numpy as np
from holder.globals import EQUATORIAL_RADIUS
from holder.vertex import Vertex


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
        return [vertex_list[i].water_depth for i in self.v_indices]
    
    def get_vertices_water_depth(self, vertex_list) -> list:
        return [vertex_list[i].water_depth for i in self.v_indices]
    
    def get_vertices_water_volume(self, vertex_list) -> list:
        return [vertex_list[i].water_volume for i in self.v_indices]
    
    def get_plate(self, vertex_list):
        plates = [vertex_list[i].plate_id for i in self.v_indices]
        count = Counter(plates)
        max_count = max(count.values())
        most_common = [plates for plates, c in count.items() if c == max_count]
        return random.choice(most_common)
    
    def __repr__(self) -> str:
        return f"Face({self.v_indices})"

    def spherical_triangle_area(self, tri):
        """Calculate area of a spherical triangle given three points"""
        # Convert to unit vectors
        a, b, c = [p/np.linalg.norm(p) for p in tri]
        
        # Calculate angles using dot products
        alpha = np.arccos(np.dot(np.cross(a, b), np.cross(a, c)) / 
                (np.linalg.norm(np.cross(a, b)) * np.linalg.norm(np.cross(a, c))))
        beta = np.arccos(np.dot(np.cross(b, a), np.cross(b, c)) / 
               (np.linalg.norm(np.cross(b, a)) * np.linalg.norm(np.cross(b, c))))
        gamma = np.arccos(np.dot(np.cross(c, a), np.cross(c, b)) / 
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
    
    def subdivide_face(self, vertex_list) -> Tuple[List['Face'], List[int], List[Vertex]]:
        plate = self.get_plate(vertex_list)
        original_vertices = self.v_indices
        n = len(original_vertices)
        new_faces = []
        new_vertex_indices = []
        new_vertices = []  # This will store the actual Vertex objects
        
        # Get the positions of the original vertices
        original_positions = [vertex_list[i].pos for i in original_vertices]
        
        if n == 3:
            # Triangle subdivision into 4 smaller triangles
            edge_midpoints = []
            for i in range(3):
                j = (i + 1) % 3
                midpoint_pos = (original_positions[i] + original_positions[j]) / 2
                midpoint_pos = midpoint_pos / np.linalg.norm(midpoint_pos) * EQUATORIAL_RADIUS
                
                # Create new Vertex object
                new_vertex = Vertex(*midpoint_pos)
                new_vertex.normalize()
                new_vertices.append(new_vertex)
                new_vertex_index = len(vertex_list) + len(new_vertices) - 1
                edge_midpoints.append(new_vertex_index)
            
            # Create 4 new triangles
            new_faces.append(Face(edge_midpoints))
            
            for i in range(3):
                j = (i + 1) % 3
                new_faces.append(Face([
                    original_vertices[i],
                    edge_midpoints[i],
                    edge_midpoints[j]
                ]))
        
        elif n == 4:
            # Quad subdivision into 4 smaller quads
            center_pos = sum(original_positions) / 4
            center_pos = center_pos / np.linalg.norm(center_pos) * EQUATORIAL_RADIUS
            
            # Create center vertex
            center_vertex = Vertex(*center_pos)
            new_vertices.append(center_vertex)
            center_index = len(vertex_list) + len(new_vertices) - 1
            
            edge_midpoints = []
            for i in range(4):
                j = (i + 1) % 4
                midpoint_pos = (original_positions[i] + original_positions[j]) / 2
                midpoint_pos = midpoint_pos / np.linalg.norm(midpoint_pos) * EQUATORIAL_RADIUS
                
                # Create new Vertex object
                new_vertex = Vertex(*midpoint_pos)
                new_vertex.normalize()
                new_vertices.append(new_vertex)
                new_vertex_index = len(vertex_list) + len(new_vertices) - 1
                edge_midpoints.append(new_vertex_index)
            
            # Create 4 new quads
            for i in range(4):
                j = (i + 1) % 4
                new_faces.append(Face([
                    original_vertices[i],
                    edge_midpoints[i],
                    center_index,
                    edge_midpoints[j]
                ]))
        
        elif n == 5:
            # Pentagon subdivision into 6 pentagons and 5 triangles
            center_pos = sum(original_positions) / 5
            center_pos = center_pos / np.linalg.norm(center_pos) * EQUATORIAL_RADIUS
            
            # Create center vertex
            center_vertex = Vertex(*center_pos)
            new_vertices.append(center_vertex)
            center_index = len(vertex_list) + len(new_vertices) - 1
            
            edge_midpoints = []
            for i in range(5):
                j = (i + 1) % 5
                midpoint_pos = (original_positions[i] + original_positions[j]) / 2
                midpoint_pos = midpoint_pos / np.linalg.norm(midpoint_pos) * EQUATORIAL_RADIUS
                
                # Create new Vertex object
                new_vertex = Vertex(*midpoint_pos)
                new_vertex.normalize()
                new_vertices.append(new_vertex)
                new_vertex_index = len(vertex_list) + len(new_vertices) - 1
                edge_midpoints.append(new_vertex_index)
            
            # Create inner ring vertices
            inner_ring = []
            for i in range(5):
                inner_pos = (original_positions[i] + center_pos * 2) / 3
                inner_pos = inner_pos / np.linalg.norm(inner_pos) * EQUATORIAL_RADIUS
                
                # Create new Vertex object
                new_vertex = Vertex(*inner_pos)
                new_vertex.normalize()
                new_vertices.append(new_vertex)
                new_vertex_index = len(vertex_list) + len(new_vertices) - 1
                inner_ring.append(new_vertex_index)
            
            # Create faces
            new_faces.append(Face(inner_ring))  # Center pentagon
            
            for i in range(5):
                j = (i + 1) % 5
                new_faces.append(Face([
                    original_vertices[i],
                    edge_midpoints[i],
                    inner_ring[i],
                    center_index,
                    inner_ring[j]
                ]))
            
            for i in range(5):
                j = (i + 1) % 5
                new_faces.append(Face([
                    edge_midpoints[i],
                    original_vertices[j],
                    edge_midpoints[j]
                ]))
        
        else:
            # N-gon subdivision (N > 5)
            center_pos = sum(original_positions) / n
            center_pos = center_pos / np.linalg.norm(center_pos) * EQUATORIAL_RADIUS
            
            # Create center vertex
            center_vertex = Vertex(*center_pos)
            new_vertices.append(center_vertex)
            center_index = len(vertex_list) + len(new_vertices) - 1
            
            edge_midpoints = []
            for i in range(n):
                j = (i + 1) % n
                midpoint_pos = (original_positions[i] + original_positions[j]) / 2
                midpoint_pos = midpoint_pos / np.linalg.norm(midpoint_pos) * EQUATORIAL_RADIUS
                
                # Create new Vertex object
                new_vertex = Vertex(*midpoint_pos)
                new_vertex.normalize()
                new_vertices.append(new_vertex)
                new_vertex_index = len(vertex_list) + len(new_vertices) - 1
                edge_midpoints.append(new_vertex_index)
            
            # Create central face
            new_faces.append(Face(edge_midpoints))
            
            # Create surrounding faces
            for i in range(n):
                j = (i + 1) % n
                new_faces.append(Face([
                    original_vertices[i],
                    edge_midpoints[i],
                    center_index,
                    edge_midpoints[j]
                ]))
                
                if i < n - 2:
                    new_faces.append(Face([
                        original_vertices[i],
                        original_vertices[j],
                        edge_midpoints[j]
                    ]))
        
        # The new_vertex_indices list isn't really needed anymore since we're returning the actual vertices
        # But we'll keep it for backward compatibility
        new_vertex_indices = [len(vertex_list) + i for i in range(len(new_vertices))]
        
        for vertex in new_vertices:
            vertex.plate_id = plate

        return new_faces, new_vertex_indices, new_vertices