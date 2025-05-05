from collections import Counter
import copy
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
        plate_id = random.choice(most_common)
        if plate_id >= 0: return plate_id
        else: return -1
    
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
    
    def subdivide_face(self, vertex_list, faces) -> Tuple[List['Face'], List[int], List[Vertex]]:
        plate = self.get_plate(vertex_list)
        original_vertices = self.v_indices
        n = len(original_vertices)
        midpoint_cache = {}
        original_positions = [vertex_list[i].pos for i in original_vertices]
        new_faces = []
        new_vertices = []
        
        if n == 3:
            # Triangle subdivision into 4 smaller triangles
            edge_midpoints = []
            for i in range(3):
                j = (i + 1) % 3
                midpoint_pos = (original_positions[i] + original_positions[j]) / 2
                midpoint_pos = midpoint_pos / np.linalg.norm(midpoint_pos)
                
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
                
            face_center_pos = np.zeros(3, dtype=float)
            for v_idx in original_vertices:
                face_center_pos += vertex_list[v_idx].pos
            face_center_pos /= n
            
            center_v = Vertex(*face_center_pos)
            center_v.normalize()
            
            new_vertices.append(center_v)
            center_idx = len(vertex_list) + len(new_vertices) - 1

            mid_indices = []
            for i in range(n):
                v1_idx = original_vertices[i]
                v2_idx = original_vertices[(i + 1) % n] 
                mid_idx = self._get_midpoint_vertex(vertex_list, v1_idx, v2_idx, midpoint_cache, vertex_list)
                mid_indices.append(mid_idx)
            
            for i in range(n):
                v_orig_idx = original_vertices[i]
                mid_curr_idx = mid_indices[i]
                mid_prev_idx = mid_indices[i - 1]
                
                new_quad = Face((v_orig_idx, mid_curr_idx, center_idx, mid_prev_idx))
                new_faces.append(new_quad)

        elif n == 5:
            # Pentagon subdivision into 6 pentagons and 5 triangles
            center_pos = sum(original_positions) / 5
            center_pos = center_pos / np.linalg.norm(center_pos)
            
            # Create center vertex
            center_vertex = Vertex(*center_pos)
            new_vertices.append(center_vertex)
            center_index = len(vertex_list) + len(new_vertices) - 1
            
            edge_midpoints = []
            for i in range(5):
                j = (i + 1) % 5
                midpoint_pos = (original_positions[i] + original_positions[j]) / 2
                midpoint_pos = midpoint_pos / np.linalg.norm(midpoint_pos) 
                
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
                inner_pos = inner_pos / np.linalg.norm(inner_pos) 
                
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
            center_pos = center_pos / np.linalg.norm(center_pos) 
            
            # Create center vertex
            center_vertex = Vertex(*center_pos)
            new_vertices.append(center_vertex)
            center_index = len(vertex_list) + len(new_vertices) - 1
            
            edge_midpoints = []
            for i in range(n):
                j = (i + 1) % n
                midpoint_pos = (original_positions[i] + original_positions[j]) / 2
                midpoint_pos = midpoint_pos / np.linalg.norm(midpoint_pos) 
                
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
        
        for vertex in new_vertices:
            vertex.plate_id = plate

        return new_faces, new_vertices

    def create_edge_midpoints(self, vertex_list):
        """Create midpoint vertices for all edges of this face"""
        new_vertices = []
        n = len(self.v_indices)
        
        for i in range(n):
            v1_idx = self.v_indices[i]
            v2_idx = self.v_indices[(i+1)%n]
            
            # Skip if we already created this midpoint
            if (v1_idx, v2_idx) in self.edge_midpoint_indices or (v2_idx, v1_idx) in self.edge_midpoint_indices:
                continue
                
            v1 = vertex_list[v1_idx]
            v2 = vertex_list[v2_idx]
            
            # Calculate midpoint position (normalized to unit sphere)
            midpoint_pos = (v1.pos + v2.pos) / 2
            midpoint_pos = midpoint_pos / np.linalg.norm(midpoint_pos)
            
            # Create new vertex with interpolated properties
            midpoint_vertex = Vertex(
                pos=midpoint_pos,
                elevation=(v1.elevation + v2.elevation)/2,
                water_depth=(v1.water_depth + v2.water_depth)/2,
                plate_id=v1.plate_id  # Or could do majority vote
                # Add other properties as needed
            )
            
            # Add to vertex list and store index
            new_index = len(vertex_list)
            vertex_list.append(midpoint_vertex)
            new_vertices.append(new_index)
            self.edge_midpoint_indices[(v1_idx, v2_idx)] = new_index
            
        return new_vertices

    def create_center_vertex(self, vertex_list):
        """Create a center vertex for this face"""
        if self.center_index is not None:
            return self.center_index
            
        points = [vertex_list[i].pos for i in self.v_indices]
        
        # Calculate centroid position (normalized to unit sphere)
        centroid = sum(points) / len(points)
        centroid = centroid / np.linalg.norm(centroid)
        
        # Get average properties from all vertices
        avg_elevation = sum(vertex_list[i].elevation for i in self.v_indices) / len(self.v_indices)
        avg_water_depth = sum(vertex_list[i].water_depth for i in self.v_indices) / len(self.v_indices)
        
        # Determine plate ID by majority vote
        plate_counts = {}
        for i in self.v_indices:
            plate_id = vertex_list[i].plate_id
            plate_counts[plate_id] = plate_counts.get(plate_id, 0) + 1
        plate_id = max(plate_counts.items(), key=lambda x: x[1])[0]
        
        # Create center vertex
        center_vertex = Vertex(
            pos=centroid,
            elevation=avg_elevation,
            water_depth=avg_water_depth,
            plate_id=plate_id
            # Add other properties as needed
        )
        
        # Add to vertex list and store index
        self.center_index = len(vertex_list)
        vertex_list.append(center_vertex)
        return self.center_index

    def get_edge_midpoint(self, vertex_list, v1_idx, v2_idx):
        """Get the midpoint vertex index between two vertices"""
        if (v1_idx, v2_idx) in self.edge_midpoint_indices:
            return self.edge_midpoint_indices[(v1_idx, v2_idx)]
        if (v2_idx, v1_idx) in self.edge_midpoint_indices:
            return self.edge_midpoint_indices[(v2_idx, v1_idx)]
        return None