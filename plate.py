import random
import numpy as np

from holder.globals import CONTINENTAL_CRUST_THICKNESS, OCEANIC_CRUST_THICKNESS, PLATE_TYPE_CONTINENTAL, PLATE_TYPE_OCEANIC


class Plate:
    def __init__(self, plate_id, conoce, is_minor=False):
        self.plate_id = plate_id
        #self.velocity = np.random.uniform(low=-1, high=1, size=3)

        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        self.speed = np.random.uniform(low=0.1, high=1.0) if not is_minor else np.random.uniform(low=0.05, high=0.5)
        self.velocity = axis * self.speed

        self.vertices = set()
        self.type = conoce #if conoce else random.choice([PLATE_TYPE_OCEANIC, PLATE_TYPE_CONTINENTAL])
        #self.type = random.choice([PLATE_TYPE_OCEANIC, PLATE_TYPE_CONTINENTAL])
        self.is_minor = is_minor
        self.growth_rate = 0.5 if is_minor else 1.0
        self.base_elevation = (OCEANIC_CRUST_THICKNESS if self.type == PLATE_TYPE_OCEANIC 
                              else CONTINENTAL_CRUST_THICKNESS)
        if self.type == PLATE_TYPE_CONTINENTAL:
            self.continental_ratio = random.uniform(0.4, 0.95)
        else:
            self.continental_ratio = random.uniform(0.0, 0.1)
        self.continental_centers = []

    def __repr__(self):
        return f'Plate({self.plate_id}, heading towards {self.velocity} with a speed of {self.speed}.)\nPlate is {"continental" if self.type else "oceanic"}, and {"is" if self.is_minor else "is not"} a minor plate'

    def resetVertices(self):
        self.vertices.clear()

    def add_vertex(self, vertex_idx, vertices = None):
        vertices[vertex_idx].plate_id = self.plate_id
        self.vertices.add(vertex_idx)
        
    def get_boundary_vertices(self, world) -> list:
        boundary = []
        for v_idx in self.vertices:
            neighbors = world._find_vertex_neighbors(v_idx)
            neighbor_plates = {world.vertices[n].plate_id for n in neighbors}
            if len(neighbor_plates) > 1: 
                boundary.append(v_idx)
        return boundary

    def set_continental_centers(self, vertex_list):
        if self.type != PLATE_TYPE_CONTINENTAL or not self.vertices:
            return  # Exit if not continental or no vertices
        
        # Find vertex with highest elevation above base
        max_elevation = -float('inf')
        best_vertex = None
        
        for v_idx in self.vertices:
            vertex = vertex_list[v_idx]
            elevation_above_base = vertex.elevation - self.base_elevation
            
            if elevation_above_base > max_elevation:
                max_elevation = elevation_above_base
                best_vertex = v_idx
        
        # Set center (fallback to random if no elevated points found)
        self.continental_centers = [best_vertex if best_vertex is not None 
                                else random.choice(list(self.vertices))]