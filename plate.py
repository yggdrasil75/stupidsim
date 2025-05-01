import random
import numpy as np

from holder.globals import CONTINENTAL_CRUST_THICKNESS, OCEANIC_CRUST_THICKNESS, PLATE_TYPE_CONTINENTAL, PLATE_TYPE_OCEANIC


class Plate:
    def __init__(self, plate_id, is_minor=False):
        self.plate_id = plate_id
        self.velocity = np.random.uniform(low=-1, high=1, size=3)
        self.vertices = set()
        self.type = random.choice([PLATE_TYPE_OCEANIC, PLATE_TYPE_CONTINENTAL])
        self.is_minor = is_minor
        self.growth_rate = 0.5 if is_minor else 1.0
        self.base_elevation = (OCEANIC_CRUST_THICKNESS if self.type == PLATE_TYPE_OCEANIC 
                              else CONTINENTAL_CRUST_THICKNESS)
        
    def resetVertices(self):
        self.vertices.clear()

    def add_vertex(self, vertex_idx, vertices):
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
