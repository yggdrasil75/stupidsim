from collections import deque
import ctypes
from datetime import time
import multiprocessing
import queue
import random
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from holder.face import Face
from holder.globals import CONTINENTAL_CRUST_THICKNESS, OCEANIC_CRUST_THICKNESS, PLATE_TYPE_CONTINENTAL, PLATE_TYPE_OCEANIC
from holder.vertex import Vertex
from plate import Plate
from util import batch_calculate_areas, plate_grow_worker_process

class world:
    def __init__(self):
        self.vertices: list[Vertex] = []
        self.faces: list[Face] = []
        self.plates: dict[int, Plate] = {}
        self._neighbor_map_initialized: bool = False
        self.sea_level: np.float64 = 0.0
        self.minheight: np.float64 = -15000
        self.maxheight: np.float64  = 15000
        self.platecount: np.int8 = 15
        self.rainfall_rate: float = 0.002
        self.evaporation_rate: float = 0.001
        self.max_water_flow: float = 0.5
        self.min_river_flow: float = 1.0
        self.min_lake_volume: float = 3.0
        self.vertex_areas = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_vertex(self, vertex):
        self.vertices.append(vertex)
        return len(self.vertices) - 1

    def add_face(self, face):
        self.faces.append(face)

    def create_world(self, radius=1.0, plates = 15, min_height = -15000, max_height = 15000):
        self.platecount = plates
        self.elevationMin = min_height
        self.elevationMax = max_height
        self._create_world(radius=1.0)

    def _create_world(self, radius = 1.0):
        raise NotImplementedError("Subclasses must implement _create_world()")
    
    def _subdivide_selected(self, faces_to_subdivide, radius=1.0, level=3):
        raise NotImplementedError("Subclasses must implement _subdivide_selected(faces_to_subdivide, radius, level)")
    
    