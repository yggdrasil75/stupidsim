from matplotlib import pyplot as plt
import numpy as np
from holder.face import Face
from holder.vertex import TerrainVertex
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class World:
    def __init__(self):
        self.vertices = np.array([], np.dtype([('id', np.int32),('vertex', TerrainVertex)]))
        self.faces = np.array([], np.dtype([('id', np.int32),('face', Face)]))
        self._create_world()

    def add_vertex(self, vertex: TerrainVertex):
        index = len(self.vertices)
        self.vertices = np.append(self.vertices, np.array((index, vertex), dtype=[('id', np.int32), ('vertex', TerrainVertex)]))
        return index
    
    def add_face(self, face):
        index = len(self.faces)
        self.faces = np.append(self.faces, np.array((index, face), dtype=[('id', np.int32), ('face', Face)]))
        return index

    def _create_world(self):
        t = 1.839286755214161
        v_data = [
            (-1, -1, -1),
            ( 1, -1, -1),
            ( 1,  1, -1),
            (-1,  1, -1),
            (-1, -1,  1),
            ( 1, -1,  1),
            ( 1,  1,  1),
            (-1,  1,  1)
                ]
        for v in v_data:
            self.add_vertex(TerrainVertex(*v))

        f_ind = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7)
        ]

        for f in f_ind:
            self.add_face(Face(f))

    def plot(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract vertex positions
        verts = [vertex.pos for (id, vertex) in self.vertices]
        
        # Create a list to hold the polygons for each face
        polygons = []
        print(self.faces)
        for (id, face) in self.faces:
            # Get the vertex indices for this face
            face_verts = [verts[i] for i in face.v_indices]
            polygons.append(face_verts)
        print(polygons)
        
        # Create the 3D polygon collection
        poly_collection = Poly3DCollection(polygons,facecolors='red', alpha=0.8, linewidths=1, edgecolor='black')
        ax.add_collection3d(poly_collection)
        
        # Set the axes limits
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.title('3D Cube Visualization')
        fig.canvas.draw_idle()
        plt.tight_layout()

world = World()
world.plot()
plt.show()