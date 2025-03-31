from matplotlib import pyplot as plt
import numpy as np
import icosphere
from mpl_toolkits.mplot3d import Axes3D

PLANET_RADIUS_KM = 6371.0
cbar_obj = None  # Global colorbar object to update


def visualize_world_spherical(vertices, faces, data, ax, data_type='elevation', elevations=None,
                            show_clouds=False, cloud_coverage=None,
                            show_storms=False, active_storms=None):
    global cbar_obj
    ax.clear()
    ax.set_zorder(1)

    # Main data visualization
    if data_type == 'elevation':
        vmin, vmax = -10000, 10000  # Define elevation range
        norm_data = (data - vmin) / (vmax - vmin)
        norm_data = np.clip(norm_data, 0, 1)
        face_data = np.mean(norm_data[faces], axis=1) # average vertex data for each face
        colors = plt.cm.terrain(face_data) # use terrain colormap
        cmap = 'terrain'
        title = 'Elevation'
        cbar_label = 'Elevation (m)'
    else:
        # Default case if data_type is not 'elevation' - can be expanded later
        face_data = np.zeros(len(faces)) # placeholder if no data_type matches
        colors = 'gray'
        cmap = 'gray'
        title = 'No Data'
        cbar_label = 'Data Value'


    # Plot the main mesh
    mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                            triangles=faces, color='white',
                            edgecolor='none', alpha=1.0)
    mesh.set_array(face_data) # set face data for coloring
    mesh.set_cmap(cmap) # apply colormap

    # Colorbar - keep it for now, even if only elevation is implemented
    if cbar_obj is None:
        cbar_obj = plt.colorbar(mesh, ax=ax, shrink=0.5)
    else:
        cbar_obj.mappable = mesh
        cbar_obj.update_normal(mesh)

    cbar_obj.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect('equal')
    ax.view_init(elev=30, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title)


if __name__ == "__main__":
    subdivisions = 3
    radius = PLANET_RADIUS_KM  # Use the global planet radius
    vertices_cpp, faces_cpp, _ = icosphere.initializeWorld(subdivisions=subdivisions, radius=radius, elevationRange=20000.0)

    # Convert C++ vectors to NumPy arrays
    num_vertices = len(vertices_cpp)
    vertices_np = np.zeros((num_vertices, 3))
    elevations_np = np.zeros(num_vertices)
    for i in range(num_vertices):
        v = vertices_cpp[i]
        vertices_np[i, :] = [v.x, v.y, v.z]
        elevations_np[i] = v.elevation

    num_faces = len(faces_cpp)
    faces_np = np.zeros((num_faces, 3), dtype=int) # Assuming faces are indexed from 0 to num_vertices-1 based on icosphere generation
    vertex_map = {}
    unique_vertices_list = []
    vertex_index = 0

    for face_cpp in faces_cpp:
        face_indices = []
        for vertex_cpp in [face_cpp.a, face_cpp.b, face_cpp.c]:
            vertex_tuple = (vertex_cpp.x, vertex_cpp.y, vertex_cpp.z) #use coordinates as key
            if vertex_tuple not in vertex_map:
                vertex_map[vertex_tuple] = vertex_index
                unique_vertices_list.append(vertex_cpp)
                face_indices.append(vertex_index)
                vertex_index += 1
            else:
                face_indices.append(vertex_map[vertex_tuple])
        faces_np[faces_cpp.index(face_cpp), :] = face_indices


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    visualize_world_spherical(vertices_np, faces_np, elevations_np, ax, data_type='elevation')

    plt.show()