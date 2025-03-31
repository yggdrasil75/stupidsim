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
                            triangles=faces,
                            array=face_data,  # Pass face_data directly as array
                            cmap=cmap,        # Apply colormap
                            linewidth=0,      # Remove edge lines for cleaner look
                            antialiased=False) # Disable antialiasing for performance and cleaner lines

    # Colorbar - keep it for now, even if only elevation is implemented
    if cbar_obj is None:
        cbar_obj = plt.colorbar(mesh, ax=ax, shrink=0.5, aspect=20) # Adjust aspect ratio
    else:
        cbar_obj.mappable = mesh
        cbar_obj.update_normal(mesh)

    cbar_obj.set_label(cbar_label)
    cbar_obj.ax.tick_params(labelsize=8) # Reduce colorbar label size

    ax.set_title(title, fontsize=12) # Reduce title fontsize
    ax.set_xlabel("X", fontsize=8)   # Reduce axis label fontsize
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6) # Reduce tick label size
    ax.set_aspect('equal')
    ax.view_init(elev=30, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title)


if __name__ == "__main__":
    subdivisions = 3
    radius = PLANET_RADIUS_KM  # Use the global planet radius
    world_state = icosphere.initializeWorld(subdivisions=subdivisions, radius=radius, elevationRange=20000.0)
    vertices_cpp = world_state.vertices
    face_history = icosphere.run_simulation(world_state, steps=10) # Run simulation for 10 steps
    final_faces_cpp = face_history[-1] # Get the faces from the last step
    vertex_indices_map = world_state.vertex_indices # Access the vertex indices map (remains the same as vertices are not reindexed)


    # Convert C++ vectors to NumPy arrays
    num_vertices = len(vertices_cpp)
    vertices_np = np.zeros((num_vertices, 3))
    elevations_np = np.zeros(num_vertices)
    for i in range(num_vertices):
        v = vertices_cpp[i]
        vertices_np[i, :] = [v.x, v.y, v.z]
        elevations_np[i] = v.elevation

    num_faces = len(final_faces_cpp) # Use final_faces_cpp here
    faces_np = np.zeros((num_faces, 3), dtype=int)
    for i in range(num_faces):
        face = final_faces_cpp[i] # Use final_faces_cpp here
        # Use the vertex_indices_map to get the indices
        faces_np[i, 0] = vertex_indices_map[face.a]
        faces_np[i, 1] = vertex_indices_map[face.b]
        faces_np[i, 2] = vertex_indices_map[face.c]


    fig = plt.figure(figsize=(8, 8)) # Adjust figure size
    ax = fig.add_subplot(111, projection='3d')

    visualize_world_spherical(vertices_np, faces_np, elevations_np, ax, data_type='elevation')

    plt.tight_layout() # Improve layout to prevent labels from overlapping
    plt.show()