from collections import defaultdict
from functools import lru_cache
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np
import icosphere
from mpl_toolkits.mplot3d import Axes3D

PLANET_RADIUS_KM = 6371.0
cbar_obj = None  # Global colorbar object to update
current_step = 0
vertex_hierarchy = None


def visualize_world_spherical(vertices, faces, data, ax, data_type='elevation', 
                            show_clouds=False, cloud_coverage=None,
                            show_storms=False, active_storms=None):
    global cbar_obj
    ax.clear()
    ax.set_zorder(1)

    if data_type == 'elevation':
        # Use vertex data directly, not face averages
        face_data = data  #data is already vertex elevations
        vmin = np.min(data)
        vmax = np.max(data)
		# Normalize data using the original data's vmin and vmax
        norm_data = (face_data - vmin) / (vmax - vmin)      
        colors = plt.cm.terrain(norm_data) #removed clipping as it caused issues with the terrain normalization
        cmap = 'terrain'
        title = 'Elevation'
        cbar_label = 'Elevation (km)'
    else:
        # Default case if data_type is not 'elevation'
        face_data = np.zeros(len(vertices))  # placeholder
        colors = 'gray'
        cmap = 'gray'
        title = 'No Data'
        cbar_label = 'Data Value'


    # Plot the mesh using vertex colors
    mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           triangles=faces,
                           color='white',
                           edgecolor='none',
                           alpha=1)
    mesh.set_array(face_data)
    mesh.set_cmap(cmap)

      # Colorbar Handling
    if cbar_obj is None:
        cbar_obj = plt.colorbar(mesh, ax=ax, shrink=0.5, aspect=20)
    else:
        cbar_obj.mappable.set_clim(vmin=vmin, vmax=vmax)  # Update colorbar range
        cbar_obj.mappable = mesh
        cbar_obj.update_normal(mesh)  # Update color

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

def get_vertex_index_old(vertex_to_find, vertex_indices_map):
    """Finds the index of a vertex in vertices_np based on its data."""
    vertex_data_to_find = np.array([vertex_to_find.x, vertex_to_find.y, vertex_to_find.z])
    for original_vertex, index in vertex_indices_map.items():
        original_vertex = np.array([original_vertex.x, original_vertex.y, original_vertex.z])
        if np.allclose(vertex_data_to_find, original_vertex):
            return index
    return None  # Should not reach here if vertex_indices_map is consistent

def build_hierarchical_vertex_map(vertices):
    """Build a 3-level dictionary: x → y → z → vertex index"""
    x_map = defaultdict(lambda: defaultdict(dict))
    
    for vertex, index in vertices.items():
        x, y, z = vertex.x, vertex.y, vertex.z
        x_map[x][y][z] = index
    
    return x_map

def get_vertex_index(vertex_to_find, x_map):
    """Find vertex index using hierarchical lookup"""
    x, y, z = vertex_to_find.x, vertex_to_find.y, vertex_to_find.z
    
    try:
        y_map = x_map[x]
        z_map = y_map[y]
        return z_map[z]
    except KeyError:
        return None

if __name__ == "__main__":
    subdivisions = 10
    radius = PLANET_RADIUS_KM  # Use the global planet radius
    num_steps = 2

    world = icosphere.run_simulation(subdivisions=subdivisions, radius=radius, elevationRange=20000.0, steps=num_steps)
    final_faces_cpp = world[-1].faces


    # Convert C++ vectors to NumPy arrays
    num_vertices = len(world[-1].vertices)
    vertices_np = np.zeros((num_vertices, 3))

    num_faces = len(final_faces_cpp)
    elevations_np = np.zeros(num_faces)
    faces_np = np.zeros((num_faces, 3), dtype=int)

    def getWorldPoint(val):
        global vertex_hierarchy
        if vertex_hierarchy is None:
            vertex_hierarchy = build_hierarchical_vertex_map(world[val].vertex_indices)
        for i, v in enumerate(world[val].vertices):
            vertices_np[i, :] = [v.x, v.y, v.z]
        for i, face in enumerate(final_faces_cpp):
            faces_np[i, 0] = get_vertex_index(face.a, vertex_hierarchy)
            faces_np[i, 1] = get_vertex_index(face.b, vertex_hierarchy)
            faces_np[i, 2] = get_vertex_index(face.c, vertex_hierarchy)
            elevations_np[i] = face.average_elevation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    getWorldPoint(0)
    visualize_world_spherical(vertices_np, faces_np, elevations_np, ax, data_type='elevation')

    def update_view():
        visualize_world_spherical(vertices_np, faces_np, elevations_np, ax, data_type='elevation')
        fig.canvas.draw_idle()


    plt.tight_layout()
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    step_slider = Slider(ax=ax_slider, label='Step', valmin=0, 
                         valmax=num_steps - 1, valinit=current_step, valstep=1)
    def update_step(val):
        global current_step
        current_step = int(step_slider.val)
        getWorldPoint(val)
        #label = radio.value_selected
        update_view()
    step_slider.on_changed(update_step)


    plt.subplots_adjust(bottom=0.15) # Add space at the bottom for the slider

    plt.show()