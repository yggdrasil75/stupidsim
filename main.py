from collections import defaultdict
from functools import lru_cache
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np
import icosphere
from mpl_toolkits.mplot3d import Axes3D

PLANET_RADIUS_KM = 6371.0
SUBDIVISIONS = 3
NUM_STEPS = 12
cbar_obj = None  # Global colorbar object to update
current_step = 0
vertex_hierarchy = None



def visualize_world_spherical(vertices, faces, data, ax, data_type):
    global cbar_obj
    ax.clear()
    ax.set_zorder(1)

    if data_type == 'elevation':
        # Use vertex data directly, not face averages
        face_data = data  #data is already vertex elevations
        vmin = np.min(data)
        vmax = np.max(data)
        print(f"Min Elevation: {vmin}, Max Elevation: {vmax}")
		# Normalize data using the original data's vmin and vmax
        norm_data = (face_data - vmin) / (vmax - vmin)      
        colors = plt.cm.terrain(norm_data) #removed clipping as it caused issues with the terrain normalization
        cmap = 'terrain'
        title = 'Elevation'
        cbar_label = 'Elevation (km)'
    elif data_type == 'Water':
        # Use vertex data directly, not face averages
        face_data = data  #data is already vertex elevations
        vmin = np.min(data)
        vmax = np.max(data)
        print(f"Min fluid: {vmin}, Max fluid: {vmax}")
		# Normalize data using the original data's vmin and vmax
        norm_data = (face_data - vmin) / (vmax - vmin)      
        colors = plt.cm.terrain(norm_data) #removed clipping as it caused issues with the terrain normalization
        cmap = 'Blues'
        title = 'Surface Water'
        cbar_label = 'water'
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
    subdivisions = SUBDIVISIONS
    radius = PLANET_RADIUS_KM  # Use the global planet radius
    num_steps = NUM_STEPS

    world = icosphere.run_simulation(subdivisions=subdivisions, elevationRange=20000.0, steps=num_steps, totalWaterZL=1386.0)
    final_faces_cpp = world[-1].faces


    # Convert C++ vectors to NumPy arrays
    num_vertices = len(world[-1].vertices)
    vertices_np = np.zeros((num_vertices, 3))

    num_faces = len(final_faces_cpp)
    elevations_np = np.zeros(num_faces)
    surfaceWater_np = np.zeros(num_faces)
    faces_np = np.zeros((num_faces, 3), dtype=int)
    current_data_type='elevation'
    figdata = np.zeros(num_faces)
    for i, v in enumerate(world[-1].vertices):
        vertices_np[i, :] = [v.x, v.y, v.z]

    def getWorldPoint(val):
        global vertex_hierarchy, figdata
        if vertex_hierarchy is None:
            vertex_hierarchy = build_hierarchical_vertex_map(world[val].vertex_indices)
        for i, face in enumerate(final_faces_cpp):
            faces_np[i, 0] = get_vertex_index(face.a, vertex_hierarchy)
            faces_np[i, 1] = get_vertex_index(face.b, vertex_hierarchy)
            faces_np[i, 2] = get_vertex_index(face.c, vertex_hierarchy)
            elevations_np[i] = face.average_elevation
            surfaceWater_np[i] = face.surface_water
            if current_data_type == 'elevation':
                figdata = elevations_np
            elif current_data_type == 'Water':
                figdata = surfaceWater_np
            #print(elevations_np[i])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create axes for the radio buttons
    rax = plt.axes([0.05, 0.7, 0.15, 0.15])
    radio = RadioButtons(rax, ('Elevation', 'Water'), active=0)
    

    getWorldPoint(0)
    visualize_world_spherical(vertices_np, faces_np, figdata, ax, data_type=current_data_type)

    def update_view():
        visualize_world_spherical(vertices_np, faces_np, data=figdata, ax=ax, data_type=current_data_type)
        fig.canvas.draw_idle()


    #plt.tight_layout()
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    step_slider = Slider(ax=ax_slider, label='Step', valmin=0, 
                         valmax=num_steps - 1, valinit=current_step, valstep=1)
    
    
    def update_data_type(label):
        global figdata, current_data_type
        if label == 'Elevation':
            current_data_type = 'elevation'
            figdata = elevations_np
        elif label == 'Water':
            current_data_type = 'Water'
            figdata = surfaceWater_np
        
        update_view()
    radio.on_clicked(update_data_type)

    def update_step(val):
        global current_step, figdata
        current_step = int(step_slider.val)
        getWorldPoint(val)
        
        # Use the current data type to determine which data to show
        if current_data_type == 'elevation':
            figdata = elevations_np
        else:
            figdata = surfaceWater_np
            
        update_view()

    step_slider.on_changed(update_step)


    plt.subplots_adjust(bottom=0.15, left=0.2) # Add space at the bottom for the slider

    plt.show()