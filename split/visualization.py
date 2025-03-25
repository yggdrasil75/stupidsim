# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from constants import cbar_obj  # Import the global variable

def visualize_world_spherical(vertices, faces, data, ax, data_type='elevation', elevations=None):
    global cbar_obj
    ax.clear()

    if data_type == 'elevation':
        # Normalize elevations for coloring (using percentile to handle outliers)
        vmin = np.percentile(data, 5)
        vmax = np.percentile(data, 95)
        norm_data = (data - vmin) / (vmax - vmin)
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors based on vertex elevations
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.terrain(face_data)
        cmap = 'terrain'
        title = 'Elevation'

    elif data_type == 'temperature':
        # Normalize temperature data (-20 to 40°C)
        norm_data = (data + 20) / 60  # Scale -20°C to 40°C to 0-1
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.coolwarm(face_data)
        cmap = 'coolwarm'
        title = 'Temperature (°C)'

    elif data_type == 'rainfall':
        # Normalize rainfall data (0 to 200mm)
        norm_data = data / 200
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.Blues(face_data)
        cmap = 'Blues'
        title = 'Rainfall (mm)'

    elif data_type == 'pressure':
        # Normalize pressure data (950 to 1050 hPa) at surface level
        # Use the temperature data passed to the function for pressure calculation
        surface_pressures = np.array([calculate_pressure_with_layers(elevations[i], data[i])
                                  for i in range(len(elevations))]) # Use 'data' which is temperatures
        norm_data = (surface_pressures - 950) / 100
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.viridis(face_data)
        cmap = 'viridis'
        title = 'Surface Pressure (hPa)'


    # Plot the mesh with face colors
    mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                          triangles=faces, color='white',
                          edgecolor='none', alpha=1.0)
    mesh.set_array(face_data)
    mesh.set_cmap(cmap)

    # Add or update colorbar
    if cbar_obj is None:
        cbar_obj = plt.colorbar(mesh, ax=ax, shrink=0.5)
    else:
        cbar_obj.mappable.set_clim(vmin=np.min(face_data), vmax=np.max(face_data)) # Optional: update color limits if needed
        cbar_obj.mappable.set_array(face_data) # Update data for the colorbar

    if data_type == 'elevation':
        cbar_obj.set_label('Elevation (m)')
    elif data_type == 'temperature':
        cbar_obj.set_label('Temperature (°C)')
    elif data_type == 'rainfall':
        cbar_obj.set_label('Rainfall (mm)')
    elif data_type == 'pressure':
        cbar_obj.set_label('Pressure (hPa)') # Set label for pressure
    cbar_obj.cmap = cmap

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect('equal')
    ax.view_init(elev=30, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title)