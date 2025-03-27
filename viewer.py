from matplotlib import pyplot as plt
import numpy as np

from pressure import calculate_pressure_with_layers
from globals import cbar_obj

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
		cbar_label = 'Elevation (m)'

	elif data_type == 'temperature':
		# Normalize temperature data (-20 to 40°C)
		norm_data = (data + 20) / 60  # Scale -20°C to 40°C to 0-1
		norm_data = np.clip(norm_data, 0, 1)

		# Create face colors
		face_data = np.mean(norm_data[faces], axis=1)
		colors = plt.cm.coolwarm(face_data)
		cmap = 'coolwarm'
		title = 'Temperature (°C)'
		cbar_label = 'Temperature (°C)'

	elif data_type == 'rainfall':
		# Normalize rainfall data (0 to 200mm)
		norm_data = data / 200
		norm_data = np.clip(norm_data, 0, 1)

		# Create face colors
		face_data = np.mean(norm_data[faces], axis=1)
		colors = plt.cm.Blues(face_data)
		cmap = 'Blues'
		title = 'Rainfall (mm)'
		cbar_label = 'Rainfall (mm)'

	elif data_type == 'pressure':
		# Convert pressure data from hPa to atm
		data_atm = data / 1013.25
		
		# Normalize between typical surface pressure range (in atm)
		vmin = np.percentile(data_atm, 5)  # Use 5th percentile as min
		vmax = np.percentile(data_atm, 95)  # Use 95th percentile as max
		norm_data = (data_atm - vmin) / (vmax - vmin)
		norm_data = np.clip(norm_data, 0, 1)

		# Create face colors
		face_data = np.mean(norm_data[faces], axis=1)
		colors = plt.cm.viridis(face_data)
		cmap = 'viridis'
		title = 'Surface Pressure (atm)'
		cbar_label = 'Pressure (atm)'

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
		cbar_obj.mappable = mesh
		cbar_obj.update_normal(mesh)
		
	cbar_obj.set_label(cbar_label)

	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.set_aspect('equal')
	ax.view_init(elev=30, azim=45)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	ax.set_title(title)