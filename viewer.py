from matplotlib import pyplot as plt
import numpy as np

from pressure import calculate_pressure_with_layers
from globals import PLANET_RADIUS_KM, cbar_obj

def visualize_world_spherical(vertices, faces, data, ax, data_type='elevation', elevations=None, 
							show_clouds=False, cloud_coverage=None, 
							show_storms=False, active_storms=None):
	global cbar_obj
	ax.clear()
	ax.set_zorder(1)

	# Main data visualization
	if data_type == 'elevation':
		vmin, vmax = -10000, 10000
		norm_data = (data - vmin) / (vmax - vmin)
		norm_data = np.clip(norm_data, 0, 1)
		face_data = np.mean(norm_data[faces], axis=1)
		colors = plt.cm.terrain(face_data)
		cmap = 'terrain'
		title = 'Elevation'
		cbar_label = 'Elevation (m)'
	elif data_type == 'temperature':
		norm_data = (data + 20) / 60
		norm_data = np.clip(norm_data, 0, 1)
		face_data = np.mean(norm_data[faces], axis=1)
		colors = plt.cm.coolwarm(face_data)
		cmap = 'coolwarm'
		title = 'Temperature (°C)'
		cbar_label = 'Temperature (°C)'
	elif data_type == 'rainfall':
		norm_data = data / 200
		norm_data = np.clip(norm_data, 0, 1)
		face_data = np.mean(norm_data[faces], axis=1)
		colors = plt.cm.Blues(face_data)
		cmap = 'Blues'
		title = 'Rainfall (mm)'
		cbar_label = 'Rainfall (mm)'
	elif data_type == 'pressure':
		data_atm = data / 1013.25
		vmin = np.percentile(data_atm, 5)
		vmax = np.percentile(data_atm, 95)
		norm_data = (data_atm - vmin) / (vmax - vmin)
		norm_data = np.clip(norm_data, 0, 1)
		face_data = np.mean(norm_data[faces], axis=1)
		colors = plt.cm.viridis(face_data)
		cmap = 'viridis'
		title = 'Surface Pressure (atm)'
		cbar_label = 'Pressure (atm)'

	# Plot the main mesh
	mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
							triangles=faces, color='white',
							edgecolor='none', alpha=1.0)
	mesh.set_array(face_data)
	mesh.set_cmap(cmap)

	# Add cloud overlay if enabled
	if show_clouds and cloud_coverage is not None:
		face_clouds = np.mean(cloud_coverage[faces], axis=1)
		cloud_colors = plt.cm.Blues_r(face_clouds)
		
		# Create a separate transparent mesh for clouds
		cloud_mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
									triangles=faces, alpha=0.0)  # Invisible geometry
		cloud_mesh.set_facecolor(cloud_colors)
		cloud_mesh.set_alpha(0.3)  # Set transparency after creating mesh
		cloud_mesh.set_zorder(0.5)

	# Add storm markers if enabled
	if show_storms and active_storms is not None:
		for storm in active_storms:
			lat_rad = np.radians(storm.center_lat)
			lon_rad = np.radians(storm.center_lon)
			
			# Calculate position with a small offset to ensure visibility
			offset_factor = 1.02  # 2% larger than planet surface
			x = np.cos(lat_rad) * np.cos(lon_rad) * (PLANET_RADIUS_KM * offset_factor + 50)
			y = np.cos(lat_rad) * np.sin(lon_rad) * (PLANET_RADIUS_KM * offset_factor + 50)
			z = np.sin(lat_rad) * (PLANET_RADIUS_KM * offset_factor + 50)
			
			# Determine storm properties
			color = 'red' if storm.pressure_anomaly < 0 else 'blue'
			size = np.clip(abs(storm.pressure_anomaly) * 2, 10, 40)  # Increased base size
			
			# Use a more visible marker with edge
			ax.scatter([x], [y], [z], 
					c=color, 
					s=size, 
					edgecolor='white',
					linewidth=1.5,
					marker='o',
					alpha=0.9)
			
			# Add a smaller inner marker for better visibility
			ax.scatter([x], [y], [z], 
					c=color, 
					s=size/3, 
					edgecolor='none',
					marker='o',
					alpha=1.0)

	# Colorbar
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