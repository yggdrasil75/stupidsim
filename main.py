import struct
from typing import Dict, List, Tuple, Set, Optional
from matplotlib import pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import RadioButtons, Slider
import numpy as np
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as colors
import random

from globals import ELEVATION_MOUNTAIN_BASE, ELEVATION_TRENCH_BASE, NORM_ELEVATION, PLATES, RADIUS, SUBDIVISIONS, VMAX, VMIN
from shape import Face, Vertex
from world import worldState, icosphere


def VisualizeWorld(world: worldState, title: str = "World Mesh",
				   color_mode: str = "elevation", # "elevation", "plates", "normal"
				   plate_colors_map: Optional[Dict[int, str]] = None):

	fig = plt.figure(figsize=(12, 10))
	ax = fig.add_subplot(111, projection='3d')

	face_polys = []
	face_colors = []
	edge_lines = []

	has_plates = bool(world.plates and world.vertex_to_plate_id)
	has_elevations = bool(world.elevations)

	if not world.faces:
		print("Warning: No faces found in the world state.")
	elif not world.vertices:
		print("Warning: No vertices found in the world state.")
	else:
		print(f"Visualizing {len(world.faces)} faces and {len(world.vertices)} vertices...")
		count = 0
		skipped_faces = 0

		cmap_elevation = colormaps['gray']
		cmap_water = colormaps['Blues'] # Colormap for water visualization
		norm_elevation_dynamic = NORM_ELEVATION

		total_water_volume_m3 = 0.0 # Initialize total water volume

		for face in world.faces:
			count += 1

			try:
				verts_idx = face.vertices
				face_verts_pos_poly = [world.vertices[i].pos * world.radius for i in verts_idx]

				# --- Determine Face Color ---
				f_color = (0.5, 0.5, 0.5, 0.5) # Default gray

				if color_mode == "elevation":
					# Average elevation of face vertices
					face_elevations = [world.elevations.get(i, 0.0) for i in verts_idx]
					face_water = [world.vertices[i].surface_water for i in verts_idx] # Get surface water

					# Check if all elevations were found
					if len(face_elevations) == 3:
						avg_elevation = sum(face_elevations) / 3.0
						avg_water = sum(face_water) / 3.0 # Average surface water

						# Estimate face area (planar triangle approximation)
						v0 = world.vertices[verts_idx[0]].pos * world.radius
						v1 = world.vertices[verts_idx[1]].pos * world.radius
						v2 = world.vertices[verts_idx[2]].pos * world.radius
						face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

						# Add to total water volume (assuming avg_water is water height in meters)
						face_water_volume = face_area * avg_water
						total_water_volume_m3 += face_water_volume


						# Elevation color (grayscale)
						if norm_elevation_dynamic:
							elevation_color = cmap_elevation(norm_elevation_dynamic(avg_elevation))
							elevation_rgb = np.array(elevation_color[:3]) # RGB for blending
							elevation_alpha = elevation_color[3] # Keep original alpha
						else: # Fallback if norm failed
							elevation_rgb = np.array([0.4, 0.4, 0.4]) # Gray RGB
							elevation_alpha = 0.6

						# Water color (blue)
						water_color_rgb = np.array([0.0, 0.0, 1.0]) # Pure blue RGB

						# Water Alpha - Based on water amount, from 0 to 0.95
						water_alpha = avg_water * 0.95 # Scale water amount to alpha range

						# --- Color Blending: Overlay Water with Alpha ---
						if avg_water > 0:
							# Blend elevation with water color using water_alpha
							blended_rgb = (1 - water_alpha) * elevation_rgb + water_alpha * water_color_rgb
							f_color = tuple(blended_rgb.tolist() + [elevation_alpha]) # Keep elevation alpha
						else:
							# No water, just use elevation color
							f_color = tuple(elevation_color)


					else: # Missing elevation data for some vertices
						f_color = (0.1, 0.1, 0.1, 0.6) # Black/dark gray for missing data

				face_polys.append(face_verts_pos_poly)
				face_colors.append(f_color)

				# --- Prepare Edges for plotting ---
				verts_idx_lines = list(face.vertices) + [face.vertices[0]] # Close the loop
				face_verts_pos_lines = [world.vertices[i].pos * world.radius for i in verts_idx_lines]
				edge_lines.append(face_verts_pos_lines)

			except IndexError:
				# print(f"Warning: Vertex index out of bounds in face {face}. Skipping face.")
				skipped_faces += 1
				continue
			except Exception as e:
				print(f"Warning: Error processing face {face}: {e}. Skipping face.")
				skipped_faces += 1
				continue

		if skipped_faces > 0:
			print(f"Skipped {skipped_faces} faces due to errors.")

		total_water_volume_exaliters = total_water_volume_m3 / 1e15 # Conversion to exaliters

		print(f"Total Surface Water Volume: {total_water_volume_exaliters:.2f} Exaliters")


	if face_polys:
		poly_collection = Poly3DCollection(face_polys, facecolors=face_colors, edgecolors='k', linewidth=0.1) # Added edgecolors='k' for black edges
		ax.add_collection3d(poly_collection)
		print(" Faces added.")
	else:
		print(" No face polygons to plot.")


	# --- Axes and Labels ---
	ax.set_box_aspect([1, 1, 1]) # Equal aspect ratio
	limit = world.radius * 1.1
	ax.set_xlim(-limit, limit)
	ax.set_ylim(-limit, limit)
	ax.set_zlim(-limit, limit)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	elev_min = min(world.elevations.values()) if world.elevations else float('nan')
	elev_max = max(world.elevations.values()) if world.elevations else float('nan')

	title_str = (f'{title}\n(Detail: {world.details}, Verts: {len(world.vertices)}, Faces: {len(world.faces)}, Mode: {color_mode})\n'
                 f'Elevation Range: {elev_min:.0f}m to {elev_max:.0f}m, Water: {total_water_volume_exaliters:.2f} EL')

	ax.set_title(title_str, pad=20) # Added pad for title
	ax.view_init(elev=30, azim=45) # Adjust viewing angle
	ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5) # Added gridlines
	# Background color
	ax.xaxis.set_pane_color((0.9, 0.9, 0.9, 0.1))
	ax.yaxis.set_pane_color((0.9, 0.9, 0.9, 0.1))
	ax.zaxis.set_pane_color((0.9, 0.9, 0.9, 0.1))


	# --- Colorbar / Legend ---
	# Adjust layout rect based on whether a legend or colorbar is added
	legend_rect = [0, 0, 0.85, 1]
	colorbar_rect = [0, 0.02, 0.88, 0.96]
	default_rect = [0, 0, 1, 1]
	current_rect = default_rect

	if color_mode == "elevation" and norm_elevation_dynamic:
		scalar_mappable = cm.ScalarMappable(norm=norm_elevation_dynamic, cmap=cmap_elevation)
		scalar_mappable.set_array([]) # Important!
		cbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.6, aspect=20, label='Elevation (m)', pad=0.08, location='right') # Adjusted location
		cbar.ax.set_ylim(VMIN, VMAX)
		current_rect = colorbar_rect

	plt.tight_layout(rect=current_rect) # Adjust layout for legend/colorbar
	plt.show()


# --- Main Execution ---

def main():
	world_radius_m = RADIUS
	num_plates = PLATES
	subdivision_level = SUBDIVISIONS
	max_plate_speed_deg_yr = 1.5 # Max rotation speed in degrees per year
	max_ang_vel_rad_yr = np.radians(max_plate_speed_deg_yr)

	# Elevation generation parameters
	# Using ridge for divergent instead of trench
	conv_elev = ELEVATION_MOUNTAIN_BASE    # Base elevation boost for convergence (mountains)
	div_elev = ELEVATION_TRENCH_BASE     # Base elevation for divergent boundaries (mid-ocean ridge)
	trans_elev = 200.0    # Minor ridges/troughs for transform faults
	rate_scale = 5.0
	diff_passes = 15      # Number of smoothing passes for elevation diffusion
	diff_factor = 0.10    # How much elevation spreads per pass (0-1)

	world_sim = icosphere(radius=world_radius_m, subdivions=subdivision_level)

	world_sim.plates = world_sim.assign_icosphere_vertices_to_plates(num_plates)


	if world_sim.plates:
		world_sim.assign_random_angular_velocities()

		world_sim._identify_boundaries()

		success = world_sim.calculate_boundary_motions(classification_threshold=0.65)

		if success:
			world_sim.assign_elevations_from_boundaries()
		else:
			print("Skipping elevation assignment due to errors in motion calculation.")

	else:
		print("\nNo plates assigned, skipping tectonic simulation.")

	world_sim.duplicateLayers()

	# Example: Increase surface water on layer 0 vertices to visualize water
	for v in world_sim.vertices:
		if v.layer_id == 0:
			if world_sim.elevations.get(world_sim.vertices.index(v), 0) < -1000: # Deeper water
				v.surface_water = 1.0 # Full water
			elif world_sim.elevations.get(world_sim.vertices.index(v), 0) < 0: # water at lower elevations
				v.surface_water = 0.8 # High water
			elif world_sim.elevations.get(world_sim.vertices.index(v), 0) < 200:
				v.surface_water = 0.3 # Medium water
			else:
				v.surface_water = 0.05 # Little water at higher elevations


	plate_colors = None
	if world_sim.elevations:
		min_elev = min(world_sim.elevations.values())
		max_elev = max(world_sim.elevations.values())
		print(f"Elevation range: min={min_elev}, max={max_elev}")
	else:
		print("world_sim.elevations is empty!")
	if world_sim.plates:
		plate_colors = {}
		num_p = len(world_sim.plates)
		cmap_plates_vis = plt.get_cmap('tab20', num_p if num_p > 0 else 1) # Use tab20, ensure lut >= 1
		for i in range(num_p):
			plate_colors[i] = cmap_plates_vis(i) # Get RGBA tuple


	VisualizeWorld(world_sim,
				   title=f"World Simulation with Elevation and Water Overlay",
				   color_mode="elevation")



if __name__ == "__main__":
	main()