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

from globals import ELEVATION_MOUNTAIN_BASE, ELEVATION_TRENCH_BASE, PLATES, RADIUS, SUBDIVISIONS
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

		cmap_elevation = colormaps['terrain']
		norm_elevation_dynamic = colors.Normalize(vmin=-15000, vmax=15000) # Fixed range
		# norm_elevation_dynamic = None # Old dynamic range
		# if color_mode == "elevation" and has_elevations:
		# 	all_elevs = list(world.elevations.values())
		# 	min_elev, max_elev = min(all_elevs), max(all_elevs)
		# 	print(f" Elevation range for coloring: {min_elev:.0f}m to {max_elev:.0f}m")
		# 	norm_elevation_dynamic = colors.Normalize(vmin=min_elev, vmax=max_elev) # Old dynamic range

		cmap_normal = colormaps['viridis']
		norm_normal = colors.Normalize(vmin=-1.0, vmax=1.0)


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
					# Check if all elevations were found
					if len(face_elevations) == 3:
						avg_elevation = sum(face_elevations) / 3.0
						if norm_elevation_dynamic:
							f_color = cmap_elevation(norm_elevation_dynamic(avg_elevation))
						else: # Fallback if norm failed
							f_color = (0.4, 0.4, 0.4, 0.6)
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
	ax.set_title(f'{title}\n(Detail: {world.details}, Verts: {len(world.vertices)}, Faces: {len(world.faces)}, Mode: {color_mode})', pad=20) # Added pad for title
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
		cbar.ax.set_ylim(-15000, 15000) 
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

	# Choose visualization mode: "elevation", "plates", "normal", "boundary_type"
	# VisualizeWorld(world_sim,
	# 			   title=f"Plate Assignments (Plates: {num_plates}, Detail: {subdivision_level})",
	# 			   color_mode="plates",
	# 			   plate_colors_map=plate_colors)

	VisualizeWorld(world_sim,
				   title=f"World Simulation with Tectonic Elevation",
				   color_mode="elevation")

	# VisualizeWorld(world_sim,
	# 			   title=f"Boundary Types",
	# 			   color_mode="boundary_type")


if __name__ == "__main__":
	main()