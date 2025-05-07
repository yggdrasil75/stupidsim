from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict

from globals import DEVICE
from holder.globals import EARTH_RADIUS_TO_AU
from holder.vertex import SpaceVertex, Vertex
from world import World

class System:
	def __init__(self, radius: float = 2.0, resolution: float = 0.05):
		self.radius = radius  # in AU
		self.resolution = resolution
		self.vertices: List[SpaceVertex] = []
		self.worlds: List[World] = []
		self.create_system()
		self._assign_neighbors()

	def create_system(self):
		"""Vectorized HCP grid generation using PyTorch"""
		a = self.resolution
		c = a * np.sqrt(8/3)  # vertical spacing
		
		# Calculate grid dimensions
		layers = int(2 * self.radius / c) + 1
		max_per_layer = int(2 * self.radius / a) + 1
		
		# Generate layer coordinates
		z_layers = torch.arange(-layers//2, layers//2 + 1, dtype=torch.float32) * c
		
		# Precompute all possible x,y positions
		i = torch.arange(-max_per_layer, max_per_layer + 1, dtype=torch.float32)
		j = torch.arange(-max_per_layer, max_per_layer + 1, dtype=torch.float32)
		ii, jj = torch.meshgrid(i, j, indexing='ij')
		
		all_vertices = []
		
		for layer_idx, z in enumerate(z_layers):
			# Alternate layer pattern
			x_offset = 0 if layer_idx % 2 == 0 else a/2
			y_offset = 0 if layer_idx % 2 == 0 else a*np.sqrt(3)/2
			
			# Calculate all positions in this layer
			x = ii * a + x_offset
			y = jj * a * np.sqrt(3) + y_offset
			
			# Combine and filter
			pos = torch.stack([x.ravel(), y.ravel(), torch.full_like(x.ravel(), z)], dim=1)
			dist = torch.norm(pos, dim=1)
			mask = dist <= self.radius
			
			all_vertices.append(pos[mask])
		
		# Combine all valid vertices
		if all_vertices:
			all_vertices = torch.cat(all_vertices).cpu().numpy()
			self.vertices = [SpaceVertex(*pos) for pos in all_vertices]
		
		# Build spatial index
		self._build_spatial_index()

	def _build_spatial_index(self):
		"""Build a spatial index using bounding boxes for faster neighbor queries"""
		if not self.vertices:
			return
			
		# Create a grid of bounding boxes
		self.cell_size = self.resolution * 1.3  # Each cell is 2x the resolution
		self.bbox_dict = defaultdict(list)
		
		for idx, vertex in enumerate(self.vertices):
			x, y, z = vertex.pos
			# Determine which bounding box cell this vertex belongs to
			cell_x = int(x // self.cell_size)
			cell_y = int(y // self.cell_size)
			cell_z = int(z // self.cell_size)
			self.bbox_dict[(cell_x, cell_y, cell_z)].append(vertex)

	def _get_nearby_cells(self, cell: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
		"""Get all neighboring cells (including diagonals) for a given cell"""
		x, y, z = cell
		nearby = []
		for dx in [-1, 0, 1]:
			for dy in [-1, 0, 1]:
				for dz in [-1, 0, 1]:
					nearby.append((x + dx, y + dy, z + dz))
		return nearby

	def _assign_neighbors(self):
		if not self.vertices:
			return

		# Create a mapping from Vertex object to its index for quick lookup.
		# This is necessary because self.bbox_dict stores Vertex objects, but
		# the Vertex.neighbors dictionary expects integer indices as keys.
		vertex_to_idx = {vertex: i for i, vertex in enumerate(self.vertices)}

		# Define a threshold for considering two vertices as neighbors.
		# In an HCP-like lattice, nearest neighbors are at distance 'a' (self.resolution).
		# We use a slightly larger threshold (e.g., 1.05 * resolution) to account
		# for floating-point inaccuracies or minor deviations from a perfect lattice.
		neighbor_dist_threshold = self.resolution * 1.05 

		for idx1, v1 in enumerate(self.vertices):
			v1_pos = v1.pos  # v1.pos is a NumPy array [x, y, z]

			# Determine the grid cell for v1.
			# self.cell_size is set in _build_spatial_index.
			# It's guaranteed to be valid because _build_spatial_index is called
			# in __init__ after create_system, and we've handled empty self.vertices.
			cell_x1 = int(v1_pos[0] // self.cell_size)
			cell_y1 = int(v1_pos[1] // self.cell_size)
			cell_z1 = int(v1_pos[2] // self.cell_size)
			current_v1_cell_coords = (cell_x1, cell_y1, cell_z1)

			# Get a list of cells to search (v1's cell and its 26 direct neighbors)
			candidate_cell_coords_list = self._get_nearby_cells(current_v1_cell_coords)

			for cell_coords_key in candidate_cell_coords_list:
				# self.bbox_dict is a defaultdict(list). If a key is accessed that
				# wasn't populated during _build_spatial_index, it yields an empty list.
				# No explicit `if cell_coords_key in self.bbox_dict:` check is strictly
				# needed due to defaultdict, but iterating an empty list is harmless.
				
				# Iterate over vertex objects (v2_obj) stored in this bounding box cell
				for v2_obj in self.bbox_dict[cell_coords_key]:
					# Get the original index of v2_obj using the precomputed map
					idx2 = vertex_to_idx[v2_obj]

					if idx1 == idx2:
						continue  # A vertex cannot be its own neighbor

					# Calculate distance using the Vertex.distance_to method.
					# This method is lru_cached.
					distance = v1.distance_to(v2_obj)

					if distance < neighbor_dist_threshold:
						# Add neighbor relationship using indices.
						# The Vertex.add_neighbor_preweighted method stores:
						#   {neighbor_id: int, distance: float}
						# This is consistent with Vertex.neighbors: dict[int, float] type hint.
						v1.add_neighbor_preweighted(idx2, distance)
						
						# Ensure symmetry: if v1 is a neighbor of v2_obj, then v2_obj
						# must also be a neighbor of v1 with the same distance.
						v2_obj.add_neighbor_preweighted(idx1, distance)

	def _segments_intersect(self, a1, a2, b1, b2):
		"""Check if line segments a1-a2 and b1-b2 intersect in 3D space using PyTorch"""
		a1 = torch.tensor(a1, dtype=torch.float32)
		a2 = torch.tensor(a2, dtype=torch.float32)
		b1 = torch.tensor(b1, dtype=torch.float32)
		b2 = torch.tensor(b2, dtype=torch.float32)
		
		da = a2 - a1
		db = b2 - b1
		cross = torch.cross(da, db)
		denom = torch.dot(cross, cross)
		
		if denom < 1e-10:
			return False
			
		diff = b1 - a1
		t = torch.dot(torch.cross(diff, db), cross) / denom
		u = torch.dot(torch.cross(diff, da), cross) / denom
		
		return 0 <= t <= 1 and 0 <= u <= 1

	def plot_system(self):
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection='3d')
		
		verts = self.vertices
		pos = np.array([v.pos for v in verts])
		ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10, alpha=0.6)
		
		# Plot connections
		for v in verts:
			if v.neighbors:
				neighbors_pos = np.array([self.vertices[n].pos for n in v.neighbors.keys()])
				for n_pos in neighbors_pos:
					ax.plot([v.pos[0], n_pos[0]], 
							[v.pos[1], n_pos[1]], 
							[v.pos[2], n_pos[2]], 'gray', alpha=0.3)
		
		ax.set_title(f"HCP System ({len(self.vertices)} vertices)")
		plt.show()


solar_system = System(radius=1.0, resolution=0.1)

System.plot_system(solar_system)