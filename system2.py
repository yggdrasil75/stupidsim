from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict

from holder.globals import EARTH_RADIUS_TO_AU
from holder.vertex import SpaceVertex, Vertex
from world import World

class System:
	def __init__(self, radius: float = 2.0, resolution: int = 1000):
		self.radius = radius #in AU
		self.density = resolution
		self.vertex_grid = defaultdict(list)
		self.vertices: List[SpaceVertex] = []
		self.worlds: List[World] = []
		self.create_system()

	def create_system(self):
		"""Generate 3D Hexagonal Close Packed grid"""
		# HCP parameters
		a = self.density  # spacing between vertices
		c = a * np.sqrt(8/3)  # vertical spacing
		
		# Calculate grid dimensions
		layers = int(2 * self.radius / c) + 1
		max_per_layer = int(2 * self.radius / a) + 1
		
		# Generate layers
		for layer in range(-layers//2, layers//2 + 1):
			z = layer * c
			# Alternate layer pattern
			x_offset = 0 if layer % 2 == 0 else a/2
			y_offset = 0 if layer % 2 == 0 else a*np.sqrt(3)/2
			
			# Generate hex pattern in layer
			for i in range(-max_per_layer, max_per_layer + 1):
				for j in range(-max_per_layer, max_per_layer + 1):
					x = i * a + x_offset
					y = j * a * np.sqrt(3) + y_offset
					
					# Check if within system radius
					if np.sqrt(x**2 + y**2 + z**2) <= self.radius:
						vertex = SpaceVertex(x, y, z)
						self.vertices.append(vertex)
						# Store for spatial lookup
						key = self._quantize_position(x, y, z)
						self.vertex_grid[key].append(vertex)
		
		# Assign neighbors after all vertices created
		self._assign_neighbors()

	def _quantize_position(self, x, y, z, precision=3):
		"""Create spatial hash key for neighbor lookup"""
		return (round(x/self.density, precision), 
				round(y/self.density, precision), 
				round(z/self.density, precision))

	def _assign_neighbors(self):
		"""Assign nearest neighbors to each vertex"""
		neighbor_offsets = [
			# Same layer neighbors (6 in hexagonal pattern)
			(1, 0, 0), (-1, 0, 0),
			(0.5, np.sqrt(3)/2, 0), (-0.5, np.sqrt(3)/2, 0),
			(0.5, -np.sqrt(3)/2, 0), (-0.5, -np.sqrt(3)/2, 0),
			
			# Adjacent layer neighbors (3 above and 3 below)
			(0, np.sqrt(3)/3, np.sqrt(8/3)/2),
			(0.5, -np.sqrt(3)/6, np.sqrt(8/3)/2),
			(-0.5, -np.sqrt(3)/6, np.sqrt(8/3)/2),
			
			(0, np.sqrt(3)/3, -np.sqrt(8/3)/2),
			(0.5, -np.sqrt(3)/6, -np.sqrt(8/3)/2),
			(-0.5, -np.sqrt(3)/6, -np.sqrt(8/3)/2)
		]
		
		for vertex in self.vertices:
			base_key = self._quantize_position(*(vertex.pos))
			
			for dx, dy, dz in neighbor_offsets:
				# Calculate neighbor position
				nx = vertex.pos[0] + dx * self.density
				ny = vertex.pos[1] + dy * self.density
				nz = vertex.pos[2] + dz * self.density
				
				# Look up potential neighbors
				neighbor_key = self._quantize_position(nx, ny, nz)
				for candidate in self.vertex_grid.get(neighbor_key, []):
					# Verify actual distance is within threshold
					dist = np.sqrt((vertex.x-candidate.x)**2 + 
										(vertex.y-candidate.y)**2 + 
										(vertex.z-candidate.z)**2)
					if dist <= 1.1 * self.density:  # Small tolerance
						vertex.add_neighbor(candidate)
	
	def plot_system(self, max_vertices=500):
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection='3d')
		
		# Plot vertices
		verts = self.vertices[:max_vertices]
		xs = [v.pos[0] for v in verts]
		ys = [v.pos[1] for v in verts]
		zs = [v.pos[2] for v in verts]
		ax.scatter(xs, ys, zs, s=10, alpha=0.6)
		
		# Plot some neighbor connections
		for v in verts[:10]:  # Just show for first 10 vertices
			for n in v.neighbors:
				ax.plot([v.x, n.x], [v.y, n.y], [v.z, n.z], 'gray', alpha=0.3)
		
		ax.set_title(f"HCP Star System ({len(self.vertices)} vertices)")
		plt.show()


solar_system = System(radius=1.0, resolution=0.15)
solar_system.create_hcp_grid()

# Examine a vertex and its neighbors
sample_vertex = solar_system.vertices[100]
print(f"Vertex: {sample_vertex}")
print(f"Neighbors: {sample_vertex.neighbors}")

# Verify neighbor count (should be 12 for interior vertices)
print(f"Number of neighbors: {len(sample_vertex.neighbors)}")

System.plot_system(solar_system)