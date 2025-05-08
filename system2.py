import heapq
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
import numpy as np
import numba
import torch
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict

from globals import DEVICE
from holder.globals import EARTH_RADIUS_TO_AU
from holder.vertex import SpaceVertex, StellarVertex, Vertex
from util import get_timing_stats, print_timing_stats, time_function
from world import World


class Star:
    def __init__(self, mass: float, radius: float, temperature: float, color: str = 'yellow'):
        self.mass = mass
        self.radius = radius  # in AU
        self.temperature = temperature  # in Kelvin
        self.color = color
        self.vertices: List[StellarVertex] = []
        self._generate_icosahedron()
        for v in self.vertices:
            v.color = self.color

    def _generate_icosahedron(self, subdivisions: int = 2):
        """Generate an icosahedron approximation of the star's surface"""
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Icosahedron vertices (normalized)
        vertices = [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
        ]
        
        # Normalize and scale vertices
        vertices = [self.radius * np.array(v) / np.linalg.norm(v) for v in vertices]
        
        self.vertices = [StellarVertex(*v) for v in vertices]

    def __repr__(self):
        return f"Star(M={self.mass}M☉, R={self.radius}R☉, T={self.temperature}K)"

@numba.jit(nopython=True)
def assign_neighbors_numba(positions: np.ndarray, resolution: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(positions)
    resolution_sq = (resolution * 1.7)**2
    v1_indices = []
    v2_indices = []
    distances = []

    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            
            if dist_sq <= resolution_sq:
                dist = np.sqrt(dist_sq)
                v1_indices.append(i)
                v2_indices.append(j)
                distances.append(dist)

    return np.array(v1_indices), np.array(v2_indices), np.array(distances)

class System:
    def __init__(self, radius: float = 2.0, resolution: float = 0.05):
        self.radius = radius  # in AU
        self.resolution = resolution
        self.vertices: List[SpaceVertex] = []
        self.worlds: List[World] = []
        self.stars: List[Star] = []
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

    @time_function
    def _build_spatial_index(self):
        """Build a spatial index using bounding boxes for faster neighbor queries"""
        if not self.vertices:
            return
            
        # Create a grid of bounding boxes
        self.cell_size = self.resolution * 2  # Each cell is 2x the resolution
        self.bbox_dict = defaultdict(list)
        
        for idx, vertex in enumerate(self.vertices):
            x, y, z = vertex.pos
            # Determine which bounding box cell this vertex belongs to
            cell_x = int(x // self.cell_size)
            cell_y = int(y // self.cell_size)
            cell_z = int(z // self.cell_size)
            self.bbox_dict[(cell_x, cell_y, cell_z)].append(idx)

    def _get_nearby_cells(self, cell: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get all neighboring cells (including diagonals) for a given cell"""
        x, y, z = cell
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nearby.append((x + dx, y + dy, z + dz))
        return nearby

    def add_star(self, star: Star):
        """Add a star to the system and remove vertices within its radius"""
        self.stars.append(star)
        
        # Remove vertices inside the star's radius
        star_pos = np.zeros(3)  # Center of system
        remaining_vertices = []
        
        for vertex in self.vertices:
            dist = np.linalg.norm(vertex.pos - star_pos)
            if dist > star.radius:
                remaining_vertices.append(vertex)
        
        self.vertices = remaining_vertices
        
        # Add stellar vertices
        self.vertices.extend(star.vertices)
        
        # Rebuild spatial index and neighbors
        self._build_spatial_index()
        self._assign_neighbors_star(star)
        
    @time_function
    # def _assign_neighbors(self):

    # 	positions = np.array([v.pos for v in self.vertices])
    # 	v1_indices, v2_indices, distances = assign_neighbors_numba(positions, self.resolution)
        
    # 	for i, j, dist in zip(v1_indices, v2_indices, distances):
    # 		v1 = self.vertices[i]
    # 		v2 = self.vertices[j]
    # 		v1.add_neighbor_preweighted(j, dist)
    # 		v2.add_neighbor_preweighted(i, dist)
    def _assign_neighbors(self):
        resolution = self.resolution * 1.7
        for cell in self.bbox_dict:
            neighborcells = self._get_nearby_cells(cell)
            vidxs = list(cell)
            for cell2 in neighborcells:
                vidxs.extend(cell2)
            vertles = [self.vertices[v] for v in vidxs]
            vertlepos = np.array([v.pos for v in vertles])
            positions = vertlepos
            diff = np.abs(positions[:, None, :] - positions[None, :, :])
            mask = np.all(diff <= resolution, axis=2)
            np.fill_diagonal(mask, False)
            v_ind, v_ids = np.where(mask)
            for v_idx, v_idx2 in zip(v_ind, v_ids):
                if v_idx < v_idx2:
                    v = self.vertices[v_idx]
                    v2 = self.vertices[v_idx2]
                    dist = v.distance_to(v2)
                    v.add_neighbor_preweighted(v_idx2, dist)
                    v2.add_neighbor_preweighted(v_idx, dist)

    @time_function
    def _assign_neighbors_star(self, star):
        """Special neighbor assignment that connects stellar and space vertices"""
        
        positions = np.array([v.pos for v in self.vertices])
        resolution = self.resolution * 1.7
        
        # First connect all non-stellar vertices normally
        space_vertex_indices = [i for i, v in enumerate(self.vertices) if not hasattr(v, 'is_stellar')]
        stellar_vertex_indices = [i for i, v in enumerate(self.vertices) if hasattr(v, 'is_stellar')]
        
        # Connect space vertices to each other
        space_positions = positions[space_vertex_indices]
        diff = np.abs(space_positions[:, None, :] - space_positions[None, :, :])
        mask = np.all(diff <= resolution, axis=2)
        np.fill_diagonal(mask, False)
        v_ind, v_ids = np.where(mask)
        
        for i, j in zip(v_ind, v_ids):
            if i < j:
                idx1 = space_vertex_indices[i]
                idx2 = space_vertex_indices[j]
                v1 = self.vertices[idx1]
                v2 = self.vertices[idx2]
                dist = v1.distance_to(v2)
                v1.add_neighbor_preweighted(idx2, dist)
                v2.add_neighbor_preweighted(idx1, dist)
        
        # Connect stellar vertices to nearby space vertices
        if stellar_vertex_indices and space_vertex_indices:
            stellar_positions = positions[stellar_vertex_indices]
            space_positions = positions[space_vertex_indices]
            
            # Find space vertices near stellar surface
            diff = np.abs(stellar_positions[:, None, :] - space_positions[None, :, :])
            distances = np.linalg.norm(diff, axis=2)
            connection_threshold = resolution * 1.5
            
            for s_idx, row in enumerate(distances):
                nearby = np.where(row <= connection_threshold)[0]
                for sp_idx in nearby:
                    stellar_idx = stellar_vertex_indices[s_idx]
                    space_idx = space_vertex_indices[sp_idx]
                    
                    v_stellar = self.vertices[stellar_idx]
                    v_space = self.vertices[space_idx]
                    
                    dist = v_stellar.distance_to(v_space)
                    v_stellar.add_neighbor_preweighted(space_idx, dist)
                    v_space.add_neighbor_preweighted(stellar_idx, dist)


    # def plot_system(self):
    # 	fig = plt.figure(figsize=(10, 8))
    # 	ax = fig.add_subplot(111, projection='3d')
        
    # 	# Separate stellar and space vertices
    # 	space_verts = [v for v in self.vertices if not hasattr(v, 'is_stellar')]
    # 	stellar_verts = [v for v in self.vertices if hasattr(v, 'is_stellar')]
        
    # 	if space_verts:
    # 		space_pos = np.array([v.pos for v in space_verts])
    # 		distances = np.linalg.norm(space_pos, axis=1)
    # 		max_dist = np.max(distances) if len(distances) > 0 else 1
    # 		norm_distances = distances / max_dist
            
    # 		sc = ax.scatter(space_pos[:, 0], space_pos[:, 1], space_pos[:, 2],
    # 						s=5, alpha=0.6, c=norm_distances, cmap='viridis')
        
    # 	if stellar_verts:
    # 		stellar_pos = np.array([v.pos for v in stellar_verts])
    # 		colors = [v.color for v in stellar_verts]
    # 		ax.scatter(stellar_pos[:, 0], stellar_pos[:, 1], stellar_pos[:, 2],
    # 					s=20, alpha=1.0, c=colors, edgecolors='black')
        
    # 	# Plot connections
    # 	for i, v in enumerate(self.vertices):
    # 		if v.neighbors:
    # 			for n_idx in v.neighbors:
    # 				n_pos = self.vertices[n_idx].pos
    # 				color = 'red' if hasattr(v, 'is_stellar') or hasattr(self.vertices[n_idx], 'is_stellar') else 'lightgray'
    # 				alpha = 0.3 if color == 'lightgray' else 0.6
    # 				ax.plot([v.pos[0], n_pos[0]], 
    # 						[v.pos[1], n_pos[1]], 
    # 						[v.pos[2], n_pos[2]], 
    # 						color=color, alpha=alpha, linewidth=0.5)
        
    # 	ax.set_title(f"System with {len(self.stars)} star(s) and {len(self.vertices)} vertices")
    # 	plt.show()


    def plot_system(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        verts = self.vertices
        pos = np.array([v.pos for v in verts])
        
        # Calculate distances from origin for all vertices
        distances = np.linalg.norm(pos, axis=1)
        
        # Normalize distances for color mapping (0 to 1)
        max_dist = np.max(distances)
        norm_distances = distances / max_dist if max_dist > 0 else distances
        
        # Create a colormap (using viridis, but you can choose any)
        cmap = plt.cm.viridis
        
        # Plot all vertices with color based on distance
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                        s=10, alpha=0.8, 
                        c=norm_distances, cmap=cmap)
        
        # Add colorbar to show the distance scale
        cbar = fig.colorbar(sc, ax=ax, shrink=0.5)
        cbar.set_label('Normalized Distance from Origin')
        
        
        # for v in verts:
        # 	if v.neighbors:
        # 		neighbors_pos = np.array([self.vertices[n].pos for n in v.neighbors.keys()])
        # 		for n_pos in neighbors_pos:
        # 			ax.plot([v.pos[0], n_pos[0]], 
        # 					[v.pos[1], n_pos[1]], 
        # 					[v.pos[2], n_pos[2]], 
        # 					color='lightgray', alpha=0.15, linewidth=0.5)
        
        ax.set_title(f"HCP System ({len(self.vertices)} vertices)")
        plt.show()


solar_system = System(radius=1.0, resolution=0.1)
#sun = Star(1, 0.1, 5778, 'yellow')
#solar_system.add_star(sun)
print_timing_stats()
solar_system.plot_system()
