import struct
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np
from collections import defaultdict
import matplotlib.cm as cm # Import colormaps
import matplotlib.colors as colors # Import color normalization

PHI = (1.0 + np.sqrt(5.0)) / 2.0

class Vertex:
	def __init__(self, x, y, z):
		self.pos = np.array([float(x), float(y), float(z)], dtype=np.float64)
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)

	def normalize(self):
		norm = np.linalg.norm(self.pos)
		# Increase epsilon slightly for robustness
		if abs(norm) > 1e-10:
			self.pos /= norm
			self.x, self.y, self.z = self.pos
		# else: # Optional: handle zero vector case if it occurs
			# print("Warning: Zero vector encountered during normalization.")
		return self

	def dotProd(self, other):
		return np.dot(self.pos, other.pos)

	def crossProduct(self, other):
		cross_pos = np.cross(self.pos, other.pos)
		return Vertex(cross_pos[0], cross_pos[1], cross_pos[2])

	def __hash__(self):
		hx = struct.unpack('I', struct.pack('f', self.x))[0]
		hy = struct.unpack('I', struct.pack('f', self.y))[0]
		hz = struct.unpack('I', struct.pack('f', self.z))[0]
		return hx ^ (hy << 1) ^ (hz << 2)

	def __eq__(self, other):
		if not isinstance(other, Vertex):
			return False
		# Use a slightly larger tolerance if needed, depends on calculations
		return np.allclose(self.pos, other.pos, atol=1e-10)

	def __repr__(self):
		return f"Vertex({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


class Face:
	def __init__(self, *vertices: int):
		if len(vertices) < 3:
			raise ValueError("Face must have at least 3 vertices")
		self.vertices = tuple(vertices)

	def get_vertices(self, world: 'worldState') -> List[Vertex]:
		"""Helper to get the actual Vertex objects for this face."""
		try:
			return [world.vertices[i] for i in self.vertices]
		except IndexError:
			print(f"Error: Vertex index out of bounds in face {self.vertices}.")
			raise # Re-raise the exception

	def normal(self, world: 'worldState') -> np.ndarray:
		"""Calculates the approximate normal vector of the face (pointing outwards)."""
		verts = self.get_vertices(world)
		if len(verts) < 3:
			return np.array([0.0, 0.0, 0.0]) # Cannot compute normal

		# Use Newell's method for robustness with non-planar faces on the sphere
		# Or simpler: use first three vertices for approximation (assumes relative planarity)
		v0 = verts[0].pos
		v1 = verts[1].pos
		v2 = verts[2].pos

		# Vector edges
		edge1 = v1 - v0
		edge2 = v2 - v0

		face_normal = np.cross(edge1, edge2)
		norm = np.linalg.norm(face_normal)
		if abs(norm) < 1e-10:
			# Handle degenerate case (collinear vertices) - might need a different approach
			# For now, return zero vector or use centroid direction
			centroid = self.centroid(world).pos
			norm_c = np.linalg.norm(centroid)
			if norm_c > 1e-9:
				return centroid / norm_c # Use direction from origin to centroid
			else:
				return np.array([0.0, 0.0, 0.0])

		face_normal /= norm

		# Ensure normal points outward (dot product with vertex position should be positive)
		if np.dot(face_normal, v0) < 0:
			face_normal *= -1.0

		return face_normal

	def centroid(self, world: 'worldState') -> Vertex:
		"""Calculates the geometric centroid of the face vertices."""
		verts = self.get_vertices(world)
		if not verts:
			return Vertex(0,0,0)
		center_pos = np.mean([v.pos for v in verts], axis=0)
		# Normalizing the centroid projects it onto the sphere surface near the face center
		return Vertex(*center_pos).normalize()

	def _validate_face(self, world):
		"""Ensure the face is simple (non-intersecting edges)"""
		n = len(self.vertices)
		if n < 3:
			raise ValueError("Face must have at least 3 vertices")
			
		# For faces with more than 3 vertices, check for self-intersections
		if n > 3:
			# Convert to 2D coordinates for simpler intersection checking
			# We'll project the 3D face onto a 2D plane
			points = [world.vertices[i].pos for i in self.vertices]
			
			# Find the normal vector of the face
			normal = np.cross(points[1] - points[0], points[2] - points[0])
			normal /= np.linalg.norm(normal)
			
			# Find the dominant axis to project onto
			dominant_axis = np.argmax(np.abs(normal))
			axes = [0, 1, 2]
			axes.remove(dominant_axis)
			
			# Project points onto 2D plane
			projected = [(p[axes[0]], p[axes[1]]) for p in points]
			
			# Check all non-adjacent edges for intersections
			for i in range(n):
				for j in range(i + 2, n):
					if (i == 0 and j == n - 1):
						continue  # Skip the closing edge
					a1, a2 = projected[i], projected[(i + 1) % n]
					b1, b2 = projected[j], projected[(j + 1) % n]
					
					if self._segments_intersect(a1, a2, b1, b2):
						raise ValueError("Face has self-intersecting edges")

	def _segments_intersect(self, a1, a2, b1, b2):
		"""Check if two line segments intersect in 2D"""
		def ccw(A, B, C):
			return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
		
		return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)
	def area(self, world):
		n = len(self.vertices)
		if n < 3:
			Exception("something broke. your ngon is 2")
		angles = []
		for i in range(n):
			prev_vert = world.vertices[self.vertices[(i-1) % n]]
			cur_vert = world.vertices[self.vertices[i]]
			nex_vert = world.vertices[self.vertices[(i + 1) % n]]

			vec_prev = prev_vert.pos - cur_vert.pos
			vec_next = nex_vert.pos - cur_vert.pos

			tan_prev = np.cross(cur_vert.pos, np.cross(cur_vert.pos, vec_prev))
			tan_nex = np.cross(cur_vert.pos, np.cross(cur_vert.pos, vec_next))

			cos_angle = np.dot(tan_nex, tan_prev)
			angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
			angles.append(angle)

		anglesum = sum(angles)
		angular_excess = anglesum - (n-2) * np.pi
		return angular_excess * world.radius**2

	def __repr__(self):
		return f"Face{self.vertices}"

class worldState:
	def __init__(self):
		self.vertices: List[Vertex] = []
		self.faces: List[Face] = []
		self.radius: float = 1.0 # Default to unit radius for easier visualization scaling
		self.timestepSeconds: int = 1
		self.gravity: float = 9.81
		self.details: int = 3
		self._subdivision_cache: Dict[Tuple[int, int], int] = {}

	# timeStep methods remain the same...
	def timeStepHour(self) -> float: return float(self.timestepSeconds) / 3600.0
	def timeStepDay(self): return float(self.timestepSeconds) / 86400.0
	def timeStepYear(self): return float(self.timestepSeconds) / 31556952.0

	def icosphereBase(self):
		# Create initial vertices (12 vertices of an icosahedron)
		vertices = [
			Vertex(-1, PHI, 0),
			Vertex(1, PHI, 0),
			Vertex(-1, -PHI, 0),
			Vertex(1, -PHI, 0),
			Vertex(0, -1, PHI),
			Vertex(0, 1, PHI),
			Vertex(0, -1, -PHI),
			Vertex(0, 1, -PHI),
			Vertex(PHI, 0, -1),
			Vertex(PHI, 0, 1),
			Vertex(-PHI, 0, -1),
			Vertex(-PHI, 0, 1)
		]
		
		# Normalize all vertices to unit length
		for v in vertices:
			v.normalize()
		
		# Create initial faces (20 faces of an icosahedron)
		faces = [
			Face(0, 11, 5),
			Face(0, 5, 1),
			Face(0, 1, 7),
			Face(0, 7, 10),
			Face(0, 10, 11),
			Face(1, 5, 9),
			Face(5, 11, 4),
			Face(11, 10, 2),
			Face(10, 7, 6),
			Face(7, 1, 8),
			Face(3, 9, 4),
			Face(3, 4, 2),
			Face(3, 2, 6),
			Face(3, 6, 8),
			Face(3, 8, 9),
			Face(4, 9, 5),
			Face(2, 4, 11),
			Face(6, 2, 10),
			Face(8, 6, 7),
			Face(9, 8, 1)
		]
		
		self.vertices = vertices
		self.faces = faces
		
		# Subdivide the mesh
		self.subdivide()
			
		# Create vertex index mapping
		self.vertex_indices = {i: v for i, v in enumerate(self.vertices)}
		
		return self
	
	def _normalize_all_vertices(self):
		for v in self.vertices:
			v.normalize()

	def _add_vertex(self, vertex: Vertex) -> int:
		"""Adds a vertex if it's not already present (using Vertex __eq__), returns index."""
		# Consider using a dictionary for faster lookups if vertex count is massive
		# For moderate counts, linear search is often acceptable.
		norm_vertex = vertex.normalize() # Ensure vertex is normalized before comparison/addition
		try:
			# Check against existing vertices
			idx = self.vertices.index(norm_vertex)
			return idx
		except ValueError:
			# Vertex not found, add it
			idx = len(self.vertices)
			self.vertices.append(norm_vertex)
			return idx

	def _get_or_create_midpoint(self, v1_idx: int, v2_idx: int) -> int:
		key = tuple(sorted((v1_idx, v2_idx)))
		if key in self._subdivision_cache:
			return self._subdivision_cache[key]

		v1 = self.vertices[v1_idx]
		v2 = self.vertices[v2_idx]
		mid_pos = (v1.pos + v2.pos) / 2.0
		# Midpoint vertex is normalized automatically in _add_vertex
		midpoint_vertex = Vertex(mid_pos[0], mid_pos[1], mid_pos[2])

		# Add vertex (handles normalization and prevents duplicates)
		mid_idx = self._add_vertex(midpoint_vertex)

		self._subdivision_cache[key] = mid_idx
		return mid_idx

	def cubeBase(self):
		self.vertices = [
			Vertex(x, y, z) for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]
		]
		self.faces = [
			Face(0, 1, 3, 2), Face(1, 5, 7, 3), Face(5, 4, 6, 7),
			Face(4, 0, 2, 6), Face(2, 3, 7, 6), Face(4, 5, 1, 0) # Check winding order if needed
		]
		self._normalize_all_vertices()
		self.subdivide()
		return self

	def _get_icosahedron_vertices(self) -> List[Vertex]:
		"""Returns the 12 vertices of a standard icosahedron."""
		return [
			Vertex(-1, PHI, 0), Vertex(1, PHI, 0), Vertex(-1, -PHI, 0), Vertex(1, -PHI, 0),
			Vertex(0, -1, PHI), Vertex(0, 1, PHI), Vertex(0, -1, -PHI), Vertex(0, 1, -PHI),
			Vertex(PHI, 0, -1), Vertex(PHI, 0, 1), Vertex(-PHI, 0, -1), Vertex(-PHI, 0, 1)
		]

	def _get_icosahedron_faces(self) -> List[Face]:
		"""Returns the 20 triangular faces of a standard icosahedron using vertex indices 0-11."""
		return [
			Face(0, 11, 5), Face(0, 5, 1), Face(0, 1, 7), Face(0, 7, 10), Face(0, 10, 11),
			Face(1, 5, 9), Face(5, 11, 4), Face(11, 10, 2), Face(10, 7, 6), Face(7, 1, 8),
			Face(3, 9, 4), Face(3, 4, 2), Face(3, 2, 6), Face(3, 6, 8), Face(3, 8, 9),
			Face(4, 9, 5), Face(2, 4, 11), Face(6, 2, 10), Face(8, 6, 7), Face(9, 8, 1)
		]

	def truncatedIcosahedronBase(self):
		"""Generates a sphere-like shape based on a truncated icosahedron (soccer ball)."""
		# Start with base icosahedron data
		icosa_verts = self._get_icosahedron_vertices()
		icosa_faces = self._get_icosahedron_faces()
		for v in icosa_verts: v.normalize()

		self.vertices = [] # Reset world vertices
		self.faces = []    # Reset world faces
		vertex_map: Dict[Tuple[float, ...], int] = {} # Cache added vertices by position tuple
		edge_vertex_map: Dict[Tuple[int, int], Tuple[int, int]] = {} # original edge -> (1/3 Vtx idx, 2/3 Vtx idx)

		def add_unique_vertex(v: Vertex) -> int:
			# Normalize before adding/checking
			v_norm = v.normalize()
			# Use rounded tuple key for lookup
			key = tuple(np.round(v_norm.pos, 7)) # Adjust precision if needed
			if key in vertex_map:
				return vertex_map[key]
			else:
				# Add the ALREADY NORMALIZED vertex
				index = len(self.vertices)
				self.vertices.append(v_norm)
				vertex_map[key] = index
				return index

		# 1. Generate vertices by trisection
		processed_edges = set()
		for face in icosa_faces:
			for i in range(3):
				idx1 = face.vertices[i]
				idx2 = face.vertices[(i + 1) % 3]
				edge_key = tuple(sorted((idx1, idx2)))

				if edge_key not in processed_edges:
					processed_edges.add(edge_key)
					v1 = icosa_verts[idx1]
					v2 = icosa_verts[idx2]

					# Linear interpolation in 3D space
					p13 = v1.pos + (v2.pos - v1.pos) / 3.0
					p23 = v2.pos + (v1.pos - v2.pos) / 3.0 # Point closer to v2

					# Create Vertex objects (normalization happens in add_unique_vertex)
					v13 = Vertex(*p13)
					v23 = Vertex(*p23)

					idx13 = add_unique_vertex(v13) # Index of vertex closer to v1 on sphere
					idx23 = add_unique_vertex(v23) # Index of vertex closer to v2 on sphere

					# Store map: smaller original index first in key
					# Store resulting indices: first index is the one closer to original edge_key[0]
					if idx1 == edge_key[0]: # v1 corresponds to the smaller index
						edge_vertex_map[edge_key] = (idx13, idx23)
					else: # v2 corresponds to the smaller index
						edge_vertex_map[edge_key] = (idx23, idx13)

		# Helper to get the vertex index 1/3 along edge v_start -> v_end
		def get_vertex_third(v_start_idx, v_end_idx):
			edge_key = tuple(sorted((v_start_idx, v_end_idx)))
			idx_near_low, idx_near_high = edge_vertex_map[edge_key]

			if v_start_idx == edge_key[0]: # We want the one near v_end_idx
				return idx_near_high
			else: # We want the one near v_end_idx (which is edge_key[0])
				return idx_near_low


		# 2. Generate faces (Pentagons and Hexagons)
		# Build neighbor map for original icosahedron vertices
		# We need neighbors in order around each vertex
		icosa_vertex_ordered_neighbors = defaultdict(list)
		processed_neighbor_edges = set()

		for face in icosa_faces:
			f_verts = face.vertices
			for i in range(3):
				v_center = f_verts[i]
				v_prev = f_verts[(i - 1 + 3) % 3]
				v_next = f_verts[(i + 1) % 3]
				# Store neighbors - this simple approach might not guarantee order initially
				# We need a robust way to order neighbors around the vertex later
				edge1_key = tuple(sorted((v_center, v_prev)))
				if edge1_key not in processed_neighbor_edges:
					processed_neighbor_edges.add(edge1_key)
					icosa_vertex_ordered_neighbors[v_center].append(v_prev)
					icosa_vertex_ordered_neighbors[v_prev].append(v_center)

				edge2_key = tuple(sorted((v_center, v_next)))
				if edge2_key not in processed_neighbor_edges:
					processed_neighbor_edges.add(edge2_key)
					icosa_vertex_ordered_neighbors[v_center].append(v_next)
					icosa_vertex_ordered_neighbors[v_next].append(v_center)

		# Create Pentagons (around original icosahedron vertices)
		added_pentagons_centers = set()
		for orig_v_idx in range(len(icosa_verts)):
			if orig_v_idx in added_pentagons_centers: continue

			neighbors = icosa_vertex_ordered_neighbors[orig_v_idx]
			if len(neighbors) != 5:
				# print(f"Warning: Icosahedron vertex {orig_v_idx} has {len(neighbors)} neighbors, expected 5.")
				continue # Should not happen for standard icosahedron

			# Order neighbors correctly (e.g., using angles on tangent plane)
			center_pos = icosa_verts[orig_v_idx].pos
			neighbor_vectors = [icosa_verts[n_idx].pos - center_pos for n_idx in neighbors]

			# Project onto tangent plane (normal = center_pos)
			tangent_vectors = [nv - np.dot(nv, center_pos) * center_pos for nv in neighbor_vectors]

			# Choose a reference vector (e.g., the first one)
			ref_vec = tangent_vectors[0]
			ref_vec /= np.linalg.norm(ref_vec)

			# Calculate angles relative to the reference vector
			angles = []
			cross_prod_ref = np.cross(ref_vec, tangent_vectors[1]) # Need a cross product for sign
			ref_perp = np.cross(center_pos, ref_vec) # Vector perpendicular to normal and ref_vec
			ref_perp /= np.linalg.norm(ref_perp)

			for i, t_vec in enumerate(tangent_vectors):
				t_vec_norm = np.linalg.norm(t_vec)
				if t_vec_norm < 1e-9: angle = 0.0 # Should not happen
				else:
					t_vec /= t_vec_norm
					cos_theta = np.clip(np.dot(ref_vec, t_vec), -1.0, 1.0)
					angle = np.arccos(cos_theta)
					# Determine sign using perpendicular vector
					if np.dot(t_vec, ref_perp) < 0:
						angle = 2 * np.pi - angle
				angles.append((angle, neighbors[i]))

			# Sort neighbors by angle
			angles.sort()
			ordered_neighbors = [neighbor_idx for angle, neighbor_idx in angles]

			# Get the vertex 1/3rd of the way from neighbor towards center for the pentagon
			pentagon_vertex_indices = [get_vertex_third(neighbor_idx, orig_v_idx) for neighbor_idx in ordered_neighbors]

			if len(pentagon_vertex_indices) == 5:
				self.faces.append(Face(*pentagon_vertex_indices))
				added_pentagons_centers.add(orig_v_idx)
			else:
				print(f"Failed to form pentagon around {orig_v_idx}")


		# Create Hexagons (one for each original icosahedron face)
		for face in icosa_faces:
			f0, f1, f2 = face.vertices
			# Vertices for the hexagon are 1/3rd along edges, moving around the face
			# Order: Start near f0 on edge f0-f1, then near f1 on f0-f1, near f1 on f1-f2, etc.
			v_f0_f1 = get_vertex_third(f0, f1) # Near f0 on edge (f0,f1)
			v_f1_f0 = get_vertex_third(f1, f0) # Near f1 on edge (f0,f1)
			v_f1_f2 = get_vertex_third(f1, f2) # Near f1 on edge (f1,f2)
			v_f2_f1 = get_vertex_third(f2, f1) # Near f2 on edge (f1,f2)
			v_f2_f0 = get_vertex_third(f2, f0) # Near f2 on edge (f2,f0)
			v_f0_f2 = get_vertex_third(f0, f2) # Near f0 on edge (f2,f0)

			# Correct order tracing the new hexagon face
			hexagon_verts = (v_f0_f1, v_f1_f0, v_f1_f2, v_f2_f1, v_f2_f0, v_f0_f2)
			self.faces.append(Face(*hexagon_verts))

		# Normalize all generated vertices AFTER creation (add_unique_vertex already does this)
		# self._normalize_all_vertices()
		self.subdivide()
		return self


	def subdivide(self):
		"""
		Subdivides each face of the mesh using a generalized approach.
		- Triangles (n=3) are subdivided into 4 smaller triangles.
		- N-gons (n>3) are subdivided into n triangles around the perimeter
		  and one central n-gon formed by the edge midpoints.
		"""
		if self.details <= 0: return

		for _ in range(self.details):
			new_faces = []
			self._subdivision_cache = {} # Clear cache for each subdivision level

			for face in self.faces:
				vert_indices = face.vertices
				n = len(vert_indices)

				# 1. Get or create midpoints for all edges of the face
				midpoint_indices = []
				for i in range(n):
					v1_idx = vert_indices[i]
					v2_idx = vert_indices[(i + 1) % n]
					mid_idx = self._get_or_create_midpoint(v1_idx, v2_idx)
					midpoint_indices.append(mid_idx)

				# 2. Create new faces based on n
				if n == 3:
					# Standard triangle subdivision -> 4 triangles
					v0, v1, v2 = vert_indices
					m01, m12, m20 = midpoint_indices
					new_faces.append(Face(v0, m01, m20))
					new_faces.append(Face(v1, m12, m01))
					new_faces.append(Face(v2, m20, m12))
					new_faces.append(Face(m01, m12, m20)) # Center triangle
				elif n > 3:
					# N-gon subdivision -> n triangles + 1 central n-gon
					# Central n-gon uses the midpoint indices in order
					new_faces.append(Face(*midpoint_indices))

					# Create n triangles connecting original vertices to adjacent midpoints
					for i in range(n):
						v_curr = vert_indices[i]
						m_next = midpoint_indices[i] # Midpoint of (v_curr, v_next)
						m_prev = midpoint_indices[(i - 1 + n) % n] # Midpoint of (v_prev, v_curr)
						new_faces.append(Face(v_curr, m_next, m_prev))
				# else: n < 3 is invalid and handled by Face constructor

			self.faces = new_faces
		# Normalization happens implicitly via _get_or_create_midpoint -> _add_vertex


	def duplicateLayers(self):
		if self.details < 1:
			return

		original_vertices = self.vertices.copy()
		original_faces = self.faces.copy()

		layers_above = self.details // 2
		layers_below = self.details // 2

		# Create layers above
		for layer in range(1, layers_above + 1):
			scale = 1.0 + (0.1 * layer)  # Scale factor for each layer
			new_vertices = []
			vertex_map = {}
			
			# Create scaled vertices
			for i, v in enumerate(original_vertices):
				new_v = Vertex(*(v.pos * scale))
				new_vertices.append(new_v)
				vertex_map[i] = len(self.vertices) + len(new_vertices) - 1
			
			# Add to main vertices list
			self.vertices.extend(new_vertices)
			
			# Create connecting faces (quads)
			for face in original_faces:
				if len(face.vertices) == 3:  # Only handle triangles
					v0, v1, v2 = face.vertices
					self.faces.append(Face(
						v0,
						v1,
						vertex_map[v1],
						vertex_map[v0]
					))
					self.faces.append(Face(
						v1,
						v2,
						vertex_map[v2],
						vertex_map[v1]
					))
					self.faces.append(Face(
						v2,
						v0,
						vertex_map[v0],
						vertex_map[v2]
					))

		# Create layers below (similar to above but with negative scale)
		for layer in range(1, layers_below + 1):
			scale = max(0.1, 1.0 - (0.1 * layer))  # Prevent negative scale
			new_vertices = []
			vertex_map = {}
			
			for i, v in enumerate(original_vertices):
				new_v = Vertex(*(v.pos * scale))
				new_vertices.append(new_v)
				vertex_map[i] = len(self.vertices) + len(new_vertices) - 1
			
			self.vertices.extend(new_vertices)
			
			for face in original_faces:
				if len(face.vertices) == 3:
					v0, v1, v2 = face.vertices
					self.faces.append(Face(
						v0,
						v1,
						vertex_map[v1],
						vertex_map[v0]
					))
					self.faces.append(Face(
						v1,
						v2,
						vertex_map[v2],
						vertex_map[v1]
					))
					self.faces.append(Face(
						v2,
						v0,
						vertex_map[v0],
						vertex_map[v2]
					))

	def validateStructure(self):
		# Create a spatial index for faster lookup
		
		# Build edge list and face list
		edges = set()
		face_edges = []
		
		for face in self.faces:
			verts = face.vertices
			face_edges.append([])
			for i in range(len(verts)):
				edge = tuple(sorted((verts[i], verts[(i+1)%len(verts)])))
				edges.add(edge)
				face_edges[-1].append(edge)
		
		# Check for edge-face intersections
		for edge in edges:
			v1 = self.vertices[edge[0]].pos
			v2 = self.vertices[edge[1]].pos
			
			for face in self.faces:
				# Skip if edge belongs to this face
				face_edge_set = set(face_edges[self.faces.index(face)])
				if edge in face_edge_set:
					continue
					
				# Get face vertices
				f_verts = [self.vertices[i].pos for i in face.vertices]
				
				# Check if edge intersects with face
				if self._edge_face_intersection(v1, v2, f_verts):
					# Try to fix by flipping edge
					if self._try_fix_edge(edge, face):
						continue
					else:
						raise ValueError("Mesh has intersecting edges/faces that cannot be easily fixed")
		
		return True

	def _edge_face_intersection(self, v1, v2, face_verts):
		# Simplified intersection check
		# In practice, you'd want a more robust implementation
		edge_dir = v2 - v1
		edge_len = np.linalg.norm(edge_dir)
		edge_dir /= edge_len
		
		# Check against each face edge
		for i in range(len(face_verts)):
			fv1 = face_verts[i]
			fv2 = face_verts[(i+1)%len(face_verts)]
			
			# Simple segment-segment intersection check
			if self._segments_intersect(v1, v2, fv1, fv2):
				return True
		
		return False

	def _segments_intersect(self, a1, a2, b1, b2):
		# Implementation of segment-segment intersection test
		# Using cross product method
		def ccw(A, B, C):
			return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
		
		return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)

	def _try_fix_edge(self, edge, face):
		# Try to flip the edge to fix intersection
		# This is a simplified approach - in practice you'd need more checks
		edge_faces = [f for f in self.faces if edge in [tuple(sorted((f.vertices[i], f.vertices[(i+1)%len(f.vertices)]))) for i in range(len(f.vertices))]]
		
		if len(edge_faces) == 2:
			# Can potentially flip the edge
			f1, f2 = edge_faces
			# Find the other two vertices to form a quad
			other_verts = []
			for f in [f1, f2]:
				verts = f.vertices
				for i in range(len(verts)):
					if tuple(sorted((verts[i], verts[(i+1)%len(verts)]))) == edge:
						other_verts.append(verts[(i+2)%len(verts)])
						break
			
			if len(other_verts) == 2:
				# Replace the two faces with two new faces using the alternative diagonal
				self.faces.remove(f1)
				self.faces.remove(f2)
				new_edge = tuple(sorted(other_verts))
				self.faces.append(Face(edge[0], other_verts[0], other_verts[1]))
				self.faces.append(Face(edge[1], other_verts[1], other_verts[0]))
				return True
		
		return False


# --- Visualization ---

# Define a colormap and normalization for face coloring based on Z-normal
cmap = cm.viridis # Or cm.coolwarm, etc.
norm = colors.Normalize(vmin=-1.0, vmax=1.0) # Map Z-normal range [-1, 1] to [0, 1]

def VisualizeWorld(world: worldState, title: str = "World Mesh"):
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')

	# Plot vertices (optional, can make plot busy)
	# if world.vertices:
	#	 x, y, z = zip(*(v.pos for v in world.vertices))
	#	 ax.scatter(x, y, z, color='blue', s=5, alpha=0.3)

	# Plot faces (edges) with color based on normal's Z component
	if world.faces:
		for face in world.faces:
			verts_idx = list(face.vertices) + [face.vertices[0]] # Close the loop
			try:
				face_verts_pos = [world.vertices[i].pos for i in verts_idx]
				xs, ys, zs = zip(*face_verts_pos)

				# Calculate face normal and its Z component
				face_normal = face.normal(world)
				z_normal = face_normal[2]

				# Get color from colormap based on Z component
				edge_color = cmap(norm(z_normal))

				ax.plot(xs, ys, zs, color=edge_color, alpha=0.6, linewidth=0.8) # Apply color
			except IndexError:
				print(f"Warning: Vertex index out of bounds in face {face}. Skipping.")
				continue
			except Exception as e:
				print(f"Warning: Error processing face {face}: {e}. Skipping.")
				continue
	else:
		print("No faces to plot.")

	# --- Enforce Spherical Aspect Ratio ---
	# Set equal aspect ratio (important for preventing distortion)
	ax.set_box_aspect([1, 1, 1]) # Try this first

	# As a backup or alternative, set manual limits for unit sphere
	limit = world.radius * 1.1 # Add a small buffer
	ax.set_xlim(-limit, limit)
	ax.set_ylim(-limit, limit)
	ax.set_zlim(-limit, limit)
	# -------------------------------------

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title(f'{title} (Level: {world.details}, Verts: {len(world.vertices)}, Faces: {len(world.faces)})')

	# Add a colorbar (optional, explains the gradient)
	# scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
	# scalar_mappable.set_array([]) # Need an array for colorbar
	# fig.colorbar(scalar_mappable, ax=ax, label='Face Normal Z Component')


	plt.tight_layout()
	plt.show()

def main():
	# Set radius to 1.0 for all examples for consistent visualization
	common_radius = 1.0

	# --- Icosphere Example ---
	print("Generating Icosphere...")
	world_ico = worldState()
	world_ico.radius = common_radius
	world_ico.details = 2
	world_ico.icosphereBase()
	print(f"Icosphere Vertices: {len(world_ico.vertices)}, Faces: {len(world_ico.faces)}")
	VisualizeWorld(world_ico, "Icosphere")


	# --- Cube Sphere Example ---
	print("\nGenerating Cube Sphere...")
	world_cube = worldState()
	world_cube.radius = common_radius
	world_cube.details = 3
	world_cube.cubeBase()
	print(f"Cube Sphere Vertices: {len(world_cube.vertices)}, Faces: {len(world_cube.faces)}")
	VisualizeWorld(world_cube, "Cube Sphere")


	# --- Truncated Icosahedron Example ---
	print("\nGenerating Truncated Icosahedron Sphere...")
	world_trunc = worldState()
	world_trunc.radius = common_radius
	world_trunc.details = 2 # Increase detail level to see subdivision effect
	world_trunc.truncatedIcosahedronBase()
	print(f"Trunc. Ico. Vertices: {len(world_trunc.vertices)}, Faces: {len(world_trunc.faces)}")
	# Verify face types after subdivision (should be triangles, pentagons, hexagons)
	face_types = defaultdict(int)
	for f in world_trunc.faces: face_types[len(f.vertices)] += 1
	print(f"Face types after subdivision: {dict(face_types)}")
	VisualizeWorld(world_trunc, "Truncated Icosahedron Sphere")


if __name__ == "__main__":
	main()