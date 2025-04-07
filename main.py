import struct
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
# Import necessary for solid 3D polygons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import RadioButtons, Slider
import numpy as np
from collections import defaultdict
import matplotlib.cm as cm # Import colormaps
import matplotlib.colors as colors # Import color normalization

PHI = (1.0 + np.sqrt(5.0)) / 2.0
cmap = cm.get_cmap('RdYlGn_r')
norm = colors.Normalize(vmin=-1.0, vmax=1.0) # Map Z-normal range [-1, 1] to [0, 1]

class Vertex:
	def __init__(self, x, y, z):
		self.pos = np.array([float(x), float(y), float(z)], dtype=np.float64)
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)

	def normalize(self):
		norm = np.linalg.norm(self.pos)
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
		rounded_pos = tuple(np.round(self.pos, 8))
		return hash(rounded_pos)

	def __eq__(self, other):
		if not isinstance(other, Vertex):
			return False
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
		except IndexError as e:
			max_idx = max(self.vertices) if self.vertices else -1
			print(f"Error: Vertex index out of bounds in face {self.vertices}. Max index: {max_idx}, Num vertices: {len(world.vertices)}. Error: {e}")
			raise # Re-raise the exception

	def normal(self, world: 'worldState') -> np.ndarray:
		"""Calculates the approximate normal vector of the face (pointing outwards)."""
		verts = self.get_vertices(world)
		if len(verts) < 3:
			return np.array([0.0, 0.0, 0.0]) # Cannot compute normal

		# Use Newell's method for robustness with non-planar faces on the sphere
		face_normal = np.zeros(3, dtype=np.float64)
		num_verts = len(verts)
		for i in range(num_verts):
			v_curr = verts[i].pos
			v_next = verts[(i + 1) % num_verts].pos
			face_normal[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
			face_normal[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
			face_normal[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

		norm = np.linalg.norm(face_normal)
		if abs(norm) < 1e-10:
			centroid = self.centroid(world).pos
			norm_c = np.linalg.norm(centroid)
			if norm_c > 1e-9:
				return centroid / norm_c
			else:
				return np.array([0.0, 0.0, 1.0])

		face_normal /= norm
		if np.dot(face_normal, self.centroid(world).pos) < 0:
			face_normal *= -1.0
		return face_normal

	def centroid(self, world: 'worldState') -> Vertex:
		"""Calculates the geometric centroid of the face vertices."""
		verts = self.get_vertices(world)
		if not verts:
			return Vertex(0,0,0)
		center_pos = np.mean([v.pos for v in verts], axis=0)
		return Vertex(*center_pos).normalize()

	# _validate_face and _segments_intersect remain the same
	def _validate_face(self, world):
		"""Ensure the face is simple (non-intersecting edges)"""
		n = len(self.vertices)
		if n < 3:
			raise ValueError("Face must have at least 3 vertices")
			
		if n > 3:
			points = [world.vertices[i].pos for i in self.vertices]
			
			normal = np.cross(points[1] - points[0], points[2] - points[0])
			norm_val = np.linalg.norm(normal)
			if norm_val < 1e-9: # Handle collinear points
				# Try different points if first 3 are collinear
				if n > 3:
					normal = np.cross(points[2] - points[1], points[3] - points[1])
					norm_val = np.linalg.norm(normal)
					if norm_val < 1e-9: # Still collinear, maybe problematic face
							# print(f"Warning: Could not determine normal for face validation {self.vertices}")
							return # Skip validation for potentially degenerate face
			normal /= norm_val

			dominant_axis = np.argmax(np.abs(normal))
			axes = [0, 1, 2]
			axes.remove(dominant_axis)
			
			projected = [(p[axes[0]], p[axes[1]]) for p in points]
			
			for i in range(n):
				for j in range(i + 2, n):
					if (i == 0 and j == n - 1):
						continue
					a1, a2 = projected[i], projected[(i + 1) % n]
					b1, b2 = projected[j], projected[(j + 1) % n]
					if self._segments_intersect(a1, a2, b1, b2):
						raise ValueError(f"Face {self.vertices} has self-intersecting edges between edge {i}-{ (i + 1) % n} and {j}-{(j + 1) % n}")

	def _segments_intersect(self, p1, p2, p3, p4):
		def orientation(p, q, r):
			val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
			if abs(val) < 1e-7: return 0
			return 1 if val > 0 else 2
		def on_segment(p, q, r):
			return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
					q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

		o1 = orientation(p1, p2, p3)
		o2 = orientation(p1, p2, p4)
		o3 = orientation(p3, p4, p1)
		o4 = orientation(p3, p4, p2)

		# General case
		if o1 != o2 and o3 != o4:
			eps = 1e-9
			if np.linalg.norm(np.array(p1)-np.array(p3)) < eps or \
			   np.linalg.norm(np.array(p1)-np.array(p4)) < eps or \
			   np.linalg.norm(np.array(p2)-np.array(p3)) < eps or \
			   np.linalg.norm(np.array(p2)-np.array(p4)) < eps:
				return False
			return True
		if o1 == 0 and on_segment(p1, p3, p2): return True
		if o2 == 0 and on_segment(p1, p4, p2): return True
		if o3 == 0 and on_segment(p3, p1, p4): return True
		if o4 == 0 and on_segment(p3, p2, p4): return True
		return False

	def area(self, world):
		n = len(self.vertices)
		if n < 3:
			raise ValueError(f"Cannot calculate area for face with {n} vertices: {self.vertices}")

		verts_pos = [world.vertices[idx].pos for idx in self.vertices]

		# Girard's theorem for spherical polygons
		angle_sum = 0.0
		for i in range(n):
			p0 = verts_pos[(i - 1 + n) % n] # Previous vertex
			p1 = verts_pos[i]               # Current vertex
			p2 = verts_pos[(i + 1) % n]     # Next vertex

			# Vectors from current vertex p1 to neighbors on the sphere surface
			v1 = p0 - p1
			v2 = p2 - p1

            # Project vectors onto the tangent plane at p1
            # Normal at p1 is p1 itself (for unit sphere)
			tangent_v1 = v1 - np.dot(v1, p1) * p1
			tangent_v2 = v2 - np.dot(v2, p1) * p1

			norm_tv1 = np.linalg.norm(tangent_v1)
			norm_tv2 = np.linalg.norm(tangent_v2)

			if norm_tv1 < 1e-10 or norm_tv2 < 1e-10:
				# Handle degenerate cases (e.g., duplicate vertices)
				# print(f"Warning: Degenerate angle calculation in face {self.vertices} at vertex {i}")
				angle = np.pi # Assume straight line if vectors are tiny/zero
			else:
				cos_angle = np.dot(tangent_v1, tangent_v2) / (norm_tv1 * norm_tv2)
				angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

			angle_sum += angle

		# Spherical excess: Area = Sum of angles - (n - 2) * pi (for unit sphere)
		area_unit_sphere = angle_sum - (n - 2) * np.pi
		# Check for small negative areas due to floating point errors
		if area_unit_sphere < 0 and abs(area_unit_sphere) < 1e-7:
		     area_unit_sphere = 0.0
		elif area_unit_sphere < 0:
		    # This might indicate a winding order issue or complex self-intersection not caught
		    # print(f"Warning: Negative spherical excess ({area_unit_sphere:.4e}) for face {self.vertices}. Angle sum: {angle_sum/np.pi:.3f}*pi")
		    # For robustness, return 0 or abs value, but investigate the cause
		    area_unit_sphere = abs(area_unit_sphere)


		return area_unit_sphere * world.radius**2

	def __repr__(self):
		return f"Face{self.vertices}"

class Plate:
	def __init__(self):
		self.vertices: List[int] = [] #list of vertex indices in plate
		pass

	def move(self, oldvert: List[int], newvert: List[int]):
		for v in oldvert:
			self.vertices.remove(v)
		self.vertices.extend(newvert)

class worldState:
	def __init__(self):
		self.vertices: List[Vertex] = []
		self.faces: List[Face] = []
		self.radius: float = 1.0
		self.timestepSeconds: int = 1
		self.gravity: float = 9.81
		self.details: int = 3
		self.elevations: dict[int, float] = {} #assign vertex to float elevation
		#TODO: assign elevations based on plates. mountains and trenches along fault lines

		# Cache for subdivision: key=sorted tuple(v_idx1, v_idx2), value=midpoint_idx
		self._subdivision_cache: Dict[Tuple[int, int], int] = {}
		self._vertex_pos_cache: Dict[Tuple[float, ...], int] = {}

	# timeStep methods remain the same
	def timeStepHour(self) -> float: return float(self.timestepSeconds) / 3600.0
	def timeStepDay(self): return float(self.timestepSeconds) / 86400.0
	def timeStepYear(self): return float(self.timestepSeconds) / 31556952.0

	def _add_vertex(self, vertex: Vertex) -> int:
		norm_vertex = vertex.normalize()
		key = tuple(np.round(norm_vertex.pos, 8))
		if key in self._vertex_pos_cache:
			return self._vertex_pos_cache[key]
		else:
			idx = len(self.vertices)
			self.vertices.append(norm_vertex)
			self._vertex_pos_cache[key] = idx
			return idx

	def _get_or_create_midpoint(self, v1_idx: int, v2_idx: int) -> int:
		key = tuple(sorted((v1_idx, v2_idx)))
		if key in self._subdivision_cache:
			return self._subdivision_cache[key]
		v1 = self.vertices[v1_idx]
		v2 = self.vertices[v2_idx]
		mid_pos = (v1.pos + v2.pos) / 2.0
		midpoint_vertex = Vertex(mid_pos[0], mid_pos[1], mid_pos[2])
		mid_idx = self._add_vertex(midpoint_vertex)
		self._subdivision_cache[key] = mid_idx
		return mid_idx

	def _normalize_all_vertices(self):
		self._vertex_pos_cache.clear()
		for i, v in enumerate(self.vertices):
			v.normalize()
			key = tuple(np.round(v.pos, 8))
			if key in self._vertex_pos_cache:
				print(f"Warning: Duplicate vertex found after normalization: Index {i} same as {self._vertex_pos_cache[key]}")
			self._vertex_pos_cache[key] = i

	# --- Base Mesh Generators ---

	def icosphereBase(self):
		self.vertices = []
		self.faces = []
		self._vertex_pos_cache = {}
		self._subdivision_cache = {}
		raw_verts = self._get_icosahedron_vertices()
		ico_indices = [self._add_vertex(v) for v in raw_verts]
		if len(self.vertices) != 12:
		     print(f"Warning: Expected 12 base vertices for icosahedron, got {len(self.vertices)}")

		# Use the indices returned by _add_vertex (should be 0-11 if cache was empty)
		self.faces = [
			Face(ico_indices[0], ico_indices[11], ico_indices[5]), Face(ico_indices[0], ico_indices[5], ico_indices[1]), Face(ico_indices[0], ico_indices[1], ico_indices[7]),
			Face(ico_indices[0], ico_indices[7], ico_indices[10]), Face(ico_indices[0], ico_indices[10], ico_indices[11]), Face(ico_indices[1], ico_indices[5], ico_indices[9]),
			Face(ico_indices[5], ico_indices[11], ico_indices[4]), Face(ico_indices[11], ico_indices[10], ico_indices[2]), Face(ico_indices[10], ico_indices[7], ico_indices[6]),
			Face(ico_indices[7], ico_indices[1], ico_indices[8]), Face(ico_indices[3], ico_indices[9], ico_indices[4]), Face(ico_indices[3], ico_indices[4], ico_indices[2]),
			Face(ico_indices[3], ico_indices[2], ico_indices[6]), Face(ico_indices[3], ico_indices[6], ico_indices[8]), Face(ico_indices[3], ico_indices[8], ico_indices[9]),
			Face(ico_indices[4], ico_indices[9], ico_indices[5]), Face(ico_indices[2], ico_indices[4], ico_indices[11]), Face(ico_indices[6], ico_indices[2], ico_indices[10]),
			Face(ico_indices[8], ico_indices[6], ico_indices[7]), Face(ico_indices[9], ico_indices[8], ico_indices[1])
		]

		self.subdivide() # Subdivide based on self.details
		# Final normalization check (should be redundant if _add_vertex works)
		# self._normalize_all_vertices() # Optional check
		return self

	def cubeBase(self):
		self.vertices = []
		self.faces = []
		self._vertex_pos_cache = {}
		self._subdivision_cache = {}

		# Add base cube vertices
		raw_verts = [Vertex(x, y, z) for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
		cube_indices = [self._add_vertex(v) for v in raw_verts] # Normalizes and adds to cache

		# Define faces using the indices returned by _add_vertex (0-7)
		# Check winding order for outward normals after normalization
		self.faces = [
			Face(cube_indices[0], cube_indices[2], cube_indices[6], cube_indices[4]), # Left (-X) - Corrected order? Check normal
			Face(cube_indices[1], cube_indices[3], cube_indices[7], cube_indices[5]), # Right (+X)
			Face(cube_indices[0], cube_indices[1], cube_indices[5], cube_indices[4]), # Bottom (-Y)
			Face(cube_indices[2], cube_indices[3], cube_indices[7], cube_indices[6]), # Top (+Y)
			Face(cube_indices[0], cube_indices[1], cube_indices[3], cube_indices[2]), # Back (-Z)
			Face(cube_indices[4], cube_indices[5], cube_indices[7], cube_indices[6])  # Front (+Z)
		]
		# Winding order check (optional but good):
		# for i, f in enumerate(self.faces):
		# 	n = f.normal(self)
		# 	c = f.centroid(self).pos
		# 	if np.dot(n, c) < 0:
		# 		print(f"Warning: Cube Face {i} {f.vertices} normal might be inward.")
				# Reverse face: f.vertices = f.vertices[::-1] # Doesn't work as tuples immutable
				# Need to recreate the face if reversal is needed: self.faces[i] = Face(*f.vertices[::-1])

		self.subdivide()
		self._normalize_all_vertices() # Optional check
		return self

	def _get_icosahedron_vertices(self) -> List[Vertex]:
		return [
			Vertex(-1, PHI, 0), Vertex(1, PHI, 0), Vertex(-1, -PHI, 0), Vertex(1, -PHI, 0),
			Vertex(0, -1, PHI), Vertex(0, 1, PHI), Vertex(0, -1, -PHI), Vertex(0, 1, -PHI),
			Vertex(PHI, 0, -1), Vertex(PHI, 0, 1), Vertex(-PHI, 0, -1), Vertex(-PHI, 0, 1)
		]

	def _get_icosahedron_faces_indices(self) -> List[Tuple[int, int, int]]:
		return [
			(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
			(1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
			(3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
			(4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
		]

	def truncatedIcosahedronBase(self):
		self.vertices = []
		self.faces = []
		self._vertex_pos_cache = {}
		self._subdivision_cache = {}
		icosa_verts_raw = self._get_icosahedron_vertices()
		icosa_faces_indices = self._get_icosahedron_faces_indices()
		icosa_vertex_map = {i: self._add_vertex(v) for i, v in enumerate(icosa_verts_raw)}
		# Check if map is identity 0->0, 1->1 etc.
		if any(k != v for k, v in icosa_vertex_map.items()):
			print("Warning: Icosahedron vertex mapping is not identity. Indices might be shifted.")

		# Use the indices FROM the map when accessing vertices
		icosa_verts = [self.vertices[icosa_vertex_map[i]] for i in range(len(icosa_verts_raw))]

		# Cache for the new vertices created by trisection
		# Key: tuple(sorted(original_v_idx1, original_v_idx2)), Value: (new_v_idx_near_v1, new_v_idx_near_v2)
		edge_vertex_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
		processed_edges = set()
		for face_indices in icosa_faces_indices:
			for i in range(3):
				orig_idx1 = face_indices[i]
				orig_idx2 = face_indices[(i + 1) % 3]
				edge_key = tuple(sorted((orig_idx1, orig_idx2)))
				if edge_key not in processed_edges:
					processed_edges.add(edge_key)
					# Get the actual Vertex objects using mapped indices
					v1 = self.vertices[icosa_vertex_map[orig_idx1]]
					v2 = self.vertices[icosa_vertex_map[orig_idx2]]

					# Linear interpolation in 3D space
					p13 = v1.pos + (v2.pos - v1.pos) / 3.0
					p23 = v1.pos + 2.0 * (v2.pos - v1.pos) / 3.0 # Corrected: 2/3 way from v1 to v2

					# Create Vertex objects (normalization happens in _add_vertex)
					v13 = Vertex(*p13) # Closer to v1
					v23 = Vertex(*p23) # Closer to v2

					# Add new vertices using _add_vertex and get their final indices
					idx13 = self._add_vertex(v13) # Index of vertex closer to v1 on sphere
					idx23 = self._add_vertex(v23) # Index of vertex closer to v2 on sphere

					# Store map: key uses ORIGINAL icosahedron indices
					# Value uses NEW world indices
					# Ensure order: value[0] is near edge_key[0], value[1] is near edge_key[1]
					if orig_idx1 == edge_key[0]: # v1 corresponds to the smaller original index
						edge_vertex_map[edge_key] = (idx13, idx23)
					else: # v2 corresponds to the smaller original index
						edge_vertex_map[edge_key] = (idx23, idx13)

		def get_vertex_third(v_start_orig_idx, v_end_orig_idx):
			edge_key = tuple(sorted((v_start_orig_idx, v_end_orig_idx)))
			idx_near_low, idx_near_high = edge_vertex_map[edge_key]

			# We want the new vertex NEAR v_start_orig_idx
			if v_start_orig_idx == edge_key[0]: # v_start is the lower original index
				return idx_near_low # Return the first index in the map value
			else: # v_start is the higher original index
				return idx_near_high # Return the second index in the map value


		# 2. Generate faces (Pentagons and Hexagons) using NEW world indices

		# Build ordered neighbor list for original icosahedron vertices
		icosa_vertex_ordered_neighbors = defaultdict(list)
		for orig_v_idx in range(len(icosa_verts_raw)):
			neighbors_unordered = []
			for face_indices in icosa_faces_indices:
				if orig_v_idx in face_indices:
					idx_in_face = face_indices.index(orig_v_idx)
					neighbors_unordered.append(face_indices[(idx_in_face - 1 + 3) % 3])
					neighbors_unordered.append(face_indices[(idx_in_face + 1) % 3])
			# Remove duplicates
			unique_neighbors = list(dict.fromkeys(neighbors_unordered)) # Preserves order roughly
			if len(unique_neighbors) != 5:
				print(f"Warning: Icosahedron vertex {orig_v_idx} has {len(unique_neighbors)} unique neighbors, expected 5.")
				continue

			# Order neighbors correctly around the central vertex
			center_v_world_idx = icosa_vertex_map[orig_v_idx]
			center_pos = self.vertices[center_v_world_idx].pos

			neighbor_world_indices = [icosa_vertex_map[n_idx] for n_idx in unique_neighbors]
			neighbor_positions = [self.vertices[nw_idx].pos for nw_idx in neighbor_world_indices]

			# Project onto tangent plane (normal = center_pos for sphere)
			tangent_vectors = [(n_pos - center_pos) - np.dot(n_pos - center_pos, center_pos) * center_pos
							   for n_pos in neighbor_positions]

			# Choose a reference vector (e.g., the first one) & create orthonormal basis on tangent plane
			ref_vec = tangent_vectors[0]
			norm_ref = np.linalg.norm(ref_vec)
			if norm_ref < 1e-9: print(f"Warning: Near-zero tangent vector for ordering neighbors of {orig_v_idx}"); continue
			ref_vec /= norm_ref

			ref_perp = np.cross(center_pos, ref_vec) # Perpendicular to normal and ref_vec

			# Calculate angles relative to the reference vector
			angles = []
			for i, t_vec in enumerate(tangent_vectors):
				norm_t = np.linalg.norm(t_vec)
				if norm_t < 1e-9: angle = 0.0
				else:
					t_vec /= norm_t
					cos_theta = np.clip(np.dot(ref_vec, t_vec), -1.0, 1.0)
					angle = np.arccos(cos_theta)
					# Determine sign using the perpendicular vector
					if np.dot(t_vec, ref_perp) < 0:
						angle = 2 * np.pi - angle
				# Store angle and the ORIGINAL neighbor index
				angles.append((angle, unique_neighbors[i]))
			angles.sort()
			ordered_neighbors_orig_indices = [neighbor_orig_idx for angle, neighbor_orig_idx in angles]
			icosa_vertex_ordered_neighbors[orig_v_idx] = ordered_neighbors_orig_indices


		added_pentagons_centers = set()
		for orig_v_idx in range(len(icosa_verts_raw)):
			if orig_v_idx in added_pentagons_centers: continue
			ordered_neighbors = icosa_vertex_ordered_neighbors.get(orig_v_idx)
			if not ordered_neighbors or len(ordered_neighbors) != 5:
				print(f"Error: Could not get 5 ordered neighbors for original vertex {orig_v_idx}")
				continue

			# Get the NEW world vertex index 1/3rd of the way from center towards neighbor
			pentagon_vertex_indices = [get_vertex_third(orig_v_idx, neighbor_orig_idx)
									   for neighbor_orig_idx in ordered_neighbors]

			if len(pentagon_vertex_indices) == 5:
				# Ensure indices are unique before adding face
				if len(set(pentagon_vertex_indices)) == 5:
					self.faces.append(Face(*pentagon_vertex_indices))
					added_pentagons_centers.add(orig_v_idx)
				else:
					print(f"Error: Duplicate vertex indices generated for pentagon around {orig_v_idx}: {pentagon_vertex_indices}")
			else:
				print(f"Failed to form pentagon around {orig_v_idx}, got {len(pentagon_vertex_indices)} vertices.")


		# Create Hexagons (one for each original icosahedron face)
		for face_indices in icosa_faces_indices:
			f0, f1, f2 = face_indices # Original indices

			# Get the NEW world vertex indices 1/3rd along edges, moving around the face
			# Use get_vertex_third(start_orig_idx, end_orig_idx) -> returns new world index near start
			v_near_f0_on_f0f1 = get_vertex_third(f0, f1)
			v_near_f1_on_f0f1 = get_vertex_third(f1, f0)
			v_near_f1_on_f1f2 = get_vertex_third(f1, f2)
			v_near_f2_on_f1f2 = get_vertex_third(f2, f1)
			v_near_f2_on_f2f0 = get_vertex_third(f2, f0)
			v_near_f0_on_f2f0 = get_vertex_third(f0, f2)

			# Assemble hexagon face using these NEW world indices in order
			hexagon_verts = (
				v_near_f0_on_f0f1, v_near_f1_on_f0f1, v_near_f1_on_f1f2,
				v_near_f2_on_f1f2, v_near_f2_on_f2f0, v_near_f0_on_f2f0
			)

			# Ensure indices are unique before adding face
			if len(set(hexagon_verts)) == 6:
				self.faces.append(Face(*hexagon_verts))
			else:
				print(f"Error: Duplicate vertex indices generated for hexagon from face {face_indices}: {hexagon_verts}")


		# Subdivide the resulting pentagons and hexagons
		self.subdivide()
		# self._normalize_all_vertices() # Optional check
		# Validate faces (optional, can be slow)
		# try:
		# 	for face in self.faces: face._validate_face(self)
		# except ValueError as e:
		# 	print(f"Validation Error in Truncated Icosahedron: {e}")

		return self

	# --- MODIFIED SUBDIVIDE METHOD ---
	def subdivide(self):
		"""
		Subdivides faces of the mesh based on detail level.
		Only subdivides faces whose area is >= 50% of the largest face's area
		at each subdivision level to promote more uniform face sizes.
		"""
		if self.details <= 0: return

		print(f"Starting subdivision process (details={self.details})")

		for level in range(self.details):
			print(f" Subdividing Level {level + 1}/{self.details}...")
			current_faces = self.faces[:] # Copy faces from previous level
			if not current_faces:
				print("  No faces to subdivide.")
				break # Stop if there are no faces

			# 1. Calculate areas of all current faces
			face_areas = []
			valid_faces_indices = [] # Keep track of indices of faces with calculable area
			print(f"  Calculating areas for {len(current_faces)} faces...")
			for i, face in enumerate(current_faces):
				try:
					area = face.area(self)
					if not np.isfinite(area) or area < 0:
					    print(f"  Warning: Invalid area ({area}) calculated for face {face.vertices}. Skipping.")
					    face_areas.append(-1.0) # Mark as invalid/skip
					else:
					    face_areas.append(area)
					    valid_faces_indices.append(i)
				except Exception as e:
					print(f"  Error calculating area for face {face.vertices}: {e}. Skipping this face.")
					face_areas.append(-1.0) # Mark as invalid/skip

			# Filter out areas of skipped faces before finding max
			valid_areas = [face_areas[i] for i in valid_faces_indices]

			if not valid_areas:
				print("  No valid face areas calculated. Stopping subdivision.")
				break

			# 2. Determine max area and threshold
			max_area = max(valid_areas) if valid_areas else 0
			# Add epsilon to max_area check to handle cases where all areas are tiny/zero
			if max_area < 1e-12:
				print("  Max face area is near zero. No subdivision will occur this level.")
				# Keep existing faces and stop further subdivision for this world
				# Or decide if you want to subdivide everything if max_area is effectively zero
				# For now, we stop subdividing if faces are negligible.
				self.details = level # Adjust detail level to reflect performed subdivisions
				break

			area_threshold = 0.5 * max_area
			print(f"  Max face area: {max_area:.4e}, Threshold for subdivision: {area_threshold:.4e}")

			# 3. Process faces: Subdivide large ones, keep small ones
			new_faces_next_level = []
			self._subdivision_cache = {} # Reset cache for this level
			faces_subdivided = 0
			faces_kept = 0

			print("  Processing faces for subdivision/keeping...")
			for i, face in enumerate(current_faces):
				# Use the pre-calculated area. Skip if marked invalid (-1.0).
				current_area = face_areas[i]
				if current_area < 0:
				    continue # Skip faces that had area calculation errors

				# Check if face should be subdivided
				if current_area >= area_threshold:
					# --- Subdivide this face ---
					faces_subdivided += 1
					vert_indices = face.vertices
					n = len(vert_indices)
					midpoint_indices = []
					try:
						for j in range(n):
							v1_idx = vert_indices[j]
							v2_idx = vert_indices[(j + 1) % n]
							if v1_idx >= len(self.vertices) or v2_idx >= len(self.vertices):
								raise IndexError(f"Vertex index out of bounds. V1:{v1_idx}, V2:{v2_idx}, Max:{len(self.vertices)-1}")
							mid_idx = self._get_or_create_midpoint(v1_idx, v2_idx)
							midpoint_indices.append(mid_idx)
					except IndexError as e:
						print(f"  Error getting midpoints for face {face.vertices}: {e}. Skipping subdivision for this face.")
						new_faces_next_level.append(face) # Keep the original face if subdivision fails
						faces_subdivided -= 1
						faces_kept += 1
						continue
					except Exception as e:
						print(f"  Unexpected error during midpoint creation for face {face.vertices}: {e}. Skipping subdivision.")
						new_faces_next_level.append(face)
						faces_subdivided -= 1
						faces_kept += 1
						continue

					# Create new smaller faces based on n
					if n == 3:
						v0, v1, v2 = vert_indices
						m01, m12, m20 = midpoint_indices
						new_faces_next_level.append(Face(v0, m01, m20))
						new_faces_next_level.append(Face(v1, m12, m01))
						new_faces_next_level.append(Face(v2, m20, m12))
						new_faces_next_level.append(Face(m01, m12, m20)) # Center triangle
					elif n > 3:
						# Central n-gon uses the midpoint indices
						new_faces_next_level.append(Face(*midpoint_indices))
						# N triangles around the edges
						for j in range(n):
							v_curr = vert_indices[j]
							m_next = midpoint_indices[j]
							m_prev = midpoint_indices[(j - 1 + n) % n]
							new_faces_next_level.append(Face(v_curr, m_next, m_prev))
					# else: n < 3 already handled by Face constructor / area calculation skip

				else:
					# --- Keep this face (it's too small) ---
					faces_kept += 1
					new_faces_next_level.append(face)

			# Update the world's faces for the next iteration or final result
			self.faces = new_faces_next_level
			print(f"  Level {level + 1} complete. Faces subdivided: {faces_subdivided}, Faces kept: {faces_kept}. Total faces: {len(self.faces)}")

		print(f"Subdivision finished. Final face count: {len(self.faces)}")
		# Final normalization isn't strictly necessary if _add_vertex works correctly,
		# but can be a safety measure after many operations.
		# print(" Normalizing all vertices post-subdivision...")
		# self._normalize_all_vertices()


	def duplicateLayers(self):
		# This function likely doesn't make sense after the spherical projection
		# and subdivision methods, as it assumes a base layer and scales it linearly.
		# It would create disconnected layers not conforming to the sphere.
		# Keeping it commented out or removing it might be best for spherical meshes.
		print("Warning: duplicateLayers is generally not suitable for spherical meshes generated by subdivision.")
		pass
		# if self.details < 1:
		# 	return
		# ... (rest of the original duplicateLayers code) ...

	def validateStructure(self):
		# Validation might be complex and slow for highly subdivided meshes.
		# The _validate_face check during generation/subdivision is often more targeted.
		# Skipping full structural validation here for performance unless specifically needed.
		print("Skipping full structure validation.")
		pass
		# ... (rest of the original validateStructure code) ...


# --- Visualization ---

# Define a colormap (Green to Red) and normalization for face coloring based on Z-normal
# RdYlGn_r reverses Red-Yellow-Green to Green-Yellow-Red
def VisualizeWorld(world: worldState, title: str = "World Mesh"):
	fig = plt.figure(figsize=(12, 10)) # Slightly larger figure
	ax = fig.add_subplot(111, projection='3d')

	face_polys = []
	face_colors = []
	edge_lines = []

	# Process faces to collect polygons, colors, and edge lines
	if world.faces:
		print(f"Visualizing {len(world.faces)} faces...")
		count = 0
		for face in world.faces:
			count += 1
			if count % 1000 == 0 : print(f" Processing face {count}/{len(world.faces)}")
			try:
				# Get vertex coordinates for the polygon
				verts_idx = list(face.vertices)
				face_verts_pos_poly = [world.vertices[i].pos for i in verts_idx]

				# Calculate face normal and its Z component for color
				face_normal = face.normal(world)
				z_normal = face_normal[2]
				f_color = cmap(norm(z_normal))

				face_polys.append(face_verts_pos_poly)
				face_colors.append(f_color)

				# Get vertex coordinates for edge lines (close the loop)
				verts_idx_lines = list(face.vertices) + [face.vertices[0]]
				face_verts_pos_lines = [world.vertices[i].pos for i in verts_idx_lines]
				edge_lines.append(face_verts_pos_lines)

			except IndexError:
				print(f"Warning: Vertex index out of bounds in face {face}. Skipping.")
				continue
			except Exception as e:
				print(f"Warning: Error processing face {face}: {e}. Skipping.")
				continue
	else:
		print("No faces to plot.")

    # Add solid faces to the plot
	if face_polys:
		print("Adding face collection...")
		poly_collection = Poly3DCollection(face_polys, alpha=0.75, facecolors=face_colors, edgecolors='none') # No auto edges
		ax.add_collection3d(poly_collection)
		print("Faces added.")

	# Plot edges in black
	if edge_lines:
		print("Adding edge lines...")
		for line_verts in edge_lines:
			xs, ys, zs = zip(*line_verts)
			ax.plot(xs, ys, zs, color='black', alpha=0.4, linewidth=0.6) # Thinner, slightly transparent black lines
		print("Edges added.")

	# Plot vertices in white
	if world.vertices:
		print("Adding vertices...")
		x, y, z = zip(*(v.pos for v in world.vertices))
		# White dots with a subtle black edge for contrast
		ax.scatter(x, y, z, color='white', s=8, alpha=0.9, edgecolors='black', linewidths=0.5, depthshade=False)
		print("Vertices added.")

	# --- Enforce Spherical Aspect Ratio ---
	ax.set_box_aspect([1, 1, 1]) # Essential for correct sphere appearance
	limit = world.radius * 1.1
	ax.set_xlim(-limit, limit)
	ax.set_ylim(-limit, limit)
	ax.set_zlim(-limit, limit)
	# -------------------------------------

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title(f'{title} (Level: {world.details}, Verts: {len(world.vertices)}, Faces: {len(world.faces)})')

	# Add a colorbar
	scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
	scalar_mappable.set_array([]) # Need an array for colorbar
	cbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.6, aspect=20, label='Face Normal Z (Green: +1, Red: -1)')

	print("Finalizing plot...")
	plt.tight_layout()
	plt.show()

def main():
	# Set radius to 1.0 for all examples for consistent visualization
	common_radius = 1.0

	# # --- Icosphere Example ---
	# print("Generating Icosphere...")
	# world_ico = worldState()
	# world_ico.radius = common_radius
	# world_ico.details = 2 # Detail level
	# world_ico.icosphereBase()
	# print(f"Icosphere Vertices: {len(world_ico.vertices)}, Faces: {len(world_ico.faces)}")
	# VisualizeWorld(world_ico, "Icosphere")


	# # --- Cube Sphere Example ---
	# print("\nGenerating Cube Sphere...")
	# world_cube = worldState()
	# world_cube.radius = common_radius
	# world_cube.details = 3 # Detail level
	# world_cube.cubeBase()
	# print(f"Cube Sphere Vertices: {len(world_cube.vertices)}, Faces: {len(world_cube.faces)}")
	# VisualizeWorld(world_cube, "Cube Sphere")


	# --- Truncated Icosahedron Example ---
	print("\nGenerating Truncated Icosahedron Sphere...")
	world_trunc = worldState()
	world_trunc.radius = common_radius
	world_trunc.details = 3 # Start with lower detail for faster generation/viz
	world_trunc.truncatedIcosahedronBase()
	print(f"Trunc. Ico. Vertices: {len(world_trunc.vertices)}, Faces: {len(world_trunc.faces)}")
	face_types = defaultdict(int)
	total_area = 0
	for f in world_trunc.faces:
		face_types[len(f.vertices)] += 1
		try:
			total_area += f.area(world_trunc) # Calculate area
		except Exception as e:
			print(f"Error calculating area for face {f.vertices}: {e}")

	print(f"Face types after subdivision: {dict(face_types)}")
	expected_area = 4 * np.pi * world_trunc.radius**2
	print(f"Total calculated face area: {total_area:.5f}")
	print(f"Expected sphere area:       {expected_area:.5f}")
	print(f"Area difference: {abs(total_area - expected_area):.5f}")
	VisualizeWorld(world_trunc, "Truncated Icosahedron Sphere")


if __name__ == "__main__":
	main()