import struct
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import RadioButtons, Slider
import numpy as np
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as colors
import random

PHI = (1.0 + np.sqrt(5.0)) / 2.0
cmap = cm.get_cmap('RdYlGn_r')
norm = colors.Normalize(vmin=-1.0, vmax=1.0)

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
			raise

	def normal(self, world: 'worldState') -> np.ndarray:
		"""Calculates the approximate normal vector of the face (pointing outwards)."""
		verts = self.get_vertices(world)
		if len(verts) < 3:
			return np.array([0.0, 0.0, 0.0])


		face_normal = np.zeros(3, dtype=np.float64)
		num_verts = len(verts)
		for i in range(num_verts):
			v_curr = verts[i].pos
			v_next = verts[(i + 1) % num_verts].pos
			face_normal[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
			face_normal[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
			face_normal[2] += (v_curr[0] - v_next[2]) * (v_curr[1] + v_next[1])

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

	def _validate_face(self, world):
		"""Ensure the face is simple (non-intersecting edges)"""
		n = len(self.vertices)
		if n < 3:
			raise ValueError("Face must have at least 3 vertices")

		if n > 3:
			points = [world.vertices[i].pos for i in self.vertices]

			normal = np.cross(points[1] - points[0], points[2] - points[0])
			norm_val = np.linalg.norm(normal)
			if norm_val < 1e-9:
				if n > 3:
					normal = np.cross(points[2] - points[1], points[3] - points[1])
					norm_val = np.linalg.norm(normal)
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

		angle_sum = 0.0
		for i in range(n):
			p0 = verts_pos[(i - 1 + n) % n]
			p1 = verts_pos[i]
			p2 = verts_pos[(i + 1) % n]

			v1 = p0 - p1
			v2 = p2 - p1

			tangent_v1 = v1 - np.dot(v1, p1) * p1
			tangent_v2 = v2 - np.dot(v2, p1) * p1

			norm_tv1 = np.linalg.norm(tangent_v1)
			norm_tv2 = np.linalg.norm(tangent_v2)

			if norm_tv1 < 1e-10 or norm_tv2 < 1e-10:
				angle = np.pi
			else:
				cos_angle = np.dot(tangent_v1, tangent_v2) / (norm_tv1 * norm_tv2)
				angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

			angle_sum += angle

		area_unit_sphere = angle_sum - (n - 2) * np.pi
		if area_unit_sphere < 0 and abs(area_unit_sphere) < 1e-7:
				area_unit_sphere = 0.0
		elif area_unit_sphere < 0:
			area_unit_sphere = abs(area_unit_sphere)


		return area_unit_sphere * world.radius**2

	def __repr__(self):
		return f"Face{self.vertices}"

class Plate:
	def __init__(self, plate_id):
		self.plate_id = plate_id
		self.vertices: List[int] = [] # List of vertex indices belonging to this plate

	def add_vertex(self, vertex_index):
		self.vertices.append(vertex_index)

	def __repr__(self):
		return f"Plate(ID: {self.plate_id}, Vertices: {len(self.vertices)})"


class worldState:
	def __init__(self):
		self.vertices: List[Vertex] = []
		self.faces: List[Face] = []
		self.radius: float = 1.0
		self.timestepSeconds: int = 1
		self.gravity: float = 9.81
		self.details: int = 3
		self.elevations: dict[int, float] = {} #assign vertex to float elevation
		self.plates: List[Plate] = [] # Add plates attribute
		self.adjacency_list: defaultdict[int, List[int]] = defaultdict(list) # Adjacency list for vertices
		#TODO: assign elevations based on plates. mountains and trenches along fault lines

		# Cache for subdivision: key=sorted tuple(v_idx1, v_idx2), value=midpoint_idx
		self._subdivision_cache: Dict[Tuple[int, int], int] = {}
		self._vertex_pos_cache: Dict[Tuple[float, ...], int] = {}

	def timeStepHour(self) -> float: return float(self.timestepSeconds) / 3600.0
	def timeStepDay(self) -> float: return float(self.timestepSeconds) / 86400.0
	def timeStepYear(self) -> float: return float(self.timestepSeconds) / 31556952.0

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

	def _build_adjacency_list(self):
		"""Builds an adjacency list for vertices based on faces."""
		self.adjacency_list.clear()
		for face in self.faces:
			for v_idx in face.vertices:
				for neighbor_idx in face.vertices:
					if v_idx != neighbor_idx:
						if neighbor_idx not in self.adjacency_list[v_idx]:
							self.adjacency_list[v_idx].append(neighbor_idx)

	def icosphereBase(self):
		self.vertices = []
		self.faces = []
		self._vertex_pos_cache = {}
		self._subdivision_cache = {}

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

		for v in vertices:
			v.normalize()

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
		self.subdivide()
		self._build_adjacency_list() # Build adjacency list after subdivision
		return self

	def assign_icosphere_vertices_to_plates(self, num_plates: int) -> List[Plate]:
		"""Assigns vertices to plates using a proximity-biased random walk with recycling."""
		plates = [Plate(i) for i in range(num_plates)]
		unassigned_vertices = set(range(len(self.vertices)))
		if num_plates > len(self.vertices):
			num_plates = len(self.vertices)
			print(f"Warning: Number of plates requested exceeds number of vertices. Reducing to {num_plates} plates.")

		start_vertices_indices = random.sample(list(unassigned_vertices), num_plates)

		for i in range(num_plates):
			start_vertex_index = start_vertices_indices[i]
			plates[i].add_vertex(start_vertex_index)
			unassigned_vertices.remove(start_vertex_index)

		plate_vertex_counts = [1] * num_plates

		iteration_count = 0
		max_iterations = 20000 # Limit iterations to prevent potential infinite loops

		while unassigned_vertices and iteration_count < max_iterations:
			iteration_count += 1
			print(f"unassigned_vertices has {len(unassigned_vertices)} left, iteration: {iteration_count}")
			plate_index = random.randrange(num_plates)
			current_plate = plates[plate_index]
			possible_expansion_vertices = set()

			for plate_vertex_index in current_plate.vertices:
				adjacent_vertices = self.adjacency_list[plate_vertex_index]
				for v_idx in adjacent_vertices:
					if v_idx in unassigned_vertices:
						possible_expansion_vertices.add(v_idx)

			possible_expansion_vertices = list(possible_expansion_vertices)

			if not possible_expansion_vertices:
				if unassigned_vertices: # Fallback: random assignment if no proximal expansion
					next_vertex_index = random.choice(list(unassigned_vertices))
					current_plate.add_vertex(next_vertex_index)
					unassigned_vertices.remove(next_vertex_index)
					continue
				else:
					break # No more unassigned vertices, exit loop


			# Calculate proximity bias
			proximity_scores = []
			for expansion_vertex_idx in possible_expansion_vertices:
				min_distance = float('inf')
				expansion_vertex_pos = self.vertices[expansion_vertex_idx].pos
				for plate_vertex_idx in current_plate.vertices:
					plate_vertex_pos = self.vertices[plate_vertex_idx].pos
					distance = 1.0 - np.dot(expansion_vertex_pos, plate_vertex_pos)
					min_distance = min(min_distance, distance)
				proximity_scores.append(1.0 / (min_distance + 1e-6))

			probabilities = np.array(proximity_scores) / np.sum(proximity_scores) if np.sum(proximity_scores) > 0 else np.ones(len(possible_expansion_vertices)) / len(possible_expansion_vertices)

			if not possible_expansion_vertices:
				continue

			try:
				next_vertex_index = np.random.choice(possible_expansion_vertices, p=probabilities)
			except ValueError as e:
				print(f"Error during weighted choice: {e}, probabilities: {probabilities}, possible vertices: {possible_expansion_vertices}")
				next_vertex_index = random.choice(possible_expansion_vertices)


			current_plate.add_vertex(next_vertex_index)
			unassigned_vertices.remove(next_vertex_index)
			plate_vertex_counts[plate_index] += 1

		print("Initial plate assignment complete. Starting vertex recycling...")

		recycled_vertices_count = 0
		recycling_iterations = 5 # Number of recycling passes

		for recycle_iter in range(recycling_iterations):
			vertices_to_recycle = []
			for plate in plates:
				plate_recyclable_vertices = []
				for vertex_index in plate.vertices:
					plate_neighbors_count = 0
					for neighbor_index in self.adjacency_list[vertex_index]:
						if neighbor_index in plate.vertices:
							plate_neighbors_count += 1
					if plate_neighbors_count <= 1: # Recycle if very few neighbors within the plate
						plate_recyclable_vertices.append(vertex_index)
				vertices_to_recycle.extend(plate_recyclable_vertices)
				plate.vertices = [v_idx for v_idx in plate.vertices if v_idx not in plate_recyclable_vertices] # Remove recycled vertices from plate


			if not vertices_to_recycle:
				print(f"No vertices to recycle in iteration {recycle_iter+1}. Recycling complete.")
				break
			else:
				print(f"Recycling {len(vertices_to_recycle)} vertices in iteration {recycle_iter+1}...")
				recycled_vertices_count += len(vertices_to_recycle)
				unassigned_vertices.update(vertices_to_recycle) # Add recycled vertices back to unassigned

				while vertices_to_recycle:
					vertex_index_to_reassign = vertices_to_recycle.pop(0)
					best_plate_index = -1
					max_proximity_score = -1.0

					neighboring_plates = defaultdict(list) # Plate index -> list of neighbor vertex indices in that plate

					for neighbor_vertex_index in self.adjacency_list[vertex_index_to_reassign]:
						for plate_idx, plate in enumerate(plates):
							if neighbor_vertex_index in plate.vertices:
								neighboring_plates[plate_idx].append(neighbor_vertex_index)
								break # Vertex can only belong to one plate

					if neighboring_plates:
						proximity_scores_reassign = []
						plate_indices_reassign = list(neighboring_plates.keys())
						for plate_idx in plate_indices_reassign:
							plate = plates[plate_idx]
							proximity_score = 0.0
							vertex_pos = self.vertices[vertex_index_to_reassign].pos
							for plate_vertex_index in plate.vertices:
								plate_vertex_pos = self.vertices[plate_vertex_index].pos
								proximity_score += 1.0 / (1.0 - np.dot(vertex_pos, plate_vertex_pos) + 1e-6) # Sum proximity to all plate vertices
							proximity_scores_reassign.append(proximity_score)

						probabilities_reassign = np.array(proximity_scores_reassign) / np.sum(proximity_scores_reassign) if np.sum(proximity_scores_reassign) > 0 else np.ones(len(plate_indices_reassign)) / len(plate_indices_reassign)

						try:
							reassign_plate_index = np.random.choice(plate_indices_reassign, p=probabilities_reassign)
						except ValueError as e:
							print(f"Error during re-assignment weighted choice: {e}, probabilities: {probabilities_reassign}, possible plates: {plate_indices_reassign}")
							reassign_plate_index = random.choice(plate_indices_reassign)

						plates[reassign_plate_index].add_vertex(vertex_index_to_reassign)
						unassigned_vertices.remove(vertex_index_to_reassign) #remove from unassigned - already added to plate

					elif unassigned_vertices: # If no neighboring plates, put back in unassigned for general assignment
						unassigned_vertices.add(vertex_index_to_reassign) # Keep in unassigned for next round if no neighbors
						continue # Go to next vertex to recycle

		print(f"Vertex recycling complete. Recycled a total of {recycled_vertices_count} vertices.")
		return plates


	def cubeBase(self):
		self.vertices = []
		self.faces = []
		self._vertex_pos_cache = {}
		self._subdivision_cache = {}

		raw_verts = [Vertex(x, y, z) for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
		cube_indices = [self._add_vertex(v) for v in raw_verts]

		self.faces = [
			Face(cube_indices[0], cube_indices[2], cube_indices[6], cube_indices[4]),
			Face(cube_indices[1], cube_indices[3], cube_indices[7], cube_indices[5]),
			Face(cube_indices[0], cube_indices[1], cube_indices[5], cube_indices[4]),
			Face(cube_indices[2], cube_indices[3], cube_indices[7], cube_indices[6]),
			Face(cube_indices[0], cube_indices[1], cube_indices[3], cube_indices[2]),
			Face(cube_indices[4], cube_indices[5], cube_indices[7], cube_indices[6])
		]

		self.subdivide()
		self._normalize_all_vertices()
		return self

	def truncatedIcosahedronBase(self):
		self.vertices = []
		self.faces = []
		self._vertex_pos_cache = {}
		self._subdivision_cache = {}
		icosa_verts_raw = [
			Vertex(-1, PHI, 0), Vertex(1, PHI, 0), Vertex(-1, -PHI, 0), Vertex(1, -PHI, 0),
			Vertex(0, -1, PHI), Vertex(0, 1, PHI), Vertex(0, -1, -PHI), Vertex(0, 1, -PHI),
			Vertex(PHI, 0, -1), Vertex(PHI, 0, 1), Vertex(-PHI, 0, -1), Vertex(-PHI, 0, 1)
		]
		icosa_faces_indices = [
			(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
			(1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
			(3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
			(4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
		]
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

	def subdivide(self):
		"""Subdivides faces of the mesh based on detail level."""
		if self.details <= 0: return

		print(f"Starting subdivision process (details={self.details})")

		for level in range(self.details):
			print(f" Subdividing Level {level + 1}/{self.details}...")
			current_faces = self.faces[:]
			if not current_faces:
				print("  No faces to subdivide.")
				break

			face_areas = []
			valid_faces_indices = []
			print(f"  Calculating areas for {len(current_faces)} faces...")
			for i, face in enumerate(current_faces):
				try:
					area = face.area(self)
					if not np.isfinite(area) or area < 0:
						print(f"  Warning: Invalid area ({area}) calculated for face {face.vertices}. Skipping.")
						face_areas.append(-1.0)
					else:
						face_areas.append(area)
						valid_faces_indices.append(i)
				except Exception as e:
					print(f"  Error calculating area for face {face.vertices}: {e}. Skipping this face.")
					face_areas.append(-1.0)

			valid_areas = [face_areas[i] for i in valid_faces_indices]

			if not valid_areas:
				print("  No valid face areas calculated. Stopping subdivision.")
				break

			max_area = max(valid_areas) if valid_areas else 0
			if max_area < 1e-12:
				print("  Max face area is near zero. No subdivision will occur this level.")
				self.details = level
				break

			area_threshold = 0.5 * max_area
			print(f"  Max face area: {max_area:.4e}, Threshold for subdivision: {area_threshold:.4e}")

			new_faces_next_level = []
			self._subdivision_cache = {}
			faces_subdivided = 0
			faces_kept = 0

			print("  Processing faces for subdivision/keeping...")
			for i, face in enumerate(current_faces):
				current_area = face_areas[i]
				if current_area < 0:
					continue

				if current_area >= area_threshold:
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
						new_faces_next_level.append(face)
						faces_subdivided -= 1
						faces_kept += 1
						continue
					except Exception as e:
						print(f"  Unexpected error during midpoint creation for face {face.vertices}: {e}. Skipping subdivision.")
						new_faces_next_level.append(face)
						faces_subdivided -= 1
						faces_kept += 1
						continue

					if n == 3:
						v0, v1, v2 = vert_indices
						m01, m12, m20 = midpoint_indices
						new_faces_next_level.append(Face(v0, m01, m20))
						new_faces_next_level.append(Face(v1, m12, m01))
						new_faces_next_level.append(Face(v2, m20, m12))
						new_faces_next_level.append(Face(m01, m12, m20))
						for nf in new_faces_next_level[-4:]:
							try:
								nf._validate_face(self)
							except ValueError as ve:
								print(f"  Validation failed for new face {nf.vertices} during subdivision: {ve}")

					elif n > 3:
						center_face = Face(*midpoint_indices)
						new_faces_next_level.append(center_face)
						try:
							center_face._validate_face(self)
						except ValueError as ve:
							print(f"  Validation failed for new center face {center_face.vertices} during subdivision (n-gon): {ve}")

						for j in range(n):
							v_curr = vert_indices[j]
							m_next = midpoint_indices[j]
							m_prev = midpoint_indices[(j - 1 + n) % n]
							edge_triangle = Face(v_curr, m_next, m_prev)
							new_faces_next_level.append(edge_triangle)
							try:
								edge_triangle._validate_face(self)
							except ValueError as ve:
								print(f"  Validation failed for new edge triangle {edge_triangle.vertices} during subdivision: {ve}")

				else:
					faces_kept += 1
					new_faces_next_level.append(face)

			self.faces = new_faces_next_level
			print(f"  Level {level + 1} complete. Faces subdivided: {faces_subdivided}, Faces kept: {faces_kept}. Total faces: {len(self.faces)}")

		print(f"Subdivision finished. Final face count: {len(self.faces)}")

	def duplicateLayers(self):
		pass

	def validateStructure(self):
		pass


def VisualizeWorld(world: worldState, title: str = "World Mesh", plate_colors: Dict[int, str] = None):
	fig = plt.figure(figsize=(12, 10))
	ax = fig.add_subplot(111, projection='3d')

	face_polys = []
	face_colors = []
	edge_lines = []

	if world.faces:
		print(f"Visualizing {len(world.faces)} faces...")
		count = 0
		for face in world.faces:
			count += 1
			if count % 1000 == 0 : print(f" Processing face {count}/{len(world.faces)}")
			try:
				verts_idx = list(face.vertices)
				face_verts_pos_poly = [world.vertices[i].pos for i in verts_idx]

				if plate_colors:
					# Determine plate_id for this face (using the first vertex as a proxy - might need refinement for complex cases)
					plate_id = -1
					for plate in world.plates:
						if verts_idx[0] in plate.vertices:
							plate_id = plate.plate_id
							break
					if plate_id != -1 and plate_id in plate_colors:
						f_color = plate_colors[plate_id]
					else:
						face_normal = face.normal(world)
						z_normal = face_normal[2]
						f_color = cmap(norm(z_normal)) # Default color if plate color not found
				else:
					face_normal = face.normal(world)
					z_normal = face_normal[2]
					f_color = cmap(norm(z_normal))


				face_polys.append(face_verts_pos_poly)
				face_colors.append(f_color)

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

	if face_polys:
		print("Adding face collection...")
		poly_collection = Poly3DCollection(face_polys, alpha=0.75, facecolors=face_colors, edgecolors='none')
		ax.add_collection3d(poly_collection)
		print("Faces added.")

	if edge_lines:
		print("Adding edge lines...")
		for line_verts in edge_lines:
			xs, ys, zs = zip(*line_verts)
			ax.plot(xs, ys, zs, color='black', alpha=0.4, linewidth=0.6)
		print("Edges added.")

	if world.vertices:
		print("Adding vertices...")
		x, y, z = zip(*(v.pos for v in world.vertices))
		ax.scatter(x, y, z, color='white', s=8, alpha=0.9, edgecolors='black', linewidths=0.5, depthshade=False)
		print("Vertices added.")

	ax.set_box_aspect([1, 1, 1])
	limit = world.radius * 1.1
	ax.set_xlim(-limit, limit)
	ax.set_ylim(-limit, limit)
	ax.set_zlim(-limit, limit)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title(f'{title} (Level: {world.details}, Verts: {len(world.vertices)}, Faces: {len(world.faces)})')

	if plate_colors:
		# Create a legend for plates (basic example - adjust as needed)
		handles = [plt.Rectangle((0,0),1,1, color=color) for color in plate_colors.values()]
		labels = [f"Plate {i}" for i in plate_colors.keys()]
		ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
	else:
		scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
		scalar_mappable.set_array([])
		cbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.6, aspect=20, label='Face Normal Z (Green: +1, Red: -1)')

	print("Finalizing plot...")
	plt.tight_layout()
	plt.show()

def main():
	common_radius = 1.0
	num_plates = 15 # Modifiable number of plates

	# --- Icosphere Example ---
	print("Generating Icosphere...")
	world_ico = worldState()
	world_ico.radius = common_radius
	world_ico.details = 2 # Detail level
	world_ico.icosphereBase()
	print(f"Icosphere Vertices: {len(world_ico.vertices)}, Faces: {len(world_ico.faces)}")
	world_ico.plates = world_ico.assign_icosphere_vertices_to_plates(num_plates) # Assign plates with proximity-biased random walk + recycling
	print(f"Plates: {world_ico.plates}") # Print plate info

	# Assign colors to plates for visualization
	plate_colors_map = {}
	random.seed(42) # for consistent colors
	distinct_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige', 'maroon']
	plate_colors = random.sample(distinct_colors, num_plates) if num_plates <= len(distinct_colors) else distinct_colors * (num_plates // len(distinct_colors) + 1)

	for i in range(num_plates):
		plate_colors_map[i] = plate_colors[i]


	VisualizeWorld(world_ico, f"Icosphere with {num_plates} Plates (Proximity Biased + Recycling)", plate_colors=plate_colors_map)



if __name__ == "__main__":
	main()