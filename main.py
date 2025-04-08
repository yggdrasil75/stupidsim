import struct
from typing import Dict, List, Tuple, Set, Optional
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import RadioButtons, Slider
import numpy as np
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as colors
import random


PHI = (1.0 + np.sqrt(5.0)) / 2.0
norm_elevation = colors.Normalize(vmin=-10000, vmax=10000) # Example range, adjust later

PLATES = 15 #earth rate
SUBDIVISIONS = 3 #3 is balanced for testing, but 5 is needed for reasonable accuracy
MAX_ANGULAR_VELOCITY_RAD_PER_YR = np.radians(1.0) # Corresponds to ~11 cm/yr at equator for Earth radius. Adjust as needed.
ELEVATION_MOUNTAIN_BASE = 4000.0 # meters
ELEVATION_TRENCH_BASE = -5000.0 # meters
ELEVATION_DIFFUSION_FACTOR = 0.15 # How much elevation spreads per pass
ELEVATION_DIFFUSION_PASSES = 10 # Number of smoothing passes
RADIUS = 6371000

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
			raise IndexError(f"Error: Vertex index out of bounds in face {self.vertices}. Max index: {max_idx}, Num vertices: {len(world.vertices)}. Error: {e}") 

	def normal(self, world: 'worldState') -> np.ndarray:
		"""Calculates the approximate normal vector of the face (pointing outwards)."""
		verts = self.get_vertices(world)
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
		self.vertices: List[int] = [] 
		self.angular_velocity: np.ndarray = np.zeros(3, dtype=np.float64) 

	def add_vertex(self, vertex_index):
		self.vertices.append(vertex_index)

	def __repr__(self):
		ang_vel_deg_yr = np.degrees(np.linalg.norm(self.angular_velocity)) # Magnitude
		return (f"Plate(ID: {self.plate_id}, Vertices: {len(self.vertices)}, "
				f"AngVel: {ang_vel_deg_yr:.2f} deg/yr)")


class worldState:
	def __init__(self, radius: float = 1.0):
		self.vertices: List[Vertex] = []
		self.faces: List[Face] = []
		self.radius: float = radius # Physical radius (e.g., in meters or km)
		self.timestepSeconds: int = 1
		self.gravity: float = 9.81
		self.details: int = 3
		self.elevations: dict[int, float] = {}
		self.plates: List[Plate] = []
		self.adjacency_list: defaultdict[int, List[int]] = defaultdict(list)
		self.vertex_to_plate_id: Dict[int, int] = {}
		self.boundary_vertices: Set[int] = set() 
		self.boundary_edges: Set[Tuple[int, int]] = set() 
		self.boundary_properties: Dict[Tuple[int, int], Dict] = {}

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
		self.vertex_to_plate_id = {} # Reset map
		start_vertices_indices = random.sample(list(unassigned_vertices), num_plates)

		# Assign starting vertices
		for i in range(num_plates):
			start_vertex_index = start_vertices_indices[i]
			plates[i].add_vertex(start_vertex_index)
			self.vertex_to_plate_id[start_vertex_index] = i
			unassigned_vertices.remove(start_vertex_index)

		iteration_count = 0
		max_iterations = len(self.vertices) * 10

		while unassigned_vertices and iteration_count < max_iterations:
			iteration_count += 1

			# Pick a plate to potentially grow (can bias towards smaller plates later if needed)
			plate_index = random.randrange(num_plates)
			current_plate = plates[plate_index]

			# Find unassigned neighbors of the current plate's vertices
			possible_expansion_vertices = set()
			for plate_vertex_index in current_plate.vertices:
				# Check if vertex index is valid in adjacency list
				if plate_vertex_index in self.adjacency_list:
					for neighbor_idx in self.adjacency_list[plate_vertex_index]:
						if neighbor_idx in unassigned_vertices:
							possible_expansion_vertices.add(neighbor_idx)


			if not possible_expansion_vertices:
				# If this plate can't expand, try another one next iteration
				continue

			expansion_candidates = list(possible_expansion_vertices)

			# --- Proximity Biased Selection ---
			proximity_scores = []
			expansion_vertex_positions = {idx: self.vertices[idx].pos for idx in expansion_candidates}
			plate_vertex_positions = [self.vertices[idx].pos for idx in current_plate.vertices]

			for exp_idx, exp_pos in expansion_vertex_positions.items():
				# Find distance to the *closest* vertex already in the plate
				min_distance_sq = float('inf')
				for plate_v_pos in plate_vertex_positions:
					# Use squared Euclidean distance for efficiency (avoids sqrt)
					# Or use angular distance (dot product) - more relevant on sphere
					dot_prod = np.dot(exp_pos, plate_v_pos)
					# Clamp dot product for safety with acos
					angle = np.arccos(np.clip(dot_prod, -1.0, 1.0))
					# distance = angle # Use angle as distance measure
					# Use 1 - dot_prod (chord length proxy) or angle directly
					# Smaller angle (closer dot_prod to 1) means closer
					distance_metric = 1.0 - dot_prod # Smaller value is closer
					min_distance_sq = min(min_distance_sq, distance_metric)

				# Score inversely proportional to distance (closer is better)
				# Add small epsilon to avoid division by zero
				proximity_scores.append(1.0 / (min_distance_sq + 1e-9))

			# Normalize scores to get probabilities
			total_score = sum(proximity_scores)
			if total_score > 1e-9 and len(expansion_candidates) > 0:
				probabilities = np.array(proximity_scores) / total_score
				# Ensure probabilities sum to 1 (handle potential float errors)
				probabilities /= probabilities.sum()
			else:
				# Fallback: Equal probability if scores are zero or no candidates
				if len(expansion_candidates) > 0:
					probabilities = np.ones(len(expansion_candidates)) / len(expansion_candidates)
				else:
					# This case shouldn't be reached if possible_expansion_vertices was checked
					continue # Skip if somehow candidates disappeared

			# Choose the next vertex based on calculated probabilities
			try:
				# Ensure probabilities match the number of candidates
				if len(probabilities) != len(expansion_candidates):
					next_vertex_index = random.choice(expansion_candidates)
				else:
					next_vertex_index = np.random.choice(expansion_candidates, p=probabilities)
			except ValueError as e:
				print("failure")
				next_vertex_index = random.choice(expansion_candidates)

			# Assign the chosen vertex to the current plate
			current_plate.add_vertex(next_vertex_index)
			self.vertex_to_plate_id[next_vertex_index] = plate_index
			unassigned_vertices.remove(next_vertex_index)

		if unassigned_vertices:
			remaining_unassigned = list(unassigned_vertices)
			random.shuffle(remaining_unassigned)
			for idx, vertex_idx in enumerate(remaining_unassigned):
				target_plate_index = idx % num_plates # Distribute somewhat evenly
				plates[target_plate_index].add_vertex(vertex_idx)
				self.vertex_to_plate_id[vertex_idx] = target_plate_index
				unassigned_vertices.remove(vertex_idx)


		if len(self.vertex_to_plate_id) != len(self.vertices):
			# Attempt to find missing vertices
			assigned_v_set = set(self.vertex_to_plate_id.keys())
			all_v_set = set(range(len(self.vertices)))
			missing = all_v_set - assigned_v_set
			if missing:
				# Assign missing ones randomly as a fallback
				for v_idx in missing:
					target_plate_index = random.randrange(num_plates)
					plates[target_plate_index].add_vertex(v_idx)
					self.vertex_to_plate_id[v_idx] = target_plate_index

		recycling_iterations = 5
		recycled_total = 0

		for recycle_iter in range(recycling_iterations):
			vertices_to_reassign = []
			# Identify vertices on the border of their current plate
			for plate_idx, plate in enumerate(plates):
				border_vertices_this_plate = []
				for vertex_index in plate.vertices:
					is_border = False
					if vertex_index not in self.adjacency_list: continue # Skip if no neighbors known

					num_neighbors_same_plate = 0
					num_neighbors_total = 0
					for neighbor_index in self.adjacency_list[vertex_index]:
						num_neighbors_total += 1
						# Check if neighbor belongs to the *same* plate
						if neighbor_index in self.vertex_to_plate_id and self.vertex_to_plate_id[neighbor_index] == plate_idx:
							num_neighbors_same_plate += 1
						else:
							is_border = True # It has at least one neighbor from another plate

					# Criteria for recycling: on the border AND maybe has few same-plate neighbors?
					# Simple: just reassign border vertices.
					# More complex: reassign if ratio num_same / num_total is low?
					if is_border: # Recycle all border vertices in this pass
						border_vertices_this_plate.append(vertex_index)

				vertices_to_reassign.extend(border_vertices_this_plate)

			if not vertices_to_reassign:
				break

			recycled_total += len(vertices_to_reassign)
			random.shuffle(vertices_to_reassign) # Process in random order

			reassigned_count = 0
			for vertex_index in vertices_to_reassign:
				current_plate_id = self.vertex_to_plate_id.get(vertex_index, -1)
				if current_plate_id == -1: continue # Should already be assigned

				# Find neighboring plates and count neighbors belonging to each
				neighboring_plate_counts = defaultdict(int)
				if vertex_index in self.adjacency_list:
					vertex_pos = self.vertices[vertex_index].pos
					for neighbor_idx in self.adjacency_list[vertex_index]:
						neighbor_plate_id = self.vertex_to_plate_id.get(neighbor_idx, -1)
						if neighbor_plate_id != -1:
							neighboring_plate_counts[neighbor_plate_id] += 1
				else: continue # Skip if no neighbors known

				if not neighboring_plate_counts:
					# Vertex has no assigned neighbors, keep its current plate
					continue

				# --- Vote / Proximity Reassignment ---
				# Option 1: Simple Majority Vote
				# best_plate_id = max(neighboring_plate_counts, key=neighboring_plate_counts.get)

				# Option 2: Weighted Vote by Proximity (similar to growth phase)
				best_plate_id = -1
				best_score = -1.0
				possible_plates = list(neighboring_plate_counts.keys())

				scores = {}
				for target_plate_id in possible_plates:
					plate_score = 0
					# Consider neighbors belonging to this target plate
					for neighbor_idx in self.adjacency_list[vertex_index]:
						if self.vertex_to_plate_id.get(neighbor_idx) == target_plate_id:
							# Add score based on proximity (dot product)
							neighbor_pos = self.vertices[neighbor_idx].pos
							dot_prod = np.dot(vertex_pos, neighbor_pos)
							plate_score += (1.0 + dot_prod) # Score higher for closer neighbors (dot ~ 1)

					# Normalize by the number of neighbors from that plate? Or just sum scores? Sum seems ok.
					scores[target_plate_id] = plate_score * neighboring_plate_counts[target_plate_id] # Weight by count too

				if scores:
					best_plate_id = max(scores, key=scores.get)
				else:
					# Fallback if scoring fails
					best_plate_id = random.choice(possible_plates) if possible_plates else current_plate_id


				# Reassign if the best neighboring plate is different from the current one
				if best_plate_id != -1 and best_plate_id != current_plate_id:
					# Remove from old plate
					if vertex_index in plates[current_plate_id].vertices:
						plates[current_plate_id].vertices.remove(vertex_index)
					# Add to new plate
					plates[best_plate_id].add_vertex(vertex_index)
					self.vertex_to_plate_id[vertex_index] = best_plate_id
					reassigned_count += 1


		final_v_count = 0
		for i, p in enumerate(plates):
			p.vertices = sorted(list(set(p.vertices))) # Ensure unique and sorted
			# Verify mapping consistency
			for v_idx in p.vertices:
				if self.vertex_to_plate_id.get(v_idx) != i:
					self.vertex_to_plate_id[v_idx] = i
			final_v_count += len(p.vertices)

		self.plates = plates
		return plates

	def subdivide(self):
		"""Subdivides faces of the mesh based on detail level."""
		if self.details <= 0: return

		for level in range(self.details):
			current_faces = self.faces[:]
			if not current_faces:
				break

			face_areas = []
			valid_faces_indices = []
			for i, face in enumerate(current_faces):
				try:
					area = face.area(self)
					if not np.isfinite(area) or area < 0:
						face_areas.append(-1.0)
					else:
						face_areas.append(area)
						valid_faces_indices.append(i)
				except Exception as e:
					face_areas.append(-1.0)

			valid_areas = [face_areas[i] for i in valid_faces_indices]

			if not valid_areas:
				break

			max_area = max(valid_areas) if valid_areas else 0
			if max_area < 1e-12:
				self.details = level
				break

			area_threshold = 0.5 * max_area

			new_faces_next_level = []
			self._subdivision_cache = {}
			faces_subdivided = 0
			faces_kept = 0

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

	def duplicateLayers(self):
		pass

	def validateStructure(self, check_intersection=False):
		"""Performs checks on the mesh structure."""
		valid = True
		num_vertices = len(self.vertices)
		num_faces = len(self.faces)

		nan_inf_verts = 0
		non_unit_verts = 0
		for i, v in enumerate(self.vertices):
			if np.any(np.isnan(v.pos)) or np.any(np.isinf(v.pos)):
				nan_inf_verts += 1
				valid = False
			norm_sq = np.dot(v.pos, v.pos)
			if abs(norm_sq - 1.0) > 1e-7: # Check if norm is close to 1
				non_unit_verts += 1
				valid = False
		if (nan_inf_verts > 0) or (non_unit_verts > 0):
			print(f"  Found {nan_inf_verts} vertices with NaN/Inf and {non_unit_verts} needing to be normalized")

		intersecting_faces = 0

		for i, face in enumerate(self.faces):
			face_valid = True
			# Check for self-intersection (optional, can be slow)
			if face_valid and check_intersection and len(face.vertices) > 3:
				try:
					face._validate_face(self) # This internal method checks intersection
				except (ValueError, IndexError) as e:
					print(f"  Error: Face {i} {face.vertices} failed validation: {e}")
					intersecting_faces += 1 # Count intersection or other validation errors from _validate_face
					valid = False
				except Exception as e:
					print(f"  Error: Unexpected error validating face {i} {face.vertices}: {e}")
					intersecting_faces += 1
					valid = False


		if intersecting_faces > 0: print(f"  Found {intersecting_faces} faces failing validation (e.g., self-intersection).")

		adj_list_ok = True
		if not self.adjacency_list and num_vertices > 0:
			adj_list_ok = False
			valid = False
		elif self.adjacency_list:
			# Basic check: ensure all keys and values are valid vertex indices
			max_adj_idx = -1
			for k, neighbors in self.adjacency_list.items():
				max_adj_idx = max(max_adj_idx, k)
				if k >= num_vertices or k < 0:
					print(f"  Error: Invalid key {k} in adjacency list.")
					adj_list_ok = False
					valid = False
				for neighbor in neighbors:
					max_adj_idx = max(max_adj_idx, neighbor)
					if neighbor >= num_vertices or neighbor < 0:
						print(f"  Error: Invalid neighbor index {neighbor} for key {k} in adjacency list.")
						adj_list_ok = False
						valid = False
					# Check symmetry: if k lists neighbor, neighbor should list k
					if neighbor not in self.adjacency_list or k not in self.adjacency_list[neighbor]:
						print(f"  Error: Adjacency list asymmetry detected between {k} and {neighbor}.")
						adj_list_ok = False
						valid = False

		if self.plates:
			plate_check_ok = True
			assigned_verts = set()
			total_verts_in_plates = 0
			if not self.vertex_to_plate_id:
				plate_check_ok = False
				valid = False
			else:
				# Check consistency between plate lists and map
				for plate_id, plate in enumerate(self.plates):
					if plate.plate_id != plate_id:
						plate_check_ok = False; valid = False
					total_verts_in_plates += len(plate.vertices)
					for v_idx in plate.vertices:
						if v_idx >= num_vertices or v_idx < 0:
							plate_check_ok = False; valid = False
						elif self.vertex_to_plate_id.get(v_idx) != plate_id:
							plate_check_ok = False; valid = False
						assigned_verts.add(v_idx)

		return valid

	def _build_vertex_plate_map(self):
		self.vertex_to_plate_id.clear()
		for plate_idx, plate in enumerate(self.plates):
			for vertex_idx in plate.vertices:
				if vertex_idx in self.vertex_to_plate_id:
					print(f"Warning: Vertex {vertex_idx} found in multiple plates ({self.vertex_to_plate_id[vertex_idx]} and {plate_idx}). Overwriting.")
				if vertex_idx >= len(self.vertices) or vertex_idx < 0:
					print(f"Warning: Plate {plate_idx} contains invalid vertex index {vertex_idx}. Skipping.")
					continue
				self.vertex_to_plate_id[vertex_idx] = plate_idx
		if len(self.vertex_to_plate_id) != len(self.vertices):
			print("Error: Mismatch between map size and vertex count!")

	def _identify_boundaries(self):
		"""Identifies vertices and edges lying on plate boundaries."""
		print("Identifying plate boundaries...")
		self.boundary_vertices.clear()
		self.boundary_edges.clear()
		if not self.vertex_to_plate_id:
			print(" Vertex-to-plate map is empty. Building it first.")
			self._build_vertex_plate_map()
			if not self.vertex_to_plate_id:
				print("Error: Cannot identify boundaries without vertex-to-plate map.")
				return

		if not self.adjacency_list:
			print(" Adjacency list is empty. Building it first.")
			self._build_adjacency_list()
			if not self.adjacency_list:
				print("Error: Cannot identify boundaries without adjacency list.")
				return

		num_vertices = len(self.vertices)
		for v1_idx in range(num_vertices):
			plate1_id = self.vertex_to_plate_id.get(v1_idx, -1)
			if plate1_id == -1:
				# print(f"Warning: Vertex {v1_idx} is not assigned to any plate.")
				continue # Cannot be a boundary if not on a plate

			if v1_idx not in self.adjacency_list:
				# print(f"Warning: Vertex {v1_idx} (Plate {plate1_id}) has no known neighbors.")
				continue

			is_boundary_vertex = False
			for v2_idx in self.adjacency_list[v1_idx]:
				plate2_id = self.vertex_to_plate_id.get(v2_idx, -1)

				if plate2_id != -1 and plate1_id != plate2_id:
					# This vertex (v1_idx) borders another plate (plate2_id)
					is_boundary_vertex = True
					# Add the edge connecting them to the boundary edges set
					edge = tuple(sorted((v1_idx, v2_idx)))
					self.boundary_edges.add(edge)
					# Mark the neighbor (v2_idx) as a boundary vertex too
					self.boundary_vertices.add(v2_idx)

			if is_boundary_vertex:
				self.boundary_vertices.add(v1_idx)

		print(f"Boundaries identified. Boundary Vertices: {len(self.boundary_vertices)}, Boundary Edges: {len(self.boundary_edges)}")

	def assign_random_angular_velocities(self, max_angular_velocity: float):
		if not self.plates:
			print("Warning: No plates exist to assign velocities to.")
			return

		for plate in self.plates:
			# Random direction (uniformly on sphere)
			phi = np.random.uniform(0, 2 * np.pi)
			costheta = np.random.uniform(-1, 1)
			theta = np.arccos(costheta)
			x = np.sin(theta) * np.cos(phi)
			y = np.sin(theta) * np.sin(phi)
			z = np.cos(theta)
			direction = np.array([x, y, z])

			# Random magnitude up to max_angular_velocity
			magnitude = np.random.uniform(0, max_angular_velocity)

			plate.angular_velocity = direction * magnitude
			# print(f" Plate {plate.plate_id}: Magnitude {np.degrees(magnitude):.3f} deg/yr")
		print("Angular velocities assigned.")

	def calculate_boundary_motions(self, classification_threshold: float = 0.7) -> bool:
		self.boundary_properties.clear() # Clear previous properties

		m_per_year_to_m_per_sec = 1.0 / (365.25 * 24 * 3600)

		successful_calculations = 0
		for edge in self.boundary_edges:
			v1_idx, v2_idx = edge

			# Get plate IDs and check validity
			plate1_id = self.vertex_to_plate_id.get(v1_idx, -1)
			plate2_id = self.vertex_to_plate_id.get(v2_idx, -1)
			if plate1_id == -1 or plate2_id == -1 or plate1_id == plate2_id:
				continue

			# Get vertices and angular velocities
			try:
				v1 = self.vertices[v1_idx]
				v2 = self.vertices[v2_idx]
				plate1 = self.plates[plate1_id]
				plate2 = self.plates[plate2_id]
			except IndexError:
				continue

			omega1 = plate1.angular_velocity
			omega2 = plate2.angular_velocity

			# Calculate relative angular velocity (Omega2 relative to Omega1)
			omega_rel = omega2 - omega1

			# Calculate linear velocity at the midpoint of the edge
			# Midpoint position (approximation, could use SLERP for perfect sphere)
			p_mid = (v1.pos + v2.pos) / 2.0
			p_mid_norm = np.linalg.norm(p_mid)
			if p_mid_norm < 1e-9:
				print(f"Warning: Midpoint of edge {edge} is near origin. Skipping motion calc.")
				continue
			p_mid_unit = p_mid / p_mid_norm # Position vector on the unit sphere
			p_mid_world = p_mid_unit * self.radius # Position vector in world units (meters)

			# Relative linear velocity: v_rel = omega_rel x p_mid_world
			# v_rel = np.cross(omega_rel, p_mid_world) # meters/year

			# Calculate relative velocity more robustly using individual velocities
			vel1 = np.cross(omega1, p_mid_world) # Velocity of point if it were on plate 1
			vel2 = np.cross(omega2, p_mid_world) # Velocity of point if it were on plate 2
			v_rel = vel2 - vel1 # Velocity of plate 2 relative to plate 1 at p_mid (meters/year)


			# --- Classify Boundary ---
			# Need a local normal vector separating the plates at the edge midpoint
			# Approx normal: radial vector x edge vector (normalized)
			edge_vec = v2.pos - v1.pos # Vector along the edge on unit sphere
			if np.linalg.norm(edge_vec) < 1e-9:
				print(f"Warning: Edge {edge} has near-zero length. Skipping classification.")
				continue

			# Radial vector at midpoint (already calculated p_mid_unit)
			# boundary_normal = np.cross(p_mid_unit, edge_vec) # Normal to edge tangent and radial vector
			# Alternative normal: Cross edge with radial, points along boundary? Need normal *separating* plates.
			# Let's use the vector *between* the vertices projected onto the tangent plane.
			# Project edge_vec onto tangent plane at p_mid:
			tangent_edge_vec = edge_vec - np.dot(edge_vec, p_mid_unit) * p_mid_unit
			norm_tangent_edge = np.linalg.norm(tangent_edge_vec)
			if norm_tangent_edge < 1e-9:
				print(f"Warning: Tangent edge vector for {edge} is near-zero. Skipping classification.")
				continue
			tangent_edge_unit = tangent_edge_vec / norm_tangent_edge

			# The normal separating the plates locally should be perpendicular to the tangent edge vector
			# and also lie on the tangent plane (i.e., perpendicular to p_mid_unit)
			boundary_normal_local = np.cross(p_mid_unit, tangent_edge_unit)
			# No need to normalize boundary_normal_local for dot product projection if v_rel is used directly.

			# Decompose v_rel into components along the boundary normal and tangent
			# Ensure v_rel is projected onto the tangent plane first (it should mostly be if omega is correct)
			v_rel_tangent = v_rel - np.dot(v_rel, p_mid_unit) * p_mid_unit

			# Component along the separating normal (Convergence/Divergence)
			# Positive dot product means moving *along* the normal (divergence)
			# Negative dot product means moving *against* the normal (convergence)
			convergence_component_scalar = np.dot(v_rel_tangent, boundary_normal_local)
			convergence_vec = convergence_component_scalar * boundary_normal_local

			# Component along the boundary edge (Transform/Strike-Slip)
			transform_component_scalar = np.dot(v_rel_tangent, tangent_edge_unit)
			transform_vec = transform_component_scalar * tangent_edge_unit

			# Magnitudes
			convergence_mag = abs(convergence_component_scalar) # meters/year
			transform_mag = abs(transform_component_scalar) # meters/year
			total_mag = np.linalg.norm(v_rel_tangent) # meters/year

			boundary_type = "undefined"
			if total_mag < 1e-9: # Plates aren't moving relative to each other here
				boundary_type = "passive" # Or "rift" if it was previously active?
			else:
				# Ratios for classification
				conv_ratio = convergence_mag / total_mag
				trans_ratio = transform_mag / total_mag

				if conv_ratio >= classification_threshold:
					if convergence_component_scalar < 0: # Moving against normal
						boundary_type = "convergent"
					else: # Moving along normal
						boundary_type = "divergent"
				elif trans_ratio >= classification_threshold:
					boundary_type = "transform"
				else:
					# Oblique motion - classify based on dominant component?
					if convergence_mag > transform_mag:
						boundary_type = "convergent-oblique" if convergence_component_scalar < 0 else "divergent-oblique"
					else:
						boundary_type = "transform-oblique"
					# Or just leave as oblique? Let's assign based on dominant for elevation.
					if convergence_mag > transform_mag:
						boundary_type = "convergent" if convergence_component_scalar < 0 else "divergent"
					else:
						boundary_type = "transform"


			# Store properties
			self.boundary_properties[edge] = {
				"type": boundary_type,
				"relative_velocity_mps": v_rel * m_per_year_to_m_per_sec, # m/s
				"convergence_rate_mps": convergence_component_scalar * m_per_year_to_m_per_sec, # Positive=divergence
				"transform_rate_mps": transform_component_scalar * m_per_year_to_m_per_sec,
				"total_rate_mps": total_mag * m_per_year_to_m_per_sec
			}
			successful_calculations += 1

		print(f"Boundary motion calculation complete. Processed {successful_calculations}/{len(self.boundary_edges)} edges.")
		if successful_calculations == 0 and len(self.boundary_edges) > 0:
			return False
		return True

	def assign_elevations_from_boundaries(self,
										  base_convergent: float = ELEVATION_MOUNTAIN_BASE,
										  base_divergent: float = ELEVATION_TRENCH_BASE,
										  base_transform: float = 100.0, # Slight ridge/fracture zone
										  rate_scaling_factor: float = 1e6, # Scale velocity (m/yr) to elevation impact
										  diffusion_passes: int = ELEVATION_DIFFUSION_PASSES,
										  diffusion_factor: float = ELEVATION_DIFFUSION_FACTOR):
		"""Assigns vertex elevations based on nearby boundary types and magnitudes, then smooths."""
		print("\nAssigning elevations based on plate boundaries...")
		if not self.boundary_properties:
			print(" Error: Boundary properties not calculated. Run calculate_boundary_motions first.")
			return
		if not self.vertices:
			print(" Error: No vertices exist.")
			return

		num_vertices = len(self.vertices)
		# Initialize elevations to 0 (or a base sea level)
		base_elevation = 0.0
		self.elevations = {i: base_elevation for i in range(num_vertices)}

		# --- Step 1: Apply direct elevation changes at boundary vertices ---
		print(f" Applying direct elevation changes to {len(self.boundary_vertices)} boundary vertices...")
		elevation_updates = defaultdict(lambda: {'sum_influence': 0.0, 'count': 0})

		# Convert rates from m/s back to m/year for potentially more intuitive scaling
		m_per_sec_to_m_per_yr = 1.0 / (1.0 / (365.25 * 24 * 3600))

		for edge, props in self.boundary_properties.items():
			v1_idx, v2_idx = edge
			boundary_type = props.get("type", "undefined")
			# Use absolute convergence rate for magnitude, sign determines type effect
			conv_rate_myr = props.get("convergence_rate_mps", 0.0) * m_per_sec_to_m_per_yr
			trans_rate_myr = abs(props.get("transform_rate_mps", 0.0)) * m_per_sec_to_m_per_yr

			elevation_change = 0.0
			if boundary_type == "convergent":
				# Negative conv_rate_myr means convergence
				magnitude = abs(conv_rate_myr) # Use absolute rate
				elevation_change = base_convergent * (1 + magnitude * rate_scaling_factor / base_convergent)
			elif boundary_type == "divergent":
				# Positive conv_rate_myr means divergence
				magnitude = abs(conv_rate_myr)
				elevation_change = base_divergent * (1 + magnitude * rate_scaling_factor / abs(base_divergent)) # Trench gets deeper
			elif boundary_type == "transform":
				magnitude = trans_rate_myr
				elevation_change = base_transform * (1 + magnitude * rate_scaling_factor / base_transform) # Minor effect, scales with slip rate


			# Apply influence to both vertices of the edge
			# We average the influence from all connected boundary edges later
			if boundary_type != "passive" and boundary_type != "undefined":
				if v1_idx in self.boundary_vertices:
					elevation_updates[v1_idx]['sum_influence'] += elevation_change
					elevation_updates[v1_idx]['count'] += 1
				if v2_idx in self.boundary_vertices:
					elevation_updates[v2_idx]['sum_influence'] += elevation_change
					elevation_updates[v2_idx]['count'] += 1

		# Assign the averaged initial elevation change to boundary vertices
		for v_idx, data in elevation_updates.items():
			if data['count'] > 0:
				avg_influence = data['sum_influence'] / data['count']
				self.elevations[v_idx] = base_elevation + avg_influence
			# else: vertex was in boundary_vertices but no edges contributed? Should not happen.


		print(f" Initial boundary elevations assigned. Max: {max(self.elevations.values()):.0f}m, Min: {min(self.elevations.values()):.0f}m")

		# --- Step 2: Diffuse elevation inland ---
		if diffusion_passes > 0 and diffusion_factor > 0:
			print(f" Performing {diffusion_passes} elevation diffusion passes (Factor: {diffusion_factor})...")
			# Use a temporary dictionary for updates to avoid overwriting during pass
			current_elevations = self.elevations.copy()
			next_elevations = self.elevations.copy()

			for i_pass in range(diffusion_passes):

				# Update all vertices *except* potentially the boundary ones?
				# Let's update all vertices, allowing boundaries to smooth slightly too.
				for v_idx in range(num_vertices):
					if v_idx not in self.adjacency_list or not self.adjacency_list[v_idx]:
						# Isolated vertex, keep its current elevation
						next_elevations[v_idx] = current_elevations.get(v_idx, base_elevation)
						continue

					neighbor_indices = self.adjacency_list[v_idx]
					sum_neighbor_elev = 0
					num_valid_neighbors = 0
					for n_idx in neighbor_indices:
						# Check if neighbor index is valid before accessing elevation
						if n_idx in current_elevations:
							sum_neighbor_elev += current_elevations[n_idx]
							num_valid_neighbors += 1
						# else: print(f"Warning: Neighbor {n_idx} not in current_elevations during diffusion.")


					if num_valid_neighbors > 0:
						avg_neighbor_elev = sum_neighbor_elev / num_valid_neighbors
						current_elev = current_elevations.get(v_idx, base_elevation) # Get current elevation

						# Weighted average: (1-f)*current + f*average_neighbor
						smoothed_elev = (1.0 - diffusion_factor) * current_elev + diffusion_factor * avg_neighbor_elev
						next_elevations[v_idx] = smoothed_elev
					else:
						# No valid neighbors, keep current elevation
						next_elevations[v_idx] = current_elevations.get(v_idx, base_elevation)


				# Update current_elevations for the next pass
				current_elevations = next_elevations.copy()


			# Assign the final smoothed elevations back to the world state
			self.elevations = current_elevations
			print(" Elevation diffusion complete.")
		else:
			print(" Skipping elevation diffusion.")

		# Final elevation range
		final_max = max(self.elevations.values()) if self.elevations else 0
		final_min = min(self.elevations.values()) if self.elevations else 0
		norm_elevation.vmin = final_min
		norm_elevation.vmax = final_max
		print(f"Final Elevation Range: {final_min:.0f}m to {final_max:.0f}m")



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

		# Determine elevation range for normalization if needed
		if color_mode == "elevation" and has_elevations:
			all_elevs = list(world.elevations.values())
			if all_elevs:
				min_elev, max_elev = min(all_elevs), max(all_elevs)
				print(f"Elevation range for coloring: {min_elev:.0f}m to {max_elev:.0f}m")
				# Update the norm dynamically based on actual data
				norm_elevation_dynamic = colors.Normalize(vmin=min_elev, vmax=max_elev)
			else:
				print("Warning: Elevation data exists but is empty. Using default elevation norm.")
				norm_elevation_dynamic = norm_elevation # Use default if no values
		elif color_mode == "elevation" and not has_elevations:
			print("Warning: Elevation color mode selected, but no elevation data found. Using default normal colors.")
			color_mode = "normal" # Fallback

		# Default normal colormap setup
		cmap_normal = cm.get_cmap('RdYlGn_r')
		norm_normal = colors.Normalize(vmin=-1.0, vmax=1.0)

		# Plate color setup
		if color_mode == "plates" and not has_plates:
			print("Warning: Plate color mode selected, but no plate data found. Using default normal colors.")
			color_mode = "normal" # Fallback
		elif color_mode == "plates" and not plate_colors_map:
			print("Warning: Plate color mode selected, but no plate_colors_map provided. Generating random colors.")
			# Generate random colors if missing
			plate_colors_map = {}
			num_plates_vis = len(world.plates)
			distinct_colors = plt.cm.get_cmap('tab20').colors # Use a colormap with distinct colors
			for i in range(num_plates_vis):
				plate_colors_map[i] = distinct_colors[i % len(distinct_colors)]


		for face in world.faces:
			count += 1
			if count % 5000 == 0 : print(f" Processing face {count}/{len(world.faces)}")

			try:
				verts_idx = list(face.vertices)
				# Get 3D positions for the polygon
				face_verts_pos_poly = [world.vertices[i].pos * world.radius for i in verts_idx]
				# Note: Scaling by world.radius here for visualization if radius != 1

				# --- Determine Face Color ---
				f_color = (0.5, 0.5, 0.5, 0.5) # Default gray

				if color_mode == "elevation":
					# Average elevation of face vertices
					face_elevations = [world.elevations.get(i, 0.0) for i in verts_idx]
					avg_elevation = sum(face_elevations) / len(face_elevations) if face_elevations else 0.0

				elif color_mode == "plates":
					# Use plate ID of the first vertex (approximation)
					first_vertex_id = verts_idx[0]
					plate_id = world.vertex_to_plate_id.get(first_vertex_id, -1)
					if plate_id != -1 and plate_colors_map and plate_id in plate_colors_map:
						f_color = plate_colors_map[plate_id]
					else:
						f_color = (0.3, 0.3, 0.3, 0.6) # Dark gray for unassigned/missing color

				elif color_mode == "normal":
					# Color by face normal's Z component
					face_normal_vec = face.normal(world)
					z_normal = face_normal_vec[2] if len(face_normal_vec) == 3 else 0.0
					f_color = cmap_normal(norm_normal(z_normal))

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
		print(" Adding face collection to plot...")
		# Use edgecolors='none' for cleaner look with many polygons
		poly_collection = Poly3DCollection(face_polys, alpha=0.85, facecolors=face_colors, edgecolors='none', linewidth=0)
		ax.add_collection3d(poly_collection)
		print(" Faces added.")
	else:
		print(" No face polygons to plot.")

	if edge_lines:
		print(" Adding edge lines to plot...")
		# Plot edges with low alpha to avoid clutter
		for line_verts in edge_lines:
			xs, ys, zs = zip(*line_verts)
			ax.plot(xs, ys, zs, color='black', alpha=0.15, linewidth=0.4)
		print(" Edges added.")
	else:
		print(" No edge lines to plot.")

	# Optionally plot vertices (can be slow for high detail)
	plot_vertices = False
	if plot_vertices and world.vertices:
		print(" Adding vertices to plot...")
		vx, vy, vz = zip(*(v.pos * world.radius for v in world.vertices))
		ax.scatter(vx, vy, vz, color='white', s=5, alpha=0.7, edgecolors='black', linewidths=0.3, depthshade=False)
		print(" Vertices added.")

	# --- Axes and Labels ---
	ax.set_box_aspect([1, 1, 1]) # Equal aspect ratio
	# Set limits based on radius
	limit = world.radius * 1.15 # Add a bit of padding
	ax.set_xlim(-limit, limit)
	ax.set_ylim(-limit, limit)
	ax.set_zlim(-limit, limit)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title(f'{title}\n(Detail: {world.details}, Verts: {len(world.vertices)}, Faces: {len(world.faces)})')
	# Basic lighting/shading improvement
	# ax.view_init(elev=30, azim=45) # Adjust viewing angle

	# --- Colorbar / Legend ---
	if color_mode == "elevation" and has_elevations and all_elevs:
		scalar_mappable = cm.ScalarMappable(norm=norm_elevation_dynamic, cmap='terrain')
		scalar_mappable.set_array([]) # Important!
		cbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.6, aspect=20, label='Elevation (m)')
	elif color_mode == "plates" and has_plates and plate_colors_map:
		handles = [plt.Rectangle((0,0),1,1, color=color) for color in plate_colors_map.values()]
		labels = [f"Plate {i}" for i in plate_colors_map.keys()]
		ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Plates")
	elif color_mode == "normal":
		scalar_mappable = cm.ScalarMappable(norm=norm_normal, cmap=cmap_normal)
		scalar_mappable.set_array([])
		cbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.6, aspect=20, label='Face Normal Z')
	plt.tight_layout(rect=[0, 0, 0.85, 1] if color_mode=='plates' else [0, 0, 1, 1]) # Adjust layout for legend/colorbar
	plt.show()


# --- Main Execution ---

def main():
	world_radius_m = RADIUS # Approx Earth radius in meters
	num_plates = PLATES
	subdivision_level = SUBDIVISIONS
	max_plate_speed_deg_yr = 1.5 # Max rotation speed in degrees per year
	max_ang_vel_rad_yr = np.radians(max_plate_speed_deg_yr)

	# Elevation generation parameters
	conv_elev = 6000.0 # Max elevation boost for convergence
	div_elev = -7000.0 # Max depth for divergence
	trans_elev = 200.0 # Minor ridges for transform
	rate_scale = 5.0e7 # How much velocity (m/yr) affects elevation magnitude
	diff_passes = 15
	diff_factor = 0.10

	world_sim = worldState(radius=world_radius_m)
	world_sim.details = subdivision_level

	world_sim.icosphereBase()

	# Validate the initial subdivided mesh structure
	world_sim.validateStructure(check_intersection=False) # Intersection check is slow

	world_sim.plates = world_sim.assign_icosphere_vertices_to_plates(num_plates)

	# Verify plate assignment consistency
	world_sim._build_vertex_plate_map() # Ensure map is built/updated
	# world_sim.validateStructure() # Re-validate after plate assignment updates map

	# --- Tectonic Simulation ---
	if world_sim.plates:
		world_sim.assign_random_angular_velocities(max_ang_vel_rad_yr)

		world_sim._identify_boundaries()

		success = world_sim.calculate_boundary_motions(classification_threshold=0.65)

		if success:
			world_sim.assign_elevations_from_boundaries(
				base_convergent=conv_elev,
				base_divergent=div_elev,
				base_transform=trans_elev,
				rate_scaling_factor=rate_scale,
				diffusion_passes=diff_passes,
				diffusion_factor=diff_factor
			)
		else:
			print("Skipping elevation assignment due to errors in motion calculation.")

		world_sim.validateStructure(check_intersection=False)

	else:
		print("\nNo plates assigned, skipping tectonic simulation.")


	plate_colors = None
	if world_sim.plates:
		plate_colors = {}
		cmap_plates_vis = cm.get_cmap('turbo', len(world_sim.plates)) # 'turbo' is good for many categories
		for i in range(len(world_sim.plates)):
			plate_colors[i] = cmap_plates_vis(i)

	VisualizeWorld(world_sim,
				   title=f"World Simulation with Tectonic Elevation (Plates: {num_plates}, Detail: {subdivision_level})",
				   color_mode="elevation") # Show elevation map

if __name__ == "__main__":
	main()