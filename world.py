

from collections import defaultdict
import random
from typing import Dict, List, Set, Tuple

import numpy as np
from globals import CONTINENTAL_BASE_ELEVATION, CONTINENTAL_PLATE_PROB, ELEVATION_DIFFUSION_FACTOR, ELEVATION_DIFFUSION_PASSES, ELEVATION_MOUNTAIN_BASE, ELEVATION_TRENCH_BASE, MAX_ANGULAR_VELOCITY_RAD_PER_YR, NORM_ELEVATION, OCEANIC_BASE_ELEVATION, PHI
from shape import Face, Vertex


#copied from globals for reference
# NORM_ELEVATION = colors.Normalize(vmin=-15000, vmax=15000) # Fixed range for colorbar
# PLATES = 15 #earth rate
# SUBDIVISIONS: int = 3 #3 is balanced for testing, but 5 is needed for reasonable accuracy
# MAX_ANGULAR_VELOCITY_RAD_PER_YR = np.radians(1.0) # Corresponds to ~11 cm/yr at equator for Earth radius. Adjust as needed.
# ELEVATION_MOUNTAIN_BASE = 10000.0 # meters
# ELEVATION_TRENCH_BASE = -11000.0 # meters
# ELEVATION_DIFFUSION_FACTOR = 0.15 # How much elevation spreads per pass
# ELEVATION_DIFFUSION_PASSES = 10 # Number of smoothing passes
# RADIUS = 6371000

# CONTINENTAL_PLATE_PROB = 0.3  # Probability of a plate being continental
# CONTINENTAL_BASE_ELEVATION = 2000.0 # meters
# OCEANIC_BASE_ELEVATION = -3000.0 # meters

# PHI = (1.0 + np.sqrt(5.0)) / 2.0

class Plate:
	def __init__(self, plate_id):
		self.plate_id = plate_id
		self.vertices: List[int] = []
		self.angular_velocity: np.ndarray = np.zeros(3, dtype=np.float64)
		self.plate_type: str = 'oceanic' # Default to oceanic, can be 'continental'

	def add_vertex(self, vertex_index):
		self.vertices.append(vertex_index)

	def __repr__(self):
		ang_vel_deg_yr = np.degrees(np.linalg.norm(self.angular_velocity)) # Magnitude
		return (f"Plate(ID: {self.plate_id}, Type: {self.plate_type}, Vertices: {len(self.vertices)}, "
				f"AngVel: {ang_vel_deg_yr:.2f} deg/yr)")

class worldState:
	def __init__(self, radius, subdivisions=3):
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
		self.details = subdivisions
		self.shapeBase()

	def shapeBase(self):
		#implement in subclass
		pass

	def timeStepHour(self) -> float: return float(self.timestepSeconds) / 3600.0
	def timeStepDay(self) -> float: return float(self.timestepSeconds) / 86400.0
	def timeStepYear(self) -> float: return float(self.timestepSeconds) / 31556952.0

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

	def assign_icosphere_vertices_to_plates(self, num_plates: int) -> List[Plate]:
		"""Assigns vertices to plates using a proximity-biased random walk with recycling."""
		plates = [Plate(i) for i in range(num_plates)]

		# Randomly assign plate types (continental/oceanic)
		for plate in plates:
			if random.random() < CONTINENTAL_PLATE_PROB:
				plate.plate_type = 'continental'
			else:
				plate.plate_type = 'oceanic'

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
					dot_prod = np.dot(exp_pos, plate_v_pos)
					distance_metric = 1.0 - dot_prod # Smaller value is closer
					min_distance_sq = min(min_distance_sq, distance_metric)

				# Score inversely proportional to distance (closer is better)
				proximity_scores.append(1.0 / (min_distance_sq + 1e-9))

			# Normalize scores to get probabilities
			total_score = sum(proximity_scores)
			if total_score > 1e-9 and len(expansion_candidates) > 0:
				probabilities = np.array(proximity_scores) / total_score
				# Ensure probabilities sum to 1 (handle potential float errors)
				probabilities /= probabilities.sum()
			else:
				probabilities = np.ones(len(expansion_candidates)) / len(expansion_candidates)

			# Choose the next vertex based on calculated probabilities
			try:
				# Ensure probabilities match the number of candidates
				if len(probabilities) != len(expansion_candidates):
					next_vertex_index = random.choice(expansion_candidates)
				else:
					next_vertex_index = np.random.choice(expansion_candidates, p=probabilities)
			except ValueError as e:
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

					if is_border and (num_neighbors_same_plate < num_neighbors_total / 3):
						border_vertices_this_plate.append(vertex_index)

				vertices_to_reassign.extend(border_vertices_this_plate)

			if not vertices_to_reassign:
				break

			unique_vertices_to_reassign = list(set(vertices_to_reassign)) # Ensure uniqueness
			recycled_total += len(unique_vertices_to_reassign)
			random.shuffle(unique_vertices_to_reassign) # Process in random order

			reassigned_count = 0
			for vertex_index in unique_vertices_to_reassign:
				current_plate_id = self.vertex_to_plate_id.get(vertex_index, -1)
				if current_plate_id == -1: continue # Should already be assigned

				# Find neighboring plates and count neighbors belonging to each
				neighboring_plate_counts = defaultdict(int)
				if vertex_index not in self.adjacency_list: continue # Skip if no neighbors

				vertex_pos = self.vertices[vertex_index].pos
				neighbor_plate_scores = defaultdict(float)
				for neighbor_idx in self.adjacency_list[vertex_index]:
					neighbor_plate_id = self.vertex_to_plate_id.get(neighbor_idx, -1)
					if neighbor_plate_id != -1:
						neighboring_plate_counts[neighbor_plate_id] += 1
						# Add score based on proximity (dot product)
						neighbor_pos = self.vertices[neighbor_idx].pos
						dot_prod = np.dot(vertex_pos, neighbor_pos)
						neighbor_plate_scores[neighbor_plate_id] += (1.0 + dot_prod) # Score higher for closer neighbors

				if not neighboring_plate_counts:
					continue # Vertex has no assigned neighbors

				# Weighted Vote by Proximity and Count
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
							plate_score += (1.0 + dot_prod) # Score higher for closer neighbors

					# Normalize by the number of neighbors from that plate? Or just sum scores? Sum seems ok.
					scores[target_plate_id] = plate_score * neighboring_plate_counts[target_plate_id] # Weight by count too

				best_plate_id = max(scores, key=scores.get)
				# Reassign if the best neighboring plate is different from the current one
				if best_plate_id != -1 and best_plate_id != current_plate_id:
					# Check if the old plate exists and vertex is in it
					if 0 <= current_plate_id < len(plates) and vertex_index in plates[current_plate_id].vertices:
						plates[current_plate_id].vertices.remove(vertex_index)
					# Check if the new plate exists before adding
					if 0 <= best_plate_id < len(plates):
						plates[best_plate_id].add_vertex(vertex_index)
						self.vertex_to_plate_id[vertex_index] = best_plate_id
						reassigned_count += 1
					else:
						# This case indicates an error (best_plate_id invalid)
						# Put the vertex back into its original plate if possible
						if 0 <= current_plate_id < len(plates):
							plates[current_plate_id].add_vertex(vertex_index) # Add back
							self.vertex_to_plate_id[vertex_index] = current_plate_id # Ensure map is correct


			# print(f"  Reassigned {reassigned_count} vertices in pass {recycle_iter + 1}.")


		# Final cleanup and validation
		final_v_count = 0
		assigned_verts_check = set()
		for i, p in enumerate(plates):
			p.vertices = sorted(list(set(p.vertices))) # Ensure unique and sorted
			# Verify mapping consistency
			for v_idx in p.vertices:
				if self.vertex_to_plate_id.get(v_idx) != i:
					# print(f"WARN: Correcting map for vertex {v_idx} (was {self.vertex_to_plate_id.get(v_idx)}, should be {i})")
					self.vertex_to_plate_id[v_idx] = i
				assigned_verts_check.add(v_idx)
			final_v_count += len(p.vertices)

		self.plates = plates
		return plates

	def duplicateLayers(self):
		pass

	def _build_vertex_plate_map(self):
		"""Builds or rebuilds the vertex_to_plate_id dictionary from plate lists."""
		# print(" Building vertex-to-plate map...")
		self.vertex_to_plate_id.clear()
		num_vertices = len(self.vertices)
		if not self.plates:
			# print(" No plates exist, map will be empty.")
			return

		vertices_assigned_count = 0
		overlapping_vertices = 0
		invalid_indices = 0

		for plate_idx, plate in enumerate(self.plates):
			for vertex_idx in plate.vertices:
				# Check index validity first
				if not (0 <= vertex_idx < num_vertices):
					# print(f"Warning: Plate {plate_idx} contains invalid vertex index {vertex_idx}. Skipping.")
					invalid_indices += 1
					continue

				if vertex_idx in self.vertex_to_plate_id:
					# Vertex found in multiple plate lists, indicates an issue upstream
					# print(f"Warning: Vertex {vertex_idx} found in multiple plates ({self.vertex_to_plate_id[vertex_idx]} and {plate_idx}). Overwriting map entry with {plate_idx}.")
					overlapping_vertices += 1
				self.vertex_to_plate_id[vertex_idx] = plate_idx
				vertices_assigned_count += 1 # Count assignments, not unique vertices yet

		map_size = len(self.vertex_to_plate_id)
		# print(f" Vertex-to-plate map built. Size: {map_size}. Overlaps found: {overlapping_vertices}. Invalid indices skipped: {invalid_indices}.")

		# Final consistency check
		if map_size != num_vertices:
			print(f"Error building map: Final map size ({map_size}) != number of vertices ({num_vertices}). {num_vertices - map_size} vertices are unmapped.")
		elif overlapping_vertices > 0:
			print(f"Warning: {overlapping_vertices} vertices were listed in multiple plates during map build.")

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
					if edge not in self.boundary_edges:
						self.boundary_edges.add(edge)
					self.boundary_vertices.add(v2_idx)

			if is_boundary_vertex:
				self.boundary_vertices.add(v1_idx)

		print(f"Boundaries identified. Boundary Vertices: {len(self.boundary_vertices)}, Boundary Edges: {len(self.boundary_edges)}")

	def assign_random_angular_velocities(self):
		if not self.plates:
			print("Warning: No plates exist to assign velocities to.")
			return

		for plate in self.plates:
			phi = np.random.uniform(0, 2 * np.pi)
			costheta = np.random.uniform(-1, 1)
			theta = np.arccos(costheta)
			x = np.sin(theta) * np.cos(phi)
			y = np.sin(theta) * np.sin(phi)
			z = np.cos(theta)
			direction = np.array([x, y, z])

			# Random magnitude up to max_angular_velocity
			magnitude = np.random.uniform(0, MAX_ANGULAR_VELOCITY_RAD_PER_YR) / self.radius

			plate.angular_velocity = direction * magnitude
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
				continue
			p_mid_unit = p_mid / p_mid_norm # Position vector on the unit sphere
			p_mid_world = p_mid_unit * self.radius # Position vector in world units (meters)

			vel1 = np.cross(omega1, p_mid_world)
			vel2 = np.cross(omega2, p_mid_world)
			v_rel = vel2 - vel1


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
			if total_mag < 1e-9:
				boundary_type = "passive"
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

	def assign_elevations_from_boundaries(self):
		"""Assigns vertex elevations based on nearby boundary types and magnitudes, then smooths."""
		base_convergent: float = ELEVATION_MOUNTAIN_BASE 
		base_divergent: float = ELEVATION_TRENCH_BASE 
		base_transform: float = 200.0 # Slight ridge/fracture zone
		rate_scaling_factor: float = 5.0e7 # Scale velocity (m/yr) to elevation impact
		diffusion_passes: int = ELEVATION_DIFFUSION_PASSES
		diffusion_factor: float = ELEVATION_DIFFUSION_FACTOR


		num_vertices = len(self.vertices)
		# Initialize elevations to 0 (or a base sea level)
		base_elevation = 0.0
		self.elevations = {i: base_elevation for i in range(num_vertices)}

		print(" Assigning base elevations based on plate type...")
		for v_idx in range(num_vertices):
			plate_id = self.vertex_to_plate_id.get(v_idx)
			if plate_id is not None:
				plate_type = self.plates[plate_id].plate_type
				if plate_type == 'continental':
					self.elevations[v_idx] += CONTINENTAL_BASE_ELEVATION# * (self.radius / 6371000)
				elif plate_type == 'oceanic':
					self.elevations[v_idx] += OCEANIC_BASE_ELEVATION# * (self.radius / 6371000)
		elevation_updates = defaultdict(lambda: {'sum_influence': 0.0, 'count': 0})

		m_per_sec_to_m_per_yr = 1.0 / (1.0 / (365.25 * 24 * 3600))

		for edge, props in self.boundary_properties.items():
			v1_idx, v2_idx = edge
			boundary_type = props.get("type", "undefined")
			conv_rate_myr = props.get("convergence_rate_mps", 0.0)
			trans_rate_myr = abs(props.get("transform_rate_mps", 0.0))

			elevation_change = 0.0
			if boundary_type == "convergent":
				# Negative conv_rate_myr means convergence
				magnitude = abs(conv_rate_myr) # Use absolute rate
				elevation_change = base_convergent * (1 + magnitude * rate_scaling_factor / base_convergent)# * (self.radius / 6371000)
			elif boundary_type == "divergent":
				# Positive conv_rate_myr means divergence
				magnitude = conv_rate_myr
				elevation_change = base_divergent * (1 + magnitude * rate_scaling_factor / base_divergent)# * (self.radius / 6371000) # Trench gets deeper
			elif boundary_type == "transform":
				magnitude = trans_rate_myr
				elevation_change = base_transform * (1 + magnitude * rate_scaling_factor / base_transform)# * (self.radius / 6371000) # Minor effect, scales with slip rate


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
				self.elevations[v_idx] += avg_influence # Add to existing base elevation
			# else: vertex was in boundary_vertices but no edges contributed? Should not happen.

		if diffusion_passes > 0 and diffusion_factor > 0:
			print(f" Performing {diffusion_passes} elevation diffusion passes (Factor: {diffusion_factor})...")
			# Use a temporary dictionary for updates to avoid overwriting during pass
			current_elevations = self.elevations.copy()
			next_elevations = self.elevations.copy()

			for i_pass in range(diffusion_passes):

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
							dist = np.linalg.norm(self.vertices[v_idx].pos - self.vertices[n_idx].pos) * self.radius
							weight = 1.0 / dist
							sum_neighbor_elev += current_elevations[n_idx] * weight
							num_valid_neighbors += 1
						# else: print(f"Warning: Neighbor {n_idx} not in current_elevations during diffusion.")


					if num_valid_neighbors > 0:
						avg_neighbor_elev = sum_neighbor_elev / num_valid_neighbors
						current_elev = current_elevations.get(v_idx, base_elevation) # Get current elevation

						# Weighted average: (1-f)*current + f*average_neighbor
						smoothed_elev = (1.0 - diffusion_factor) * current_elev + diffusion_factor * avg_neighbor_elev
						next_elevations[v_idx] = smoothed_elev
					else:
						# No valid neighbors (shouldn't happen if not isolated), keep current
						next_elevations[v_idx] = current_elevations.get(v_idx, base_elevation)

				# Update current_elevations for the next pass (use copy)
				current_elevations = next_elevations.copy()
				# Optional: Print min/max elevation after each pass
				# if i_pass % 5 == 0 and current_elevations:
				#    print(f"    Pass {i_pass+1} Elev Range: {min(current_elevations.values()):.0f} to {max(current_elevations.values()):.0f}")


			# Assign the final smoothed elevations back to the world state
			self.elevations = current_elevations
			print(" Elevation diffusion complete.")
		else:
			print(" Skipping elevation diffusion.")

		final_min = min(self.elevations.values())
		final_max = max(self.elevations.values())
		NORM_ELEVATION.vmin = -15000 # Fixed range
		NORM_ELEVATION.vmax = 15000 # Fixed range
		print(f"Final Elevation Range: {final_min:.0f}m to {final_max:.0f}m")


class icosphere(worldState):
	def __init__(self, radius, subdivions):
		super().__init__(radius, subdivions)
	
	def shapeBase(self):
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
			# Assumes face.vertices is a tuple of 3 indices (v0, v1, v2)
			v0, v1, v2 = face.vertices
			# Add edges (v0,v1), (v1,v2), (v2,v0)
			if v1 not in self.adjacency_list[v0]: self.adjacency_list[v0].append(v1)
			if v0 not in self.adjacency_list[v1]: self.adjacency_list[v1].append(v0)

			if v2 not in self.adjacency_list[v1]: self.adjacency_list[v1].append(v2)
			if v1 not in self.adjacency_list[v2]: self.adjacency_list[v2].append(v1)

			if v0 not in self.adjacency_list[v2]: self.adjacency_list[v2].append(v0)
			if v2 not in self.adjacency_list[v0]: self.adjacency_list[v0].append(v2)

	def subdivide(self):
		"""Subdivides faces of the mesh"""
		if self.details <= 0: return

		for _ in range(self.details):
			new_faces_next_level = []
			self._subdivision_cache = {} # Clear cache for each level

			current_faces = self.faces[:] # Copy current faces
			if not current_faces: break # Stop if no faces exist

			for face in current_faces:
				# Assumes face is always a triangle Face(v0, v1, v2)
				v0_idx, v1_idx, v2_idx = face.vertices

				# Get or create midpoints for the three edges
				try:
					m01_idx = self._get_or_create_midpoint(v0_idx, v1_idx)
					m12_idx = self._get_or_create_midpoint(v1_idx, v2_idx)
					m20_idx = self._get_or_create_midpoint(v2_idx, v0_idx)
				except IndexError as e:
					print(f"  Error getting vertices for midpoint creation in face {face.vertices}: {e}. Skipping subdivision for this face.")
					new_faces_next_level.append(face) # Keep original face if error
					continue
				except Exception as e:
					print(f"  Unexpected error during midpoint creation for face {face.vertices}: {e}. Skipping subdivision.")
					new_faces_next_level.append(face)
					continue

				# Create the four new triangular faces
				new_faces_next_level.append(Face(v0_idx, m01_idx, m20_idx))
				new_faces_next_level.append(Face(v1_idx, m12_idx, m01_idx))
				new_faces_next_level.append(Face(v2_idx, m20_idx, m12_idx))
				new_faces_next_level.append(Face(m01_idx, m12_idx, m20_idx)) # Center face

			self.faces = new_faces_next_level # Update faces list for the next level

		# print(f"Subdivision complete. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")
