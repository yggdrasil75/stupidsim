from collections import defaultdict
import random
from typing import Dict, List, Set, Tuple

import numpy as np
from globals import CONTINENTAL_BASE_ELEVATION, CONTINENTAL_PLATE_PROB, ELEVATION_DIFFUSION_FACTOR, ELEVATION_DIFFUSION_PASSES, ELEVATION_MOUNTAIN_BASE, ELEVATION_TRENCH_BASE, MAX_ANGULAR_VELOCITY_RAD_PER_YR, NORM_ELEVATION, OCEANIC_BASE_ELEVATION, PHI, VMAX, VMIN
from shape import Face, Vertex


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
		self.details: int = subdivisions
		self.elevations: dict[int, float] = {}
		self.plates: List[Plate] = []
		self.adjacency_list: defaultdict[int, List[int]] = defaultdict(list)
		self.vertex_to_plate_id: Dict[int, int] = {}
		self.boundary_vertices: Set[int] = set()
		self.boundary_edges: Set[Tuple[int, int]] = set()
		self.boundary_properties: Dict[Tuple[int, int], Dict] = {}
		self.total_water_volume_surface: float = 0.0 # Total surface water volume in Petaliters (PL)
		self.total_water_volume_ground: float = 0.0  # Total groundwater volume in Petaliters (PL)
		self.total_water_volume_ice: float = 0.0     # Total ice water volume in Petaliters (PL)
		self.total_water_volume_atmosphere: float = 0.0 # Total atmospheric water in some unit

		# Water dictionaries to store water per vertex ID
		self.surface_water: Dict[int, float] = {}
		self.groundwater: Dict[int, float] = {}
		self.ice_water: Dict[int, float] = {}
		self.atmospheric_water: Dict[int, float] = {}

		# Cache for subdivision: key=sorted tuple(v_idx1, v_idx2), value=midpoint_idx
		self._subdivision_cache: Dict[Tuple[int, int], int] = {}
		self._vertex_pos_cache: Dict[Tuple[float, ...], int] = {}

	def shapeBase(self):
		#implement in subclass
		raise NotImplementedError("shapeBase must be implemented in a subclass")

	def timeStepHour(self) -> float: return float(self.timestepSeconds) / 3600.0
	def timeStepDay(self) -> float: return float(self.timestepSeconds) / 86400.0
	def timeStepYear(self) -> float: return float(self.timestepSeconds) / 31556952.0

	# This function is for the initial sphere generation, relies on normalized positions
	def _add_vertex(self, vertex: Vertex) -> int:
		"""Adds a vertex if unique normalized position, returns index. Assumes vertex is on unit sphere."""
		norm_vertex = vertex # Assume vertex is already normalized or should be treated as such for caching
		key = tuple(np.round(norm_vertex.pos, 8)) # Use rounded normalized position as key
		if key in self._vertex_pos_cache:
			return self._vertex_pos_cache[key]
		else:
			idx = len(self.vertices)
			self.vertices.append(norm_vertex) # Add the (normalized) vertex
			self._vertex_pos_cache[key] = idx
			return idx

	# This function is used during subdivision, creates midpoints
	def _get_or_create_midpoint(self, v1_idx: int, v2_idx: int) -> int:
		key = tuple(sorted((v1_idx, v2_idx)))
		if key in self._subdivision_cache:
			return self._subdivision_cache[key]
		v1 = self.vertices[v1_idx]
		v2 = self.vertices[v2_idx]
		mid_pos = (v1.pos + v2.pos) / 2.0
		midpoint_vertex = Vertex(mid_pos[0], mid_pos[1], mid_pos[2])
		midpoint_vertex.normalize()
		mid_idx = self._add_vertex(midpoint_vertex)
		self._subdivision_cache[key] = mid_idx
		return mid_idx

	def duplicateLayers(self):
		"""Creates concentric layers of vertices based on the initial subdivided sphere."""
		if self.details <= 0:
			print("No subdivisions used (details=0), cannot create layers.")
			return

		print("Starting layer duplication...")
		# Keep copies of original data linked to ORIGINAL indices BEFORE duplication
		# These represent the state of the surface layer (layer 0)
		original_vertices = self.vertices[:] # Copy vertex objects
		original_faces = self.faces[:]       # Copy face objects
		original_vertex_to_plate_id = self.vertex_to_plate_id.copy()
		original_elevations = self.elevations.copy()
		original_plates_info = [(p.plate_id, p.angular_velocity.copy(), p.plate_type) for p in self.plates] # Store essential info

		num_original_vertices = len(original_vertices)
		if num_original_vertices == 0:
			print("Error: No vertices exist to duplicate.")
			return

		# Clear instance variables that will be completely rebuilt
		self.vertices = []
		self.faces = [] # Faces will ONLY be rebuilt for layer 0
		self.adjacency_list = defaultdict(list) # Will be rebuilt
		self.vertex_to_plate_id = {} # Will be rebuilt ONLY for layer 0
		self.elevations = {}         # Will be rebuilt ONLY for layer 0
		self.plates = []             # Will be rebuilt with ONLY layer 0 vertices
		# self._vertex_pos_cache.clear() # Clear this cache as it's for normalized unit sphere vertices
		# We don't need a position cache for layers as each vertex is unique by definition (orig_idx, layer_id)

		num_layers = self.details * 2 + 1 # e.g., details=1 -> layers -1, 0, 1 (3 layers)
		layers_vertices = [[] for _ in range(num_layers)] # Stores NEW vertex indices per layer
		# Relative shift factor: Adjust as needed. 0.05 means layers are 5% of radius apart.
		layer_shift_factor = 0.01 # Smaller shift to avoid large gaps initially

		# Map original vertex index to the NEW index of that vertex in Layer 0
		new_vertex_index_map_layer0 = {}

		print(f" Creating {num_layers} layers for {num_original_vertices} original vertices...")

		# --- Create All Vertices for All Layers ---
		current_new_vertex_index = 0
		for layer_idx in range(num_layers):
			# Layer IDs go from -details to +details, with 0 being the surface
			layer_id = layer_idx - self.details
			# The scaling factor for radius for this layer
			radius_multiplier = 1.0 + layer_id * layer_shift_factor
			# print(f"  Processing Layer {layer_id} (Index {layer_idx}), Radius Multiplier: {radius_multiplier:.4f}")

			layer_new_indices = [] # Store new indices for this specific layer
			for original_v_idx, original_vertex in enumerate(original_vertices):
				# Original position should already be normalized unit vector from icosphere generation
				original_norm_pos = original_vertex.pos # Assuming it's already normalized

				# Calculate the new position for this layer - DO NOT NORMALIZE
				new_pos = original_norm_pos * radius_multiplier

				# Create the new vertex object WITH the layer position
				new_vertex = Vertex(new_pos[0], new_pos[1], new_pos[2])

				# Assign attributes BEFORE adding to list
				new_vertex.layer_id = layer_id
				new_vertex.original_vertex_index = original_v_idx

				# Add the new vertex to the main list
				# The index is simply the current length before appending
				new_vertex_index = current_new_vertex_index
				self.vertices.append(new_vertex)
				layer_new_indices.append(new_vertex_index)

				# If this is the surface layer (layer 0), map original index to its new index
				if layer_id == 0:
					new_vertex_index_map_layer0[original_v_idx] = new_vertex_index

				current_new_vertex_index += 1

			# Store the list of new indices created for this layer
			layers_vertices[layer_idx] = layer_new_indices

		print(f" Created {len(self.vertices)} total vertices across {num_layers} layers.")
		if len(self.vertices) != num_original_vertices * num_layers:
			print(f" Warning: Vertex count mismatch! Expected {num_original_vertices * num_layers}, got {len(self.vertices)}")


		# --- Rebuild Data Structures focusing on Layer 0 (Surface) ---
		print(" Rebuilding surface layer (Layer 0) specific data...")

		# 1. Rebuild Faces for Layer 0 ONLY
		# Use the original faces but substitute original vertex indices with their new Layer 0 indices
		new_faces_layer0 = []
		if not original_faces:
			print(" Warning: No original faces found to rebuild for layer 0.")
		else:
			for face in original_faces:
				try:
					# Map old vertex indices in the face to new indices from layer 0 map
					new_face_vertices = tuple(new_vertex_index_map_layer0[v_idx] for v_idx in face.vertices)
					new_faces_layer0.append(Face(*new_face_vertices))
				except KeyError as e:
					print(f" Error: Original vertex index {e} not found in new_vertex_index_map_layer0 while rebuilding faces. Original face: {face.vertices}. Skipping face.")
					continue
		self.faces = new_faces_layer0 # Assign ONLY layer 0 faces
		print(f" Rebuilt {len(self.faces)} faces for layer 0.")


		# 2. Rebuild vertex_to_plate_id map using NEW indices for Layer 0 vertices ONLY
		self.vertex_to_plate_id = {}
		for original_v_idx, plate_id in original_vertex_to_plate_id.items():
			if original_v_idx in new_vertex_index_map_layer0:
				new_v_idx_layer0 = new_vertex_index_map_layer0[original_v_idx]
				self.vertex_to_plate_id[new_v_idx_layer0] = plate_id
			else:
				# This should not happen if the mapping was built correctly for all original vertices
				print(f" Warning: Original vertex index {original_v_idx} not found in new_vertex_index_map_layer0 while rebuilding plate map.")
		print(f" Rebuilt vertex_to_plate_id map for layer 0 (Size: {len(self.vertex_to_plate_id)}). Should match original number of vertices.")


		# 3. Rebuild Plates list (assigning NEW Layer 0 vertex indices ONLY to plate.vertices)
		self.plates = [] # Start fresh
		# Create new Plate objects using the stored info
		new_plates_dict = {pid: Plate(pid) for pid, _, _ in original_plates_info}
		for pid, ang_vel, ptype in original_plates_info:
			new_plates_dict[pid].angular_velocity = ang_vel
			new_plates_dict[pid].plate_type = ptype

		# Assign the correct NEW Layer 0 vertices to each new Plate object
		for new_v_idx_layer0, plate_id in self.vertex_to_plate_id.items():
			if plate_id in new_plates_dict:
				new_plates_dict[plate_id].add_vertex(new_v_idx_layer0)
			else:
				print(f" Warning: Plate ID {plate_id} referenced by vertex {new_v_idx_layer0} not found in original plate info.")

		# Sort vertices within plates and add to the final list
		self.plates = sorted(new_plates_dict.values(), key=lambda p: p.plate_id)
		for plate in self.plates:
			plate.vertices.sort() # Keep vertices sorted

		print(f" Rebuilt {len(self.plates)} plates with new vertex indices for layer 0.")
		# Verify total vertices in plates match layer 0 vertices
		plate_vertex_count = sum(len(p.vertices) for p in self.plates)
		if plate_vertex_count != len(new_vertex_index_map_layer0):
			print(f" Warning: Mismatch between vertices in plates ({plate_vertex_count}) and layer 0 vertices ({len(new_vertex_index_map_layer0)})")


		# 4. Rebuild elevations map using NEW indices for Layer 0 vertices ONLY
		self.elevations = {}
		for original_v_idx, elevation in original_elevations.items():
			if original_v_idx in new_vertex_index_map_layer0:
				new_v_idx_layer0 = new_vertex_index_map_layer0[original_v_idx]
				self.elevations[new_v_idx_layer0] = elevation
			else:
				print(f" Warning: Original vertex index {original_v_idx} not found in new_vertex_index_map_layer0 while rebuilding elevations.")
		print(f" Rebuilt elevations map for layer 0 (Size: {len(self.elevations)}).")


		# --- Build Adjacency List ---
		print(" Building adjacency list...")
		# 1. Horizontal connections on Layer 0 (using the rebuilt self.faces)
		self._build_adjacency_list() # This now correctly uses only Layer 0 faces
		print(f" Built horizontal adjacency for layer 0 (Entries: {len(self.adjacency_list)}).")

		# 2. Vertical connections between layers
		print(" Connecting layers vertically...")
		for layer_idx in range(num_layers):
			# Connect layer_idx to layer_idx + 1
			if layer_idx < num_layers - 1:
				current_layer_new_indices = layers_vertices[layer_idx]
				upper_layer_new_indices = layers_vertices[layer_idx + 1]

				# The vertices should correspond one-to-one based on the creation order
				if len(current_layer_new_indices) != len(upper_layer_new_indices):
					print(f" FATAL Error: Layer size mismatch between layer index {layer_idx} ({len(current_layer_new_indices)}) and {layer_idx+1} ({len(upper_layer_new_indices)}). Cannot connect vertically.")
					# Depending on severity, you might want to raise an exception or return
					return # Stop further processing

				for i in range(len(current_layer_new_indices)):
					v_lower_idx = current_layer_new_indices[i]
					v_upper_idx = upper_layer_new_indices[i]

					# Add bidirectional connections to the adjacency list
					if v_upper_idx not in self.adjacency_list[v_lower_idx]:
						self.adjacency_list[v_lower_idx].append(v_upper_idx)
					if v_lower_idx not in self.adjacency_list[v_upper_idx]:
						self.adjacency_list[v_upper_idx].append(v_lower_idx)

		print(f" Added vertical connections between layers.")
		print(f" Layer duplication complete. Total vertices: {len(self.vertices)}, Faces (layer 0): {len(self.faces)}, Layers: {num_layers}")

		# --- Final Sanity Checks ---
		num_layer0_vertices_check = len(layers_vertices[self.details])
		if len(self.elevations) != num_layer0_vertices_check:
			print(f" Post-Check Warning: Elevation map size ({len(self.elevations)}) != Expected Layer 0 vertices ({num_layer0_vertices_check})")
		if len(self.vertex_to_plate_id) != num_layer0_vertices_check:
			print(f" Post-Check Warning: Vertex-to-plate map size ({len(self.vertex_to_plate_id)}) != Expected Layer 0 vertices ({num_layer0_vertices_check})")
		if len(self.faces) != len(original_faces):
			print(f" Post-Check Warning: Number of faces ({len(self.faces)}) != Number of original faces ({len(original_faces)})")


	def _build_vertex_plate_map(self):
		"""(Re)Builds the vertex_to_plate_id dictionary from self.plates lists."""
		self.vertex_to_plate_id.clear()
		num_vertices = len(self.vertices)
		if not self.plates: return

		overlapping_vertices = 0
		invalid_indices = 0
		mapped_count = 0

		for plate_idx, plate in enumerate(self.plates):
			for vertex_idx in plate.vertices:
				if not (0 <= vertex_idx < num_vertices):
					invalid_indices += 1
					continue
				if vertex_idx in self.vertex_to_plate_id:
					overlapping_vertices += 1
					# Optional: Resolve overlap, e.g., keep first encountered plate ID
					# print(f"Warning: Vertex {vertex_idx} in multiple plates ({self.vertex_to_plate_id[vertex_idx]} and {plate_idx}). Keeping {self.vertex_to_plate_id[vertex_idx]}.")
				else:
					self.vertex_to_plate_id[vertex_idx] = plate_idx
					mapped_count +=1

		map_size = len(self.vertex_to_plate_id)


	def _identify_boundaries(self):
		"""Identifies vertices and edges lying on plate boundaries."""
		print("Identifying plate boundaries...")
		self.boundary_vertices.clear()
		self.boundary_edges.clear()
		if not self.vertex_to_plate_id:
			print("Error: Cannot identify boundaries without vertex-to-plate map.")
			return
		if not self.adjacency_list:
			print("Error: Cannot identify boundaries without adjacency list.")
			return

		# Iterate through ONLY the vertices known to be on plates (likely Layer 0)
		relevant_vertices = list(self.vertex_to_plate_id.keys())
		print(f" Checking {len(relevant_vertices)} vertices with plate assignments for boundaries...")

		for v1_idx in relevant_vertices:
			plate1_id = self.vertex_to_plate_id[v1_idx] # Known to exist

			if v1_idx not in self.adjacency_list: continue

			is_boundary_vertex = False
			# Check neighbors for different plate IDs
			for v2_idx in self.adjacency_list[v1_idx]:
				# IMPORTANT: Only consider neighbors that are ALSO on a plate (i.e., likely Layer 0)
				plate2_id = self.vertex_to_plate_id.get(v2_idx, -1)

				if plate2_id != -1 and plate1_id != plate2_id:
					# This is a boundary edge between two mapped vertices
					is_boundary_vertex = True
					edge = tuple(sorted((v1_idx, v2_idx)))
					self.boundary_edges.add(edge)
					# The neighbor v2_idx is also a boundary vertex
					self.boundary_vertices.add(v2_idx)

			if is_boundary_vertex:
				# v1_idx is also a boundary vertex
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

			magnitude = np.random.uniform(0, MAX_ANGULAR_VELOCITY_RAD_PER_YR)

			plate.angular_velocity = direction * magnitude
		print("Angular velocities assigned.")

	def calculate_boundary_motions(self, classification_threshold: float = 0.7) -> bool:
		self.boundary_properties.clear() # Clear previous properties

		m_per_year_to_m_per_sec = 1.0 / (365.25 * 24 * 3600)

		successful_calculations = 0
		skipped_edges = 0

		for edge in self.boundary_edges:
			v1_idx, v2_idx = edge

			plate1_id = self.vertex_to_plate_id.get(v1_idx, -1)
			plate2_id = self.vertex_to_plate_id.get(v2_idx, -1)

			# Ensure both vertices belong to different, valid plates
			if not (0 <= plate1_id < len(self.plates) and 0 <= plate2_id < len(self.plates) and plate1_id != plate2_id):
				# print(f" Skipping edge {edge}: Invalid plate assignments ({plate1_id}, {plate2_id}).")
				skipped_edges += 1
				continue

			try:
				v1 = self.vertices[v1_idx]
				v2 = self.vertices[v2_idx]
				plate1 = self.plates[plate1_id]
				plate2 = self.plates[plate2_id]
			except IndexError:
				print(f" Skipping edge {edge}: Vertex index out of bounds.")
				skipped_edges += 1
				continue

			# Angular velocities (radians per year)
			omega1 = plate1.angular_velocity
			omega2 = plate2.angular_velocity

			p_mid = (v1.pos + v2.pos) / 2.0
			p_mid_norm = np.linalg.norm(p_mid)
			if p_mid_norm < 1e-9:
				continue
			p_mid /= p_mid_norm # Normalized direction
			p_mid_world = p_mid * self.radius # Position vector in meters

			# Linear velocities v = omega x r (result in meters per year)
			vel1_myr = np.cross(omega1, p_mid_world)
			vel2_myr = np.cross(omega2, p_mid_world)
			v_rel_myr = vel2_myr - vel1_myr # Relative velocity in meters per year

			# --- Classify Boundary ---
			# Local coordinate system at p_mid_world:
			# Radial: p_mid_unit
			# Tangent 1 (along edge approx): Vector from v1 to v2, projected onto tangent plane
			edge_vec_unit = v2.pos - v1.pos # Vector along edge on unit sphere
			if np.linalg.norm(edge_vec_unit) < 1e-9: continue
			# Project edge_vec onto tangent plane at p_mid_unit
			tangent_edge_vec = edge_vec_unit - np.dot(edge_vec_unit, p_mid) * p_mid
			norm_tangent_edge = np.linalg.norm(tangent_edge_vec)
			if norm_tangent_edge < 1e-9: continue
			tangent_edge_unit = tangent_edge_vec / norm_tangent_edge # Unit vector along boundary tangent

			# Tangent 2 (normal to boundary): Cross radial with tangent_edge_unit
			boundary_normal_local = np.cross(p_mid, tangent_edge_unit) # Unit vector normal to boundary

			# Project relative velocity onto the tangent plane (remove radial component)
			v_rel_tangent_myr = v_rel_myr - np.dot(v_rel_myr, p_mid) * p_mid

			# Decompose tangent velocity into components
			# Convergence component: Projection onto boundary_normal_local
			# Negative value means convergence (moving opposite to the normal)
			# Positive value means divergence (moving along the normal)
			convergence_component_scalar_myr = np.dot(v_rel_tangent_myr, boundary_normal_local)

			# Transform component: Projection onto tangent_edge_unit
			transform_component_scalar_myr = np.dot(v_rel_tangent_myr, tangent_edge_unit)

			# Magnitudes (m/yr)
			convergence_mag_myr = abs(convergence_component_scalar_myr)
			transform_mag_myr = abs(transform_component_scalar_myr)
			total_mag_myr = np.linalg.norm(v_rel_tangent_myr)

			# Classify based on magnitudes and sign of convergence
			boundary_type = "undefined"
			# Use a small threshold for 'passive' based on meters per year (e.g., < 1 mm/yr)
			passive_threshold_myr = 0.001
			if total_mag_myr < passive_threshold_myr:
				boundary_type = "passive"
			else:
				# Ratios for classification
				conv_ratio = convergence_mag_myr / total_mag_myr
				trans_ratio = transform_mag_myr / total_mag_myr

				# Check dominant component against threshold
				if conv_ratio >= classification_threshold:
					boundary_type = "convergent" if convergence_component_scalar_myr < 0 else "divergent"
				elif trans_ratio >= classification_threshold:
					boundary_type = "transform"
				else:
					# Oblique motion - classify by the more dominant component
					if convergence_mag_myr > transform_mag_myr:
						boundary_type = "convergent" if convergence_component_scalar_myr < 0 else "divergent"
					else:
						boundary_type = "transform"

			# Store properties in m/s
			self.boundary_properties[edge] = {
				"type": boundary_type,
				"relative_velocity_mps": v_rel_myr * m_per_year_to_m_per_sec,
				"convergence_rate_mps": convergence_component_scalar_myr * m_per_year_to_m_per_sec, # Negative=Convergent
				"transform_rate_mps": transform_component_scalar_myr * m_per_year_to_m_per_sec, # Signed slip rate
				"total_rate_mps": total_mag_myr * m_per_year_to_m_per_sec # Magnitude of slip
			}
			successful_calculations += 1

		print(f"Boundary motion calculation complete. Processed {successful_calculations}/{len(self.boundary_edges)} edges. Skipped {skipped_edges}.")
		if successful_calculations == 0 and len(self.boundary_edges) > 0:
			print(" Warning: No boundary motions could be calculated.")
			return False
		return True

	def assign_elevations_from_boundaries(self):
		"""Assigns vertex elevations (Layer 0) based on plate type and boundary influence, then smooths."""
		print("Assigning elevations (Layer 0)...")

		# Ensure elevations dict exists, default to 0 or base value
		# Operate ONLY on vertices that have plate assignments (Layer 0)
		layer0_vertices = list(self.vertex_to_plate_id.keys())
		if not layer0_vertices:
			print("Warning: No Layer 0 vertices found with plate assignments. Cannot assign elevations.")
			return

		# Initialize elevations ONLY for Layer 0 vertices based on plate type
		self.elevations = {} # Clear previous elevations
		print(" Assigning base elevations based on plate type...")
		for v_idx in layer0_vertices:
			plate_id = self.vertex_to_plate_id[v_idx]
			try:
				plate_type = self.plates[plate_id].plate_type
				if plate_type == 'continental':
					self.elevations[v_idx] = CONTINENTAL_BASE_ELEVATION
				elif plate_type == 'oceanic':
					self.elevations[v_idx] = OCEANIC_BASE_ELEVATION
				else: # Default case if plate type is somehow invalid
					self.elevations[v_idx] = OCEANIC_BASE_ELEVATION
			except IndexError:
				print(f"Warning: Plate ID {plate_id} invalid for vertex {v_idx}. Assigning default elevation.")
				self.elevations[v_idx] = OCEANIC_BASE_ELEVATION # Default fallback

		# --- Apply boundary influence ---
		print(" Applying boundary influences...")
		# Use m/s rates from boundary_properties
		# Convert bases to be comparable (or adjust scaling factor)
		base_convergent: float = ELEVATION_MOUNTAIN_BASE # meters (positive)
		base_divergent: float = ELEVATION_TRENCH_BASE # meters (negative)
		base_transform: float = 200.0 # Small positive bump for transforms

		# Scaling factor to convert velocity (m/s) to elevation impact (m)
		# Example: 1 cm/yr = 1e-2 m / (3.15e7 s) ~= 3e-10 m/s
		# If we want 1 cm/yr convergence (~3e-10 m/s) to add, say, 1000m elevation:
		# Scaling factor = 1000m / 3e-10 m/s ~= 3e12 s
		# Let's try a large scaling factor and see
		rate_scaling_factor: float = 3.0e12 # Units: seconds (so rate_mps * factor -> meters)

		elevation_updates = defaultdict(lambda: {'sum_influence': 0.0, 'count': 0})

		if not self.boundary_properties:
			print(" No boundary properties calculated, skipping boundary influence.")
		else:
			for edge, props in self.boundary_properties.items():
				v1_idx, v2_idx = edge
				boundary_type = props.get("type", "undefined")
				# Use absolute convergence rate for magnitude, sign determines type
				conv_rate_mps = props.get("convergence_rate_mps", 0.0) # Negative = convergence
				trans_rate_mps = props.get("transform_rate_mps", 0.0) # Signed rate

				influence = 0.0
				if boundary_type == "convergent": # conv_rate_mps is negative
					magnitude_mps = abs(conv_rate_mps)
					influence = base_convergent * (magnitude_mps * rate_scaling_factor / abs(base_convergent)) # Scale uplift by rate
				elif boundary_type == "divergent": # conv_rate_mps is positive
					magnitude_mps = conv_rate_mps
					# Make trench deeper based on rate (less negative)
					# Start from base_divergent (e.g., -10000) and add positive influence scaled by rate
					# This addition will make the number less negative (shallower trench for faster spreading?)
					# Or should faster spreading make a deeper rift? Let's assume deeper.
					# Influence = base_divergent * (magnitude_mps * rate_scaling_factor / abs(base_divergent))
					# Example: base=-10k. rate=1cm/yr -> mag=3e-10. factor=3e12. infl = -10k * (3e-10 * 3e12 / 10k) = -10k * (0.9 / 10k) = -0.9 ??? Needs rethink.
					# Let's make influence proportional to rate, added to base.
					# A positive rate means divergence. We want more negative elevation.
					influence = base_divergent * (magnitude_mps * rate_scaling_factor / 1.0 ) # Scale downward pull by rate? Needs tuning. Let's try simpler: make proportional.
					influence = -abs(base_divergent) * (magnitude_mps * rate_scaling_factor) # More negative for faster divergence

				elif boundary_type == "transform":
					magnitude_mps = abs(trans_rate_mps)
					influence = base_transform * (magnitude_mps * rate_scaling_factor / 1.0) # Small bump scaled by slip rate

				# Apply influence to both vertices of the edge
				# Ensure these vertices are actually in Layer 0 and have elevations initialized
				if boundary_type != "passive" and boundary_type != "undefined":
					if v1_idx in self.elevations:
						elevation_updates[v1_idx]['sum_influence'] += influence
						elevation_updates[v1_idx]['count'] += 1
					if v2_idx in self.elevations:
						elevation_updates[v2_idx]['sum_influence'] += influence
						elevation_updates[v2_idx]['count'] += 1

			# Add the averaged influence to the base elevations
			for v_idx, data in elevation_updates.items():
				if data['count'] > 0 and v_idx in self.elevations:
					avg_influence = data['sum_influence'] / data['count']
					self.elevations[v_idx] += avg_influence


		# --- Diffusion/Smoothing ---
		diffusion_passes: int = ELEVATION_DIFFUSION_PASSES
		diffusion_factor: float = ELEVATION_DIFFUSION_FACTOR

		if diffusion_passes > 0 and diffusion_factor > 0 and len(self.elevations) > 1:
			print(f" Performing {diffusion_passes} elevation diffusion passes (Factor: {diffusion_factor}) on {len(self.elevations)} vertices...")
			# Operate only on the vertices that have elevations (Layer 0)
			current_elevations = self.elevations # Work directly on the dict
			vertices_to_smooth = list(current_elevations.keys())

			for i_pass in range(diffusion_passes):
				next_elevations = current_elevations.copy() # Create copy for next state

				for v_idx in vertices_to_smooth:
					if v_idx not in self.adjacency_list or not self.adjacency_list[v_idx]:
						continue # Isolated vertex

					neighbor_indices = self.adjacency_list[v_idx]
					sum_neighbor_elev = 0.0
					sum_weights = 0.0
					num_valid_neighbors = 0

					# Only consider neighbors that are ALSO in Layer 0 (have elevations)
					for n_idx in neighbor_indices:
						if n_idx in current_elevations:
							neighbor_elev = current_elevations[n_idx]
							# Simple average (weight=1)
							weight = 1.0
							# # Optional: Distance weighting (more complex)
							# dist_sq = np.sum((self.vertices[v_idx].pos - self.vertices[n_idx].pos)**2)
							# weight = 1.0 / (dist_sq + 1e-6) # Inverse distance weight

							sum_neighbor_elev += neighbor_elev * weight
							sum_weights += weight
							num_valid_neighbors += 1

					if num_valid_neighbors > 0 and sum_weights > 1e-9:
						avg_neighbor_elev = sum_neighbor_elev / sum_weights
						current_elev = current_elevations[v_idx]

						# Weighted average: (1-f)*current + f*average_neighbor
						smoothed_elev = (1.0 - diffusion_factor) * current_elev + diffusion_factor * avg_neighbor_elev
						next_elevations[v_idx] = smoothed_elev
					# else: keep original elevation next_elevations[v_idx] = current_elevations[v_idx]

				# Update current_elevations for the next pass
				current_elevations = next_elevations # Point to the results of this pass
				# Optional: Print min/max elevation after each pass
				# if i_pass % 1 == 0 and current_elevations:
				#    min_el = min(current_elevations.values())
				#    max_el = max(current_elevations.values())
				#    print(f"    Pass {i_pass+1} Elev Range: {min_el:.0f} to {max_el:.0f}")

			# Assign the final smoothed elevations back
			self.elevations = current_elevations
			print(" Elevation diffusion complete.")
		else:
			print(" Skipping elevation diffusion.")

		if not self.elevations:
			print("Warning: No elevations were assigned.")
			return

		# Update normalization range based on actual data or keep fixed
		# Using fixed range based on globals VMIN, VMAX
		NORM_ELEVATION.vmin = VMIN
		NORM_ELEVATION.vmax = VMAX
		final_min = min(self.elevations.values())
		final_max = max(self.elevations.values())
		print(f"Final Elevation Range (Layer 0): {final_min:.0f}m to {final_max:.0f}m (Colorbar range: {VMIN} to {VMAX})")

	def distribute_water(self, total_water_zettaliters: float):
		"""Distributes water over the world layers based on elevation and layer type."""
		total_water_petaliters = total_water_zettaliters * 1e6 # 1 ZL = 10^6 PL
		print(f"Distributing {total_water_zettaliters:.3f} ZL ({total_water_petaliters:.0f} PL) of water...")

		num_vertices = len(self.vertices)
		if num_vertices == 0:
			print("Warning: No vertices to distribute water to.")
			return

		# Initialize water dictionaries for ALL vertices
		self.surface_water = {i: 0.0 for i in range(num_vertices)}
		self.groundwater = {i: 0.0 for i in range(num_vertices)}
		self.ice_water = {i: 0.0 for i in range(num_vertices)}
		self.atmospheric_water = {i: 0.0 for i in range(num_vertices)}
		self.total_water_volume_surface = 0.0
		self.total_water_volume_ground = 0.0
		self.total_water_volume_ice = 0.0
		self.total_water_volume_atmosphere = 0.0

		# Define fractions (adjust these to sum to 1.0)
		surface_water_fraction = 0.90  # Majority on the surface
		groundwater_fraction = 0.08 # Below surface
		ice_water_fraction = 0.015 # Surface, high/cold places
		atmosphere_water_fraction = 0.005 # Above surface

		# --- 1. Surface Water (Layer 0) ---
		total_surface_water_volume = total_water_petaliters * surface_water_fraction
		layer0_vertices = []
		potential_well_vertices = [] # Vertices below sea level (0 elevation)
		total_potential = 0.0

		for v_idx in range(num_vertices):
			if self.vertices[v_idx].layer_id == 0:
				layer0_vertices.append(v_idx)
				elevation = self.elevations.get(v_idx, 0.0) # Get elevation for Layer 0 vertices
				if elevation <= 0:
					# Calculate potential based on depth (larger positive number for deeper)
					potential = -elevation + 1.0 # Add 1 to give sea level some potential
					potential_well_vertices.append((v_idx, potential))
					total_potential += potential

		if not layer0_vertices:
			print("Warning: No Layer 0 vertices found for surface water distribution.")
		elif total_potential > 0:
			print(f" Distributing {total_surface_water_volume:.1f} PL surface water based on elevation potential...")
			distributed_surface = 0.0
			for v_idx, potential in potential_well_vertices:
				fraction = potential / total_potential
				water_vol = total_surface_water_volume * fraction
				self.surface_water[v_idx] = water_vol
				distributed_surface += water_vol
			self.total_water_volume_surface = distributed_surface
			print(f"  -> Assigned {distributed_surface:.1f} PL to {len(potential_well_vertices)} vertices below sea level.")
			# Distribute remaining water evenly? Or let it flow later? For now, only fill depressions.
			remaining_surface_water = total_surface_water_volume - distributed_surface
			if remaining_surface_water > 1e-3 and len(layer0_vertices) > 0:
				print(f"  -> Distributing remaining {remaining_surface_water:.1f} PL evenly over all {len(layer0_vertices)} surface vertices.")
				per_vertex_extra = remaining_surface_water / len(layer0_vertices)
				for v_idx in layer0_vertices:
					self.surface_water[v_idx] += per_vertex_extra
				self.total_water_volume_surface = total_surface_water_volume # Update total assigned
		elif len(layer0_vertices) > 0: # If no potential wells, distribute evenly
				print(f" Distributing {total_surface_water_volume:.1f} PL surface water evenly...")
				per_vertex = total_surface_water_volume / len(layer0_vertices)
				for v_idx in layer0_vertices:
					self.surface_water[v_idx] = per_vertex
				self.total_water_volume_surface = total_surface_water_volume


		# --- 2. Groundwater (Layers < 0) ---
		total_groundwater_volume = total_water_petaliters * groundwater_fraction
		ground_vertices = [v_idx for v_idx in range(num_vertices) if self.vertices[v_idx].layer_id < 0]
		if not ground_vertices:
			print("Warning: No ground layers found for groundwater distribution.")
			self.total_water_volume_surface += total_groundwater_volume # Add to surface if no ground layers
		else:
			print(f" Distributing {total_groundwater_volume:.1f} PL groundwater evenly over {len(ground_vertices)} ground vertices...")
			per_vertex = total_groundwater_volume / len(ground_vertices)
			for v_idx in ground_vertices:
				self.groundwater[v_idx] = per_vertex
			self.total_water_volume_ground = total_groundwater_volume


		# --- 3. Ice/Glaciers (High Latitude/Elevation on Layer 0) ---
		total_ice_water_volume = total_water_petaliters * ice_water_fraction
		ice_candidates = []
		total_ice_potential = 0.0
		# Define conditions for ice formation (example thresholds)
		min_latitude_for_ice = 0.7 # Corresponds to ~45 degrees (sin(45) ~ 0.707)
		min_elevation_for_ice = 3000.0 # Meters

		for v_idx in layer0_vertices: # Only consider surface layer for ice
			vertex = self.vertices[v_idx]
			elevation = self.elevations.get(v_idx, 0.0)
			# Use z-coordinate normalized as proxy for |sin(latitude)|
			abs_z_norm = abs(vertex.pos[2] / np.linalg.norm(vertex.pos)) if np.linalg.norm(vertex.pos) > 0 else 0

			is_high_latitude = abs_z_norm >= min_latitude_for_ice
			is_high_elevation = elevation >= min_elevation_for_ice

			if is_high_latitude or is_high_elevation:
				# Simple potential: 1 point for meeting condition, more if both met?
				potential = 0
				if is_high_latitude: potential += 1
				if is_high_elevation: potential += (elevation / min_elevation_for_ice) # Scale by how high
				ice_candidates.append((v_idx, potential))
				total_ice_potential += potential

		if not ice_candidates:
			print("Warning: No suitable locations found for ice distribution.")
			self.total_water_volume_surface += total_ice_water_volume # Add to surface
		elif total_ice_potential > 0:
			print(f" Distributing {total_ice_water_volume:.1f} PL ice water based on potential over {len(ice_candidates)} locations...")
			distributed_ice = 0.0
			for v_idx, potential in ice_candidates:
				fraction = potential / total_ice_potential
				water_vol = total_ice_water_volume * fraction
				self.ice_water[v_idx] = water_vol
				distributed_ice += water_vol
			self.total_water_volume_ice = distributed_ice
		# Handle case where total_ice_potential is 0 but candidates exist? Distribute evenly.
		elif len(ice_candidates) > 0:
			per_vertex = total_ice_water_volume / len(ice_candidates)
			for v_idx, _ in ice_candidates:
				self.ice_water[v_idx] = per_vertex
			self.total_water_volume_ice = total_ice_water_volume


		# --- 4. Atmospheric Water (Layers > 0) ---
		total_atmospheric_water_volume = total_water_petaliters * atmosphere_water_fraction
		atmosphere_vertices = [v_idx for v_idx in range(num_vertices) if self.vertices[v_idx].layer_id > 0]
		if not atmosphere_vertices:
			print("Warning: No atmospheric layers found for atmospheric water distribution.")
			self.total_water_volume_surface += total_atmospheric_water_volume # Add to surface
		else:
			print(f" Distributing {total_atmospheric_water_volume:.1f} PL atmospheric water evenly over {len(atmosphere_vertices)} atmosphere vertices...")
			per_vertex = total_atmospheric_water_volume / len(atmosphere_vertices)
			for v_idx in atmosphere_vertices:
				self.atmospheric_water[v_idx] = per_vertex
			self.total_water_volume_atmosphere = total_atmospheric_water_volume

		# --- Final Summary ---
		total_distributed = (self.total_water_volume_surface +
							 self.total_water_volume_ground +
							 self.total_water_volume_ice +
							 self.total_water_volume_atmosphere)
		print(f"Water distribution complete.")
		print(f"  Surface Water: {self.total_water_volume_surface:.3f} PL")
		print(f"  Groundwater:   {self.total_water_volume_ground:.3f} PL")
		print(f"  Ice Water:     {self.total_water_volume_ice:.3f} PL")
		print(f"  Atmosphere:    {self.total_water_volume_atmosphere:.3f} PL")
		print(f"  ---------------------------------")
		print(f"  Total Distributed: {total_distributed:.3f} PL")
		if abs(total_distributed - total_water_petaliters) > 1e-3:
			print(f"Warning: Distributed water ({total_distributed:.3f} PL) does not match target ({total_water_petaliters:.3f} PL). Difference: {total_distributed - total_water_petaliters:.3f} PL")

	def _build_adjacency_list(self):
		"""Builds an adjacency list for vertices based ONLY on self.faces."""
		# self.adjacency_list.clear() # Clearing should happen in the caller if needed
		num_face_edges = 0
		for face in self.faces:
			# Assumes face.vertices is a tuple of 3 indices (v0, v1, v2)
			v0, v1, v2 = face.vertices
			# Add edges (v0,v1), (v1,v2), (v2,v0)
			# Check validity of indices just in case
			max_idx = max(v0, v1, v2)
			if max_idx >= len(self.vertices):
				print(f"Error building adjacency: Face {face.vertices} contains invalid index {max_idx} (max vertex index: {len(self.vertices)-1}). Skipping face.")
				continue

			if v1 not in self.adjacency_list[v0]: self.adjacency_list[v0].append(v1); num_face_edges+=1
			if v0 not in self.adjacency_list[v1]: self.adjacency_list[v1].append(v0) # No double count edges

			if v2 not in self.adjacency_list[v1]: self.adjacency_list[v1].append(v2); num_face_edges+=1
			if v1 not in self.adjacency_list[v2]: self.adjacency_list[v2].append(v1)

			if v0 not in self.adjacency_list[v2]: self.adjacency_list[v2].append(v0); num_face_edges+=1
			if v2 not in self.adjacency_list[v0]: self.adjacency_list[v0].append(v0)
		# print(f" Built adjacency from {len(self.faces)} faces, added {num_face_edges} unique face edges.")




class icosphere(worldState):
	def __init__(self, radius, subdivisions):
		# Call superclass init first
		super().__init__(radius, subdivisions)
		# Now call shapeBase which requires self.details to be set
		print(f"Initializing icosphere with radius {radius}, subdivisions {self.details}")
		self.shapeBase()
		print(f"Icosphere base created. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")
		# Build adjacency list for the base sphere
		self._build_adjacency_list()
		print(f"Initial adjacency list built.")


	def shapeBase(self):
		"""Creates the base icosahedron and subdivides it."""
		self.vertices = [] # Ensure lists are clear
		self.faces = []
		self._vertex_pos_cache = {} # Clear cache for new shape
		self._subdivision_cache = {}

		# Create 12 base vertices of icosahedron
		vertices_raw = [
			(-1, PHI, 0), (1, PHI, 0), (-1, -PHI, 0), (1, -PHI, 0),
			(0, -1, PHI), (0, 1, PHI), (0, -1, -PHI), (0, 1, -PHI),
			(PHI, 0, -1), (PHI, 0, 1), (-PHI, 0, -1), (-PHI, 0, 1)
		]

		# Normalize and add using _add_vertex to populate self.vertices and cache
		vertex_indices = []
		for x, y, z in vertices_raw:
			v = Vertex(x, y, z)
			v.normalize() # Normalize before adding
			idx = self._add_vertex(v) # Use the cache-aware method
			vertex_indices.append(idx)
		# print(f" Added {len(self.vertices)} base vertices.")

		# Create 20 base faces using the indices returned by _add_vertex
		faces_raw = [
			(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
			(1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
			(3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
			(4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
		]

		self.faces = [Face(v0, v1, v2) for v0, v1, v2 in faces_raw]
		# print(f" Created {len(self.faces)} base faces.")

		# Subdivide according to self.details
		self.subdivide()
		# No need to build adjacency here, it's built after __init__ completes.
		return self # Return self for potential chaining, though not strictly necessary


	def subdivide(self):
		"""Subdivides faces of the mesh according to self.details."""
		if self.details <= 0:
			# print(" No subdivisions requested.")
			return

		print(f" Starting subdivision for {self.details} level(s)...")
		for level in range(self.details):
			new_faces_next_level = []
			self._subdivision_cache = {} # Clear midpoint cache for each level

			current_faces = self.faces[:] # Operate on a copy
			if not current_faces:
				print(" Warning: No faces to subdivide.")
				break

			# print(f"  Level {level+1}/{self.details}: Subdividing {len(current_faces)} faces...")
			for face in current_faces:
				v0_idx, v1_idx, v2_idx = face.vertices

				try:
					# Get or create midpoints (normalized, using _add_vertex internally)
					m01_idx = self._get_or_create_midpoint(v0_idx, v1_idx)
					m12_idx = self._get_or_create_midpoint(v1_idx, v2_idx)
					m20_idx = self._get_or_create_midpoint(v2_idx, v0_idx)
				except IndexError as e:
					print(f"  Error finding vertices for midpoint creation in face {face.vertices}: {e}. Skipping subdivision for this face.")
					new_faces_next_level.append(face) # Keep original face if error
					continue
				except Exception as e:
					print(f"  Unexpected error during midpoint creation for face {face.vertices}: {e}. Skipping subdivision.")
					new_faces_next_level.append(face)
					continue

				# Create the four new faces using the new indices
				new_faces_next_level.append(Face(v0_idx, m01_idx, m20_idx))
				new_faces_next_level.append(Face(v1_idx, m12_idx, m01_idx))
				new_faces_next_level.append(Face(v2_idx, m20_idx, m12_idx))
				new_faces_next_level.append(Face(m01_idx, m12_idx, m20_idx)) # Center face

			self.faces = new_faces_next_level # Update faces list for the next level or final result
			# print(f"  Level {level+1} complete. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")

		print(f"Subdivision complete. Final Vertices: {len(self.vertices)}, Final Faces: {len(self.faces)}")


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
		if not unassigned_vertices:
			print("Warning: No vertices to assign to plates.")
			return []

		if num_plates > len(unassigned_vertices):
			print(f"Warning: More plates ({num_plates}) requested than vertices ({len(unassigned_vertices)}). Reducing plates.")
			num_plates = len(unassigned_vertices)
			plates = plates[:num_plates]
		if num_plates == 0:
			print("Warning: Number of plates is zero.")
			return []

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
			# Use a sample of plate vertices for performance on large plates
			sample_size = min(len(current_plate.vertices), 100 + int(np.sqrt(len(current_plate.vertices))))
			sampled_plate_vertices = random.sample(current_plate.vertices, sample_size)

			for plate_vertex_index in sampled_plate_vertices: # current_plate.vertices:
				# Check if vertex index is valid in adjacency list
				if plate_vertex_index in self.adjacency_list:
					for neighbor_idx in self.adjacency_list[plate_vertex_index]:
						if neighbor_idx in unassigned_vertices:
							possible_expansion_vertices.add(neighbor_idx)


			if not possible_expansion_vertices:
				# If this plate can't expand from the sample, try another one next iteration
				# Could add a check here to see if *any* vertex has unassigned neighbors
				continue

			expansion_candidates = list(possible_expansion_vertices)

			# --- Proximity Biased Selection ---
			# Simplified: Just pick a random candidate for now
			# The proximity bias adds complexity and may not be strictly needed for initial assignment
			next_vertex_index = random.choice(expansion_candidates)


			# Assign the chosen vertex to the current plate
			current_plate.add_vertex(next_vertex_index)
			self.vertex_to_plate_id[next_vertex_index] = plate_index
			unassigned_vertices.remove(next_vertex_index)

		# --- Handle Leftovers ---
		if unassigned_vertices:
			print(f"Warning: {len(unassigned_vertices)} vertices remained unassigned after walk. Assigning randomly.")
			remaining_unassigned = list(unassigned_vertices)
			random.shuffle(remaining_unassigned)
			for idx, vertex_idx in enumerate(remaining_unassigned):
				target_plate_index = idx % num_plates # Distribute somewhat evenly
				plates[target_plate_index].add_vertex(vertex_idx)
				self.vertex_to_plate_id[vertex_idx] = target_plate_index
				# unassigned_vertices.remove(vertex_idx) # Not needed, just iterating

		# --- Recycling/Smoothing Pass ---
		recycling_iterations = 5 # Smooth boundaries
		print(f" Starting {recycling_iterations} plate boundary smoothing passes...")
		for recycle_iter in range(recycling_iterations):
			vertices_to_reassign = []
			# Identify vertices potentially worth reassigning (near borders)
			for plate_idx, plate in enumerate(plates):
				for vertex_index in plate.vertices:
					if vertex_index not in self.adjacency_list: continue
					is_border = False
					neighbor_plate_ids = set()
					for neighbor_index in self.adjacency_list[vertex_index]:
						neighbor_plate_id = self.vertex_to_plate_id.get(neighbor_index, -1)
						if neighbor_plate_id != -1 and neighbor_plate_id != plate_idx:
							is_border = True
							neighbor_plate_ids.add(neighbor_plate_id)
					# Reassign if it's on a border and has diverse neighbors? Simpler: reassign all border nodes.
					if is_border:
						vertices_to_reassign.append(vertex_index)

			if not vertices_to_reassign:
				print("  No border vertices found to reassign, stopping smoothing early.")
				break

			unique_vertices_to_reassign = list(set(vertices_to_reassign))
			random.shuffle(unique_vertices_to_reassign)
			reassigned_count = 0

			for vertex_index in unique_vertices_to_reassign:
				current_plate_id = self.vertex_to_plate_id.get(vertex_index, -1)
				if current_plate_id == -1: continue

				neighboring_plate_counts = defaultdict(int)
				if vertex_index not in self.adjacency_list: continue

				for neighbor_idx in self.adjacency_list[vertex_index]:
					neighbor_plate_id = self.vertex_to_plate_id.get(neighbor_idx, -1)
					if neighbor_plate_id != -1:
						neighboring_plate_counts[neighbor_plate_id] += 1

				if not neighboring_plate_counts: continue

				# Assign to the plate with the most neighbors
				# (Could add proximity weighting here too)
				best_plate_id = max(neighboring_plate_counts, key=neighboring_plate_counts.get)

				if best_plate_id != current_plate_id:
					# Remove from old plate (if valid)
					if 0 <= current_plate_id < len(plates):
						try:
							plates[current_plate_id].vertices.remove(vertex_index)
						except ValueError:
							pass # Wasn't in the list, maybe already moved
					# Add to new plate (if valid)
					if 0 <= best_plate_id < len(plates):
						plates[best_plate_id].add_vertex(vertex_index)
						self.vertex_to_plate_id[vertex_index] = best_plate_id
						reassigned_count += 1
					else:
						# Put back in old plate if new one is invalid
						if 0 <= current_plate_id < len(plates):
							plates[current_plate_id].add_vertex(vertex_index)
							self.vertex_to_plate_id[vertex_index] = current_plate_id

			print(f"  Smoothing Pass {recycle_iter + 1}: Reassigned {reassigned_count} vertices.")


		# Final cleanup and validation
		final_v_count = 0
		assigned_verts_check = set()
		# Ensure vertex_to_plate_id matches plate.vertices lists
		temp_map = {}
		for i, p in enumerate(plates):
			p.vertices = sorted(list(set(p.vertices))) # Ensure unique and sorted
			for v_idx in p.vertices:
				if v_idx in temp_map:
					print(f"Warning: Vertex {v_idx} found in multiple plates ({temp_map[v_idx]} and {i}) after smoothing.")
					# Decide how to handle - e.g., keep first assignment or assign randomly
				temp_map[v_idx] = i
				assigned_verts_check.add(v_idx)
			final_v_count += len(p.vertices)

		self.vertex_to_plate_id = temp_map # Update map based on final plate lists

		if len(self.vertex_to_plate_id) != len(self.vertices):
			print(f"ERROR: Plate assignment incomplete! {len(self.vertices) - len(self.vertex_to_plate_id)} vertices unmapped.")
		elif final_v_count != len(self.vertices):
			print(f"ERROR: Vertex count mismatch in plates! Sum of plate vertices ({final_v_count}) != total vertices ({len(self.vertices)}).")


		self.plates = plates
		print(f"Plate assignment complete. {len(plates)} plates assigned.")
		return plates
