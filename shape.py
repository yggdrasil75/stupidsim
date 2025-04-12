from typing import List
import numpy as np

class Vertex:
	def __init__(self, x, y, z):
		self.pos = np.array([float(x), float(y), float(z)], dtype=np.float64)
		self.layer_id = 0
		self.original_vertex_index: int = -1 

	def normalize(self):
		norm = np.linalg.norm(self.pos)
		if abs(norm) > 1e-10:
			self.pos /= norm
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
		return f"Vertex({self.pos[0]:.4f}, {self.pos[1]:.4f}, {self.pos[2]:.4f})"

class Face:
	def __init__(self, *vertices: int):
		self.vertices = tuple(vertices)

	def get_vertices(self, world) -> List[Vertex]:
		return [world.vertices[i] for i in self.vertices]

	def normal(self, world) -> np.ndarray:
		verts = self.get_vertices(world)

		p0 = verts[0].pos
		p1 = verts[1].pos
		p2 = verts[2].pos

		edge1 = p1 - p0
		edge2 = p2 - p0
		face_normal = np.cross(edge1, edge2)

		norm = np.linalg.norm(face_normal)

		face_normal /= norm

		if np.dot(face_normal, p0) < 0:
			face_normal *= -1.0
		return face_normal

	def centroid(self, world) -> Vertex:
		"""Calculates the geometric centroid of the face vertices."""
		verts = self.get_vertices(world)
		center_pos = np.mean([v.pos for v in verts], axis=0)
		return Vertex(*center_pos).normalize()

	def area(self, world) -> float:
		"""Calculates the area of the spherical triangle using L'Huilier's Theorem."""
		# Assumes vertices are normalized (on unit sphere)
		verts_pos = [world.vertices[idx].pos for idx in self.vertices]
		if len(verts_pos) != 3:
			raise ValueError(f"Face {self.vertices} is not a triangle!")

		p0, p1, p2 = verts_pos

		# Calculate the lengths of the sides (as angles on the unit sphere)
		# Use clip to avoid domain errors in acos due to floating point inaccuracies
		a = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
		b = np.arccos(np.clip(np.dot(p0, p2), -1.0, 1.0))
		c = np.arccos(np.clip(np.dot(p0, p1), -1.0, 1.0))

		# Calculate the semi-perimeter
		s = (a + b + c) / 2.0

		# Calculate the spherical excess (area on unit sphere) using L'Huilier's Theorem
		# Check for degenerate cases where s might be equal to a, b, or c
		tan_E_over_4_sq = np.tan(s / 2.0) * \
						  np.tan(np.clip((s - a)/2.0, 0, np.pi/2)) * \
						  np.tan(np.clip((s - b)/2.0, 0, np.pi/2)) * \
						  np.tan(np.clip((s - c)/2.0, 0, np.pi/2))

		# Handle potential negative values due to floating point errors before sqrt
		if tan_E_over_4_sq < 0:
			if abs(tan_E_over_4_sq) < 1e-12: # Allow small negative values near zero
				tan_E_over_4_sq = 0.0
			else:
				# This might indicate a more significant issue (e.g., invalid triangle)
				# For now, let's return 0 area for robustness, but could warn
				return 0.0


		# The spherical excess E is the area on the unit sphere
		area_unit_sphere = 4.0 * np.arctan(np.sqrt(tan_E_over_4_sq))
		# Check for negative area due to potential floating point issues
		if area_unit_sphere < 0 and abs(area_unit_sphere) < 1e-9:
				area_unit_sphere = 0.0
		elif area_unit_sphere < 0:
			# If significantly negative, indicates an issue, but return 0 for robustness
			area_unit_sphere = 0.0

		# Scale area by the square of the world radius
		return area_unit_sphere * (world.radius**2)

	def __repr__(self):
		return f"Face{self.vertices}"
