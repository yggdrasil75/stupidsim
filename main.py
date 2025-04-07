import struct
from typing import Dict, List
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0

class Vertex:
	def __init__(self, x, y, z):
		self.pos = np.array([x, y, z], dtype=np.float64)
		self.x = x
		self.y = y
		self.z = z
		self.neighbors:np.array
		
	def normalize(self):
		norm = np.linalg.norm(self.pos)
		if norm > 0:
			self.pos /= norm
			self.x, self.y, self.z = self.pos
		return self

	def dotProd(self, other):
		return self.x * other.x + self.y * other.y + self.z * other.z
	
	def crossProduct(self, other):
		x = self.y * other.z - self.z * other.y
		y = self.z * other.x - self.x * other.z
		z = self.x * other.y - self.y * other.x
		return Vertex(x, y, z)
	
	def __hash__(self):
		hx = struct.unpack('I', struct.pack('f', self.x))[0]
		hy = struct.unpack('I', struct.pack('f', self.y))[0]
		hz = struct.unpack('I', struct.pack('f', self.z))[0]
		return hx ^ (hy << 1) ^ (hz << 2)

	def __eq__(self, other):
		if not isinstance(other, Vertex):
			return False
		return (self.x == other.x and 
				self.y == other.y and 
				self.z == other.z)

class Face:
	def __init__(self, *vertices):
		self.vertices = vertices
		
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

class worldState:
	def __init__(self):
		self.vertices: List[Vertex] = []
		self.faces: List[Face] = []
		self.vertex_indices: Dict[int, Vertex] = {}
		self.radius: float = 6377.9
		self.timestepSeconds: int = 1
		self.gravity: float = 9.81
		self.details: int = 3  # subdivision level

	def timeStepHour(self) -> float:
		return float(self.timestepSeconds) / 3600.0
	
	def timeStepDay(self):
		return float(self.timestepSeconds) / 86400.0
	
	def timeStepYear(self):
		return float(self.timestepSeconds) / 31556952.0
	
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
	
	def cubeBase(self):
		# Create initial vertices (8 vertices of a cube)
		vertices = []
		for x in [-1, 1]:
			for y in [-1, 1]:
				for z in [-1, 1]:
					vertices.append(Vertex(x, y, z))
		
		self.vertices = vertices
		
		# Create initial faces (6 quad faces of a cube)
		faces = [
			Face(0, 1, 3, 2),  # Front
			Face(1, 5, 7, 3),  # Right
			Face(5, 4, 6, 7),  # Back
			Face(4, 0, 2, 6),  # Left
			Face(2, 3, 7, 6),  # Top
			Face(4, 5, 1, 0)   # Bottom
		]
		
		self.faces = faces
		
		# Validate faces
		#for face in self.faces:
		#	face._validate_face(self)
		
		# Rest of the method remains the same...
		self.subdivide()
			
		# Create vertex index mapping
		self.vertex_indices = {i: v for i, v in enumerate(self.vertices)}
		
		return self

	def subdivide(self):
		for _ in range(self.details):
			new_vertices = self.vertices.copy()
			vertex_map = {}  # Maps edge pairs to new vertex indices
			new_faces = []
			
			# Helper function to get or create midpoint vertex
			def get_midpoint(v1_idx, v2_idx):
				key = tuple(sorted((v1_idx, v2_idx)))
				if key not in vertex_map:
					v1 = self.vertices[v1_idx]
					v2 = self.vertices[v2_idx]
					midpoint = Vertex(*(v1.pos + v2.pos) / 2)
					midpoint.normalize()  # Project back onto sphere
					vertex_map[key] = len(new_vertices)
					new_vertices.append(midpoint)
				return vertex_map[key]
			
			# Process each face
			for face in self.faces:
				vert_count = len(face.vertices)
				
				if vert_count == 3:  # Triangle subdivision
					# Get vertex indices
					v0, v1, v2 = face.vertices
					
					# Create midpoint vertices for each edge
					a = get_midpoint(v0, v1)
					b = get_midpoint(v1, v2)
					c = get_midpoint(v2, v0)
					
					# Create 4 new faces from the subdivided triangle
					new_faces.append(Face(v0, a, c))
					new_faces.append(Face(v1, b, a))
					new_faces.append(Face(v2, c, b))
					new_faces.append(Face(a, b, c))
					
				elif vert_count == 4:  # Quad subdivision
					# Get vertex indices
					v0, v1, v2, v3 = face.vertices
					
					# Create midpoint vertices for each edge and face center
					a = get_midpoint(v0, v1)
					b = get_midpoint(v1, v2)
					c = get_midpoint(v2, v3)
					d = get_midpoint(v3, v0)
					e = get_midpoint(v0, v2)  # Face center (could also average all 4)
					
					# Create 4 new quad faces from the subdivided quad
					new_faces.append(Face(v0, a, e, d))
					new_faces.append(Face(a, v1, b, e))
					new_faces.append(Face(e, b, v2, c))
					new_faces.append(Face(d, e, c, v3))
					
				else:
					# Skip faces with more than 4 vertices (though base cube shouldn't have these)
					continue
			for vert in new_vertices:
				vert.normalize()
			
			# Update for next iteration
			self.vertices = new_vertices
			self.faces = new_faces
			
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


def VisualizeWorld(world: worldState):
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')

	# Plot vertices
	x = [v.pos[0] for v in world.vertices]
	y = [v.pos[1] for v in world.vertices]
	z = [v.pos[2] for v in world.vertices]
	ax.scatter(x, y, z, color='b', s=10)

	# Plot faces
	for face in world.faces:
		verts = [world.vertices[i].pos for i in face.vertices]
		verts.append(verts[0])  # Close the polygon
		xs, ys, zs = zip(*verts)
		ax.plot(xs, ys, zs, color='k', alpha=0.5)

	# Set equal aspect ratio
	ax.set_box_aspect([1, 1, 1])
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title(f'Icosphere (Subdivision Level: {world.details})')

	plt.tight_layout()
	plt.show()

def main():
	# Create a new world
	world = worldState()
	world.radius = 1.0  # For visualization, keep radius small
	world.details = 3   # Subdivision level

	# Generate the icosphere
	world.cubeBase()

	# Validate the structure
	# try:
	# 	world.validateStructure()
	# 	print("World structure validated successfully")
	# except ValueError as e:
	# 	print(f"Validation error: {e}")
	# 	return

	# Visualize the world
	VisualizeWorld(world)
	world.icosphereBase()
	VisualizeWorld(world)

if __name__ == "__main__":
	main()