from enum import Enum
import time
from typing import Optional
import dearpygui.dearpygui as dpg
import sys
import numpy as np
import torch
from dataclasses import dataclass, field

DEVICE = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device=DEVICE)
torch.set_default_dtype(torch.float32)
G = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float32, device=DEVICE)
PHYSICSSTEPMULT = 8

# Helper function to create a rotation matrix to align one vector with another
def get_rotation_matrix(v_from, v_to):
    v_from = v_from / torch.norm(v_from)
    v_to = v_to / torch.norm(v_to)
    
    axis = torch.linalg.cross(v_from, v_to)
    axis_norm = torch.norm(axis)
    
    # If vectors are parallel, no rotation is needed
    if axis_norm < 1e-8:
        # Check if they are pointing in opposite directions
        if torch.dot(v_from, v_to) < -0.9999:
            # Return 180 degree rotation around an arbitrary orthogonal axis
            # Find an arbitrary vector not parallel to v_from
            if torch.abs(v_from[0]) < 0.9:
                ortho = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
            else:
                ortho = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
            axis = torch.linalg.cross(v_from, ortho)
            axis = axis / torch.norm(axis)
            cos_a = -1.0
            sin_a = 0.0
        else: # pointing in the same direction
            return torch.eye(3)
    else:
        axis = axis / axis_norm
        cos_a = torch.dot(v_from, v_to)
        sin_a = torch.sqrt(1.0 - cos_a * cos_a)

    # Rodrigues' rotation formula
    I = torch.eye(3)
    K = torch.tensor([[0, -axis[2], axis[1]],[axis[2], 0, -axis[0]],[-axis[1], axis[0], 0]], dtype=torch.float32, device=DEVICE)
    
    return I + sin_a * K + (1 - cos_a) * torch.matmul(K, K)

@dataclass
class block:
    id: int
    _vertices: torch.Tensor
    #tris: list[tuple[int,int,int]]
    tris: torch.Tensor
    # Updated color to include alpha channel for transparency
    color: torch.Tensor = field(default_factory=lambda: torch.tensor([200,0,0,255], dtype=torch.uint8))
    physics: bool = True
    heldblocks: list[int] = field(default_factory=list)

    velocity: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float32, device=DEVICE))
    position: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float32, device=DEVICE))
    orientation: torch.Tensor = field(default_factory=lambda: torch.eye(3, dtype=torch.float32, device=DEVICE))
    angular_velocity: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float32, device=DEVICE))
    mass: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.float32, device=DEVICE))
    restitution: torch.Tensor = field(default_factory=lambda: torch.tensor(0.3, dtype=torch.float32, device=DEVICE))
    inertia_tensor_inv: torch.Tensor = field(init=False)

    @property
    def vertices(self) -> torch.Tensor:
        return torch.matmul(self._vertices, self.orientation.T) + self.position 
    
    def __post_init__(self):
        self.inertia_tensor_inv = self.calculate_inverse_inertia()

    def calculate_inverse_inertia(self) -> torch.Tensor:
        if self.mass <= 0 or len(self.vertices) == 0:
            return torch.zeros((3, 3))
            
        min_coords = torch.min(self.vertices, dim=0).values
        max_coords = torch.max(self.vertices, dim=0).values
        size = max_coords - min_coords
        w, h, d = size[0], size[1], size[2]
        
        # Clamp dimensions to avoid zero inertia on flat objects
        w, h, d = torch.clamp(w, min=1e-6), torch.clamp(h, min=1e-6), torch.clamp(d, min=1e-6)

        I_xx = (1.0 / 12.0) * self.mass * (h*h + d*d)
        I_yy = (1.0 / 12.0) * self.mass * (w*w + d*d)
        I_zz = (1.0 / 12.0) * self.mass * (w*w + h*h)

        inertia_tensor = torch.diag(torch.tensor([I_xx, I_yy, I_zz], dtype=torch.float32, device=DEVICE))
        
        # Return the inverse of the diagonal tensor
        return torch.inverse(inertia_tensor)

    def collision(self, other: 'block') -> tuple[bool, Optional[torch.Tensor], float] | None:
        min_s = torch.min(self.vertices, dim=0).values
        max_s = torch.max(self.vertices, dim=0).values
        min_o = torch.min(other.vertices, dim=0).values
        max_o = torch.max(other.vertices, dim=0).values

        if torch.any(max_s < min_o) or torch.any(max_o < min_s):
            return False, None, 0.0
        
        overlaps = torch.min(max_s, max_o) - torch.max(min_s, min_o)
        min_pen, axis_idx = torch.min(overlaps, dim=0)
        # center_s = (min_s + max_s) / 2.0
        # center_o = (min_o + max_o) / 2.0
        # direction = center_s - center_o
        direction = self.position - other.position
        normal = torch.zeros(3)
        if direction[axis_idx] < 0:
            normal[axis_idx] = -1.0
        else:
            normal[axis_idx] = 1.0
        
        return True, normal, min_pen.item()

    def apply_gravity(self, delta_time: float):
        if not self.physics or self.mass <= 0 or self.heldblocks:
            return
        self.velocity += G * delta_time
        
        self.position += self.velocity * delta_time
        omega = self.angular_velocity
        omega_norm = torch.norm(omega)
        if omega_norm > 1e-6:
            angle = omega_norm * delta_time
            axis = omega / omega_norm
            K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=torch.float32, device=DEVICE)
            I = torch.eye(3)
            step_rotation = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
            self.orientation = torch.matmul(step_rotation, self.orientation)
        
    @classmethod
    def _orient_and_translate_mesh(cls, vertices, start_point, end_point, default_axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)):
        direction = end_point - start_point
        length = torch.norm(direction)
        if length == 0:
            center = torch.mean(vertices, dim=0)
            return vertices - center

        target_direction = direction / length
        
        min_y = torch.min(vertices[:, 1])
        max_y = torch.max(vertices[:, 1])
        mesh_height = max_y - min_y
        scale_factor = length / mesh_height if mesh_height > 0 else 1.0
        
        center_offset = torch.tensor([0, (max_y + min_y) / 2, 0], dtype=torch.float32, device=DEVICE)
        scaled_vertices = (vertices - center_offset) * torch.tensor([1.0, scale_factor, 1.0], dtype=torch.float32, device=DEVICE)

        rot_matrix = get_rotation_matrix(default_axis, target_direction)
        rotated_vertices = torch.matmul(scaled_vertices, rot_matrix.T)
        
        # Translate to the midpoint of the start and end points
        #midpoint = (start_point + end_point) / 2
        return rotated_vertices

    @classmethod
    def quads_to_tris(cls, quads: torch.Tensor):
        num_quads = quads.shape[0]
        tris = torch.zeros((num_quads * 2, 3), dtype=torch.long, device=DEVICE)
        tris[::2, 0] = quads[:, 0]
        tris[::2, 1] = quads[:, 1]
        tris[::2, 2] = quads[:, 2]
        tris[1::2, 0] = quads[:, 0]
        tris[1::2, 1] = quads[:, 2]
        tris[1::2, 2] = quads[:, 3]
        return tris

    @classmethod
    def subdivide_tri(cls, vertices: torch.Tensor, tris: torch.Tensor):
        numTris = tris.shape[0]
        numVerts = vertices.shape[0]
        edges = torch.stack([
            torch.stack([tris[:,0],tris[:,1]], dim=1),
            torch.stack([tris[:,1],tris[:,2]], dim=1),
            torch.stack([tris[:,2],tris[:,0]], dim=1)
            ], dim=0)
        edges = edges.reshape(-1, 2)
        edgesSorted, _ = torch.sort(edges, dim=1)
        uniqueEdges, edgeIndices = torch.unique(edgesSorted, dim=0, return_inverse=True)
        edgeVerts = (vertices[uniqueEdges[:, 0]] + vertices[uniqueEdges[:, 1]]) / 2
        newVerts = torch.cat([vertices, edgeVerts], dim=0)
        edgeIndices = edgeIndices.reshape(3, numTris) + numVerts
        newTris = torch.zeros((numTris * 4, 3), dtype=torch.long, device=DEVICE)
        v0 = tris[:, 0]
        v1 = tris[:, 1]
        v2 = tris[:, 2]
        e0 = edgeIndices[0]
        e1 = edgeIndices[1]
        e2 = edgeIndices[2]
        newTris[0::4, 0] = v0
        newTris[0::4, 1] = e0
        newTris[0::4, 2] = e2

        newTris[1::4, 0] = e0
        newTris[1::4, 1] = v1
        newTris[1::4, 2] = e1
        
        newTris[2::4, 0] = e2
        newTris[2::4, 1] = e1
        newTris[2::4, 2] = v2
        
        newTris[3::4, 0] = e0
        newTris[3::4, 1] = e1
        newTris[3::4, 2] = e2
        
        return newVerts, newTris

    @classmethod
    def subdivide_quad(cls, vertices: torch.Tensor, quads: torch.Tensor):
        numQuads = quads.shape[0]
        numVerts = vertices.shape[0]
        edges = torch.stack([
            torch.stack([quads[:, 0], quads[:, 1]], dim=1),
            torch.stack([quads[:, 1], quads[:, 2]], dim=1),
            torch.stack([quads[:, 2], quads[:, 3]], dim=1),
            torch.stack([quads[:, 3], quads[:, 0]], dim=1)
        ], dim=0)
        edges = edges.reshape(-1, 2)
        edgesSorted, _ = torch.sort(edges, dim=1)
        uniqueEdges, edgeIndices = torch.unique(edgesSorted, dim=0, return_inverse=True)
        edgeIndices = edgeIndices.reshape(4, numQuads)
        edgeVerts = (vertices[uniqueEdges[:, 0]] + vertices[uniqueEdges[:, 1]]) / 2
        faceCenters = torch.mean(vertices[quads], dim=1)
        newVerts = torch.cat([vertices, edgeVerts, faceCenters], dim=0)
        edgeIndices = edgeIndices.reshape(4, numQuads) + numVerts
        faceIndices = torch.arange(numQuads, device=DEVICE) + numVerts + uniqueEdges.shape[0]
        newQuads = torch.zeros((numQuads * 4, 4), dtype=torch.long, device=DEVICE)

        v0 = quads[:, 0]
        v1 = quads[:, 1]
        v2 = quads[:, 2]
        v3 = quads[:, 3]
        e0 = edgeIndices[0]
        e1 = edgeIndices[1]
        e2 = edgeIndices[2]
        e3 = edgeIndices[3]

        newQuads[0::4, 0] = v0
        newQuads[0::4, 1] = e0
        newQuads[0::4, 2] = faceIndices
        newQuads[0::4, 3] = e3
        
        newQuads[1::4, 0] = e0
        newQuads[1::4, 1] = v1
        newQuads[1::4, 2] = e1
        newQuads[1::4, 3] = faceIndices
        
        newQuads[2::4, 0] = faceIndices
        newQuads[2::4, 1] = e1
        newQuads[2::4, 2] = v2
        newQuads[2::4, 3] = e2
        
        newQuads[3::4, 0] = e3
        newQuads[3::4, 1] = faceIndices
        newQuads[3::4, 2] = e2
        newQuads[3::4, 3] = v3
        
        return vertices, newQuads

    @classmethod
    def create_ground(cls, id: int, size: float = sys.float_info.max, height: float = 1):
        size = size / 2.0
        vertices = torch.tensor([
            [-size, -height, -size],
            [size, -height, -size],
            [size, -height, size],
            [-size, -height, size],
            [-size, 0, -size],
            [size, 0, -size],
            [size, 0, size],
            [-size, 0, size]
        ], dtype=torch.float32, device=DEVICE)
        quads = torch.tensor([
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7]
        ], dtype=torch.long, device=DEVICE)
        for _ in range(3):
            vertices, quads = cls.subdivide_quad(vertices, quads)
        tris = cls.quads_to_tris(quads)
        cl = cls(id=id, _vertices=vertices, tris=tris)
        cl.color = torch.tensor([0,200,0,255], dtype=torch.uint8)
        cl.physics = False
        cl.mass = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        return cl

    @classmethod
    def create_sphere(cls, id: int, radius: float = 1.0, subdivisions: int = 3, center: torch.Tensor = torch.zeros(3)):
        vertices = torch.tensor([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ], dtype=torch.float32, device=DEVICE)

        quads = torch.tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 4, 7, 3],
            [1, 5, 6, 2],
            [0, 1, 5, 4],
            [3, 2, 6, 7]
        ], dtype=torch.long, device=DEVICE)
        tris = cls.quads_to_tris(quads)
        
        for _ in range(subdivisions):
            vertices, tris = cls.subdivide_tri(vertices, tris)
        
        norms = torch.norm(vertices, dim=1, keepdim=True)
        vertices = vertices / norms * radius + center
        return cls(id=id, _vertices=vertices, tris=tris)

    @classmethod
    def create_sphere_quaddivide(cls, id: int, radius: float = 1.0, subdivisions: int = 3):
        # Start with a cube (8 vertices, 6 quad faces)
        vertices = torch.tensor([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ], dtype=torch.float32, device=DEVICE)
        
        # Cube faces (6 quads)
        quads = torch.tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 4, 7, 3],
            [1, 5, 6, 2],
            [0, 1, 5, 4],
            [3, 2, 6, 7]
        ], dtype=torch.long, device=DEVICE)
        
        for _ in range(subdivisions):
            vertices, quads = cls.subdivide_quad(vertices, quads)
        
        # Convert quads to triangles (2 per quad)
        tris = cls.quads_to_tris(quads)
        
        # Normalize vertices to make them spherical
        norms = torch.norm(vertices, dim=1, keepdim=True)
        vertices = vertices / norms * radius
        
        # Create the block
        return cls(id=id, _vertices=vertices, tris=tris)
    
    @classmethod
    def _create_cylinder_mesh(cls, radius: float, height: float, radial_segments: int):
        """Creates a cylinder mesh along the Y-axis."""
        verticesL = []
        trisL = []
        
        # Top and bottom center vertices
        bottom_center_idx = 0
        top_center_idx = 1
        verticesL.append(torch.tensor([0, -height/2, 0], dtype=torch.float32, device=DEVICE))
        verticesL.append(torch.tensor([0,  height/2, 0], dtype=torch.float32, device=DEVICE))

        # Create bottom and top rings
        for i in range(radial_segments):
            angle = torch.tensor(2 * torch.pi * i / radial_segments, dtype=torch.float32, device=DEVICE)
            x = radius * torch.cos(angle)
            z = radius * torch.sin(angle)
            verticesL.append(torch.tensor([x, -height/2, z], dtype=torch.float32, device=DEVICE)) # Bottom ring
            verticesL.append(torch.tensor([x,  height/2, z], dtype=torch.float32, device=DEVICE)) # Top ring

        vertices = torch.stack(verticesL)
        # Create tris for bottom and top caps and sides
        for i in range(radial_segments):
            b_idx = 2 + i * 2
            t_idx = 3 + i * 2
            b_next_idx = 2 + ((i + 1) % radial_segments) * 2
            t_next_idx = 3 + ((i + 1) % radial_segments) * 2

            trisL.append([bottom_center_idx, b_next_idx, b_idx])
            trisL.append([top_center_idx, t_idx, t_next_idx])
            trisL.append([b_idx, b_next_idx, t_idx])
            trisL.append([b_next_idx, t_next_idx, t_idx])
        tris = torch.tensor(trisL, dtype=torch.long, device=DEVICE)
        return vertices, tris

    @classmethod
    def create_bone(cls, id: int, start_point: torch.Tensor, end_point: torch.Tensor, radius: float = 0.2):
        length = torch.norm(end_point - start_point)
        if length == 0: return cls(id=id, _vertices=torch.empty(0,3), tris=torch.empty(0,3, dtype=torch.long))

        # 1. Create cylinder part
        cyl_verts, cyl_tris = cls._create_cylinder_mesh(radius, float(length), radial_segments=12)

        # 2. Create sphere caps
        sphere1 = cls.create_sphere(id=-1, radius=radius, subdivisions=2)
        sphere2 = cls.create_sphere(id=-1, radius=radius, subdivisions=2)

        # Move sphere caps to the ends of the cylinder
        sphere1._vertices += torch.tensor([0, -length/2, 0], dtype=torch.float32, device=DEVICE)
        sphere2._vertices += torch.tensor([0, length/2, 0], dtype=torch.float32, device=DEVICE)

        # 3. Combine meshes
        num_cyl_verts = len(cyl_verts)
        num_s1_verts = len(sphere1.vertices)

        #s1_tris_offset = [(v[0] + num_cyl_verts, v[1] + num_cyl_verts, v[2] + num_cyl_verts) for v in sphere1.tris]
        #s2_tris_offset = [(v[0] + num_cyl_verts + num_s1_verts, v[1] + num_cyl_verts + num_s1_verts, v[2] + num_cyl_verts + num_s1_verts) for v in sphere2.tris]
        s1Tris = sphere1.tris + num_cyl_verts
        s2Tris = sphere2.tris + num_cyl_verts + num_s1_verts

        all_vertices = torch.cat([cyl_verts, sphere1.vertices, sphere2.vertices], dim=0)
        all_tris = torch.cat([cyl_tris, s1Tris, s2Tris], dim=0)

        # 4. Orient and translate the final mesh
        final_vertices = cls._orient_and_translate_mesh(all_vertices, start_point, end_point)
        midpoint = (start_point + end_point) / 2

        bone = cls(id=id, _vertices=final_vertices, tris=all_tris, position=midpoint)
        bone.color = torch.tensor([230, 230, 210, 255], dtype=torch.uint8)
        return bone

    @classmethod
    def create_joint(cls, id: int, center: torch.Tensor, radius: float, angle_limit_deg_in: float = 45.0):
        #radius: torch.Tensor = torch.tensor(radius_in)
        angle_limit_deg: torch.Tensor = torch.tensor(angle_limit_deg_in, dtype=torch.float32, device=DEVICE)
        joint_sphere = cls.create_sphere(id=-1, radius=radius, subdivisions=2, center=center)
        
        cone_height = radius * 2
        cone_radius = cone_height * torch.tan(torch.deg2rad(angle_limit_deg))
        cone_verts, cone_tris = cls._create_cylinder_mesh(float(cone_radius), cone_height, radial_segments=16)
        
        cone_verts[1] = cone_verts[0]
        for i in range(16):
            cone_verts[3 + i * 2] = cone_verts[0]

        cone1_verts = cone_verts.clone()
        cone2_verts = cone_verts.clone()
        
        # Orient and place cones
        cone1_verts = cls._orient_and_translate_mesh(
            cone1_verts, center, center + torch.tensor([cone_height,0,0], dtype=torch.float32, device=DEVICE) * 0.75
        )
        cone2_verts = cls._orient_and_translate_mesh(
            cone2_verts, center, center + torch.tensor([-cone_height,0,0], dtype=torch.float32, device=DEVICE) * 0.75
        )
        
        # 3. Combine all meshes
        num_sphere_verts = len(joint_sphere.vertices)
        num_cone1_verts = len(cone1_verts)
        
        # cone1_tris_offset = [(v[0] + num_sphere_verts, v[1] + num_sphere_verts, v[2] + num_sphere_verts) for v in cone_tris]
        # cone2_tris_offset = [(v[0] + num_sphere_verts + num_cone1_verts, v[1] + num_sphere_verts + num_cone1_verts, v[2] + num_sphere_verts + num_cone1_verts) for v in cone_tris]

        cone1Tris = cone_tris + num_sphere_verts
        cone2Tris = cone_tris + num_sphere_verts + num_cone1_verts
        
        all_vertices = torch.cat([joint_sphere.vertices, cone1_verts, cone2_verts], dim=0)
        all_tris = torch.cat([joint_sphere.tris, cone1Tris, cone2Tris], dim=0)
        
        joint = cls(id=id, _vertices=all_vertices, tris=all_tris)
        joint.color = torch.tensor([150, 150, 255, 100], dtype=torch.uint8) # Semi-transparent blue
        return joint

    @classmethod
    def create_node(cls, id: int, center: torch.Tensor, radii: torch.Tensor):
        """Creates a transparent ovular/ellipsoid shape."""
        # Start with a unit sphere
        node_sphere = cls.create_sphere(id=-1, radius=1.0, subdivisions=3)
        
        # Scale vertices to form an ellipsoid and translate to center
        node_sphere._vertices = node_sphere.vertices * radii + center
        
        node = cls(id=id, _vertices=node_sphere.vertices, tris=node_sphere.tris)
        node.color = torch.tensor([100, 200, 250, 120], dtype=torch.uint8) # Light blue, transparent
        return node
        
    @classmethod
    def create_muscle(cls, id: int, start_point: torch.Tensor, end_point: torch.Tensor, max_radius: float = 0.15, activation: float = 0.0):
        """Creates a spindle-shaped muscle that can change color with activation."""
        length = torch.norm(end_point - start_point)
        if length == 0: return cls(id=id, _vertices=torch.empty(0,3), tris=torch.empty(0,3, dtype=torch.long))

        num_segments = 12
        radial_segments = 10
        vertices = []
        tris = []

        # Create vertex rings along the muscle length (Y-axis)
        for i in range(num_segments + 1):
            t = i / num_segments
            # Spindle shape using a sine wave
            current_radius = max_radius * np.sin(np.pi * t)
            y = (t - 0.5) * length

            if i == 0 or i == num_segments: # Tip points
                vertices.append(torch.tensor([0, y, 0], dtype=torch.float32, device=DEVICE))
                continue
            
            for j in range(radial_segments):
                angle = 2 * np.pi * j / radial_segments
                x = current_radius * np.cos(angle)
                z = current_radius * np.sin(angle)
                vertices.append(torch.tensor([x, y, z], dtype=torch.float32, device=DEVICE))

        vertices = torch.stack(vertices)
        # Connect first ring to start tip
        triList = []
        start_tip_idx = 0
        for j in range(radial_segments):
            v1 = 1 + j
            v2 = 1 + (j + 1) % radial_segments
            triList.append([start_tip_idx, v2, v1])
        
        # Connect middle rings
        for i in range(num_segments - 2):
            ring_start_idx = 1 + i * radial_segments
            next_ring_start_idx = 1 + (i + 1) * radial_segments
            for j in range(radial_segments):
                v1 = ring_start_idx + j
                v2 = ring_start_idx + (j + 1) % radial_segments
                v3 = next_ring_start_idx + j
                v4 = next_ring_start_idx + (j + 1) % radial_segments
                triList.append([v1, v2, v3])
                triList.append([v2, v4, v3])

        # Connect last ring to end tip
        end_tip_idx = len(vertices) - 1
        last_ring_start_idx = 1 + (num_segments - 2) * radial_segments
        for j in range(radial_segments):
            v1 = last_ring_start_idx + j
            v2 = last_ring_start_idx + (j + 1) % radial_segments
            triList.append([end_tip_idx, v1, v2])
        
        tris = torch.tensor(triList, dtype=torch.long, device=DEVICE)
        final_vertices = cls._orient_and_translate_mesh(vertices, start_point, end_point)

        muscle = cls(id=id, _vertices=final_vertices, tris=tris)
        
        # Color based on activation (lerp between blue/relaxed and red/contracted)
        red = int(200 * activation + 50 * (1 - activation))
        blue = int(50 * activation + 200 * (1 - activation))
        muscle.color = torch.tensor([red, 80, blue, 255], dtype=torch.uint8)
        return muscle

    def project_2d(self, eye: torch.Tensor, lookat: torch.Tensor, up: torch.Tensor, fov: float = 90,
                    res: tuple[int,int] = (800,600), near: float = 1.0, far: float = 1000) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        worldverts = self.vertices

        zAxis: torch.Tensor = lookat - eye
        zAxis = zAxis / torch.norm(zAxis)
        xAxis: torch.Tensor = torch.linalg.cross(up, zAxis)
        xAxis = xAxis / torch.norm(xAxis)
        yAxis: torch.Tensor = torch.linalg.cross(zAxis, xAxis)

        viewMatrix = torch.eye(4)
        viewMatrix[:3, 0] = xAxis
        viewMatrix[:3, 1] = yAxis
        viewMatrix[:3, 2] = -zAxis
        viewMatrix[:3, 3] = eye
        viewMatrix = torch.inverse(viewMatrix)

        aspectRatio = res[0] / res[1]
        fovRad = np.radians(fov)
        tanhalf = np.tan(fovRad / 2.0)
        f = 1.0 / tanhalf

        projMatrix = torch.zeros((4,4))
        projMatrix[0,0] = 1.0 / (aspectRatio * tanhalf)
        projMatrix[1,1] = f
        projMatrix[2,2] = -(far + near) / (near - far)
        projMatrix[2,3] = (-2 * far * near) / (near - far)
        projMatrix[3,2] = -1.0
        
        homogenousVerts = torch.cat([worldverts, torch.ones((worldverts.shape[0], 1))], dim=1)
        viewProjMatrix = torch.matmul(projMatrix, viewMatrix)
        projVerts = torch.matmul(homogenousVerts, viewProjMatrix.T)
        projVerts = projVerts / projVerts[:, 3].unsqueeze(1)
        
        screenVerts = torch.empty_like(projVerts[:, :2])
        screenVerts[:, 0] = (projVerts[:, 0] + 1) * 0.5 * res[0]
        screenVerts[:, 1] = (1 - (projVerts[:, 1] + 1) * 0.5) * res[1]

        viewVerts = torch.matmul(homogenousVerts, viewMatrix.T)[:, :3]
        triNorms = torch.cross(
            worldverts[self.tris[:, 1]] - worldverts[self.tris[:, 0]],
            worldverts[self.tris[:, 2]] - worldverts[self.tris[:, 0]]
        )
        viewDirs = (worldverts[self.tris[:, 0]] - eye).unsqueeze(1)
        dotProds = torch.bmm(triNorms.unsqueeze(1), viewDirs).squeeze()
        visibleMask = dotProds < 0
        visibleTris = self.tris[visibleMask]

        if len(visibleTris) > 0:
            visibleDepths = torch.mean(viewVerts[visibleTris], dim=1)
        else:
            visibleDepths = torch.empty(0, dtype=torch.float32, device=DEVICE)
        
        return screenVerts, visibleTris, visibleDepths

def resolve_collision(block_a: block, block_b: block, normal: torch.Tensor, penetration: float):
    if (not block_a.physics and not block_b.physics) or (block_a.mass <= 0 and block_b.mass <= 0):
        return
        
    inv_mass_a = 1.0 / block_a.mass if block_a.mass > 0 and block_a.physics else 0.0
    inv_mass_b = 1.0 / block_b.mass if block_b.mass > 0 and block_b.physics else 0.0
    total_inv_mass = inv_mass_a + inv_mass_b
    if total_inv_mass == 0: return

    correction = normal * max(penetration - 0.01, 0.0) / total_inv_mass
    block_a.position += correction * inv_mass_a
    block_b.position -= correction * inv_mass_b
        
    min_a = torch.min(block_a.vertices, dim=0).values; max_a = torch.max(block_a.vertices, dim=0).values
    min_b = torch.min(block_b.vertices, dim=0).values; max_b = torch.max(block_b.vertices, dim=0).values
    contact_point = (torch.max(min_a, min_b) + torch.min(max_a, max_b)) / 2.0
    
    r_a = contact_point - block_a.position
    r_b = contact_point - block_b.position
    
    v_a_contact = block_a.velocity + torch.linalg.cross(block_a.angular_velocity, r_a)
    v_b_contact = block_b.velocity + torch.linalg.cross(block_b.angular_velocity, r_b)
    relative_velocity = v_a_contact - v_b_contact
    
    vel_along_normal = torch.dot(relative_velocity, normal)
    if vel_along_normal > 0: return # Objects are already separating

    e = torch.min(block_a.restitution, block_b.restitution)
    
    # Calculate world-space inverse inertia tensors
    I_inv_a = torch.matmul(torch.matmul(block_a.orientation, block_a.inertia_tensor_inv), block_a.orientation.T) if inv_mass_a > 0 else torch.zeros_like(block_a.orientation)
    I_inv_b = torch.matmul(torch.matmul(block_b.orientation, block_b.inertia_tensor_inv), block_b.orientation.T) if inv_mass_b > 0 else torch.zeros_like(block_b.orientation)
    
    # Calculate impulse magnitude (j)
    term3 = torch.dot(torch.linalg.cross(torch.matmul(I_inv_a, torch.linalg.cross(r_a, normal)), r_a), normal) if inv_mass_a > 0 else 0
    term4 = torch.dot(torch.linalg.cross(torch.matmul(I_inv_b, torch.linalg.cross(r_b, normal)), r_b), normal) if inv_mass_b > 0 else 0
    denominator = total_inv_mass + term3 + term4
    if denominator < 1e-6: return
        
    j = -(1.0 + e) * vel_along_normal / denominator
    impulse_vec = j * normal
    
    # Apply impulse to update linear and angular velocities
    if block_a.physics and block_a.mass > 0:
        block_a.velocity += inv_mass_a * impulse_vec
        block_a.angular_velocity += torch.matmul(I_inv_a, torch.linalg.cross(r_a, impulse_vec))
        
    if block_b.physics and block_b.mass > 0:
        block_b.velocity -= inv_mass_b * impulse_vec
        block_b.angular_velocity -= torch.matmul(I_inv_b, torch.linalg.cross(r_b, impulse_vec))

class BlockRenderer:
    def __init__(self):
        self.blocks: list[block] = []
        self.lookat = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
        self.currentResolution: tuple[int,int] = (800,600)
        self.currentID = 0

        self.eye = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32, device=DEVICE)
        self.up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
        self.orbitSpeed = 0.5
        self.orbitRadius = 8.0
        self.orbitAngles = [np.pi/4, np.pi/4, 0.0]
        self.orbitEnable = [False, False, False]
        
        ground = block.create_ground(id=self.idCounter(), size=50)
        self.add_block(ground)

        hip_joint = block.create_joint(id=self.idCounter(), center=torch.tensor([0, 2.5, 0], dtype=torch.float32, device=DEVICE), radius=0.3)
        bone1_start = torch.tensor([0, 2.5, 0], dtype=torch.float32, device=DEVICE)
        bone1_end = torch.tensor([1, 1, 0], dtype=torch.float32, device=DEVICE)
        femur = block.create_bone(id=self.idCounter(), start_point=bone1_start, end_point=bone1_end, radius=0.15)
        
        knee_joint = block.create_joint(id=self.idCounter(), center=bone1_end, radius=0.2, angle_limit_deg_in=25)
        
        bone2_end = torch.tensor([1, 0.2, 0.5], dtype=torch.float32, device=DEVICE)
        tibia = block.create_bone(id=self.idCounter(), start_point=bone1_end, end_point=bone2_end, radius=0.12)

        # Muscles connecting the bones
        muscle1 = block.create_muscle(id=self.idCounter(), start_point=bone1_start + torch.tensor([-0.2,0.2,0], dtype=torch.float32, device=DEVICE), end_point=bone1_end + torch.tensor([0,0.2,0]), max_radius=0.1, activation=0.8)
        muscle2 = block.create_muscle(id=self.idCounter(), start_point=bone1_start + torch.tensor([0.2,-0.2,0], dtype=torch.float32, device=DEVICE), end_point=bone2_end, max_radius=0.1, activation=0.2)
        
        # A floating node
        node1 = block.create_node(id=self.idCounter(), center=torch.tensor([-2, 2, -1], dtype=torch.float32, device=DEVICE), radii=torch.tensor([0.5, 1.0, 0.5], dtype=torch.float32, device=DEVICE))

        self.add_block(hip_joint)
        self.add_block(femur)
        self.add_block(knee_joint)
        self.add_block(tibia)
        self.add_block(muscle1)
        self.add_block(muscle2)
        self.add_block(node1)
        
        self.target_fps = 60
        self.frame_time_target = 1.0 / self.target_fps
        self.last_frame_time = time.time()

    def idCounter(self):
        self.currentID += 1
        return self.currentID
        
    def add_block(self, block):
        self.blocks.append(block)
        
    def render(self):
        dpg.create_context()
        dpg.create_viewport(title='Creature sim', width=800, height=600, resizable=True)
        
        with dpg.window(label="3D View", tag="mainView"):
            with dpg.drawlist(width=-1, height=-1, tag="drawlist"):
                pass
        
        with dpg.window(label="Camera Controls", width=300,height=200):
            dpg.add_text('Orbit Controls')
            dpg.add_checkbox(label='Orbit X (Azimuth)', tag="orbit_x", callback=lambda: self.toggle_orbit(0))
            dpg.add_checkbox(label='Orbit Y (Elevation)', tag="orbit_y", callback=lambda: self.toggle_orbit(1))
            dpg.add_checkbox(label='Orbit Z (Roll)', tag="orbit_z", callback=lambda: self.toggle_orbit(2))
            dpg.add_slider_float(label="Orbit Speed", tag="orbitspeed", min_value=0.1, max_value=2.0, default_value=self.orbitSpeed, callback=self.update_orbit_speed)
            dpg.add_slider_float(label="Orbit Radius", tag="orbitradius", min_value=1.0, max_value=20.0, default_value=self.orbitRadius, callback=self.update_orbit_radius)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("mainView", True)
        dpg.set_viewport_resize_callback(callback=self._on_resize)
        
        while dpg.is_dearpygui_running():
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if elapsed >= self.frame_time_target:
                self.update_camera_position()
                self.updatePhysics()
                self.render_frame()
                dpg.render_dearpygui_frame()
                frame_time = time.time() - current_time
                sleep_time = max(0, self.frame_time_target - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.last_frame_time = current_time
            else:
                dpg.render_dearpygui_frame()
        dpg.destroy_context()
        
    def toggle_orbit(self, axis):
        self.orbitEnable[axis] = not self.orbitEnable[axis]

    def update_orbit_speed(self):
        self.orbitSpeed = dpg.get_value("orbitspeed")

    def update_orbit_radius(self):
        self.orbitRadius = dpg.get_value("orbitradius")

    def update_camera_position(self):
        angle_step = self.orbitSpeed / self.target_fps

        for i in range(3):
            if self.orbitEnable[i]:
                self.orbitAngles[i] = (self.orbitAngles[i] + angle_step) % (2 * np.pi)
        
        xAngle, yAngle, zAngle = self.orbitAngles
        yAngle = np.clip(yAngle, -np.pi / 2.0 + 1e-6, np.pi / 2.0 - 1e-6)
        self.orbitAngles[1] = yAngle

        eye_x = self.orbitRadius * np.cos(yAngle) * np.sin(xAngle)
        eye_y = self.orbitRadius * np.sin(yAngle)
        eye_z = self.orbitRadius * np.cos(yAngle) * np.cos(xAngle)
        self.eye = torch.tensor([eye_x, eye_y, eye_z], dtype=torch.float32, device=DEVICE) + self.lookat

        forward = self.lookat - self.eye
        forward = forward / torch.norm(forward)

        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
        right = torch.linalg.cross(forward, world_up)
        right = right / torch.norm(right)
        
        true_up = torch.linalg.cross(right, forward)

        cos_roll = np.cos(zAngle)
        sin_roll = np.sin(zAngle)
        
        self.up = true_up * cos_roll + right * sin_roll

    def updatePhysics(self):
        delta_time = self.frame_time_target
        if delta_time <= 0: return
        sub_dt = delta_time / PHYSICSSTEPMULT
        for _ in range(PHYSICSSTEPMULT):
            for b in self.blocks:
                b.apply_gravity(sub_dt)
            for i in range(len(self.blocks)):
                for j in range(i + 1, len(self.blocks)):
                    block_a = self.blocks[i]
                    block_b = self.blocks[j]
                    if block_a.mass == 0 and block_b.mass == 0:
                        continue
                    collided, normal, penetration = block_a.collision(block_b)
                    if collided:
                        resolve_collision(block_a, block_b, torch.tensor(normal, dtype=torch.float32, device=DEVICE), torch.tensor(penetration, dtype=torch.float32, device=DEVICE))

    def _on_resize(self):
        windowWidth = dpg.get_item_width("mainView") or 1
        windowHeight = dpg.get_item_height("mainView") or 1
        dpg.configure_item("drawlist", width=windowWidth, height=windowHeight)
        self.currentResolution = (windowWidth, windowHeight)

    def render_frame(self):
        dpg.delete_item("drawlist", children_only=True)
        triangles_to_draw = []

        # Separate opaque and transparent objects for correct rendering
        opaque_triangles = []
        transparent_triangles = []

        for block in self.blocks:
            # Simple check for transparency
            is_transparent = block.color[3] < 255

            screen_verts, visible_tris, depths = block.project_2d(
                eye=self.eye,
                lookat=self.lookat,
                up=self.up,
                res=self.currentResolution,
                near=0.1,
                far=10000
            )
            
            screen_verts_np = screen_verts.cpu().numpy()
            color_np = block.color.cpu().numpy().tolist()
            depths_np = depths.cpu().numpy()
            
            for i, tri in enumerate(visible_tris):
                v0 = screen_verts_np[tri[0]]
                v1 = screen_verts_np[tri[1]]
                v2 = screen_verts_np[tri[2]]
                
                triangle_data = {
                    'points': [(v0[0], v0[1]), (v1[0], v1[1]), (v2[0], v2[1])],
                    'color': color_np,
                    'depth': depths_np[i]
                }
                
                if is_transparent:
                    transparent_triangles.append(triangle_data)
                else:
                    opaque_triangles.append(triangle_data)

        # Sort opaque triangles from front-to-back (painter's algorithm)
        opaque_triangles.sort(key=lambda x: x['depth'], reverse=True)
        # Sort transparent triangles from back-to-front
        transparent_triangles.sort(key=lambda x: x['depth'], reverse=False)

        # Combine lists: draw all opaque first, then all transparent
        all_triangles = opaque_triangles + transparent_triangles
  
        with dpg.draw_node(parent="drawlist"):
            for triangle in all_triangles:
                dpg.draw_triangle(
                    p1=triangle['points'][0], p2=triangle['points'][1], p3=triangle['points'][2],
                    color=triangle['color'],
                    fill=triangle['color']
                )
        
if __name__ == "__main__":
    renderer = BlockRenderer()
    renderer.render()