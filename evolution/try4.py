from dataclasses import dataclass, field
import datetime
import sys
from typing import Optional
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader
from dearpygui import dearpygui as dpg

# --- Helper functions for 3D Math (keep the same as before) ---

def dot(v1, v2):
    return np.dot(v1, v2)

def rotate_vector_by_euler(vec: np.ndarray, euler_angles_rad: np.ndarray) -> np.ndarray:
    """Rotates a 3D vector by Euler angles (rx, ry, rz in radians)."""
    rx, ry, rz = euler_angles_rad
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx @ vec

def create_perspective_matrix(fov: float, aspect_ratio: float, near: float, far: float) -> np.ndarray:
    """Creates a perspective projection matrix."""
    f = 1.0 / np.tan(fov * 0.5)
    return np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

def create_fps_matrix(position: np.ndarray, pitch: float, yaw: float) -> np.ndarray:
    """Creates a first-person view matrix."""
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    Ry = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    R = Rx @ Ry
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ position
    return T

# --- Constants and Classes (keep the same as before) ---
MAX_FLOAT = sys.float_info.max
GROUND_HEIGHT = 0.0
BONECOLOR = np.array([255, 255, 255]) / 255.0
JOINTCOLOR = np.array([0, 255, 0]) / 255.0
MUSCLECOLOR = np.array([255, 0, 0]) / 255.0
NODECOLOR = np.array([0, 0, 255]) / 255.0

@dataclass
class SolidObject:
    """Represents a static object in the environment that can collide with creatures."""
    id: int
    vertices: list[np.ndarray]  # List of 3D points
    triangles: list[tuple]      # List of vertex indices forming triangles
    color: np.ndarray = field(default_factory=lambda: np.array([200, 200, 200]))
    
    @classmethod
    def create_ground_plane(cls, id: int, size: float = MAX_FLOAT, height: float = GROUND_HEIGHT):
        """Creates an infinite ground plane at y=height."""
        # For visualization, we'll create a large but finite plane
        visual_size = size if size < 1000 else 1000  # Limit visual size
        half_size = visual_size / 2
        
        vertices = [
            np.array([-half_size, height, -half_size]),
            np.array([half_size, height, -half_size]),
            np.array([half_size, height, half_size]),
            np.array([-half_size, height, half_size])
        ]
        
        triangles = [(0, 1, 2), (0, 2, 3)]
        return cls(id, vertices, triangles, np.array([150, 150, 150]))  # Gray color

    def get_aabb(self) -> tuple:
        """Returns the axis-aligned bounding box as (min, max) points."""
        if not self.vertices:
            return (np.array([0,0,0]), np.array([0,0,0]))
        
        min_coords = np.array([MAX_FLOAT, MAX_FLOAT, MAX_FLOAT])
        max_coords = np.array([-MAX_FLOAT, -MAX_FLOAT, -MAX_FLOAT])
        
        for v in self.vertices:
            min_coords = np.minimum(min_coords, v)
            max_coords = np.maximum(max_coords, v)
            
        return min_coords, max_coords

@dataclass 
class Bone:
    # bone cant change structure. its position and rotation can be changed by muscles
    id: int
    length: float | np.floating = field(default=10.0, metadata={"sigma": 0.1})
    thickness: float = field(default=1.0, metadata={"sigma": 0.01})
    density: float = field(default=1.9)
    youngModulus: float = field(default=10000.0)
    maxCompression: float = field(default=170.0)
    maxTension: float = field(default=120.0)

    @property
    def mass(self):
        volume = np.pi * (self.thickness * 100 / 2) ** 2 * self.length * 100
        return self.density * volume / 1000
    
    @property
    def stiffness(self):
        crossSection = np.pi * (self.thickness * 1000 / 2) ** 2
        return (self.youngModulus * crossSection) / (self.length * 1000)
    
    def calculate_stress(self, force: float):
        crossSection = np.pi * (self.thickness * 1000 / 2) ** 2
        return force / crossSection
    
    def canWithstand(self, force: float):
        stress = self.calculate_stress(abs(force))
        return stress < (self.maxCompression if force < 0 else self.maxTension)

@dataclass
class Joint:
    id: int
    bone1: int # id of bone
    bone2: int # id of bone
    rotation: float = field(default=0.0)
    flexion: float = field(default=0.0)
    abduction: float = field(default=0.0)
    stiffness: float = field(default=100.0) # newton meters per radian
    damping: float = field(default=5.0) # newton meter seconds per radian
    friction: float = field(default=0.1)
    limits: dict = field(default_factory=lambda: {'flexion': (-30, 20), 'abduction': (-20, 10), 'rotation': (-10, 10)})

    def rotate(self, flexionDelta: float=0, abductionDelta: float=0, rotationDelta: float=0):
        self.flexion = np.clip(self.flexion + flexionDelta, *self.limits['flexion'])
        self.abduction = np.clip(self.abduction + abductionDelta, *self.limits['abduction'])
        self.rotation = np.clip(self.rotation + rotationDelta, *self.limits['rotation'])
        
    def get_transform_matrix(self) -> np.ndarray:
        flex_rad = np.radians(self.flexion)
        abd_rad = np.radians(self.abduction)
        rot_rad = np.radians(self.rotation)
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(flex_rad), -np.sin(flex_rad)],
            [0, np.sin(flex_rad), np.cos(flex_rad)]
        ])
        
        Ry = np.array([
            [np.cos(abd_rad), 0, np.sin(abd_rad)],
            [0, 1, 0],
            [-np.sin(abd_rad), 0, np.cos(abd_rad)]
        ])
        
        Rz = np.array([
            [np.cos(rot_rad), -np.sin(rot_rad), 0],
            [np.sin(rot_rad), np.cos(rot_rad), 0],
            [0, 0, 1]
        ])
        
        # Changed np.matmul to @ operator for consistency with other numpy matrix multiplications
        return Rz @ Ry @ Rx 
    
    def calculate_restoring_torque(self):
        angles = np.radians([self.flexion, self.abduction, self.rotation])
        return -self.stiffness * angles

@dataclass
class Muscle:
    # muscle does most of the effort here, moving bones, joints, etc.
    id: int
    maxForce: float = field(default=300.0) # in newtons
    optimalLength: float | np.floating = field(default=0.3) # in meters
    tendenRatio: float = field(default=0.1) # percent ratio for tendon length

    activation: float = field(default=0.0)
    excitation: float = field(default=0.0)
    activationTime: float = field(default=0.05) #seconds to reach activation
    deactivationTime: float = field(default=0.02)
    length: float = field(init=False)
    velocity: float = field(default=0.0)
    stiffness: float = field(default=50.0) # in newtons per meter
    damping: float = field(default=2.0) # in newton second per meter

    def __post_init__(self):
        self.length = float(self.restLength)

    @property
    def restLength(self):
        return self.optimalLength * (1 + self.tendenRatio)
    
    def step(self, excitation: float, dt: float):
        self.excitation = np.clip(excitation, 0, 1)
        self._update_activation(dt)

    def _update_activation(self, dt: float):
        target = self.excitation
        if target > self.activation:
            tau = self.activationTime / 3.0
        else:
            tau = self.deactivationTime / 3.0
        self.activation += (target - self.activation) * dt / tau

    def compute_force(self, newLength: float, dt: float):
        self.velocity = (newLength - self.length) / dt
        self.length = newLength
        normLength = self.length / self.restLength
        fl = np.exp(-2.5 * (normLength - 1.0) ** 2)
        if self.velocity < 0:
            fv = 1.4 / (1 - self.velocity / self.restLength)
        else:
            fv = 1.8 - 0.8/(1 + self.velocity / self.restLength)
        activeForce = self.activation * self.maxForce * fl * fv
        
        strain = (self.length - self.restLength) / self.restLength
        passiveForce = self.stiffness * strain + self.damping * self.velocity
        return np.clip(activeForce + passiveForce, 0, self.maxForce * 1.5)
    
    @property
    def minLength(self):
        return self.optimalLength * 0.6 + self.tendenRatio
    
    @property
    def tendonLength(self):
        return self.optimalLength * self.tendenRatio
    
@dataclass
class Node:
    # node is a "holder" for multiple joints, muscles, or bones. the primary goal is to be able to store future parts of the creature such as organs.
    # these future parts do not matter until we can get the creature mobile at all.
    id: int
    connected_bones: list[int] = field(default_factory=list)  # IDs of connected bones
    connected_joints: list[int] = field(default_factory=list)  # IDs of connected joints
    connected_muscles: list[int] = field(default_factory=list)  # IDs of connected muscles
    
    def add_connection(self, elementType: type, elementID: int):
        if elementType == Bone:
            if elementID not in self.connected_bones:
                self.connected_bones.append(elementID)
        elif elementType == Joint:
            if elementID not in self.connected_joints:
                self.connected_joints.append(elementID)
        elif elementType == Muscle:
            if elementID not in self.connected_muscles:
                self.connected_muscles.append(elementID)

@dataclass
class Creature:
    # nodes, muscles, joints, and bones are defined in this. 
    # the position of those objects will be held in a numpy array and relative to a fixed point in the creature,
    # which will define the creatures position in the world
    id: int
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angularVelocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bones: dict[int, dict] = field(default_factory=dict)
    joints: dict[int, dict] = field(default_factory=dict)
    muscles: dict[int, dict] = field(default_factory=dict)
    nodes: dict[int, dict] = field(default_factory=dict)

    mass: float = 0.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3))
    damping: float = 0.1  # Linear damping
    angular_damping: float = 0.1  # Angular damping
    restitution: float = 0.2 

    def __post_init__(self):
        self._update_mass_properties()
    
    def _update_mass_properties(self):
        self.mass = sum(bd['bone'].mass for bd in self.bones.values())
        # Simplified inertia calculation
        total_inertia = np.zeros((3, 3))
        for bone_data in self.bones.values():
            bone = bone_data['bone']
            # Approximate each bone as a rod rotating about its center
            I_xx = (1/12) * bone.mass * bone.length**2
            total_inertia += np.diag([I_xx, I_xx, I_xx])
        self.inertia = total_inertia

    def add_bone(self, bone: Bone, pos: np.ndarray, node1_id: int, node2_id: int, rotation: np.ndarray = np.array([0,0,0])):
        if bone.id in self.bones:
            raise ValueError(f"Bone with ID {bone.id} already exists")
            
        self.bones[bone.id] = {
            'pos': pos,
            'bone': bone,
            'nodes': [node1_id, node2_id],
            'rotation': rotation  # Euler angles
        }
        
        # Update node connections
        if node1_id in self.nodes:
            self.nodes[node1_id]['node'].connected_bones.append(bone.id)
        if node2_id in self.nodes:
            self.nodes[node2_id]['node'].connected_bones.append(bone.id)
            
        self._update_mass_properties()
    
    def add_joint(self, joint: Joint, pos: np.ndarray):
        if joint.id in self.joints:
            raise ValueError(f"Joint with ID {joint.id} already exists")
            
        self.joints[joint.id] = {
            'pos': pos,
            'joint': joint,
            'connected_bones': [joint.bone1, joint.bone2]
        }
    
    def add_muscle(self, muscle: Muscle, attachment_positions: list[np.ndarray], attachment_nodes: list[int]):
        if muscle.id in self.muscles:
            raise ValueError(f"Muscle with ID {muscle.id} already exists")
            
        if len(attachment_positions) < 2:
            raise ValueError("Muscle needs at least 2 attachment points")
            
        self.muscles[muscle.id] = {
            'muscle': muscle,
            'attachments': attachment_positions,
            'nodes': attachment_nodes
        }
        
        for node_id in attachment_nodes:
            if node_id in self.nodes:
                self.nodes[node_id]['node'].connected_muscles.append(muscle.id)
    
    def add_node(self, node: Node, pos: np.ndarray):
        if node.id in self.nodes:
            raise ValueError(f"Node with ID {node.id} already exists")
            
        self.nodes[node.id] = {
            'pos': pos,
            'node': node
        }
    
    def _get_creature_model_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = self.position
        rx, ry, rz = self.rotation
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        rotationMatrix = Rz @ Ry @ Rx
        return T @ rotationMatrix
    
    def get_global_position(self, local_pos: np.ndarray) -> np.ndarray:
        """Converts a local position to world space coordinates."""
        local_pos_hom = np.append(local_pos, 1.0)  # Convert to homogeneous coordinates
        model_matrix = self._get_creature_model_matrix()
        global_pos_hom = model_matrix @ local_pos_hom
        return global_pos_hom[:3]  # Convert back to 3D coordinates
    
    def step(self, dt: float, excitations: Optional[dict[int, float]] = None):
        excitations = excitations or {}
        
        for muscle_id, excitation in excitations.items():
            if muscle_id in self.muscles:
                self.muscles[muscle_id]['muscle'].step(excitation, dt)
        muscle_forces = self._calculate_muscle_forces(dt)
        self._apply_forces(muscle_forces, dt)
        self._update_physics(dt)
    
    def _calculate_muscle_forces(self, dt: float) -> dict[int, float]:
        forces = {}
        for muscle_id, muscle_data in self.muscles.items():
            muscle: Muscle = muscle_data['muscle']
            # Calculate current length based on attachment points
            start_pos_world = self.get_global_position(muscle_data['attachments'][0])
            end_pos_world = self.get_global_position(muscle_data['attachments'][1])
            current_length = np.linalg.norm(end_pos_world - start_pos_world)
            # Compute force based on length and activation
            force = muscle.compute_force(float(current_length), dt)
            forces[muscle_id] = force
        return forces
    
    def _apply_forces(self, muscle_forces: dict[int, float], dt: float):
        for muscle_id, force in muscle_forces.items():
            muscle_data = self.muscles[muscle_id]
            direction = muscle_data['attachments'][1].astype(float) - muscle_data['attachments'][0].astype(float)
            direction /= np.linalg.norm(direction)
            
            # Apply force to connected bones/nodes
            for node_id in muscle_data['nodes']:
                if node_id in self.nodes:
                    pass
            
    def calculate_3d_coordinates(self) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[tuple[int, int, int, str]]]:
        """
        Calculates the 3D coordinates of all vertices in the creature's structure and
        returns a tuple containing:
        1. A dictionary mapping vertex IDs to their (3D coordinates, color)
        2. A list of triangles (each as a tuple of 3 vertex IDs and component type)
        """
        vertices = {}
        triangles = []
        vertex_id_counter = 0
        
        # Helper function to add a vertex and return its ID
        def add_vertex(pos: np.ndarray, color: np.ndarray) -> int:
            nonlocal vertex_id_counter
            vertices[vertex_id_counter] = (pos.copy(), color.copy())  # Ensure we store copies
            vertex_id_counter += 1
            return vertex_id_counter - 1
        
        # Process bones (represented as cylinders)
        for bone_id, bone_data in self.bones.items():
            bone = bone_data['bone']
            pos = bone_data['pos']
            rot = bone_data['rotation']
            
            # Bone direction vector (before rotation)
            bone_dir = np.array([0, bone.length, 0])
            
            # Rotate the bone direction
            rotated_dir = rotate_vector_by_euler(bone_dir, np.radians(rot))
            
            # Calculate start and end points
            start_point = pos
            end_point = pos + rotated_dir
            
            # Add vertices for the bone cylinder
            cylinder_resolution = 8
            radius = bone.thickness / 2
            
            # Create vertices around start and end points
            start_vertex_ids = []
            end_vertex_ids = []
            
            # Calculate perpendicular vectors for cylinder cross-section
            if np.allclose(rotated_dir, [0, 0, 1]):
                perp1 = np.array([1, 0, 0])
            else:
                perp1 = np.linalg.norm(np.cross(rotated_dir, np.array([0, 0, 1])))
            perp2 = np.linalg.norm(np.cross(rotated_dir, perp1))
            
            for i in range(cylinder_resolution):
                angle = 2 * np.pi * i / cylinder_resolution
                offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                
                # Start circle vertices
                start_vert = start_point + offset
                start_id = add_vertex(start_vert, BONECOLOR)
                start_vertex_ids.append(start_id)
                
                # End circle vertices
                end_vert = end_point + offset
                end_id = add_vertex(end_vert, BONECOLOR)
                end_vertex_ids.append(end_id)
            
            # Create triangles for the cylinder sides
            for i in range(cylinder_resolution):
                next_i = (i + 1) % cylinder_resolution
                # Two triangles per side segment
                triangles.append((start_vertex_ids[i], end_vertex_ids[i], end_vertex_ids[next_i], 'bone'))
                triangles.append((start_vertex_ids[i], end_vertex_ids[next_i], start_vertex_ids[next_i], 'bone'))
            
            # Create triangles for the end caps
            center_start = add_vertex(start_point, BONECOLOR)
            center_end = add_vertex(end_point, BONECOLOR)
            for i in range(cylinder_resolution):
                next_i = (i + 1) % cylinder_resolution
                # Start cap
                triangles.append((center_start, start_vertex_ids[next_i], start_vertex_ids[i], 'bone'))
                # End cap
                triangles.append((center_end, end_vertex_ids[i], end_vertex_ids[next_i], 'bone'))
        
        # Process joints (represented as spheres)
        for joint_id, joint_data in self.joints.items():
            pos = joint_data['pos']
            joint = joint_data['joint']
            
            # Use the average thickness of connected bones for sphere size
            bone1_thickness = self.bones.get(joint.bone1, {}).get('bone', Bone(0)).thickness
            bone2_thickness = self.bones.get(joint.bone2, {}).get('bone', Bone(0)).thickness
            radius = max(bone1_thickness, bone2_thickness) * 0.8
            
            # Create a sphere with 3 levels of subdivision (42 vertices)
            # Using a simple icosphere approach
            t = (1.0 + np.sqrt(5.0)) / 2.0
            vertices_pos = [
                [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
            ]
            
            # Normalize and scale vertices
            base_vertex_ids = []
            for v in vertices_pos:
                v_norm = np.linalg.norm(np.array(v))
                vertex_pos = pos + v_norm * radius
                base_vertex_ids.append(add_vertex(vertex_pos, JOINTCOLOR))
            
            # Base icosahedron triangles
            sphere_tris = [
                (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
                (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
                (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
            ]
            
            # Add base triangles
            for tri in sphere_tris:
                triangles.append((
                    base_vertex_ids[tri[0]],
                    base_vertex_ids[tri[1]],
                    base_vertex_ids[tri[2]],
                    'joint'
                ))
        
        # Process muscles (represented as tapered cylinders between attachment points)
        for muscle_id, muscle_data in self.muscles.items():
            if len(muscle_data['attachments']) < 2:
                continue
                
            start_pos = muscle_data['attachments'][0]
            end_pos = muscle_data['attachments'][1]
            direction = end_pos - start_pos
            
            # Muscle thickness based on max force (visual only)
            thickness = 0.5 + (muscle_data['muscle'].maxForce / 300.0) * 0.5
            
            # Create a tapered cylinder (cone) between points
            cylinder_resolution = 8
            start_vertex_ids = []
            end_vertex_ids = []
            
            # Calculate perpendicular vectors
            if np.allclose(direction, [0, 0, 1]):
                perp1 = np.array([1, 0, 0])
            else:
                perp1 = np.linalg.norm(np.cross(direction, np.array([0, 0, 1])))
            perp2 = np.linalg.norm(np.cross(direction, perp1))
            
            for i in range(cylinder_resolution):
                angle = 2 * np.pi * i / cylinder_resolution
                offset_start = (thickness * 0.8) * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                offset_end = (thickness * 0.5) * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                
                # Start circle vertices
                start_vert = start_pos + offset_start
                start_id = add_vertex(start_vert, MUSCLECOLOR)
                start_vertex_ids.append(start_id)
                
                # End circle vertices
                end_vert = end_pos + offset_end
                end_id = add_vertex(end_vert, MUSCLECOLOR)
                end_vertex_ids.append(end_id)
            
            # Create triangles for the muscle body
            for i in range(cylinder_resolution):
                next_i = (i + 1) % cylinder_resolution
                # Two triangles per side segment
                triangles.append((start_vertex_ids[i], end_vertex_ids[i], end_vertex_ids[next_i], 'muscle'))
                triangles.append((start_vertex_ids[i], end_vertex_ids[next_i], start_vertex_ids[next_i], 'muscle'))
        
        return vertices, triangles
        
    @classmethod
    def generate_random_creature(cls, id: int = 1, 
                                num_bones: int = 3, 
                                num_joints: int = 2, 
                                num_muscles: int = 2) -> 'Creature':
        """
        Generates a random creature with the specified number of components.
        
        Args:
            id: ID for the creature
            num_bones: Number of bones to generate (minimum 1)
            num_joints: Number of joints to generate (minimum 1)
            num_muscles: Number of muscles to generate (minimum 1)
            
        Returns:
            A new Creature instance with randomly generated components
        """
        # Ensure minimum counts
        num_bones = max(1, num_bones)
        num_joints = max(1, num_joints)
        num_muscles = max(1, num_muscles)
        
        creature = cls(id=id)
        
        # Generate nodes first (one more node than bones)
        nodes = []
        for i in range(num_bones + 1):
            node_id = i + 1
            # Position nodes in a roughly vertical line with some random variation
            pos = np.array([
                np.random.uniform(-0.5, 0.5),
                i * 1.5,
                np.random.uniform(-0.5, 0.5)
            ])
            node = Node(id=node_id)
            creature.add_node(node, pos)
            nodes.append((node_id, pos))
        
        # Generate bones between consecutive nodes
        bones = []
        for i in range(num_bones):
            bone_id = i + 1
            node1_id, node1_pos = nodes[i]
            node2_id, node2_pos = nodes[i + 1]
            
            # Calculate length from node positions
            length = np.linalg.norm(node2_pos - node1_pos)
            
            bone = Bone(
                id=bone_id,
                length=float(length),
                thickness=np.random.uniform(0.2, 0.5),
                density=np.random.uniform(1.5, 2.5),
                youngModulus=np.random.uniform(8000, 12000),
                maxCompression=np.random.uniform(150, 200),
                maxTension=np.random.uniform(100, 150)
            )
            
            # Random rotation (slightly tilted)
            rotation = np.array([
                np.random.uniform(-15, 15),
                np.random.uniform(-15, 15),
                np.random.uniform(-15, 15)
            ])
            
            creature.add_bone(bone, node1_pos, node1_id, node2_id, rotation)
            bones.append((bone_id, node1_id, node2_id))
        
        # Generate joints between random bones
        for i in range(num_joints):
            joint_id = i + 1
            
            # Select two random bones that share a node
            rid = np.random.choice(len(bones))
            bone1_id, bone1_node1, bone1_node2 = bones[rid]
            possible_connected = [b for b in bones if b[1] == bone1_node2 or b[2] == bone1_node1]
            
            if not possible_connected:
                # If no connected bones, just pick another random bone
                bone2_id, _, _ = np.random.choice([b[0] for b in bones if b[0] != bone1_id]) # Ensured bone2_id is scalar
            else:
                pcid = np.random.choice(len(possible_connected))
                bone2_id, _, _ = possible_connected[pcid]
            
            # Joint position at the connection point between bones
            bone1_data = creature.bones[bone1_id]
            bone2_data = creature.bones[bone2_id]
            
            # Find common node position
            common_nodes = set(bone1_data['nodes']) & set(bone2_data['nodes'])
            if common_nodes:
                joint_pos = creature.nodes[next(iter(common_nodes))]['pos']
            else:
                # If bones don't share a node (shouldn't happen with our generation), use midpoint
                joint_pos = (bone1_data['pos'] + bone2_data['pos']) / 2
            
            joint = Joint(
                id=joint_id,
                bone1=bone1_id,
                bone2=bone2_id,
                rotation=np.random.uniform(-10, 10),
                flexion=np.random.uniform(-20, 20),
                abduction=np.random.uniform(-10, 10),
                stiffness=np.random.uniform(30, 100),
                damping=np.random.uniform(1, 10),
                friction=np.random.uniform(0.05, 0.2),
                limits={
                    'flexion': (np.random.uniform(-45, -20), np.random.uniform(20, 45)),
                    'abduction': (np.random.uniform(-30, -10), np.random.uniform(10, 30)),
                    'rotation': (np.random.uniform(-20, -5), np.random.uniform(5, 20))
                }
            )
            creature.add_joint(joint, joint_pos)
        
        # Generate muscles between random nodes
        for i in range(num_muscles):
            muscle_id = i + 1
            
            # Select two distinct random nodes
            node1_idx, node2_idx = np.random.choice(len(nodes), 2, replace=False)
            node1_id, node1_pos = nodes[node1_idx]
            node2_id, node2_pos = nodes[node2_idx]
            
            # Calculate optimal length based on distance between nodes
            distance = np.linalg.norm(node2_pos - node1_pos)
            optimal_length = distance * np.random.uniform(0.8, 1.2)
            
            # Offset attachment points slightly from node centers
            offset1 = np.random.uniform(-0.3, 0.3, size=3)
            offset2 = np.random.uniform(-0.3, 0.3, size=3)
            attachment1 = node1_pos + offset1
            attachment2 = node2_pos + offset2
            
            muscle = Muscle(
                id=muscle_id,
                maxForce=np.random.uniform(200, 500),
                optimalLength=float(optimal_length),
                tendenRatio=np.random.uniform(0.05, 0.2),
                activationTime=np.random.uniform(0.03, 0.08),
                deactivationTime=np.random.uniform(0.01, 0.05),
                stiffness=np.random.uniform(30, 80),
                damping=np.random.uniform(1, 5)
            )
            
            creature.add_muscle(muscle, [attachment1, attachment2], [node1_id, node2_id])
        
        return creature
    
    def _check_ground_collision(self, dt: float):
        """Checks for and resolves collisions with the ground plane."""
        ground_height = GROUND_HEIGHT
        
        # Simple collision - just check if any bone is below ground
        for bone_data in self.bones.values():
            bone_pos = bone_data['pos']
            global_pos = self.get_global_position(bone_pos)
            
            # Check if bone is below ground
            if global_pos[1] < ground_height:
                penetration = ground_height - global_pos[1]
                
                # Apply correction to position
                self.position[1] += penetration * 1.1  # Small over-correction to prevent sticking
                
                # Apply bounce (reflect velocity with restitution)
                if self.velocity[1] < 0:
                    self.velocity[1] = -self.velocity[1] * self.restitution
                
                # Apply friction to horizontal motion
                self.velocity[0] *= (1 - 0.5 * dt)
                self.velocity[2] *= (1 - 0.5 * dt)
                
                # Small angular velocity change from hitting ground
                self.angularVelocity[0] += np.random.uniform(-1, 1)
                self.angularVelocity[2] += np.random.uniform(-1, 1)
                break
    
    def _update_physics(self, dt: float):
        # Apply gravity
        self.velocity[1] -= 9.8 * dt  # Earth gravity
        
        # Update position and rotation
        self.velocity *= (1 - self.damping * dt)
        self.angularVelocity *= (1 - self.angular_damping * dt)
        self.position += self.velocity * dt
        self.rotation += self.angularVelocity * dt
        self.rotation = np.mod(self.rotation + np.pi, 2 * np.pi) - np.pi  # Wrap to [-π, π]
        
        # Check for collisions
        self._check_ground_collision(dt)

    @classmethod
    def generate_cat_creature(cls, id: int = 1) -> 'Creature':
        """
        Generates a creature that approximates a cat's skeletal structure.
        """
        creature = cls(id=id)
        
        # Define cat proportions (in meters)
        body_length = 0.5
        leg_length = 0.2
        tail_length = 0.4
        head_size = 0.15
        spine_segments = 5
        
        # Create nodes for the spine
        spine_nodes = []
        for i in range(spine_segments):
            node_id = i + 1
            # Position nodes along a slight curve to simulate spine curvature
            pos = np.array([
                0,
                i * (body_length / (spine_segments-1)),
                np.sin(i/(spine_segments-1) * np.pi) * -0.05  # slight downward curve
            ])
            node = Node(id=node_id)
            creature.add_node(node, pos)
            spine_nodes.append((node_id, pos))
        
        # Create bones for the spine
        spine_bones = []
        for i in range(spine_segments - 1):
            bone_id = i + 1
            node1_id, node1_pos = spine_nodes[i]
            node2_id, node2_pos = spine_nodes[i + 1]
            
            length = np.linalg.norm(node2_pos - node1_pos)
            
            bone = Bone(
                id=bone_id,
                length=float(length),
                thickness=0.05 if i < 2 else 0.04,  # thicker near shoulders
                density=1.8,
                youngModulus=10000,
                maxCompression=180,
                maxTension=130
            )
            
            # Slight rotation to follow spine curve
            direction = node2_pos - node1_pos
            rotation = np.array([
                np.degrees(np.arctan2(direction[2], direction[1])),
                0,
                0
            ])
            
            creature.add_bone(bone, node1_pos, node1_id, node2_id, rotation)
            spine_bones.append((bone_id, node1_id, node2_id))
        
        # Add head
        head_node_id = spine_segments + 1
        head_pos = spine_nodes[-1][1] + np.array([0, head_size/2, -0.02])
        head_node = Node(id=head_node_id)
        creature.add_node(head_node, head_pos)
        
        neck_bone_id = spine_segments
        neck_length = np.linalg.norm(head_pos - spine_nodes[-1][1])
        neck_bone = Bone(
            id=neck_bone_id,
            length=float(neck_length),
            thickness=0.04,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(neck_bone, spine_nodes[-1][1], spine_nodes[-1][0], head_node_id)
        
        # Add legs (front and hind)
        leg_nodes = []
        # Front legs (attached to 2nd spine node)
        shoulder_node_id = spine_nodes[1][0]
        shoulder_pos = spine_nodes[1][1]
        
        # Right front leg
        upper_leg_pos = shoulder_pos + np.array([0.08, -0.05, 0])
        lower_leg_pos = upper_leg_pos + np.array([0, -leg_length*0.6, 0])
        foot_pos = lower_leg_pos + np.array([0, -leg_length*0.4, 0.02])
        
        # Create nodes and bones for right front leg
        r_upper_leg_node_id = head_node_id + 1
        r_lower_leg_node_id = r_upper_leg_node_id + 1
        r_foot_node_id = r_lower_leg_node_id + 1
        
        creature.add_node(Node(r_upper_leg_node_id), upper_leg_pos)
        creature.add_node(Node(r_lower_leg_node_id), lower_leg_pos)
        creature.add_node(Node(r_foot_node_id), foot_pos)
        
        # Upper leg bone
        upper_leg_bone = Bone(
            id=neck_bone_id + 1,
            length=float(np.linalg.norm(upper_leg_pos - shoulder_pos)),
            thickness=0.03,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(upper_leg_bone, shoulder_pos, shoulder_node_id, r_upper_leg_node_id)
        
        # Lower leg bone
        lower_leg_bone = Bone(
            id=neck_bone_id + 2,
            length=float(np.linalg.norm(lower_leg_pos - upper_leg_pos)),
            thickness=0.025,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(lower_leg_bone, upper_leg_pos, r_upper_leg_node_id, r_lower_leg_node_id)
        
        # Foot bone
        foot_bone = Bone(
            id=neck_bone_id + 3,
            length=float(np.linalg.norm(foot_pos - lower_leg_pos)),
            thickness=0.02,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(foot_bone, lower_leg_pos, r_lower_leg_node_id, r_foot_node_id)
        
        # Left front leg (mirror of right)
        upper_leg_pos = shoulder_pos + np.array([-0.08, -0.05, 0])
        lower_leg_pos = upper_leg_pos + np.array([0, -leg_length*0.6, 0])
        foot_pos = lower_leg_pos + np.array([0, -leg_length*0.4, 0.02])
        
        l_upper_leg_node_id = r_foot_node_id + 1
        l_lower_leg_node_id = l_upper_leg_node_id + 1
        l_foot_node_id = l_lower_leg_node_id + 1
        
        creature.add_node(Node(l_upper_leg_node_id), upper_leg_pos)
        creature.add_node(Node(l_lower_leg_node_id), lower_leg_pos)
        creature.add_node(Node(l_foot_node_id), foot_pos)
        
        # Upper leg bone
        upper_leg_bone = Bone(
            id=neck_bone_id + 4,
            length=float(np.linalg.norm(upper_leg_pos - shoulder_pos)),
            thickness=0.03,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(upper_leg_bone, shoulder_pos, shoulder_node_id, l_upper_leg_node_id)
        
        # Lower leg bone
        lower_leg_bone = Bone(
            id=neck_bone_id + 5,
            length=float(np.linalg.norm(lower_leg_pos - upper_leg_pos)),
            thickness=0.025,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(lower_leg_bone, upper_leg_pos, l_upper_leg_node_id, l_lower_leg_node_id)
        
        # Foot bone
        foot_bone = Bone(
            id=neck_bone_id + 6,
            length=float(np.linalg.norm(foot_pos - lower_leg_pos)),
            thickness=0.02,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(foot_bone, lower_leg_pos, l_lower_leg_node_id, l_foot_node_id)
        
        # Hind legs (attached to 4th spine node)
        hip_node_id = spine_nodes[-2][0]
        hip_pos = spine_nodes[-2][1]
        
        # Right hind leg
        upper_leg_pos = hip_pos + np.array([0.07, -0.08, 0])
        lower_leg_pos = upper_leg_pos + np.array([0, -leg_length*0.7, 0])
        foot_pos = lower_leg_pos + np.array([0, -leg_length*0.3, 0.03])
        
        r_upper_leg_node_id = l_foot_node_id + 1
        r_lower_leg_node_id = r_upper_leg_node_id + 1
        r_foot_node_id = r_lower_leg_node_id + 1
        
        creature.add_node(Node(r_upper_leg_node_id), upper_leg_pos)
        creature.add_node(Node(r_lower_leg_node_id), lower_leg_pos)
        creature.add_node(Node(r_foot_node_id), foot_pos)
        
        # Upper leg bone
        upper_leg_bone = Bone(
            id=neck_bone_id + 7,
            length=float(np.linalg.norm(upper_leg_pos - hip_pos)),
            thickness=0.035,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(upper_leg_bone, hip_pos, hip_node_id, r_upper_leg_node_id)
        
        # Lower leg bone
        lower_leg_bone = Bone(
            id=neck_bone_id + 8,
            length=float(np.linalg.norm(lower_leg_pos - upper_leg_pos)),
            thickness=0.03,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(lower_leg_bone, upper_leg_pos, r_upper_leg_node_id, r_lower_leg_node_id)
        
        # Foot bone
        foot_bone = Bone(
            id=neck_bone_id + 9,
            length=float(np.linalg.norm(foot_pos - lower_leg_pos)),
            thickness=0.025,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(foot_bone, lower_leg_pos, r_lower_leg_node_id, r_foot_node_id)
        
        # Left hind leg (mirror of right)
        upper_leg_pos = hip_pos + np.array([-0.07, -0.08, 0])
        lower_leg_pos = upper_leg_pos + np.array([0, -leg_length*0.7, 0])
        foot_pos = lower_leg_pos + np.array([0, -leg_length*0.3, 0.03])
        
        l_upper_leg_node_id = r_foot_node_id + 1
        l_lower_leg_node_id = l_upper_leg_node_id + 1
        l_foot_node_id = l_lower_leg_node_id + 1
        
        creature.add_node(Node(l_upper_leg_node_id), upper_leg_pos)
        creature.add_node(Node(l_lower_leg_node_id), lower_leg_pos)
        creature.add_node(Node(l_foot_node_id), foot_pos)
        
        # Upper leg bone
        upper_leg_bone = Bone(
            id=neck_bone_id + 10,
            length=np.linalg.norm(upper_leg_pos - hip_pos),
            thickness=0.035,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(upper_leg_bone, hip_pos, hip_node_id, l_upper_leg_node_id)
        
        # Lower leg bone
        lower_leg_bone = Bone(
            id=neck_bone_id + 11,
            length=np.linalg.norm(lower_leg_pos - upper_leg_pos),
            thickness=0.03,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(lower_leg_bone, upper_leg_pos, l_upper_leg_node_id, l_lower_leg_node_id)
        
        # Foot bone
        foot_bone = Bone(
            id=neck_bone_id + 12,
            length=np.linalg.norm(foot_pos - lower_leg_pos),
            thickness=0.025,
            density=1.8,
            youngModulus=10000,
            maxCompression=180,
            maxTension=130
        )
        creature.add_bone(foot_bone, lower_leg_pos, l_lower_leg_node_id, l_foot_node_id)
        
        # Add tail
        tail_segments = 4
        tail_nodes = []
        prev_node_id = spine_nodes[-1][0]
        prev_pos = spine_nodes[-1][1]
        
        for i in range(tail_segments):
            node_id = l_foot_node_id + 1 + i
            # Position tail segments curving upward
            pos = prev_pos + np.array([
                0,
                0,
                tail_length/tail_segments * (i+1)/tail_segments
            ])
            node = Node(id=node_id)
            creature.add_node(node, pos)
            tail_nodes.append((node_id, pos))
            
            # Add tail bone
            if i > 0:
                bone_id = neck_bone_id + 12 + i
                length = np.linalg.norm(pos - prev_pos)
                tail_bone = Bone(
                    id=bone_id,
                    length=float(length),
                    thickness=0.02 * (1 - i/tail_segments),  # Taper tail
                    density=1.8,
                    youngModulus=10000,
                    maxCompression=180,
                    maxTension=130
                )
                creature.add_bone(tail_bone, prev_pos, prev_node_id, node_id)
            
            prev_node_id = node_id
            prev_pos = pos
        
        # Add joints at key locations
        # Shoulder joints (front legs)
        shoulder_joint = Joint(
            id=1,
            bone1=spine_bones[1][0],  # Second spine bone
            bone2=neck_bone_id + 1,    # Right upper leg bone
            flexion=0,
            abduction=0,
            rotation=0,
            stiffness=80,
            damping=5,
            friction=0.1,
            limits={
                'flexion': (-45, 45),
                'abduction': (-30, 30),
                'rotation': (-15, 15)
            }
        )
        creature.add_joint(shoulder_joint, shoulder_pos)
        
        # Hip joints (hind legs)
        hip_joint = Joint(
            id=2,
            bone1=spine_bones[-2][0],  # Second-to-last spine bone
            bone2=neck_bone_id + 7,     # Right upper hind leg bone
            flexion=0,
            abduction=0,
            rotation=0,
            stiffness=80,
            damping=5,
            friction=0.1,
            limits={
                'flexion': (-45, 45),
                'abduction': (-30, 30),
                'rotation': (-15, 15)
            }
        )
        creature.add_joint(hip_joint, hip_pos)
        
        # Knee joints (front legs)
        knee_joint = Joint(
            id=3,
            bone1=neck_bone_id + 2,    # Right lower leg bone
            bone2=neck_bone_id + 3,    # Right foot bone
            flexion=0,
            abduction=0,
            rotation=0,
            stiffness=100,
            damping=8,
            friction=0.15,
            limits={
                'flexion': (-120, 0),  # Knees can't bend backward
                'abduction': (-10, 10),
                'rotation': (-5, 5)
            }
        )
        creature.add_joint(knee_joint, lower_leg_pos)
        
        # Add muscles for key movement groups
        # Back muscles (along spine)
        for i in range(spine_segments - 2):
            muscle_id = i + 1
            start_pos = spine_nodes[i][1] + np.array([0, 0, 0.02])
            end_pos = spine_nodes[i+2][1] + np.array([0, 0, 0.02])
            
            muscle = Muscle(
                id=muscle_id,
                maxForce=200,
                optimalLength=np.linalg.norm(end_pos - start_pos),
                tendenRatio=0.1,
                activationTime=0.05,
                deactivationTime=0.02,
                stiffness=50,
                damping=2
            )
            creature.add_muscle(muscle, [start_pos, end_pos], 
                               [spine_nodes[i][0], spine_nodes[i+2][0]])
        
        # Leg muscles (biceps/triceps equivalents)
        # Right front leg biceps
        muscle_id = spine_segments
        start_pos = shoulder_pos + np.array([0.05, 0, 0])
        end_pos = lower_leg_pos + np.array([0.02, 0.05, 0])
        
        muscle = Muscle(
            id=muscle_id,
            maxForce=300,
            optimalLength=np.linalg.norm(end_pos - start_pos),
            tendenRatio=0.15,
            activationTime=0.04,
            deactivationTime=0.01,
            stiffness=60,
            damping=3
        )
        creature.add_muscle(muscle, [start_pos, end_pos], 
                           [shoulder_node_id, r_lower_leg_node_id])
        
        # Right front leg triceps
        muscle_id += 1
        start_pos = shoulder_pos + np.array([-0.05, 0, 0])
        end_pos = lower_leg_pos + np.array([-0.02, 0.05, 0])
        
        muscle = Muscle(
            id=muscle_id,
            maxForce=300,
            optimalLength=np.linalg.norm(end_pos - start_pos),
            tendenRatio=0.15,
            activationTime=0.04,
            deactivationTime=0.01,
            stiffness=60,
            damping=3
        )
        creature.add_muscle(muscle, [start_pos, end_pos], 
                           [shoulder_node_id, r_lower_leg_node_id])
        
        # Add similar muscles for other legs...
        
        return creature

class CreatureViewer2D:
    def __init__(self, creature, width=800, height=600):
        self.creature = creature
        self.width = width
        self.height = height
        self.scale = 50.0  # Pixels per meter
        self.offset = np.array([width/2, height/2])  # Center of view
        self.drag_start_pos = None
        self.pan_offset = np.array([0.0, 0.0])
        self.selected_component = None
        
        # Create DPG context and window
        #dpg.create_context()
        self.setup_ui()
        
    def setup_ui(self):
        dpg.create_context()
        dpg.create_viewport(title='Creature Renderer', width=800, height=600)
        # Main window
        with dpg.window(label="Main_Window"):
            # Add a drawing canvas
            with dpg.drawlist(width=self.width-20, height=self.height-20, 
                            tag="drawing_canvas"):
                pass
                
            # Controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset View", callback=self.reset_view)
                dpg.add_slider_float(label="Zoom", default_value=self.scale, 
                                   min_value=10, max_value=200, 
                                   callback=self.update_scale)
                                   
        # Set mouse callbacks
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, 
                                      callback=self.on_drag)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, 
                                         callback=self.on_drag_end)
            dpg.add_mouse_wheel_handler(callback=self.on_mouse_wheel)
        dpg.setup_dearpygui()
        #dpg.set_primary_window(window="Main_Window", value=True)
        dpg.show_viewport()
            
    def reset_view(self):
        self.scale = 50.0
        self.pan_offset = np.array([0.0, 0.0])
        self.draw_creature()
        
    def update_scale(self, sender, app_data):
        self.scale = app_data
        self.draw_creature()
        
    def on_drag(self, sender, app_data):
        if dpg.is_item_hovered("drawing_canvas"):
            if self.drag_start_pos is None:
                self.drag_start_pos = np.array([app_data[1], app_data[2]])
            else:
                current_pos = np.array([app_data[1], app_data[2]])
                delta = (current_pos - self.drag_start_pos) / self.scale
                self.pan_offset += delta * np.array([1, -1])  # Invert y-axis
                self.drag_start_pos = current_pos
                self.draw_creature()
                
    def on_drag_end(self, sender, app_data):
        self.drag_start_pos = None
        
    def on_mouse_wheel(self, sender, app_data):
        # Zoom in/out based on mouse wheel
        zoom_factor = 1.1 if app_data > 0 else 0.9
        self.scale *= zoom_factor
        self.scale = np.clip(self.scale, 10, 200)
        dpg.set_value("Zoom", self.scale)
        self.draw_creature()
        
    def world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates"""
        # Note: We're using X-Z plane for 2D side view (ignoring Y)
        x = pos[0] * self.scale + self.offset[0] + self.pan_offset[0] * self.scale
        y = -pos[2] * self.scale + self.offset[1] + self.pan_offset[1] * self.scale  # Invert Z for screen Y
        return [x, y]
        
    def draw_creature(self):
        """Draw the creature in 2D"""
        dpg.delete_item("drawing_canvas", children_only=True)
        
        # Draw ground plane
        ground_y = self.world_to_screen([0, 0, 0])[1]
        dpg.draw_line([0, ground_y], [self.width, ground_y], 
                      color=(150, 150, 150, 255), parent="drawing_canvas")
        
        # Draw bones as lines
        for bone_id, bone_data in self.creature.bones.items():
            start_pos = bone_data['pos']
            # Calculate end position based on bone length and rotation
            bone_dir = np.array([0, bone_data['bone'].length, 0])
            rotated_dir = rotate_vector_by_euler(bone_dir, np.radians(bone_data['rotation']))
            end_pos = start_pos + rotated_dir
            
            # Convert to screen coordinates
            start_screen = self.world_to_screen(start_pos)
            end_screen = self.world_to_screen(end_pos)
            
            # Draw bone
            dpg.draw_line(start_screen, end_screen, 
                          color=(255, 255, 255, 255), 
                          thickness=bone_data['bone'].thickness * self.scale * 0.5,
                          parent="drawing_canvas")
            
            # Draw bone endpoints (joints)
            dpg.draw_circle(start_screen, radius=5, color=(0, 255, 0, 255),
                           parent="drawing_canvas")
            dpg.draw_circle(end_screen, radius=5, color=(0, 255, 0, 255),
                           parent="drawing_canvas")
            
        # Draw muscles as colored lines
        for muscle_id, muscle_data in self.creature.muscles.items():
            if len(muscle_data['attachments']) < 2:
                continue
                
            start_screen = self.world_to_screen(muscle_data['attachments'][0])
            end_screen = self.world_to_screen(muscle_data['attachments'][1])
            
            # Color based on activation level
            activation = muscle_data['muscle'].activation
            color = (255, int(255 * (1 - activation)), int(255 * (1 - activation)), 255)
            
            dpg.draw_line(start_screen, end_screen, 
                          color=color, 
                          thickness=3,
                          parent="drawing_canvas")
        
        # Draw creature position indicator
        creature_pos_screen = self.world_to_screen([0, 0, 0])
        dpg.draw_circle(creature_pos_screen, radius=3, color=(255, 0, 0, 255),
                       parent="drawing_canvas")
        
        
    def update(self, creature=None):
        if creature is not None:
            self.creature = creature
        self.draw_creature()
        
    def run(self):
        self.draw_creature()
        while dpg.is_dearpygui_running():
            current_time = dpg.get_total_time()
            #print(current_time)
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def rotate_vector_by_euler(v, euler_angles):
        """Rotate vector by euler angles (XYZ order)"""
        rx, ry, rz = euler_angles
        
        # X rotation
        x_rot = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # Y rotation
        y_rot = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # Z rotation
        z_rot = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        return z_rot @ y_rot @ x_rot @ v

# Run the application
if __name__ == "__main__":
    creature = Creature.generate_cat_creature()
    viewer = CreatureViewer2D(creature)
    viewer.run()