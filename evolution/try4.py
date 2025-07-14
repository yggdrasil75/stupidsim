from dataclasses import dataclass, field
import math
from typing import Optional
import dearpygui.dearpygui as dpg
import numpy as np


# --- Helper functions for 3D Math ---
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def cross(v1, v2):
    return np.cross(v1, v2)

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
    
    # Apply rotations in ZYX order (common for object rotation, first X, then Y, then Z)
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
    # Convert angles to radians
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    # Calculate rotation matrices
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
    
    # Combine rotations and translation
    R = Rx @ Ry
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ position
    return T

def project_point(point: np.ndarray, view_matrix: np.ndarray, proj_matrix: np.ndarray, viewport_size: tuple) -> tuple:
    """Projects a 3D point to 2D screen coordinates."""
    # Convert to homogeneous coordinates
    point_hom = np.append(point, 1.0)
    
    # Apply view and projection matrices
    view_space = view_matrix @ point_hom
    clip_space = proj_matrix @ view_space
    
    # Perspective division
    ndc_space = clip_space[:3] / clip_space[3]
    
    # Convert to screen coordinates
    screen_x = (ndc_space[0] * 0.5 + 0.5) * viewport_size[0]
    screen_y = (1 - (ndc_space[1] * 0.5 + 0.5)) * viewport_size[1]
    
    return (screen_x, screen_y)

@dataclass 
class Bone:
    # bone cant change structure. its position and rotation can be changed by muscles
    id: int
    length: float = field(default=10.0, metadata={"sigma": 0.1})
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
        
        return np.matmul(np.matmul(Rz, Ry), Rx)
    
    def calculate_restoring_torque(self):
        angles = np.radians([self.flexion, self.abduction, self.rotation])
        return -self.stiffness * angles

@dataclass
class Muscle:
    # muscle does most of the effort here, moving bones, joints, etc.
    id: int
    maxForce: float = field(default=300.0) # in newtons
    optimalLength: float = field(default=0.3) # in meters
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
        self.length = self.restLength

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
        local_pos_hom = np.append(local_pos, 1.0)
        model_matrix = self._get_creature_model_matrix()
        global_pos_hom = model_matrix @ local_pos_hom
        return global_pos_hom[:3]
    
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
    
    def _update_physics(self, dt: float):
        self.velocity *= (1 - self.damping * dt)
        self.angularVelocity *= (1 - self.angular_damping * dt)
        self.position += self.velocity * dt
        
        self.rotation += self.angularVelocity * dt
        self.rotation = np.mod(self.rotation + np.pi, 2 * np.pi) - np.pi  # Wrap to [-π, π]
        
    def calculate_3d_coordinates(self) -> tuple[dict[int, np.ndarray], list[tuple[int, int, int]]]:
        """
        Calculates the 3D coordinates of all vertices in the creature's structure and
        returns a tuple containing:
        1. A dictionary mapping vertex IDs to their 3D coordinates
        2. A list of triangles (each as a tuple of 3 vertex IDs)
        
        The function creates vertices for:
        - Bone endpoints (cylinders)
        - Joint centers (spheres)
        - Muscle attachment points
        """
        vertices = {}
        triangles = []
        vertex_id_counter = 0
        
        # Helper function to add a vertex and return its ID
        def add_vertex(pos: np.ndarray) -> int:
            nonlocal vertex_id_counter
            vertices[vertex_id_counter] = pos
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
            # We'll create a cylinder with 8 sides for simplicity
            cylinder_resolution = 8
            radius = bone.thickness / 2
            
            # Create vertices around start and end points
            start_vertex_ids = []
            end_vertex_ids = []
            
            # Calculate perpendicular vectors for cylinder cross-section
            if np.allclose(rotated_dir, [0, 0, 1]):
                perp1 = np.array([1, 0, 0])
            else:
                perp1 = normalize(np.cross(rotated_dir, np.array([0, 0, 1])))
            perp2 = normalize(np.cross(rotated_dir, perp1))
            
            for i in range(cylinder_resolution):
                angle = 2 * np.pi * i / cylinder_resolution
                offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                
                # Start circle vertices
                start_vert = start_point + offset
                start_id = add_vertex(start_vert)
                start_vertex_ids.append(start_id)
                
                # End circle vertices
                end_vert = end_point + offset
                end_id = add_vertex(end_vert)
                end_vertex_ids.append(end_id)
            
            # Create triangles for the cylinder sides
            for i in range(cylinder_resolution):
                next_i = (i + 1) % cylinder_resolution
                # Two triangles per side segment
                triangles.append((start_vertex_ids[i], end_vertex_ids[i], end_vertex_ids[next_i]))
                triangles.append((start_vertex_ids[i], end_vertex_ids[next_i], start_vertex_ids[next_i]))
            
            # Create triangles for the end caps
            center_start = add_vertex(start_point)
            center_end = add_vertex(end_point)
            for i in range(cylinder_resolution):
                next_i = (i + 1) % cylinder_resolution
                # Start cap
                triangles.append((center_start, start_vertex_ids[next_i], start_vertex_ids[i]))
                # End cap
                triangles.append((center_end, end_vertex_ids[i], end_vertex_ids[next_i]))
        
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
                v_norm = normalize(np.array(v))
                vertex_pos = pos + v_norm * radius
                base_vertex_ids.append(add_vertex(vertex_pos))
            
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
                    base_vertex_ids[tri[2]]
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
                perp1 = normalize(np.cross(direction, np.array([0, 0, 1])))
            perp2 = normalize(np.cross(direction, perp1))
            
            for i in range(cylinder_resolution):
                angle = 2 * np.pi * i / cylinder_resolution
                offset_start = (thickness * 0.8) * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                offset_end = (thickness * 0.5) * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                
                # Start circle vertices
                start_vert = start_pos + offset_start
                start_id = add_vertex(start_vert)
                start_vertex_ids.append(start_id)
                
                # End circle vertices
                end_vert = end_pos + offset_end
                end_id = add_vertex(end_vert)
                end_vertex_ids.append(end_id)
            
            # Create triangles for the muscle body
            for i in range(cylinder_resolution):
                next_i = (i + 1) % cylinder_resolution
                # Two triangles per side segment
                triangles.append((start_vertex_ids[i], end_vertex_ids[i], end_vertex_ids[next_i]))
                triangles.append((start_vertex_ids[i], end_vertex_ids[next_i], start_vertex_ids[next_i]))
        
        return vertices, triangles
    
    
class CreatureRenderer:
    def __init__(self):
        self.camera_pos = np.array([0, 0, -5])
        self.camera_pitch = 0
        self.camera_yaw = 0
        self.fov = 60.0
        self.near_plane = 0.1
        self.far_plane = 100.0
        
        # Lighting parameters
        self.light_dir = normalize(np.array([1, -1, -1]))
        self.ambient = 0.2
        self.diffuse = 0.7
        
        # Control state
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
    def create_simple_creature(self) -> Creature:
        """Creates a simple test creature with 2 bones, 1 joint, and 1 muscle."""
        creature = Creature(id=1)
        
        # Add nodes
        node1 = Node(id=1)
        node2 = Node(id=2)
        node3 = Node(id=3)
        
        creature.add_node(node1, np.array([0, 0, 0]))
        creature.add_node(node2, np.array([0, 2, 0]))
        creature.add_node(node3, np.array([0, 4, 0]))
        
        # Add bones
        bone1 = Bone(id=1, length=2.0, thickness=0.3)
        bone2 = Bone(id=2, length=2.0, thickness=0.3)
        
        creature.add_bone(bone1, np.array([0, 0, 0]), 1, 2)
        creature.add_bone(bone2, np.array([0, 2, 0]), 2, 3)
        
        # Add joint
        joint = Joint(id=1, bone1=1, bone2=2, 
                     flexion=0, abduction=0, rotation=0,
                     stiffness=50, damping=2,
                     limits={'flexion': (-45, 45), 'abduction': (-30, 30), 'rotation': (-30, 30)})
        creature.add_joint(joint, np.array([0, 2, 0]))
        
        # Add muscle
        muscle = Muscle(id=1, maxForce=300, optimalLength=2.0, tendenRatio=0.1)
        creature.add_muscle(muscle, [np.array([0.5, 0, 0]), np.array([0.5, 4, 0])], [1, 3])
        
        return creature
    
    def setup_dpg(self):
        """Sets up Dear PyGui context and windows."""
        dpg.create_context()
        dpg.create_viewport(title='Creature Renderer', width=800, height=600)
        
        with dpg.window(tag="Primary Window"):
            with dpg.group(horizontal=True):
                # Viewport for 3D rendering
                with dpg.child_window(tag="Viewport", width=600, height=600):
                    dpg.add_drawlist(tag="Canvas", width=600, height=600)
                
                # Control panel
                with dpg.child_window(tag="Controls", width=200, height=600):
                    dpg.add_text("Camera Controls")
                    dpg.add_slider_float(label="FOV", default_value=self.fov, min_value=10, max_value=120, callback=lambda s: setattr(self, 'fov', s))
                    dpg.add_slider_float(label="Pitch", default_value=self.camera_pitch, min_value=-89, max_value=89, callback=lambda s: setattr(self, 'camera_pitch', s))
                    dpg.add_slider_float(label="Yaw", default_value=self.camera_yaw, min_value=-180, max_value=180, callback=lambda s: setattr(self, 'camera_yaw', s))
                    
                    dpg.add_text("Lighting")
                    dpg.add_slider_float(label="Ambient", default_value=self.ambient, min_value=0, max_value=1, callback=lambda s: setattr(self, 'ambient', s))
                    dpg.add_slider_float(label="Diffuse", default_value=self.diffuse, min_value=0, max_value=1, callback=lambda s: setattr(self, 'diffuse', s))
                    
                    dpg.add_button(label="Reset View", callback=self.reset_camera)
        
        # Register handlers
        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=self._on_mouse_down)
            dpg.add_mouse_release_handler(callback=self._on_mouse_release)
            dpg.add_mouse_drag_handler(callback=self._on_mouse_drag)
            dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def reset_camera(self):
        """Resets camera to default position."""
        self.camera_pos = np.array([0, 0, -5])
        self.camera_pitch = 0
        self.camera_yaw = 0
        self.fov = 60.0
    
    def _on_mouse_down(self):
        self.mouse_dragging = True
        self.last_mouse_pos = dpg.get_mouse_pos()
    
    def _on_mouse_release(self):
        self.mouse_dragging = False
    
    def _on_mouse_drag(self, sender, app_data):
        if self.mouse_dragging:
            current_mouse_pos = dpg.get_mouse_pos()
            dx = current_mouse_pos[0] - self.last_mouse_pos[0]
            dy = current_mouse_pos[1] - self.last_mouse_pos[1]
            
            self.camera_yaw -= dx * 0.5
            self.camera_pitch -= dy * 0.5
            self.camera_pitch = np.clip(self.camera_pitch, -89, 89)
            
            self.last_mouse_pos = current_mouse_pos
    
    def _on_mouse_wheel(self, sender, app_data):
        # Zoom in/out
        self.camera_pos[2] += app_data * 0.2
    
    def render_creature(self, creature: Creature):
        """Renders the creature in the viewport."""
        dpg.delete_item("Canvas", children_only=True)
        
        # Get viewport size
        viewport_width = dpg.get_item_width("Canvas")
        viewport_height = dpg.get_item_height("Canvas")
        aspect_ratio = viewport_width / viewport_height
        
        # Create matrices
        #view_matrix = dpg.create_fps_matrix(eye=self.camera_pos.tolist(), pitch=self.camera_pitch, yaw=self.camera_yaw)
        view_matrix = create_fps_matrix(self.camera_pos, self.camera_pitch, self.camera_yaw)
        proj_matrix = create_perspective_matrix(np.radians(self.fov), aspect_ratio, self.near_plane, self.far_plane)
        
        # Get creature geometry
        vertices, triangles = creature.calculate_3d_coordinates()
        
        # Transform vertices to world space
        world_vertices = {}
        for v_id, v_pos in vertices.items():
            world_vertices[v_id] = creature.get_global_position(v_pos)
        
        # Draw each triangle
        for tri in triangles:
            v0 = world_vertices[tri[0]]
            v1 = world_vertices[tri[1]]
            v2 = world_vertices[tri[2]]
            
            # Calculate normal for lighting
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = normalize(cross(edge1, edge2))
            
            # Calculate lighting (simple Lambertian)
            light_intensity = self.ambient + self.diffuse * max(0, dot(normal, -self.light_dir))
            color = (int(255 * light_intensity), int(255 * light_intensity), int(255 * light_intensity), 255)
            
            # Project vertices to screen space
            p0 = project_point(v0, view_matrix, proj_matrix, (viewport_width, viewport_height))
            p1 = project_point(v1, view_matrix, proj_matrix, (viewport_width, viewport_height))
            p2 = project_point(v2, view_matrix, proj_matrix, (viewport_width, viewport_height))
            
            # Draw triangle
            dpg.draw_triangle(p0, p1, p2, color=color, parent="Canvas")
        
        # Draw coordinate axes for reference
        self._draw_axes(view_matrix, proj_matrix, (viewport_width, viewport_height))
    
    def _draw_axes(self, view_matrix, proj_matrix, viewport_size):
        """Draws XYZ axes at the origin for reference."""
        origin = np.array([0, 0, 0])
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        o = project_point(origin, view_matrix, proj_matrix, viewport_size)
        x = project_point(x_axis, view_matrix, proj_matrix, viewport_size)
        y = project_point(y_axis, view_matrix, proj_matrix, viewport_size)
        z = project_point(z_axis, view_matrix, proj_matrix, viewport_size)
        
        dpg.draw_line(o, x, color=(255, 0, 0, 255), thickness=2, parent="Canvas")
        dpg.draw_line(o, y, color=(0, 255, 0, 255), thickness=2, parent="Canvas")
        dpg.draw_line(o, z, color=(0, 0, 255, 255), thickness=2, parent="Canvas")
    
    def run(self):
        """Main application loop."""
        creature = self.create_simple_creature()
        
        while dpg.is_dearpygui_running():
            # Update creature (simple animation for demo)
            current_time = dpg.get_total_time()
            excitation = (np.sin(current_time) + 1) / 2  # 0-1 oscillation
            creature.step(0.016, {1: excitation})
            
            # Render
            self.render_creature(creature)
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()

# Run the application
if __name__ == "__main__":
    renderer = CreatureRenderer()
    renderer.setup_dpg()
    renderer.run()