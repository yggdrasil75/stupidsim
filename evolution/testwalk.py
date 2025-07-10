import json
import os
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dearpygui.dearpygui as dpg
import math
import random
from collections import defaultdict
import string
from collections import deque
from typing import List, Dict, Tuple, Optional

# Simulation parameters
SIM_WIDTH = 1200
SIM_HEIGHT = 800
GRAVITY = torch.tensor([0.0, 0.5])
FRICTION = 0.99
TIME_STEP = 0.1
MUSCLE_CYCLE_LENGTH = 100
CREATURE_DB_FILE = "creatures_db.json"
TRAINING_TIMESTEPS = 10000  # 10 seconds at normal speed
TRAINING_SPEEDUP = 10  # Run training 10x faster than real-time
MAX_TRIALS = 3  # Number of trials per generation

class CONNECTION_TYPES(Enum):
    node_center = 0
    node_edge = 1
    bone_center = 2

# Helper functions for JSON serialization
def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    elif isinstance(tensor, (list, tuple)):
        return [tensor_to_list(x) for x in tensor]
    elif isinstance(tensor, dict):
        return {k: tensor_to_list(v) for k, v in tensor.items()}
    return tensor

def list_to_tensor(lst):
    if isinstance(lst, (list, tuple)):
        if all(isinstance(x, (int, float)) for x in lst):  # It's a simple list of numbers
            return torch.tensor(lst, dtype=torch.float32)
        else:  # It's a nested structure
            return [list_to_tensor(x) for x in lst]
    elif isinstance(lst, dict):
        return {k: list_to_tensor(v) for k, v in lst.items()}
    return lst

class NeuralController(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralController, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Output between 0 and 1 for muscle tension
        return x

class Node:
    def __init__(self, position, radius: float=10, mass=1.0):
        self.position = torch.tensor(position, dtype=torch.float32, requires_grad=False)
        self.velocity = torch.zeros(2, dtype=torch.float32)
        self.force = torch.zeros(2, dtype=torch.float32)
        self.radius: float = radius
        self.mass = mass
        self.fixed = False
        
        # Angle restrictions for connected bones (in radians)
        # Each entry is (bone_id, min_angle, max_angle, natural_min, natural_max)
        self.angle_restrictions: List[Tuple[int, float, float, float, float]] = []
        
    def add_angle_restriction(self, bone_id: int, 
                             min_angle: float = -math.pi, 
                             max_angle: float = math.pi,
                             natural_min: Optional[float] = None,
                             natural_max: Optional[float] = None):
        if natural_min is None:
            natural_min = min_angle
        if natural_max is None:
            natural_max = max_angle
            
        self.angle_restrictions.append((bone_id, min_angle, max_angle, natural_min, natural_max))
        
    def get_angle_restriction(self, bone_id: int) -> Tuple[float, float, float, float]:
        for b_id, min_a, max_a, nat_min, nat_max in self.angle_restrictions:
            if b_id == bone_id:
                return (min_a, max_a, nat_min, nat_max)
        return (-math.pi, math.pi, -math.pi, math.pi)  # Default to no restrictions
        
    def apply_force(self, force):
        self.force += force
        
    def update(self, gravity_enabled=True, gravity_type="node"):
        if not self.fixed:
            # Apply gravity if enabled and type is node
            if gravity_enabled and gravity_type == "node":
                self.apply_force(GRAVITY * self.mass)
            
            # Update velocity and position
            acceleration = self.force / self.mass
            self.velocity += acceleration * TIME_STEP
            self.velocity *= FRICTION
            self.position += self.velocity * TIME_STEP
            
            # Reset force
            self.force = torch.zeros(2, dtype=torch.float32)
            
            # Simple ground collision
            if self.position[1] > SIM_HEIGHT - self.radius - 20:
                self.position[1] = SIM_HEIGHT - self.radius - 20
                self.velocity[1] *= -0.5

    def to_dict(self):
        return {
            'position': tensor_to_list(self.position),
            'velocity': tensor_to_list(self.velocity),
            'force': tensor_to_list(self.force),
            'radius': self.radius,
            'mass': self.mass,
            'fixed': self.fixed,
            'angle_restrictions': self.angle_restrictions
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(
            position=data['position'],
            radius=data['radius'],
            mass=data['mass']
        )
        node.velocity = list_to_tensor(data['velocity'])
        node.force = list_to_tensor(data['force'])
        node.fixed = data['fixed']
        node.angle_restrictions = [tuple(ar) for ar in data['angle_restrictions']]
        return node
    
    def get_angle_ranges(self) -> List[Tuple[float, float]]:
        """Return a list of (start_angle, end_angle) tuples for visualization"""
        ranges = []
        for bone_id, min_angle, max_angle, natural_min, natural_max in self.angle_restrictions:
            ranges.append((min_angle, max_angle))
        return ranges

class Bone:
    def __init__(self, node1_id: int, node2_id: int, length=1, stiffness=1.0):
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.stiffness = stiffness
        self.id: int = -1  # Will be set when added to organism
        self.rest_length = length  # This might be None initially
        self.mass = 0.5  # Bones have some mass too
        self.angle_restriction_strength = 0.5  # Strength of angle restriction enforcement
        
    def update_angle_restrictions(self, nodes: Dict[int, Node]):
        """Update angle restrictions based on current bone positions."""
        node1 = nodes[self.node1_id]
        node2 = nodes[self.node2_id]
        
        # Calculate direction vector
        direction = node2.position - node1.position
        current_angle = math.atan2(direction[1], direction[0])
        
        # Clear existing restrictions
        node1.angle_restrictions = [ar for ar in node1.angle_restrictions if ar[0] != self.id]
        node2.angle_restrictions = [ar for ar in node2.angle_restrictions if ar[0] != self.id]
        
        # For node1, the angle is current_angle
        # For node2, the angle is current_angle + math.pi (opposite direction)
        node1.add_angle_restriction(self.id, 
                                  current_angle - math.pi/3,  # 60 degrees range
                                  current_angle + math.pi/3,
                                  current_angle - math.pi/6,   # 30 degrees natural range
                                  current_angle + math.pi/6)
        
        node2.add_angle_restriction(self.id, 
                                  current_angle + math.pi - math.pi/3,
                                  current_angle + math.pi + math.pi/3,
                                  current_angle + math.pi - math.pi/6,
                                  current_angle + math.pi + math.pi/6)
        
    def apply_constraint(self, nodes: Dict[int, Node]):
        """Enforce rigid bone constraint (fixed length)."""
        node1 = nodes[self.node1_id]
        node2 = nodes[self.node2_id]
        
        direction = node2.position - node1.position
        distance = torch.norm(direction)
        if distance > 0:
            direction = direction / distance
            
            # Calculate displacement from rest length
            displacement = distance - self.rest_length
            force = direction * displacement * self.stiffness
            
            node1.apply_force(force)
            node2.apply_force(-force)
                
            # Enforce angle restrictions
            self.enforce_angle_constraints(nodes)
            
    def enforce_angle_constraints(self, nodes: Dict[int, Node]):
        """Apply forces to keep angles within restrictions."""
        node1 = nodes[self.node1_id]
        node2 = nodes[self.node2_id]
        
        # Get all connected bones for both nodes
        for node in [node1, node2]:
            for bone_id, min_angle, max_angle, nat_min, nat_max in node.angle_restrictions:
                if bone_id == self.id:
                    continue  # Skip self
                    
                # Get the other bone
                # other_bone = next((b for b in nodes.values() if hasattr(b, 'id') and b.id == bone_id), None)
                other_bone = next((b for b in nodes.values() if hasattr(b, 'id') and b.id == bone_id), None)
                if other_bone is None:
                    continue
                    
                # Get the connected node for the other bone
                if other_bone.node1_id == self.node1_id or other_bone.node1_id == self.node2_id:
                    other_node_id = other_bone.node2_id
                else:
                    other_node_id = other_bone.node1_id
                
                # Calculate vectors for both bones
                vec1 = node2.position - node1.position
                vec2 = nodes[other_node_id].position - node.position
                
                # Calculate angle between vectors
                angle1 = math.atan2(vec1[1], vec1[0])
                angle2 = math.atan2(vec2[1], vec2[0])
                angle_diff = (angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi  # Normalized to [-π, π]
                
                # Calculate correction needed
                correction = 0.0
                if angle_diff < nat_min:
                    # Below natural minimum - push toward natural range
                    correction = (nat_min - angle_diff) * self.angle_restriction_strength
                elif angle_diff > nat_max:
                    # Above natural maximum - push toward natural range
                    correction = (nat_max - angle_diff) * self.angle_restriction_strength
                elif angle_diff < min_angle:
                    # Below absolute minimum - strong push
                    correction = (min_angle - angle_diff) * self.angle_restriction_strength * 2
                elif angle_diff > max_angle:
                    # Above absolute maximum - strong push
                    correction = (max_angle - angle_diff) * self.angle_restriction_strength * 2
                
                if correction != 0.0:
                    # Apply corrective forces
                    # Calculate perpendicular vectors for torque
                    perp_vec1 = torch.tensor([-vec1[1], vec1[0]])
                    perp_vec2 = torch.tensor([vec2[1], -vec2[0]])
                    
                    # Normalize perpendicular vectors
                    if torch.norm(perp_vec1) > 0:
                        perp_vec1 = perp_vec1 / torch.norm(perp_vec1)
                    if torch.norm(perp_vec2) > 0:
                        perp_vec2 = perp_vec2 / torch.norm(perp_vec2)
                    
                    # Apply forces to create torque
                    force_magnitude = abs(correction) * 0.1
                    node1.apply_force(perp_vec1 * force_magnitude * (1 if correction > 0 else -1))
                    node2.apply_force(-perp_vec1 * force_magnitude * (1 if correction > 0 else -1))
                    
                    # Also apply force to the other bone's nodes
                    node.apply_force(perp_vec2 * force_magnitude * (-1 if correction > 0 else 1))
                    nodes[other_node_id].apply_force(-perp_vec2 * force_magnitude * (-1 if correction > 0 else 1))
                        
    def apply_gravity(self, nodes: Dict[int, Node]):
        """Apply gravity to the bone based on its center of mass"""
        node1 = nodes[self.node1_id]
        node2 = nodes[self.node2_id]
        center = (node1.position + node2.position) / 2
        gravity_force = GRAVITY * self.mass
        
        # Apply half to each node
        node1.apply_force(gravity_force / 2)
        node2.apply_force(gravity_force / 2)

    def to_dict(self):
        return {
            'node1_id': self.node1_id,
            'node2_id': self.node2_id,
            'stiffness': self.stiffness,
            'id': self.id,
            'rest_length': tensor_to_list(self.rest_length),
            'mass': self.mass
        }

    @classmethod
    def from_dict(cls, data):
        bone = cls(
            node1_id=data['node1_id'],
            node2_id=data['node2_id'],
            length=data['rest_length'],
            stiffness=data['stiffness']
        )
        bone.id = data['id']
        bone.mass = data['mass']
        bone.rest_length = list_to_tensor(data["rest_length"])
        return bone

class Muscle:
    def __init__(self, attachment1: Tuple[int, CONNECTION_TYPES], attachment2: Tuple[int, CONNECTION_TYPES], 
                 min_length: float, max_length: float, min_tension: float = 0.0, max_tension: float = 1.0,
                 strength: float = 0.1, phase: float = 0.0, frequency: float = 1.0):
        self.attachment1: tuple[int, CONNECTION_TYPES] = attachment1
        self.attachment2: tuple[int, CONNECTION_TYPES] = attachment2
        self.min_length: float = min_length
        self.max_length: float = max_length
        self.min_tension: float = min_tension
        self.max_tension: float = max_tension
        self.strength: float = strength
        self.phase: float = phase
        self.current_tension: float = 0.0
        self.id = None  # Will be set when added to organism
        self.frequency: float = frequency
        self.current_tension: float = 0.0
        self.id = None
        self.control_signal: float = 0.0  # Added for neural control
        self.manual_control: bool = False  # Added to toggle between manual and automatic control
        
    def get_connection_point(self, target_id: int, connection_type: CONNECTION_TYPES, nodes: Dict[int, Node], bones: Dict[int, Bone]) -> torch.Tensor:
        if target_id in nodes:
            node = nodes[target_id]
            if connection_type == CONNECTION_TYPES.node_center:
                return node.position
            elif connection_type == CONNECTION_TYPES.node_edge:
                # For edge connection, we need to find a connected bone
                connected_bones = [b for b in bones.values() if b.node1_id == target_id or b.node2_id == target_id]
                if connected_bones:
                    # Use the first connected bone to determine edge position
                    bone = connected_bones[0]
                    other_node_id = bone.node2_id if bone.node1_id == target_id else bone.node1_id
                    direction = (nodes[other_node_id].position - node.position)
                    if torch.norm(direction) > 0:
                        direction = direction / torch.norm(direction)
                    return node.position + direction * node.radius
                return node.position  # Fallback if no connected bones
        elif target_id in bones:
            bone = bones[target_id]
            if connection_type == CONNECTION_TYPES.bone_center:
                node1 = nodes[bone.node1_id]
                node2 = nodes[bone.node2_id]
                return (node1.position + node2.position) / 2
        return torch.zeros(2)  # Default fallback
        
    def update(self, time_step, nodes: Dict[int, Node], bones: Dict[int, Bone]):
        if self.manual_control:
            # Use neural control signal directly
            self.current_tension = self.control_signal
        else:
            # Original oscillating behavior
            self.phase = (self.phase + 0.01 * self.frequency) % (2 * math.pi)
            self.current_tension = self.min_tension + (self.max_tension - self.min_tension) * ((math.sin(self.phase) + 1) / 2)
        
        # Rest of the update method remains the same...
        self.point1 = self.get_connection_point(self.attachment1[0], self.attachment1[1], nodes, bones)
        self.point2 = self.get_connection_point(self.attachment2[0], self.attachment2[1], nodes, bones)
        
        target_length = self.min_length + (self.max_length - self.min_length) * (1 - self.current_tension)
        current_length = torch.dist(self.point1, self.point2)
        
        direction = self.point2 - self.point1
        if torch.norm(direction) > 0:
            direction = direction / torch.norm(direction)
            
            displacement = current_length - target_length
            force = direction * displacement * self.strength
            
            self.apply_force_to_attachment(self.attachment1, force, nodes, bones)
            self.apply_force_to_attachment(self.attachment2, -force, nodes, bones)
            
    def apply_force_to_attachment(self, attachment: Tuple[int, CONNECTION_TYPES], force: torch.Tensor, 
                                 nodes: Dict[int, Node], bones: Dict[int, Bone]):
        target_id, connection_type = attachment
        
        if target_id in nodes:
            node = nodes[target_id]
            if connection_type == 'node_center':
                node.apply_force(force)
            elif connection_type == 'node_edge':
                # For edge connection, apply torque to rotate the node
                pass  # Implement torque application
                    
        elif target_id in bones:
            bone = bones[target_id]
            if connection_type == 'bone_center':
                # Apply half force to each bone node
                node1 = nodes[bone.node1_id]
                node2 = nodes[bone.node2_id]
                node1.apply_force(force / 2)
                node2.apply_force(force / 2)

    def to_dict(self):
        return {
            'attachment1': (self.attachment1[0], self.attachment1[1].value),
            'attachment2': (self.attachment2[0], self.attachment2[1].value),
            'min_length': self.min_length,
            'max_length': self.max_length,
            'min_tension': self.min_tension,
            'max_tension': self.max_tension,
            'strength': self.strength,
            'phase': self.phase,
            'current_tension': self.current_tension,
            'id': self.id
        }

    @classmethod
    def from_dict(cls, data):
        muscle = cls(
            attachment1=(data['attachment1'][0], CONNECTION_TYPES(data['attachment1'][1])),
            attachment2=(data['attachment2'][0], CONNECTION_TYPES(data['attachment2'][1])),
            min_length=data['min_length'],
            max_length=data['max_length'],
            min_tension=data['min_tension'],
            max_tension=data['max_tension'],
            strength=data['strength'],
            phase=data['phase']
        )
        muscle.current_tension = data['current_tension']
        muscle.id = data['id']
        return muscle

class Organism:
    def __init__(self, position, node_count=5, bone_count=8, muscle_density=0.5, name=None, is_mobile=True):
        self.nodes: Dict[int, Node] = {}
        self.bones: Dict[int, Bone] = {}
        self.muscles: Dict[int, Muscle] = {}
        self.position = torch.tensor(position, dtype=torch.float32)
        self.velocity = torch.zeros(2, dtype=torch.float32)
        self.name = name if name else self.generate_random_name()
        self.next_id = 0
        self.mass = 1.0  # Organism has mass too
        self.muscle_density = muscle_density
        self.is_mobile = is_mobile  # Track whether this is a mobile instance
        
        self.controller = None
        self.distance_traveled = 0.0
        self.last_position = None
        self.training_mode = False
        self.fitness = 0.0
        self.time = 0.0 
        
        # Only build body if we're not loading from JSON
        if node_count is not None and bone_count is not None:
            self.build_body(node_count, bone_count, muscle_density)
        

    def setup_neural_controller(self):
        if not self.muscles:
            return
            
        # Inputs: current phase (time), muscle lengths, node positions
        input_size = 1 + len(self.muscles) * 2 + len(self.nodes) * 2
        output_size = len(self.muscles)
        hidden_size = 16  # Small network for simple creatures
        
        self.controller = NeuralController(input_size, hidden_size, output_size)
        
        # Enable manual control for all muscles
        for muscle in self.muscles.values():
            muscle.manual_control = True
            
    def neural_control_step(self, time_step):
        if not self.controller or not self.muscles:
            return
            
        # Prepare input features
        inputs = []
        
        # 1. Time signal
        inputs.append(math.sin(self.time * 0.1))  # Oscillating time signal
        
        # 2. Muscle lengths
        for muscle in self.muscles.values():
            point1 = muscle.get_connection_point(muscle.attachment1[0], muscle.attachment1[1], self.nodes, self.bones)
            point2 = muscle.get_connection_point(muscle.attachment2[0], muscle.attachment2[1], self.nodes, self.bones)
            length = torch.dist(point1, point2)
            inputs.append(length.item())
            inputs.append(muscle.current_tension)
        
        # 3. Node positions (relative to center)
        com = self.position
        for node in self.nodes.values():
            rel_pos = node.position - com
            inputs.extend(rel_pos.tolist())
        
        # Convert to tensor and get control signals
        input_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        control_signals = self.controller(input_tensor).squeeze(0)
        
        # Apply control signals to muscles
        for i, (muscle_id, muscle) in enumerate(self.muscles.items()):
            muscle.control_signal = control_signals[i].item()
            
    def update_fitness(self):
        if self.last_position is None:
            self.last_position = self.position.clone()
            return
            
        # Calculate distance moved in x direction (ignore vertical movement)
        distance = (self.position[0] - self.last_position[0]).abs().item()
        self.distance_traveled += distance
        self.fitness += distance
        self.last_position = self.position.clone()

    def generate_random_name(self):
        prefix = random.choice(["Species", "Creature", "Organism", "Being", "Lifeform"])
        number = random.randint(100, 99999)
        return f"{prefix}-{number}"
        
    def get_next_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1
        
    def build_body(self, node_count, bone_count, muscle_density):
        center = torch.tensor([150, 600], dtype=torch.float32)
        radius: float = 50 + random.random() * 50
        min_y = SIM_HEIGHT - 100
        max_y = SIM_HEIGHT - 150
        
        # Create nodes
        for _ in range(node_count):
            angle = 2 * math.pi * len(self.nodes) / node_count + random.uniform(-0.2, 0.2)
            offset = torch.tensor([
                math.cos(angle) * radius * random.uniform(0.8, 1.2),
                math.sin(angle) * radius * random.uniform(0.8, 1.2)
            ])
            
            node_y = min(max((center + offset)[1].item(), 50), max_y)
            node_pos = [(center + offset)[0].item(), node_y]
            
            node = Node(node_pos, 
                    radius=8 + random.random() * 4,
                    mass=0.5 + random.random() * 1.5)
            
            node_id = self.get_next_id()
            self.nodes[node_id] = node
        
        # First create a minimum spanning tree to ensure connectivity
        node_ids = list(self.nodes.keys())
        random.shuffle(node_ids)
        
        # Start with one node
        connected_nodes = {node_ids[0]}
        remaining_nodes = set(node_ids[1:])
        
        while remaining_nodes:
            # Find closest pair between connected and remaining nodes
            closest_pair = None
            min_dist = float('inf')
            
            for c_node in connected_nodes:
                for r_node in remaining_nodes:
                    dist = torch.dist(self.nodes[c_node].position, self.nodes[r_node].position)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (c_node, r_node)
            
            if closest_pair:
                # Create bone between them
                new_bone = Bone(closest_pair[0], closest_pair[1], stiffness=1.0)
                node1 = self.nodes[closest_pair[0]]
                node2 = self.nodes[closest_pair[1]]
                new_bone.rest_length = torch.dist(node1.position, node2.position)
                bone_id = self.get_next_id()
                new_bone.id = bone_id
                self.bones[bone_id] = new_bone
                new_bone.update_angle_restrictions(self.nodes)
                
                # Update connected sets
                connected_nodes.add(closest_pair[1])
                remaining_nodes.remove(closest_pair[1])
        
        # Now add remaining bones randomly, ensuring they don't create multipart creatures
        attempts = 0
        while len(self.bones) < bone_count and attempts < bone_count * 2:
            attempts += 1
            node1_id, node2_id = random.sample(node_ids, 2)
            
            # Skip if already connected
            if any(b for b in self.bones.values() if 
                (b.node1_id == node1_id and b.node2_id == node2_id) or 
                (b.node1_id == node2_id and b.node2_id == node1_id)):
                continue
                
            node1 = self.nodes[node1_id]
            node2 = self.nodes[node2_id]
            dist = torch.dist(node1.position, node2.position)
            if dist > 120:
                continue
                
            # Create new bone
            new_bone = Bone(node1_id, node2_id, stiffness=1.0)
            new_bone.rest_length = dist
            bone_id = self.get_next_id()
            new_bone.id = bone_id
            self.bones[bone_id] = new_bone
            new_bone.update_angle_restrictions(self.nodes)
        
        # Final check - if we somehow ended up with a multipart creature, fix it
        if not self.is_connected():
            self.fix_multipart_creature()
        
        self.remove_unconnected_nodes()
        
        # Create muscles with various connection types
        bone_ids = list(self.bones.keys())
        
        for bone_id in bone_ids:
            if random.random() < muscle_density:
                bone = self.bones[bone_id]
                
                # Choose random connection types
                conn1_type = random.choice(list(CONNECTION_TYPES))
                conn2_type = random.choice(list(CONNECTION_TYPES))
                
                # For node connections, choose a random connected node
                if conn1_type in [CONNECTION_TYPES.node_center, CONNECTION_TYPES.node_edge]:
                    conn1_node_id = random.choice([bone.node1_id, bone.node2_id])
                    conn1 = (conn1_node_id, conn1_type)
                else:
                    conn1 = (bone_id, conn1_type)
                    
                if conn2_type in [CONNECTION_TYPES.node_center, CONNECTION_TYPES.node_edge]:
                    # Connect to another node (could be same bone or different)
                    other_bone_id = random.choice(bone_ids)
                    other_bone = self.bones[other_bone_id]
                    if other_bone_id != bone_id:
                        conn2_node_id = random.choice([other_bone.node1_id, other_bone.node2_id])
                        conn2 = (conn2_node_id, conn2_type)
                    else:
                        # If same bone, choose the other node
                        other_nodes = [n_id for n_id in [other_bone.node1_id, other_bone.node2_id] 
                                      if n_id != conn1[0] or not conn1_type in [CONNECTION_TYPES.node_center, CONNECTION_TYPES.node_edge]]
                        if other_nodes:
                            conn2 = (random.choice(other_nodes), conn2_type)
                        else:
                            continue  # Skip if no valid connection
                else:
                    other_bone_id = random.choice(bone_ids)
                    conn2 = (other_bone_id, conn2_type)
                
                # Calculate reasonable length parameters
                point1 = self.get_connection_point(*conn1)
                point2 = self.get_connection_point(*conn2)
                current_len = torch.dist(point1, point2)
                
                min_len = current_len * (0.7 + random.random() * 0.2)
                max_len = current_len * (1.1 + random.random() * 0.3)
                min_tension = random.random() * 0.3
                max_tension = 0.7 + random.random() * 0.3
                strength = 0.1 + random.random() * 0.2
                phase = random.random() * 2 * math.pi
                
                muscle = Muscle(
                    conn1, conn2,
                    min_len, max_len,
                    min_tension, max_tension,
                    strength, phase
                )
                muscle_id = self.get_next_id()
                muscle.id = muscle_id
                self.muscles[muscle_id] = muscle
        
        # Position the organism
        for node in self.nodes.values():
            node.position += self.position

    def fix_multipart_creature(self):
        if not self.nodes or not self.bones:
            return
            
        # Find connected components
        adjacency = defaultdict(set)
        for bone in self.bones.values():
            adjacency[bone.node1_id].add(bone.node2_id)
            adjacency[bone.node2_id].add(bone.node1_id)
        
        components = []
        visited = set()
        
        for node_id in self.nodes:
            if node_id not in visited:
                # New component found
                component = set()
                queue = deque([node_id])
                visited.add(node_id)
                
                while queue:
                    current = queue.popleft()
                    component.add(current)
                    
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
        
        # If only one component, nothing to fix
        if len(components) <= 1:
            return
            
        # Connect components by finding closest pairs between them
        for i in range(len(components)-1):
            comp1 = components[i]
            comp2 = components[i+1]
            
            # Find closest pair between components
            closest_pair = None
            min_dist = float('inf')
            
            for node1 in comp1:
                for node2 in comp2:
                    dist = torch.dist(self.nodes[node1].position, self.nodes[node2].position)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (node1, node2)
            
            if closest_pair:
                # Create bone between them
                new_bone = Bone(closest_pair[0], closest_pair[1], stiffness=1.0)
                node1 = self.nodes[closest_pair[0]]
                node2 = self.nodes[closest_pair[1]]
                new_bone.rest_length = torch.dist(node1.position, node2.position)
                bone_id = self.get_next_id()
                new_bone.id = bone_id
                self.bones[bone_id] = new_bone
                new_bone.update_angle_restrictions(self.nodes)

    def get_connection_point(self, target_id: int, connection_type: CONNECTION_TYPES) -> torch.Tensor:
        if target_id in self.nodes:
            node = self.nodes[target_id]
            if connection_type == 'node_center':
                return node.position
            elif connection_type == 'node_edge':
                return node.position  # Placeholder
        elif target_id in self.bones:
            bone = self.bones[target_id]
            if connection_type == 'bone_center':
                node1 = self.nodes[bone.node1_id]
                node2 = self.nodes[bone.node2_id]
                return (node1.position + node2.position) / 2
        return torch.zeros(2)  # Default fallback
        
    def remove_unconnected_nodes(self):
        # Build adjacency list
        connections = defaultdict(set)
        for bone in self.bones.values():
            connections[bone.node1_id].add(bone.node2_id)
            connections[bone.node2_id].add(bone.node1_id)
        
        # Check for connected nodes
        to_keep = []
        for node_id, node in self.nodes.items():
            if node_id in connections or any(node_id in connections[n_id] for n_id in connections):
                to_keep.append(node_id)
        
        # Update our node list
        self.nodes = {n_id: self.nodes[n_id] for n_id in to_keep}
        
        # Remove any bones that reference removed nodes
        self.bones = {
            b_id: bone for b_id, bone in self.bones.items()
            if bone.node1_id in self.nodes and bone.node2_id in self.nodes
        }
        
        # Ensure we have at least 2 nodes and 1 bone
        if len(self.nodes) < 2 or len(self.bones) < 1:
            # If we don't have enough, just create a simple connected pair
            self.nodes = {}
            self.bones = {}
            self.next_id = 0
            
            node1_id = self.get_next_id()
            node2_id = self.get_next_id()
            
            self.nodes[node1_id] = Node([0, 600])
            self.nodes[node2_id] = Node([50, 600])
            
            bone_id = self.get_next_id()
            new_bone = Bone(node1_id, node2_id)
            new_bone.rest_length = torch.dist(self.nodes[node1_id].position, self.nodes[node2_id].position)
            new_bone.id = bone_id
            self.bones[bone_id] = new_bone

    def is_connected(self) -> bool:
        if not self.nodes or not self.bones:
            return False
        
        # Build adjacency list
        adjacency = defaultdict(set)
        for bone in self.bones.values():
            adjacency[bone.node1_id].add(bone.node2_id)
            adjacency[bone.node2_id].add(bone.node1_id)
        
        # Perform BFS to check connectivity
        visited = set()
        queue = deque()
        
        # Start with first node
        start_node = next(iter(self.nodes.keys()))
        queue.append(start_node)
        visited.add(start_node)
        
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # If we visited all nodes, it's connected
        return len(visited) == len(self.nodes)
        
    def draw(self, offset_x, gravity_enabled=True, scale=1.0, box_center=None):
        if box_center is not None:
            # Calculate center of mass
            com = torch.zeros(2)
            total_mass = 0
            for node in self.nodes.values():
                com += node.position * node.mass
                total_mass += node.mass
            com /= total_mass
            
            # Calculate offset to move creature to box center
            offset = torch.tensor(box_center) - com
            
            # Apply offset and scale to all nodes
            for node in self.nodes.values():
                node.position = (node.position + offset) * scale
                
            # Apply scale to bone rest lengths
            for bone in self.bones.values():
                bone.rest_length *= scale
                
            # Apply scale to muscle lengths
            for muscle in self.muscles.values():
                muscle.min_length *= scale
                muscle.max_length *= scale
                
            # Apply scale to node radii
            for node in self.nodes.values():
                node.radius *= scale
        
        # Draw bones
        for bone in self.bones.values():
            node1 = self.nodes[bone.node1_id]
            node2 = self.nodes[bone.node2_id]
            pos1 = node1.position.detach().numpy() - np.array([offset_x, 0])
            pos2 = node2.position.detach().numpy() - np.array([offset_x, 0])
            dpg.draw_line(pos1, pos2, color=(255, 255, 255, 255), thickness=2)
            
        # Draw nodes
        for node_id, node in self.nodes.items():
            pos = node.position.detach().numpy() - np.array([offset_x, 0])
            
            # Draw angle ranges for each connected bone
            angle_ranges = node.get_angle_ranges()
            for start_angle, end_angle in angle_ranges:
                # Draw a partial circle (arc) for the range
                arc_radius = node.radius * 1.5
                segments = 20
                
                # Calculate points along the arc
                points = []
                for i in range(segments + 1):
                    angle = start_angle + (end_angle - start_angle) * (i / segments)
                    x = pos[0] + math.cos(angle) * arc_radius
                    y = pos[1] + math.sin(angle) * arc_radius
                    points.append([x, y])
                
                # Draw the arc
                if len(points) > 1:
                    dpg.draw_polyline(points, color=(255, 255, 255, 100), thickness=1)
            
            # Draw the node itself (smaller now since we have arcs)
            color = (0, 255, 0, 255)
            dpg.draw_circle(pos, node.radius * 0.7, color=color, fill=color)
            
            # Draw a small dot in the center to indicate the exact node position
            dpg.draw_circle(pos, 2, color=(255, 255, 255, 255), fill=(255, 255, 255, 255))
            
        # Draw muscles as curves
        for muscle in self.muscles.values():
            # Get the connection points
            point1 = muscle.get_connection_point(muscle.attachment1[0], muscle.attachment1[1], self.nodes, self.bones)
            point2 = muscle.get_connection_point(muscle.attachment2[0], muscle.attachment2[1], self.nodes, self.bones)
            
            pos1 = point1.detach().numpy() - np.array([offset_x, 0])
            pos2 = point2.detach().numpy() - np.array([offset_x, 0])
            tension = muscle.current_tension
            
            # Calculate direction between points
            direction = np.array([pos2[0]-pos1[0], pos2[1]-pos1[1]])
            length = np.linalg.norm(direction)
            
            if length > 0:
                direction = direction / length
                perpendicular = np.array([-direction[1], direction[0]])
                
                # Control point will be in the middle, offset perpendicularly based on tension
                mid_point = [(pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2]
                offset_amount = 20 * tension * min(1, length/100)  # Scale offset by relative length
                control_point = [mid_point[0] + perpendicular[0] * offset_amount, 
                                mid_point[1] + perpendicular[1] * offset_amount]
                
                # Color based on tension (red when tense, blue when relaxed)
                color = (255, int(255 * (1 - tension)), int(255 * (0.5 + tension/2)), 255)
                
                # Draw the bezier curve
                dpg.draw_bezier_quadratic(
                    pos1, control_point, pos2,
                    color=color,
                    thickness=2 + tension * 2,  # Thicker when tense
                    segments=20
                )
            
        # Draw name if not in gravity (viewing in creature tab)
        if not gravity_enabled and box_center is not None:
            pos = (torch.tensor(box_center) * scale).numpy() - np.array([offset_x, 0])
            dpg.draw_text([pos[0] - 50, pos[1] - 60], self.name, color=(255, 255, 255, 255), size=15)

    def create_immobile_copy(self):
        """Create an immobile copy of this organism for display purposes"""
        copy = Organism(
            [0, 0],  # Position doesn't matter for immobile copies
            node_count=None,
            bone_count=None,
            muscle_density=self.muscle_density,
            name=self.name,
            is_mobile=False
        )
        
        # Copy all components
        copy.nodes = {nid: Node.from_dict(node.to_dict()) for nid, node in self.nodes.items()}
        copy.bones = {bid: Bone.from_dict(bone.to_dict()) for bid, bone in self.bones.items()}
        copy.muscles = {mid: Muscle.from_dict(muscle.to_dict()) for mid, muscle in self.muscles.items()}
        copy.next_id = self.next_id
        
        # Make all nodes fixed in the copy
        for node in copy.nodes.values():
            node.fixed = True
            
        return copy
        
    def update(self, time_step, gravity_enabled=True, gravity_type="creature"):
        if not self.is_mobile:
            return
        
        self.time += time_step

        total_mass = sum(node.mass for node in self.nodes.values()) + \
                    sum(bone.mass for bone in self.bones.values())
        self.mass = total_mass
        
        # Apply gravity based on type
        if gravity_enabled:
            if gravity_type == "creature":
                # Apply gravity to entire creature as a whole
                gravity_force = GRAVITY * self.mass
                for node in self.nodes.values():
                    node.apply_force(gravity_force / len(self.nodes))
            elif gravity_type == "bone":
                # Apply gravity to each bone separately
                for bone in self.bones.values():
                    bone.apply_gravity(self.nodes)
            elif gravity_type == "node":
                # Gravity will be applied in node.update()
                pass
        
        # Update muscles
        for muscle in self.muscles.values():
            muscle.update(time_step, self.nodes, self.bones)
            
        # Apply bone constraints
        for bone in self.bones.values():
            bone.apply_constraint(self.nodes)
            
        # Update nodes with the specified gravity type
        for node in self.nodes.values():
            node.update(gravity_enabled, gravity_type)
            
        # Calculate center of mass and velocity
        com = torch.zeros(2)
        total_velocity = torch.zeros(2)
        
        for node in self.nodes.values():
            com += node.position * node.mass
            total_velocity += node.velocity * node.mass
            
        com /= total_mass
        self.velocity = total_velocity / total_mass
        self.position = com

    def to_dict(self):
        return {
            'name': self.name,
            'position': tensor_to_list(self.position),
            'velocity': tensor_to_list(self.velocity),
            'mass': self.mass,
            'muscle_density': self.muscle_density,
            'is_mobile': self.is_mobile,
            'next_id': self.next_id,
            'nodes': {str(nid): node.to_dict() for nid, node in self.nodes.items()},  # Convert keys to strings
            'bones': {str(bid): bone.to_dict() for bid, bone in self.bones.items()},
            'muscles': {str(mid): muscle.to_dict() for mid, muscle in self.muscles.items()},
            'has_controller': self.controller is not None,
            'controller_state': self.controller.state_dict() if self.controller else None,
            'fitness': self.fitness
        }

    @classmethod
    def from_dict(cls, data):
        """Create an organism from a dictionary (JSON deserialization)"""
        org = cls(
            position=data['position'],
            node_count=None,
            bone_count=None,
            muscle_density=data['muscle_density'],
            name=data['name'],
            is_mobile=data['is_mobile']
        )
        
        # Restore all components
        org.position = list_to_tensor(data['position'])
        org.velocity = list_to_tensor(data['velocity'])
        org.mass = data['mass']
        org.next_id = data['next_id']
        
        # Restore nodes (convert string keys back to integers)
        org.nodes = {int(nid): Node.from_dict(node_data) for nid, node_data in data['nodes'].items()}
        
        # Restore bones
        org.bones = {int(bid): Bone.from_dict(bone_data) for bid, bone_data in data['bones'].items()}
        
        # Restore muscles
        org.muscles = {int(mid): Muscle.from_dict(muscle_data) for mid, muscle_data in data['muscles'].items()}
        
        if data['has_controller'] and data['controller_state']:
            org.setup_neural_controller()
            org.controller.load_state_dict(data['controller_state'])
            org.controller.optimizer = optim.Adam(org.controller.parameters(), lr=0.01)
            
            # Enable manual control for muscles
            for muscle in org.muscles.values():
                muscle.manual_control = True
                
        org.fitness = data.get('fitness', 0.0)

        return org

class Simulation:
    def __init__(self):
        self.organisms: list[Organism] = []  # Mobile instances in simulation
        self.all_creatures: list[Organism] = []  # Stores all creature templates (immobile)
        self.time = 0
        self.camera_offset_x = 0
        self.follow_organism = None
        self.creation_params = {
            'min_nodes': 4,
            'max_nodes': 8,
            'min_bones': 6,
            'max_bones': 15,
            'muscle_density': 0.6
        }
        self.gravity_type = "creature"  # Default to creature-level gravity
        
        self.training = False
        self.generation = 0
        self.best_fitness = 0.0
        self.best_organism = None

        # Initialize Dear PyGui
        dpg.create_context()
        dpg.create_viewport(title='Organism Walking Simulation', width=SIM_WIDTH, height=SIM_HEIGHT)
        dpg.setup_dearpygui()
        
        # Create main window
        with dpg.window(tag="Primary Window"):
            with dpg.tab_bar():
                # Main simulation tab
                with dpg.tab(label="Simulation"):
                    # Create a splitter for side panel and simulation view
                    with dpg.group(horizontal=True):
                        # Side panel for controls
                        with dpg.child_window(width=300):
                            dpg.add_text("Controls")
                            dpg.add_button(label="Create Random Creature", callback=self.create_organism)
                            dpg.add_button(label="Reset Simulation", callback=self.reset_simulation)
                            dpg.add_button(label="Save Creatures", callback=self.save_creatures)
                            dpg.add_button(label="Load Creatures", callback=self.load_creatures)
                            
                            with dpg.collapsing_header(label="Gravity Settings"):
                                dpg.add_radio_button(
                                    items=["Creature", "Bone", "Node"],
                                    label="Gravity Type",
                                    default_value="Creature",
                                    callback=self.set_gravity_type
                                )

                            with dpg.collapsing_header(label="Creature Parameters"):
                                dpg.add_slider_int(label="Min Nodes", min_value=3, max_value=10, default_value=self.creation_params['min_nodes'], 
                                                  callback=lambda s, a: self.param_update('min_nodes', a))
                                dpg.add_slider_int(label="Max Nodes", min_value=4, max_value=15, default_value=self.creation_params['max_nodes'], 
                                                  callback=lambda s, a: self.param_update('max_nodes', a))
                                dpg.add_slider_int(label="Min Bones", min_value=3, max_value=20, default_value=self.creation_params['min_bones'], 
                                                  callback=lambda s, a: self.param_update('min_bones', a))
                                dpg.add_slider_int(label="Max Bones", min_value=4, max_value=30, default_value=self.creation_params['max_bones'], 
                                                  callback=lambda s, a: self.param_update('max_bones', a))
                                dpg.add_slider_float(label="Muscle Density", min_value=0.1, max_value=1.0, default_value=self.creation_params['muscle_density'], 
                                                   callback=lambda s, a: self.param_update('muscle_density', a))
                            
                            with dpg.collapsing_header(label="Learning Controls"):
                                dpg.add_button(label="Setup Neural Control", callback=self.setup_neural_control)
                                dpg.add_button(label="Start Training", callback=self.start_training)
                                dpg.add_button(label="Stop Training", callback=self.stop_training)
                                dpg.add_text("Fitness: 0.0", tag="fitness_display")

                            dpg.add_text("Camera Controls")
                            dpg.add_slider_float(label="Camera Speed", default_value=5.0, min_value=1.0, max_value=20.0, callback=self.set_camera_speed)
                            self.camera_speed = 5.0
                            
                        # Simulation view
                        with dpg.child_window(tag="simulation_view"):
                            with dpg.drawlist(width=SIM_WIDTH-300, height=SIM_HEIGHT) as self.draw_node:
                                pass
                
                # Creature gallery tab
                with dpg.tab(label="Creature Gallery"):
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=300, tag="creature_list"):
                            dpg.add_text("Saved Creatures")
                            with dpg.group() as self.creature_list_container:
                                self.creature_list = dpg.add_listbox(
                                    items=[],
                                    width=280,
                                    num_items=10,
                                    callback=self.select_creature_from_list
                                )
                            dpg.add_button(label="Add to Simulation", callback=self.add_selected_to_simulation)
                            dpg.add_button(label="Delete Creature", callback=self.delete_selected_creature)
                        
                        with dpg.child_window(tag="creature_view"):
                            with dpg.drawlist(width=SIM_WIDTH-300, height=SIM_HEIGHT) as self.creature_draw_node:
                                pass
        
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)
        
        # Load creatures from file if it exists
        self.load_creatures()

    def setup_neural_control(self):
        if not self.organisms:
            return
            
        for org in self.organisms:
            org.setup_neural_controller()
            
    def start_training(self):
        if not self.organisms:
            return
            
        self.training = True
        self.generation = 0
        self.best_fitness = 0.0
        self.current_trial = 0
        self.trial_start_time = 0
        self.training_timesteps = 0
        
        # Reset all organisms
        for org in self.organisms:
            org.training_mode = True
            org.distance_traveled = 0.0
            org.fitness = 0.0
            org.last_position = None
            org.fitness_history = []  # Track fitness across trials
            
    def update(self):
        # Update all organisms in simulation
        for organism in self.organisms:
            if organism.training_mode and organism.controller:
                organism.neural_control_step(TIME_STEP)
                organism.update_fitness()
                
            organism.update(TIME_STEP, gravity_enabled=True, gravity_type=self.gravity_type)
            
        if self.training and self.organisms:
            self.training_timesteps += 1
            
            # Update fitness display with current best
            current_best = max(org.fitness for org in self.organisms)
            dpg.set_value("fitness_display", 
                         f"Gen {self.generation} Trial {self.current_trial+1}/{MAX_TRIALS}\n"
                         f"Best: {self.best_fitness:.2f} Current: {current_best:.2f}")
            
            # Check if trial should end (after TRAINING_TIMESTEPS)
            if self.training_timesteps >= TRAINING_TIMESTEPS:
                self.end_trial()
                
                # If we've completed all trials, go to next generation
                if self.current_trial >= MAX_TRIALS:
                    self.next_generation()
            
        # Update all creatures in gallery (without gravity)
        for creature in self.all_creatures:
            creature.update(TIME_STEP, gravity_enabled=True)
            
        # Camera follow logic (only in non-training or first trial)
        if self.follow_organism and (not self.training or self.current_trial == 0):
            target_x = self.follow_organism.position[0].item() - SIM_WIDTH/3
            self.camera_offset_x += (target_x - self.camera_offset_x) * 0.1 * self.camera_speed * TIME_STEP
            
        # Draw everything
        self.draw_simulation()
        self.draw_creature_gallery()
        
        # Accelerate training by running multiple updates per frame
        if self.training and self.training_timesteps < TRAINING_TIMESTEPS:
            # Run additional updates to speed up training
            for _ in range(TRAINING_SPEEDUP - 1):
                for organism in self.organisms:
                    if organism.training_mode and organism.controller:
                        organism.neural_control_step(TIME_STEP)
                        organism.update_fitness()
                    organism.update(TIME_STEP, gravity_enabled=True, gravity_type=self.gravity_type)
                self.training_timesteps += 1
                
    def end_trial(self):
        """Record fitness for current trial and prepare for next trial"""
        self.current_trial += 1
        
        # Record fitness for each organism
        for org in self.organisms:
            org.fitness_history.append(org.fitness)
            
        # Reset for next trial
        self.training_timesteps = 0
        for org in self.organisms:
            org.fitness = 0.0
            org.distance_traveled = 0.0
            org.last_position = None
            # Reset position but keep slight randomness
            org.position = torch.tensor([
                random.uniform(-50, 50), 
                SIM_HEIGHT - 150
            ], dtype=torch.float32)
            # Reset velocities
            for node in org.nodes.values():
                node.velocity = torch.zeros(2, dtype=torch.float32)
                
    def next_generation(self):
        """Create next generation after all trials are complete"""
        if not self.organisms:
            return
            
        self.generation += 1
        self.current_trial = 0
        
        # Calculate average fitness for each organism
        for org in self.organisms:
            if org.fitness_history:
                org.avg_fitness = sum(org.fitness_history) / len(org.fitness_history)
            else:
                org.avg_fitness = 0.0
                
        # Sort by average fitness
        self.organisms.sort(key=lambda x: x.avg_fitness, reverse=True)
        best_org = self.organisms[0]
        
        if best_org.avg_fitness > self.best_fitness:
            self.best_fitness = best_org.avg_fitness
            self.best_organism = best_org
            
        # Create new generation through mutation
        new_organisms = []
        for i in range(len(self.organisms)):
            # Create a copy of the best organism
            new_org = self.copy_organism(best_org)
            
            # Only mutate if the organism has a controller
            if new_org.controller:
                self.mutate_controller(new_org.controller)
            
            # Reset for new generation
            new_org.fitness = 0.0
            new_org.distance_traveled = 0.0
            new_org.last_position = None
            new_org.fitness_history = []
            new_org.position = torch.tensor([0, SIM_HEIGHT - 150], dtype=torch.float32)
            # Reset velocities
            for node in new_org.nodes.values():
                node.velocity = torch.zeros(2, dtype=torch.float32)
                
            new_organisms.append(new_org)
            
        self.organisms = new_organisms
            
    def stop_training(self):
        self.training = False
        for org in self.organisms:
            org.training_mode = False
            
    def param_update(self, param, value):
        self.creation_params[param] = value
        
    def set_camera_speed(self, sender, app_data):
        self.camera_speed = app_data
        
    def create_organism(self):
        # Remove current organism from simulation (but keep in gallery)
        if self.organisms:
            self.organisms = []
        
        # Get random values within specified ranges
        node_count = random.randint(
            self.creation_params['min_nodes'],
            self.creation_params['max_nodes']
        )
        bone_count = random.randint(
            self.creation_params['min_bones'],
            self.creation_params['max_bones']
        )
        
        # Generate a unique name
        existing_names = {creature.name for creature in self.all_creatures}
        new_name = None
        while new_name is None or new_name in existing_names:
            new_name = f"Species-{random.randint(1000, 9999)}"
        
        start_y = 0
        # Create mobile instance for simulation
        new_org = Organism(
            [0, start_y],
            node_count=node_count,
            bone_count=bone_count,
            muscle_density=self.creation_params['muscle_density'],
            name=new_name,
            is_mobile=True
        )
        
        # Create immobile copy for gallery
        gallery_copy = new_org.create_immobile_copy()
        
        self.organisms.append(new_org)
        self.all_creatures.append(gallery_copy)
        self.follow_organism = new_org
        self.camera_offset_x = 0
        
        # Update creature list
        self.update_creature_list()
        
    def reset_simulation(self):
        self.organisms = []
        self.follow_organism = None
        self.camera_offset_x = 0
        
    def update_creature_list(self):
        try:
            creature_names = [creature.name for creature in self.all_creatures]
            dpg.configure_item(self.creature_list, items=creature_names)
        except Exception as e:
            print(f"Error updating creature list: {e}")
            
    def select_creature_from_list(self, sender, app_data):
        # Convert app_data to integer if it's a string
        if isinstance(app_data, str):
            try:
                app_data = int(app_data)
            except ValueError:
                return
                
        if isinstance(app_data, int) and 0 <= app_data < len(self.all_creatures):
            self.selected_creature = self.all_creatures[app_data]
            
    def add_selected_to_simulation(self):
        if hasattr(self, 'selected_creature'):
            # Remove current organism from simulation
            self.organisms = []
            
            # Create a mobile instance from the selected template
            selected = self.selected_creature
            new_org = Organism(
                [0, SIM_HEIGHT - 150],  # Start above ground
                node_count=None,
                bone_count=None,
                muscle_density=selected.muscle_density,
                name=selected.name,
                is_mobile=True
            )
            
            # Copy all components from the selected creature
            new_org.nodes = {nid: Node.from_dict(node.to_dict()) for nid, node in selected.nodes.items()}
            new_org.bones = {bid: Bone.from_dict(bone.to_dict()) for bid, bone in selected.bones.items()}
            new_org.muscles = {mid: Muscle.from_dict(muscle.to_dict()) for mid, muscle in selected.muscles.items()}
            new_org.next_id = selected.next_id
            
            # Add to simulation
            self.organisms.append(new_org)
            self.follow_organism = new_org
            self.camera_offset_x = 0
            
    def delete_selected_creature(self):
        if hasattr(self, 'selected_creature'):
            # Remove from both lists
            self.all_creatures = [c for c in self.all_creatures if c.name != self.selected_creature.name]
            self.organisms = [o for o in self.organisms if o.name != self.selected_creature.name]
            
            # Update the list
            self.update_creature_list()
            
            # Clear selection
            if hasattr(self, 'selected_creature'):
                del self.selected_creature

    def set_gravity_type(self, sender, app_data):
        self.gravity_type = app_data.lower()

    def save_creatures(self):
        """Save all creatures to JSON file"""
        try:
            # Convert all creatures to dictionaries
            creatures_data = [creature.to_dict() for creature in self.all_creatures]
            
            # Save to file
            with open(CREATURE_DB_FILE, 'w') as f:
                json.dump(creatures_data, f, indent=2)
                
            print(f"Saved {len(creatures_data)} creatures to {CREATURE_DB_FILE}")
        except Exception as e:
            print(f"Error saving creatures: {e}")

    def load_creatures(self):
        """Load creatures from JSON file if it exists"""
        try:
            if os.path.exists(CREATURE_DB_FILE):
                with open(CREATURE_DB_FILE, 'r') as f:
                    creatures_data = json.load(f)
                
                # Clear current lists
                self.all_creatures = []
                self.organisms = []
                
                # Create organisms from data
                for creature_data in creatures_data:
                    org = Organism.from_dict(creature_data)
                    self.all_creatures.append(org)
                
                # Update the creature list
                self.update_creature_list()
                print(f"Loaded {len(self.all_creatures)} creatures from {CREATURE_DB_FILE}")
        except Exception as e:
            print(f"Error loading creatures: {e}")

    def draw_simulation(self):
        dpg.delete_item(self.draw_node, children_only=True)
        
        with dpg.mutex():
            with dpg.draw_node(parent=self.draw_node):
                # Draw infinite floor
                floor_y = SIM_HEIGHT - 20
                
                # Calculate visible range based on camera offset
                start_x = self.camera_offset_x - 100  # Some padding
                end_x = self.camera_offset_x + SIM_WIDTH + 100
                
                # Draw floor rectangle for visible area
                dpg.draw_rectangle([0, floor_y], [SIM_WIDTH, SIM_HEIGHT], 
                                color=(100, 80, 60, 255), fill=(100, 80, 60, 255))
                
                # Draw distance markers (aligned to world space, not screen space)
                first_marker = int((start_x) // 100) * 100
                last_marker = int((end_x) // 100) * 100
                
                for x in range(first_marker, last_marker + 100, 100):
                    screen_x = x - self.camera_offset_x
                    
                    # Only draw if visible
                    if -100 < screen_x < SIM_WIDTH + 100:
                        # Main marker line
                        dpg.draw_line([screen_x, floor_y], [screen_x, floor_y - 15], 
                                    color=(200, 200, 200, 255), thickness=1)
                        
                        # Smaller markers at 50, 25, and 75 positions
                        if screen_x + 50 < SIM_WIDTH + 100:
                            dpg.draw_line([screen_x + 50, floor_y], [screen_x + 50, floor_y - 10], 
                                        color=(200, 200, 200, 200), thickness=1)
                        if screen_x + 25 < SIM_WIDTH + 100:
                            dpg.draw_line([screen_x + 25, floor_y], [screen_x + 25, floor_y - 7], 
                                        color=(200, 200, 200, 150), thickness=1)
                        if screen_x + 75 < SIM_WIDTH + 100:
                            dpg.draw_line([screen_x + 75, floor_y], [screen_x + 75, floor_y - 7], 
                                        color=(200, 200, 200, 150), thickness=1)
                        
                        # Distance text every 100 pixels
                        if x % 100 == 0 and x > 0:
                            dpg.draw_text([screen_x - 10, floor_y - 30], str(x), 
                                        color=(255, 255, 255, 255), size=15)
                
                # Draw organisms in simulation
                for organism in self.organisms:
                    organism.draw(self.camera_offset_x, gravity_enabled=True)
    
    def draw_creature_gallery(self):
        dpg.delete_item(self.creature_draw_node, children_only=True)
        
        with dpg.mutex():
            with dpg.draw_node(parent=self.creature_draw_node):
                # Draw a neutral background
                dpg.draw_rectangle([0, 0], [SIM_WIDTH-300, SIM_HEIGHT], 
                                color=(50, 50, 50, 255), fill=(50, 50, 50, 255))
                
                # Draw all creatures in a grid of 100x100 boxes
                box_size = 100
                padding = 20
                cols = (SIM_WIDTH - 300) // (box_size + padding)
                rows = math.ceil(len(self.all_creatures) / cols)
                
                for i, creature in enumerate(self.all_creatures):
                    row = i // cols
                    col = i % cols
                    
                    # Calculate box position
                    box_x = col * (box_size + padding) + padding
                    box_y = row * (box_size + padding) + padding
                    box_center = [box_x + box_size//2, box_y + box_size//2]
                    
                    # Calculate bounding box of creature
                    min_x = min(node.position[0].item() for node in creature.nodes.values())
                    max_x = max(node.position[0].item() for node in creature.nodes.values())
                    min_y = min(node.position[1].item() for node in creature.nodes.values())
                    max_y = max(node.position[1].item() for node in creature.nodes.values())
                    
                    # Calculate required scale to fit in box (with 10px margin)
                    margin = 10
                    creature_width = max_x - min_x
                    creature_height = max_y - min_y
                    
                    scale_x = (box_size - 2*margin) / max(creature_width, 1)
                    scale_y = (box_size - 2*margin) / max(creature_height, 1)
                    scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down
                    
                    # Draw the box border
                    dpg.draw_rectangle(
                        [box_x, box_y],
                        [box_x + box_size, box_y + box_size],
                        color=(100, 100, 100, 255),
                        thickness=1
                    )
                    
                    # Draw the creature centered and scaled in the box
                    creature.draw(0, gravity_enabled=False, scale=scale, box_center=box_center)
                    
                    # Draw creature name below the box
                    dpg.draw_text(
                        [box_x, box_y + box_size + 5],
                        creature.name,
                        color=(255, 255, 255, 255),
                        size=12
                    )

    def run(self):
        # Main loop
        while dpg.is_dearpygui_running():
            self.update()
            dpg.render_dearpygui_frame()
            
        # Save creatures when closing
        self.save_creatures()
        dpg.destroy_context()

    def copy_organism(self, org):
        # Create a new organism with the same structure
        new_org = Organism(
            [0, SIM_HEIGHT - 150],
            node_count=None,
            bone_count=None,
            muscle_density=org.muscle_density,
            name=f"{org.name}-gen{self.generation}",
            is_mobile=True
        )
        
        # Copy all components
        new_org.nodes = {nid: Node.from_dict(node.to_dict()) for nid, node in org.nodes.items()}
        new_org.bones = {bid: Bone.from_dict(bone.to_dict()) for bid, bone in org.bones.items()}
        new_org.muscles = {mid: Muscle.from_dict(muscle.to_dict()) for mid, muscle in org.muscles.items()}
        new_org.next_id = org.next_id
        
        # Copy the neural controller if it exists
        if org.controller:
            new_org.setup_neural_controller()  # This initializes the controller with the right architecture
            new_org.controller.load_state_dict(org.controller.state_dict())
            new_org.controller.optimizer = optim.Adam(new_org.controller.parameters(), lr=0.01)
            
            # Enable manual control
            for muscle in new_org.muscles.values():
                muscle.manual_control = True
                
        return new_org
        
    def mutate_controller(self, controller, mutation_rate=0.1, mutation_scale=0.2):
        if controller is None:
            return
            
        with torch.no_grad():
            for param in controller.parameters():
                if random.random() < mutation_rate:
                    param.add_(torch.randn_like(param) * mutation_scale)

# Run the simulation
if __name__ == "__main__":
    sim = Simulation()
    sim.run()