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
class CONNECTION_TYPES(Enum):
    node_center = 0
    node_edge = 1
    bone_center = 2


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

class Bone:
    def __init__(self, node1_id: int, node2_id: int, length=1, stiffness=1.0):
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.stiffness = stiffness
        self.id: int = -1  # Will be set when added to organism
        self.rest_length = length  # This might be None initially
        self.mass = 0.5  # Bones have some mass too
        
    def update_angle_restrictions(self, nodes: Dict[int, Node]):
        """Update angle restrictions based on current bone positions."""
        node1 = nodes[self.node1_id]
        node2 = nodes[self.node2_id]
        
        # Calculate direction vector
        direction = node2.position - node1.position
        current_angle = math.atan2(direction[1], direction[0])
        
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
                other_bone = next((b for b in nodes.values() if hasattr(b, 'id') and b.id == bone_id), None)
                if other_bone is None:
                    continue
                    
                # Calculate current angle between this bone and the other bone
                vec1 = node2.position - node1.position
                vec2 = nodes[other_bone.node2_id].position - nodes[other_bone.node1_id].position
                
                angle = math.atan2(vec2[1], vec2[0]) - math.atan2(vec1[1], vec1[0])
                angle = (angle + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π]
                
                # Check if angle is outside natural range
                if angle < nat_min or angle > nat_max:
                    # Calculate correction force
                    correction_factor = 0.0
                    if angle < min_angle:
                        correction_factor = (min_angle - angle) * 0.1
                    elif angle > max_angle:
                        correction_factor = (max_angle - angle) * 0.1
                    elif angle < nat_min:
                        correction_factor = (nat_min - angle) * 0.05
                    elif angle > nat_max:
                        correction_factor = (nat_max - angle) * 0.05
                        
                    if correction_factor != 0.0:
                        # Apply corrective torque to both nodes
                        torque_dir = 1 if correction_factor > 0 else -1
                        perp1 = torch.tensor([-vec1[1], vec1[0]])
                        perp2 = torch.tensor([vec1[1], -vec1[0]])
                        
                        node1.apply_force(perp1 * torque_dir * abs(correction_factor) * 0.1)
                        node2.apply_force(perp2 * torque_dir * abs(correction_factor) * 0.1)
                        
    def apply_gravity(self, nodes: Dict[int, Node]):
        """Apply gravity to the bone based on its center of mass"""
        node1 = nodes[self.node1_id]
        node2 = nodes[self.node2_id]
        center = (node1.position + node2.position) / 2
        gravity_force = GRAVITY * self.mass
        
        # Apply half to each node
        node1.apply_force(gravity_force / 2)
        node2.apply_force(gravity_force / 2)

class Muscle:
    def __init__(self, attachment1: Tuple[int, CONNECTION_TYPES], attachment2: Tuple[int, CONNECTION_TYPES], min_length: float, max_length: float,
                 min_tension: float = 0.0, max_tension: float = 1.0, strength: float = 0.1, phase: float = 0.0):
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
        # Oscillate tension based on time within min/max range
        self.phase = (self.phase + 0.01) % (2 * math.pi)
        self.current_tension = self.min_tension + (self.max_tension - self.min_tension) * ((math.sin(self.phase) + 1) / 2)
        
        # Update connection points (in case nodes have moved)
        self.point1 = self.get_connection_point(self.attachment1[0], self.attachment1[1], nodes, bones)
        self.point2 = self.get_connection_point(self.attachment2[0], self.attachment2[1], nodes, bones)
        
        # Calculate target length based on current tension
        target_length = self.min_length + (self.max_length - self.min_length) * (1 - self.current_tension)
        current_length = torch.dist(self.point1, self.point2)
        
        # Apply force to reach target length
        direction = self.point2 - self.point1
        if torch.norm(direction) > 0:
            direction = direction / torch.norm(direction)
            
            displacement = current_length - target_length
            force = direction * displacement * self.strength
            
            # Apply forces to the appropriate attachment points
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
        self.build_body(node_count, bone_count, muscle_density)

        self.is_mobile = is_mobile  # Track whether this is a mobile instance
        
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
            
            # if random.random() < 0.2 and len(self.nodes) > 1:
            #     node.fixed = True
                
            node_id = self.get_next_id()
            self.nodes[node_id] = node
        
        # Create bones with rigid constraints
        attempts = 0
        node_ids = list(self.nodes.keys())
        
        while len(self.bones) < bone_count and attempts < bone_count * 2:
            attempts += 1
            node1_id, node2_id = random.sample(node_ids, 2)
            
            if any(b for b in self.bones.values() if 
                  (b.node1_id == node1_id and b.node2_id == node2_id) or 
                  (b.node1_id == node2_id and b.node2_id == node1_id)):
                continue
                
            node1 = self.nodes[node1_id]
            node2 = self.nodes[node2_id]
            dist = torch.dist(node1.position, node2.position)
            if dist > 120:
                continue
                
            # Create rigid bone (stiffness = 1.0)
            new_bone = Bone(node1_id, node2_id, stiffness=1.0)
            new_bone.rest_length = dist  # Set the rest length to current distance
            bone_id = self.get_next_id()
            new_bone.id = bone_id
            self.bones[bone_id] = new_bone
            new_bone.update_angle_restrictions(self.nodes)
        
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
        for node in self.nodes.values():
            pos = node.position.detach().numpy() - np.array([offset_x, 0])
            color = (0, 255, 0, 255)
            dpg.draw_circle(pos, node.radius, color=color)
            
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
            node_count=len(self.nodes),
            bone_count=len(self.bones),
            muscle_density=self.muscle_density,
            name=self.name,
            is_mobile=False
        )
        
        # Make all nodes fixed in the copy
        for node in copy.nodes.values():
            node.fixed = True
            
        return copy
        
    def update(self, time_step, gravity_enabled=True, gravity_type="creature"):
        if not self.is_mobile:
            return  # Skip update for immobile instances
            
        # Rest of the update logic remains the same...
        # Calculate total mass for the organism
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
                        
                        with dpg.child_window(tag="creature_view"):
                            with dpg.drawlist(width=SIM_WIDTH-300, height=SIM_HEIGHT) as self.creature_draw_node:
                                pass
        
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)
        
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
                [0, 0],
                node_count=len(selected.nodes),
                bone_count=len(selected.bones),
                muscle_density=self.creation_params['muscle_density'],
                name=selected.name,
                is_mobile=True
            )
            
            # Add to simulation
            self.organisms.append(new_org)
            self.follow_organism = new_org
            self.camera_offset_x = 0
            
    def set_gravity_type(self, sender, app_data):
        self.gravity_type = app_data.lower()

    def update(self):
        self.time += TIME_STEP
        
        # Update all organisms in simulation
        for organism in self.organisms:
            organism.fixed = False
            organism.update(TIME_STEP, gravity_enabled=True, gravity_type=self.gravity_type)
            
        # Update all creatures in gallery (without gravity)
        for creature in self.all_creatures:
            #creature.fixed = True
            creature.update(TIME_STEP, gravity_enabled=True)
            
        # Camera follow logic
        if self.follow_organism:
            target_x = self.follow_organism.position[0].item() - SIM_WIDTH/3
            self.camera_offset_x += (target_x - self.camera_offset_x) * 0.1 * self.camera_speed * TIME_STEP
            
        # Draw everything
        self.draw_simulation()
        self.draw_creature_gallery()
                
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
                    #organism.fixed = False
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
            
        dpg.destroy_context()

# Run the simulation
if __name__ == "__main__":
    sim = Simulation()
    sim.run()