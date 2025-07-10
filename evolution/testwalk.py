import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dearpygui.dearpygui as dpg
import math
import random
from collections import defaultdict

# Simulation parameters
SIM_WIDTH = 1200
SIM_HEIGHT = 800
GRAVITY = torch.tensor([0.0, 0.5])
FRICTION = 0.99
TIME_STEP = 0.1
MUSCLE_CYCLE_LENGTH = 100

class Node:
    def __init__(self, position, radius: float=10, mass=1.0):
        self.position = torch.tensor(position, dtype=torch.float32, requires_grad=False)
        self.velocity = torch.zeros(2, dtype=torch.float32)
        self.force = torch.zeros(2, dtype=torch.float32)
        self.radius: float = radius
        self.mass = mass
        self.fixed = False
        
    def apply_force(self, force):
        self.force += force
        
    def update(self):
        if not self.fixed:
            # Apply gravity
            self.apply_force(GRAVITY * self.mass)
            
            # Update velocity and position
            acceleration = self.force / self.mass
            self.velocity += acceleration * TIME_STEP
            self.velocity *= FRICTION
            self.position += self.velocity * TIME_STEP
            
            # Reset force
            self.force = torch.zeros(2, dtype=torch.float32)
            
            # Simple ground collision
            if self.position[1] > SIM_HEIGHT - self.radius - 20:  # Account for floor thickness
                self.position[1] = SIM_HEIGHT - self.radius - 20
                self.velocity[1] *= -0.5

class Bone:
    def __init__(self, node1, node2, length=None, stiffness=0.5):
        self.node1 = node1
        self.node2 = node2
        self.rest_length = length if length is not None else torch.dist(node1.position, node2.position)
        self.stiffness = stiffness
        
    def apply_constraint(self):
        direction = self.node2.position - self.node1.position
        distance = torch.norm(direction)
        if distance > 0:
            direction = direction / distance
            
            displacement = distance - self.rest_length
            force = direction * displacement * self.stiffness
            
            if not self.node1.fixed:
                self.node1.apply_force(force)
            if not self.node2.fixed:
                self.node2.apply_force(-force)

class Muscle:
    def __init__(self, bone, min_length, max_length, strength=0.1, phase=0.0):
        self.bone = bone
        self.min_length = min_length
        self.max_length = max_length
        self.strength = strength
        self.phase = phase
        self.current_tension = 0.0
        
    def update(self, time_step):
        # Oscillate tension based on time
        self.phase = (self.phase + 0.01) % (2 * math.pi)
        self.current_tension = (math.sin(self.phase) + 1) / 2  # 0-1 range
        
        target_length = self.min_length + (self.max_length - self.min_length) * self.current_tension
        current_length = torch.dist(self.bone.node1.position, self.bone.node2.position)
        
        # Apply force to reach target length
        direction = self.bone.node2.position - self.bone.node1.position
        if torch.norm(direction) > 0:
            direction = direction / torch.norm(direction)
            
            displacement = current_length - target_length
            force = direction * displacement * self.strength
            
            if not self.bone.node1.fixed:
                self.bone.node1.apply_force(force)
            if not self.bone.node2.fixed:
                self.bone.node2.apply_force(-force)

class Organism:
    def __init__(self, position, node_count=5, bone_count=8, muscle_density=0.5):
        self.nodes = []
        self.bones = []
        self.muscles = []
        self.position = torch.tensor(position, dtype=torch.float32)
        self.velocity = torch.zeros(2, dtype=torch.float32)
        self.build_body(node_count, bone_count, muscle_density)
        
    def build_body(self, node_count, bone_count, muscle_density):
        # Create nodes in a roughly circular pattern
        center = torch.tensor([150, 600], dtype=torch.float32)
        radius: float = 50 + random.random() * 50
        
        for i in range(node_count):
            # Distribute nodes around a circle with some randomness
            angle = 2 * math.pi * i / node_count + random.uniform(-0.2, 0.2)
            offset = torch.tensor([
                math.cos(angle) * radius * random.uniform(0.8, 1.2),
                math.sin(angle) * radius * random.uniform(0.8, 1.2)
            ])
            
            node = Node((center + offset).tolist(), 
                       radius=8 + random.random() * 4,
                       mass=0.5 + random.random() * 1.5)
            
            # Small chance to make a node fixed (like a foot)
            if random.random() < 0.2 and i > 1:  # Don't fix first few nodes
                node.fixed = True
                
            self.nodes.append(node)
        
        # Create bones - try to connect nodes in a reasonable way
        attempts = 0
        while len(self.bones) < bone_count and attempts < bone_count * 2:
            attempts += 1
            node1, node2 = random.sample(self.nodes, 2)
            
            # Don't create duplicate bones
            if any(b for b in self.bones if 
                  (b.node1 == node1 and b.node2 == node2) or 
                  (b.node1 == node2 and b.node2 == node1)):
                continue
                
            # Don't create very long bones
            dist = torch.dist(node1.position, node2.position)
            if dist > 120:
                continue
                
            stiffness = 0.3 + random.random() * 0.4
            self.bones.append(Bone(node1, node2, stiffness=stiffness))
        
        # Create muscles for some bones
        for bone in self.bones:
            if random.random() < muscle_density:
                min_len = bone.rest_length * (0.7 + random.random() * 0.2)
                max_len = bone.rest_length * (1.1 + random.random() * 0.3)
                strength = 0.1 + random.random() * 0.2
                phase = random.random() * 2 * math.pi
                self.muscles.append(Muscle(bone, min_len, max_len, strength, phase))
        
        # Position the organism
        for node in self.nodes:
            node.position += self.position
            
    def update(self, time_step):
        # Update muscles
        for muscle in self.muscles:
            muscle.update(time_step)
            
        # Apply bone constraints
        for bone in self.bones:
            bone.apply_constraint()
            
        # Update nodes
        for node in self.nodes:
            node.update()
            
        # Calculate center of mass and velocity
        total_mass = sum(node.mass for node in self.nodes)
        com = torch.zeros(2)
        total_velocity = torch.zeros(2)
        
        for node in self.nodes:
            com += node.position * node.mass
            total_velocity += node.velocity * node.mass
            
        com /= total_mass
        self.velocity = total_velocity / total_mass
        self.position = com
        
    def draw(self, offset_x):
        # Draw bones
        for bone in self.bones:
            pos1 = bone.node1.position.detach().numpy() - np.array([offset_x, 0])
            pos2 = bone.node2.position.detach().numpy() - np.array([offset_x, 0])
            dpg.draw_line(pos1, pos2, color=(255, 255, 255, 255), thickness=2)
            
        # Draw nodes
        for node in self.nodes:
            pos = node.position.detach().numpy() - np.array([offset_x, 0])
            color = (0, 255, 0, 255) if not node.fixed else (255, 0, 0, 255)
            dpg.draw_circle(pos, node.radius, color=color)
            
        # Draw muscles (as colored bones)
        for i, muscle in enumerate(self.muscles):
            bone = muscle.bone
            pos1 = bone.node1.position.detach().numpy() - np.array([offset_x, 0])
            pos2 = bone.node2.position.detach().numpy() - np.array([offset_x, 0])
            tension = muscle.current_tension
            color = (255, int(255 * (1 - tension)), int(255 * tension), 255)
            dpg.draw_line(pos1, pos2, color=color, thickness=3)

class Simulation:
    def __init__(self):
        self.organisms = []
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
        
        # Initialize Dear PyGui
        dpg.create_context()
        dpg.create_viewport(title='Organism Walking Simulation', width=SIM_WIDTH, height=SIM_HEIGHT)
        dpg.setup_dearpygui()
        
        # Create main window
        with dpg.window(tag="Primary Window"):
            # Create a splitter for side panel and simulation view
            with dpg.group(horizontal=True):
                # Side panel for controls
                with dpg.child_window(width=300):
                    dpg.add_text("Controls")
                    dpg.add_button(label="Create Random Creature", callback=self.create_organism)
                    dpg.add_button(label="Reset Simulation", callback=self.reset_simulation)
                    
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
        
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)
        
    def param_update(self, param, value):
        self.creation_params[param] = value
        
    def set_camera_speed(self, sender, app_data):
        self.camera_speed = app_data
        
    def create_organism(self):
        # Get random values within specified ranges
        node_count = random.randint(
            self.creation_params['min_nodes'],
            self.creation_params['max_nodes']
        )
        bone_count = random.randint(
            self.creation_params['min_bones'],
            self.creation_params['max_bones']
        )
        
        # Create new organism at starting position
        new_org = Organism(
            [0, 600],
            node_count=node_count,
            bone_count=bone_count,
            muscle_density=self.creation_params['muscle_density']
        )
        self.organisms.append(new_org)
        self.follow_organism = new_org
        self.camera_offset_x = 0
        
    def reset_simulation(self):
        self.organisms = []
        self.follow_organism = None
        self.camera_offset_x = 0
        
    def update(self):
        self.time += TIME_STEP
        
        # Update all organisms
        for organism in self.organisms:
            organism.update(TIME_STEP)
            
        # Camera follow logic
        if self.follow_organism:
            target_x = self.follow_organism.position[0].item() - SIM_WIDTH/3
            self.camera_offset_x += (target_x - self.camera_offset_x) * 0.1 * self.camera_speed * TIME_STEP
            
        # Draw everything
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
                
                # Draw organisms
                for organism in self.organisms:
                    organism.draw(self.camera_offset_x)
                
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