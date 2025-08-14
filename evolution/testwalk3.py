import json
import os
import dearpygui.dearpygui as dpg
import math
import random
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional, Any, Set
import time
from threading import Thread, Lock

# --- Configuration Constants ---
class Config:
    def __init__(self):
        # Simulation environment and physics constants
        self.SIM_WIDTH = 1200
        self.SIM_HEIGHT = 800
        self.CREATURE_DB_FILE = "evolved_creatures_db.json"

        self.HAVE_GROUND = True
        self.HIST_BARS_PER_METER = 5
        self.OPERATION_NAMES = ["#", "time", "px", "py", "+", "-", "*", "÷", "%", "sin", "sig", "pres"]
        self.OPERATION_AXONS = [0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 0] # Number of axons for each operation
        self.OPERATION_COUNT = len(self.OPERATION_NAMES)
        self.FITNESS_UNIT = "m"
        self.FITNESS_NAME = "Distance"
        self.BASELINE_ENERGY = 0.0
        self.ENERGY_DIRECTION = 1 # 1 for energy consumption, -1 for energy gain
        self.FRICTION = 4.0 # Ground friction

        self.BIG_MUTATION_CHANCE = 0.06 # Probability of a "large" mutation

        self.PRESSURE_UNIT = 500.0 / 2.37
        self.ENERGY_UNIT = 20.0 # Multiplier for energy consumption
        self.NAUSEA_UNIT = 5.0 # Multiplier for "nausea" based on acceleration
        self.MIN_BAR = -10
        self.MAX_BAR = 100
        self.BAR_LEN = self.MAX_BAR - self.MIN_BAR

        # Physics constants
        self.GRAVITY = 0.005 # Applied per simulation step
        self.AIR_FRICTION = 0.95
        
        # Creature/Evolution parameters
        self.SIMULATION_STEPS_PER_CREATURE = 900 # Fixed duration for creature simulation
        self.INITIAL_NODE_RANGE = (3, 5) # Inclusive
        self.MUSCLE_ADD_TRIES = 13
        self.NODE_ADD_TRIES = 18
        self.MIN_NODES = 3 # Minimum nodes before removal is considered
        self.MIN_MUSCLES = 1 # Minimum muscles before removal is considered

        self.PERCENTILE_INDICES = [ # Percentile indices for creature selection from sorted list (0-999)
            0, 9, 19, 29, 39, 49, 59, 69, 79, 89,
            99, 199, 299, 399, 499, 599, 699, 799, 899, 909, 919, 929, 939, 949, 959, 969, 979, 989, 999
        ]
        
        # Initial rectangles for demonstration (e.g., obstacles)
        self.RECTS: List['Rectangle'] = [
            # Rectangle(1.0, -1.0, 1.5, -0.5), # Example rectangle
            # Rectangle(-1.5, -0.5, -1.0, 0.0)
        ]

# Helper functions for random numbers
def r() -> float:
    return math.pow(random.uniform(-1, 1), 19)

def r_int() -> int:
    return random.randint(0, 1)

def to_muscle_usable(val: float) -> float:
    return val * 1.0 + 0.5

# --- Data Structures for Creatures ---
class Rectangle:
    def __init__(self, tx1: float, ty1: float, tx2: float, ty2: float):
        # Ensure x1 < x2 and y1 < y2 for proper rectangle definition
        self.x1 = min(tx1, tx2)
        self.y1 = min(ty1, ty2)
        self.x2 = max(tx1, tx2)
        self.y2 = max(ty1, ty2)

    def to_dict(self):
        return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2}

    @classmethod
    def from_dict(cls, data):
        return cls(data['x1'], data['y1'], data['x2'], data['y2'])


class Node:
    def __init__(self, conf: Config, tx: float, ty: float, tvx: float, tvy: float, tm: float, tf: float,
                 val: float, op: int, a1: int, a2: int):
        self.conf = conf
        # Position and velocity
        self.prev_x = tx # Previous x (for nausea)
        self.x = tx
        self.prev_y = ty # Previous y (for nausea)
        self.y = ty
        self.pvx = tvx # Previous velocity x
        self.vx = tvx
        self.pvy = tvy # Previous velocity y
        self.vy = tvy
        # Physical properties
        self.m = tm # mass (also used as diameter, so radius is m/2)
        self.f = tf # friction (0.0 to 1.0)
        # Neural properties
        self.value = val # Current output value of this neuron
        self.value_to_be = val # Next output value (for double buffering)
        self.operation = op # Index of the operation this neuron performs
        self.axon1 = a1 # Index of the first input neuron
        self.axon2 = a2 # Index of the second input neuron
        self.safe_input = False # Used in check_for_bad_axons for dependency analysis
        self.pressure = 0.0 # Pressure applied by collisions

    def copy_node(self) -> 'Node':
        # vx, vy, pvx, pvy are not persistent states we want to carry over for a 'copy'
        # prev_x, prev_y are also transient for nausea calculation.
        # We copy the 'blueprint' data: position, mass, friction, neuron params.
        return Node(self.conf, self.x, self.y, 0, 0, self.m, self.f, self.value, self.operation, self.axon1, self.axon2)

    def modify_node(self, mutability: float, node_num: int) -> 'Node':
        new_x = self.x + r() * 0.5 * mutability
        new_y = self.y + r() * 0.5 * mutability
        new_m = 0.4 # Mass is fixed at 0.4 as per original Processing code.

        new_v = self.value * (1 + r() * 0.2 * mutability) # Mutate initial value for constant nodes
        new_operation = self.operation
        new_axon1 = self.axon1
        new_axon2 = self.axon2

        if random.uniform(0, 1) < self.conf.BIG_MUTATION_CHANCE * mutability:
            new_operation = math.floor(random.uniform(0, self.conf.OPERATION_COUNT))
        if random.uniform(0, 1) < self.conf.BIG_MUTATION_CHANCE * mutability:
            new_axon1 = math.floor(random.uniform(0, max(1, node_num)))
        if random.uniform(0, 1) < self.conf.BIG_MUTATION_CHANCE * mutability:
            new_axon2 = math.floor(random.uniform(0, max(1, node_num)))

        # Operations that reset/determine value directly, not from axons
        if new_operation == 1: new_v = 0.0 # Time
        elif new_operation == 2: new_v = new_x * 0.2 # X position
        elif new_operation == 3: new_v = -new_y * 0.2 # Y position (negative because Y increases downwards)
        elif new_operation == 11: new_v = 0.0 # Pressure

        new_f = min(max(self.f + r() * 0.1 * mutability, 0.0), 1.0) # Mutate friction

        return Node(self.conf, new_x, new_y, 0, 0, new_m, new_f, new_v, new_operation, new_axon1, new_axon2)

    def to_dict(self):
        return {
            'x': self.x, 'y': self.y, 'm': self.m, 'f': self.f,
            'value': self.value, 'operation': self.operation,
            'axon1': self.axon1, 'axon2': self.axon2
        }

    @classmethod
    def from_dict(cls, data, conf: Config):
        # We don't restore transient physics states like vx, vy, prev_x etc.
        # These will be initialized to 0 by the Node constructor or during simulation setup.
        node = cls(
            conf=conf,
            tx=data['x'], ty=data['y'], tvx=0.0, tvy=0.0,
            tm=data['m'], tf=data['f'], val=data['value'],
            op=data['operation'], a1=data['axon1'], a2=data['axon2']
        )
        return node

class Muscle:
    def __init__(self, conf: Config, taxon: int, tc1: int, tc2: int, tlen: float, trigidity: float):
        self.conf = conf
        self.axon = taxon # Index of the neuron controlling this muscle. -1 means no neural control.
        self.previous_target = tlen # For energy calculation
        self.len = tlen # Base length (when neuron value is 0.5)
        self.c1 = tc1 # Index of first connected node
        self.c2 = tc2 # Index of second connected node
        self.rigidity = trigidity # Stiffness/strength of the muscle
        self.force = 0.0 # Not persistent, but for debugging/display

    def copy_muscle(self) -> 'Muscle':
        return Muscle(self.conf, self.axon, self.c1, self.c2, self.len, self.rigidity)

    def modify_muscle(self, node_num: int, mutability: float) -> 'Muscle':
        new_c1 = self.c1
        new_c2 = self.c2
        new_axon = self.axon

        effective_node_num = max(1, node_num) # Avoid division by zero if node_num is 0
        
        # Mutate connections
        if random.uniform(0, 1) < self.conf.BIG_MUTATION_CHANCE * mutability:
            new_c1 = math.floor(random.uniform(0, effective_node_num))
        if random.uniform(0, 1) < self.conf.BIG_MUTATION_CHANCE * mutability:
            new_c2 = math.floor(random.uniform(0, effective_node_num))
        
        # Mutate axon connection
        if random.uniform(0, 1) < self.conf.BIG_MUTATION_CHANCE * mutability:
            new_axon = Muscle.get_new_muscle_axon(effective_node_num)

        new_r = min(max(self.rigidity * (1 + r() * 0.9 * mutability), 0.01), 0.08) # Mutate rigidity
        new_len = min(max(self.len + r() * mutability, 0.4), 1.25) # Mutate base length

        return Muscle(self.conf, new_axon, new_c1, new_c2, new_len, new_r)

    @staticmethod
    def get_new_muscle_axon(node_num: int) -> int:
        """Returns a new axon index for a muscle. Can be -1 (no neural control) or a node index."""
        if random.uniform(0, 1) < 0.5:
            return math.floor(random.uniform(0, max(1, node_num)))
        else:
            return -1 # No axon control

    def to_dict(self):
        return {
            'axon': self.axon, 'len': self.len,
            'c1': self.c1, 'c2': self.c2, 'rigidity': self.rigidity
        }

    @classmethod
    def from_dict(cls, data, conf: Config):
        muscle = cls(
            conf=conf,
            taxon=data['axon'], tc1=data['c1'], tc2=data['c2'],
            tlen=data['len'], trigidity=data['rigidity']
        )
        # previous_target and force are transient and reset during simulation
        return muscle

class Creature:
    def __init__(self, conf: Config, tid: int, tn: List[Node], tm: List[Muscle], td: float, talive: bool, tct: float, tmut: float):
        self.conf = conf
        self.id = tid
        self.muscles = tm
        self.nodes = tn
        self.fitness = td
        self.alive = talive # Not strictly used in this combined version
        self.heartbeat = tct # Original creature "heartbeat" or simulation duration setting
        self.mutability = tmut # How much this creature mutates its children

    def _draw_creature(self, node_data: Optional[Dict[str, np.ndarray]], muscle_data: Optional[Dict[str, np.ndarray]],
                    drawlist_tag: str,
                    render_offset_x_dpg: float, render_offset_y_dpg: float,
                    scale: float,
                    draw_rects: bool = False):
        """
        Draws the state of a creature from its NumPy data onto a DearPyGUI drawlist.
        Coordinates are relative to `render_offset_x_dpg`, `render_offset_y_dpg` which represent sim (0,0).
        This function should only *add* drawings, not clear the drawlist.
        """
        if node_data is None or muscle_data is None:
            return

        # Draw Rectangles (obstacles) - only for main simulation view
        if draw_rects and self.config.RECTS:
            for r_obj in self.config.RECTS:
                x1_draw, y1_draw = self._sim_to_draw_coords(r_obj.x1, r_obj.y1, render_offset_x_dpg, render_offset_y_dpg, scale)
                x2_draw, y2_draw = self._sim_to_draw_coords(r_obj.x2, r_obj.y2, render_offset_x_dpg, render_offset_y_dpg, scale)
                
                # Ensure proper top-left, bottom-right order for DPG rectangle
                draw_rect_x1 = min(x1_draw, x2_draw)
                draw_rect_y1 = min(y1_draw, y2_draw)
                draw_rect_x2 = max(x1_draw, x2_draw)
                draw_rect_y2 = max(y1_draw, y2_draw)
                
                dpg.draw_rectangle((draw_rect_x1, draw_rect_y1), (draw_rect_x2, draw_rect_y2), 
                                color=(200, 200, 0, 255), fill=(200, 200, 0, 100), 
                                parent=drawlist_tag)

        # Draw muscles first (behind nodes)
        for i in range(len(muscle_data['axon'])):
            c1_idx = muscle_data['c1'][i]
            c2_idx = muscle_data['c2'][i]

            # Ensure node indices are valid
            if c1_idx < 0 or c1_idx >= len(node_data['x']) or \
            c2_idx < 0 or c2_idx >= len(node_data['x']):
                continue  # Skip invalid muscle connections

            x1_sim = node_data['x'][c1_idx]
            y1_sim = node_data['y'][c1_idx]
            x2_sim = node_data['x'][c2_idx]
            y2_sim = node_data['y'][c2_idx]

            x1_draw, y1_draw = self._sim_to_draw_coords(x1_sim, y1_sim, render_offset_x_dpg, render_offset_y_dpg, scale)
            x2_draw, y2_draw = self._sim_to_draw_coords(x2_sim, y2_sim, render_offset_x_dpg, render_offset_y_dpg, scale)

            # Safely get force, default to 0 for static rendering (e.g., gallery)
            force = muscle_data.get('force', np.zeros_like(muscle_data['axon'], dtype=np.float64))[i] 
            color = (100, 100, 100, 255)  # Default grey
            if force > 0.05:  # Pushing (expanding)
                color = (0, 200, 0, 255)  # Green
            elif force < -0.05:  # Pulling (contracting)
                color = (200, 0, 0, 255)  # Red

            dpg.draw_line((x1_draw, y1_draw), (x2_draw, y2_draw), 
                        color=color, thickness=2, parent=drawlist_tag)

        # Draw nodes
        for i in range(len(node_data['x'])):
            x_sim = node_data['x'][i]
            y_sim = node_data['y'][i]
            m_sim = node_data['m'][i]  # Mass acts as diameter in simulation

            radius_draw = m_sim / 2 * scale  # Node radius is m/2
            x_draw, y_draw = self._sim_to_draw_coords(x_sim, y_sim, render_offset_x_dpg, render_offset_y_dpg, scale)

            dpg.draw_circle((x_draw, y_draw), radius_draw, 
                            color=(150, 150, 150, 255), fill=(150, 150, 150, 200), 
                            parent=drawlist_tag)
            
    def _sim_to_draw_coords(self, x_sim: float, y_sim: float,
                        draw_offset_x: float, draw_offset_y: float,
                        scale: float) -> tuple[float, float]:
        """
        Converts simulation coordinates (x, y) to DearPyGUI drawlist coordinates
        given an offset and scale.

        Args:
            x_sim: Simulation X coordinate of the point.
            y_sim: Simulation Y coordinate of the point.
            draw_offset_x: The DPG X coordinate that maps to simulation X=0.
            draw_offset_y: The DPG Y coordinate that maps to simulation Y=0.
            scale: Pixels per simulation unit.
        """
        # Apply scale and then offset
        draw_x = draw_offset_x + x_sim * scale
        draw_y = draw_offset_y + y_sim * scale  # Y-axis direction is consistent (downwards)

        return draw_x, draw_y
    
    def modified(self, new_id: int) -> 'Creature':
        # Create a new creature based on this one, with mutations
        modified_creature = Creature(self.conf,
                                     new_id,
                                     [], # New empty lists for nodes and muscles
                                     [],
                                     0.0, True, self.heartbeat + r() * 16 * self.mutability, # Mutate heartbeat/duration
                                     min(self.mutability * random.uniform(0.8, 1.25), 2.0)) # Mutate mutability
        
        current_node_count = len(self.nodes)
        for node in self.nodes:
            modified_creature.nodes.append(node.modify_node(self.mutability, current_node_count))
        for muscle in self.muscles:
            modified_creature.muscles.append(muscle.modify_muscle(current_node_count, self.mutability))

        # Add nodes
        for _ in range(self.conf.NODE_ADD_TRIES):
            if random.uniform(0, 1) < self.conf.BIG_MUTATION_CHANCE * self.mutability or len(modified_creature.nodes) <= self.conf.MIN_NODES:
                modified_creature.add_random_node()

        # Add muscles
        for _ in range(self.conf.MUSCLE_ADD_TRIES):
            if random.uniform(0, 1) < self.conf.BIG_MUTATION_CHANCE * self.mutability:
                modified_creature.add_random_muscle(-1, -1) # -1 means random nodes

        # Remove nodes
        if random.uniform(0, 1) < self.conf.BIG_MUTATION_CHANCE * self.mutability and len(modified_creature.nodes) >= self.conf.MIN_NODES + 1:
            modified_creature.remove_random_node()
        
        # Remove muscles
        if random.uniform(0, 1) < self.conf.BIG_MUTATION_CHANCE * self.mutability and len(modified_creature.muscles) >= self.conf.MIN_MUSCLES + 1:
            modified_creature.remove_random_muscle()
            
        modified_creature.check_for_overlap()
        modified_creature.check_for_lone_nodes()
        modified_creature.check_for_bad_axons()
        return modified_creature

    def check_for_overlap(self):
        """Removes muscles that connect a node to itself or are duplicates."""
        bads: Set[int] = set()
        for i in range(len(self.muscles)):
            if self.muscles[i].c1 == self.muscles[i].c2: # Muscle connects to itself
                bads.add(i)
            for j in range(i + 1, len(self.muscles)):
                # Duplicate muscle (order doesn't matter for connections)
                if (self.muscles[i].c1 == self.muscles[j].c1 and self.muscles[i].c2 == self.muscles[j].c2) or \
                   (self.muscles[i].c1 == self.muscles[j].c2 and self.muscles[i].c2 == self.muscles[j].c1):
                    bads.add(i) 
                    
        for b_idx in sorted(list(bads), reverse=True): # Remove from high index to low to avoid shifting issues
            if b_idx < len(self.muscles): 
                self.muscles.pop(b_idx)

    def check_for_lone_nodes(self):
        """Ensures all nodes are connected by at least two muscles to prevent isolated nodes."""
        if len(self.nodes) >= self.conf.MIN_NODES: # Only apply if enough nodes exist
            for i in range(len(self.nodes)):
                connections = 0
                for muscle in self.muscles:
                    if muscle.c1 == i or muscle.c2 == i:
                        connections += 1
                
                # If node has 0 or 1 connection, add a new muscle to connect it to another random node.
                # (Original logic: <= 1. If 0, it's isolated. If 1, it's a dangling end. Both are unstable.)
                if connections <= 1:
                    new_connection_node = math.floor(random.uniform(0, len(self.nodes)))
                    if len(self.nodes) > 1:
                        while new_connection_node == i: # Ensure it doesn't connect to itself
                            new_connection_node = math.floor(random.uniform(0, len(self.nodes)))
                    self.add_random_muscle(i, new_connection_node)

    def check_for_bad_axons(self):
        """
        Validates neuron axon connections and operation dependencies.
        Ensures that an axon doesn't point to an out-of-bounds index and
        that neuron calculations don't create infinite loops or unresolvable dependencies.
        """
        # First, ensure all axon indices are within bounds
        for node in self.nodes:
            if node.axon1 >= len(self.nodes) or node.axon1 < 0:
                node.axon1 = math.floor(random.uniform(0, max(1, len(self.nodes))))
            if node.axon2 >= len(self.nodes) or node.axon2 < 0:
                node.axon2 = math.floor(random.uniform(0, max(1, len(self.nodes))))
        
        for muscle in self.muscles:
            if muscle.axon >= len(self.nodes) or muscle.axon < -1: # -1 is a valid 'no axon' state
                muscle.axon = Muscle.get_new_muscle_axon(len(self.nodes))

        # Perform dependency analysis using a propagation approach
        # A node is "safe" if its inputs are constant or come from other safe nodes.
        for node in self.nodes:
            # Nodes with 0 axons (constants, time, position, pressure) are inherently safe
            node.safe_input = (self.conf.OPERATION_AXONS[node.operation] == 0)

        iterations = 0
        did_something = True 

        # Propagate 'safe_input' status until no more changes or max iterations reached
        while did_something and iterations < 1000: # Max iterations to prevent infinite loop for cyclic dependencies
            did_something = False
            for node_idx, node in enumerate(self.nodes):
                if not node.safe_input: 
                    op_type = self.conf.OPERATION_AXONS[node.operation]
                    
                    is_axon1_safe = (op_type >= 1 and 0 <= node.axon1 < len(self.nodes) and self.nodes[node.axon1].safe_input)
                    is_axon2_safe = (op_type == 2 and 0 <= node.axon2 < len(self.nodes) and self.nodes[node.axon2].safe_input)

                    if (op_type == 1 and is_axon1_safe) or \
                       (op_type == 2 and is_axon1_safe and is_axon2_safe):
                        node.safe_input = True
                        did_something = True
            iterations += 1
        
        # If any node is still not 'safe_input' after propagation, it indicates a dependency issue.
        # Revert such nodes to a safe, constant operation.
        for node in self.nodes:
            if not node.safe_input:
                node.operation = 0 # Revert to constant operation
                node.value = random.uniform(0, 1) # Assign a random initial value for the constant

    def add_random_node(self):
        """Adds a new node and connects it to existing nodes."""
        if not self.nodes: # If no nodes exist, create the first one at origin
            self.nodes.append(Node(self.conf, 0, 0, 0, 0, 0.4, random.uniform(0, 1), random.uniform(0,1), 0, 0, 0))
            return

        parent_node_idx = math.floor(random.uniform(0, len(self.nodes)))
        parent_node = self.nodes[parent_node_idx]
        
        # Position the new node near an existing one
        ang1 = random.uniform(0, 2 * math.pi)
        distance_from_parent = math.sqrt(random.uniform(0, 1))
        x = parent_node.x + math.cos(ang1) * 0.5 * distance_from_parent
        y = parent_node.y + math.sin(ang1) * 0.5 * distance_from_parent

        new_node_count_potential = len(self.nodes) + 1 # For axon bounds calculation
        self.nodes.append(Node(self.conf, x, y, 0, 0, 0.4, random.uniform(0, 1), random.uniform(0,1),
                          math.floor(random.uniform(0, self.conf.OPERATION_COUNT)),
                          math.floor(random.uniform(0, new_node_count_potential)),
                          math.floor(random.uniform(0, new_node_count_potential))))
        
        new_node_idx = len(self.nodes) - 1 

        # Find the next closest node to connect to (excluding the parent)
        next_closest_node_idx = -1
        record = float('inf')
        if len(self.nodes) > 1: 
            for i in range(len(self.nodes) - 1): # Check all but the newly added node
                if i != parent_node_idx: 
                    current_dist = np.sqrt(((self.nodes[i].x - x) ** 2) + ((self.nodes[i].y - y) ** 2))
                    if current_dist < record:
                        record = current_dist
                        next_closest_node_idx = i

        # Connect the new node to its parent and the next closest node
        self.add_random_muscle(parent_node_idx, new_node_idx)
        if next_closest_node_idx != -1: 
            self.add_random_muscle(next_closest_node_idx, new_node_idx)

    def add_random_muscle(self, tc1: int, tc2: int):
        """Adds a new muscle, connecting two (possibly random) nodes."""
        if len(self.nodes) == 0:
            return 
        
        if tc1 == -1 or tc2 == -1: # If -1, choose random nodes
            tc1 = math.floor(random.uniform(0, len(self.nodes)))
            tc2 = tc1 
            while tc2 == tc1 and len(self.nodes) > 1: # Ensure different nodes if possible
                tc2 = math.floor(random.uniform(0, len(self.nodes)))
        
        # Ensure indices are within bounds
        tc1 = max(0, min(tc1, len(self.nodes) - 1))
        tc2 = max(0, min(tc2, len(self.nodes) - 1))

        len_val = random.uniform(0.5, 1.5) # Default length if nodes are at same spot
        if tc1 < len(self.nodes) and tc2 < len(self.nodes): # Check if indices are valid before accessing
            distance_between_nodes = np.sqrt(((self.nodes[tc1].x - self.nodes[tc2].x) ** 2) + ((self.nodes[tc1].y - self.nodes[tc2].y) ** 2))
            if distance_between_nodes > 0:
                len_val = distance_between_nodes # Set initial length to current distance
            else:
                len_val = 0.5 # Default if nodes overlap

        axon = Muscle.get_new_muscle_axon(len(self.nodes))

        self.muscles.append(Muscle(self.conf, axon, tc1, tc2, len_val, random.uniform(0.02, 0.08)))

    def remove_random_node(self):
        """Removes a random node and its associated muscles."""
        if len(self.nodes) <= self.conf.MIN_NODES: # Don't remove if below minimum
            return

        choice = math.floor(random.uniform(0, len(self.nodes)))
        
        # Remove muscles connected to the chosen node
        i = len(self.muscles) - 1
        while i >= 0:
            if self.muscles[i].c1 == choice or self.muscles[i].c2 == choice:
                self.muscles.pop(i)
            i -= 1
        
        # Remove the node
        self.nodes.pop(choice)

        # Update node and axon indices for remaining muscles and nodes
        for muscle in self.muscles:
            if muscle.c1 > choice: 
                muscle.c1 -= 1
            if muscle.c2 > choice:
                muscle.c2 -= 1
            if muscle.axon != -1 and muscle.axon > choice: 
                muscle.axon -= 1
                if muscle.axon >= len(self.nodes) or muscle.axon < 0: # Re-roll if it went out of bounds
                    muscle.axon = Muscle.get_new_muscle_axon(len(self.nodes))

        for node in self.nodes:
            if node.axon1 > choice:
                node.axon1 -= 1
            if node.axon2 > choice:
                node.axon2 -= 1
            # Re-roll if it went out of bounds
            if node.axon1 >= len(self.nodes) or node.axon1 < 0:
                node.axon1 = math.floor(random.uniform(0, max(1, len(self.nodes))))
            if node.axon2 >= len(self.nodes) or node.axon2 < 0:
                node.axon2 = math.floor(random.uniform(0, max(1, len(self.nodes))))

    def remove_random_muscle(self):
        """Removes a random muscle."""
        if len(self.muscles) <= self.conf.MIN_MUSCLES: # Don't remove if below minimum
            return

        choice = math.floor(random.uniform(0, len(self.muscles)))
        self.muscles.pop(choice)

    def stabilize_configuration(self):
        """Simulates for a short period to allow creature to settle, without gravity/wall collisions."""
        temp_simulator = CreatureSimulator(self.conf) # Create a temporary simulator instance
        node_data, muscle_data = temp_simulator._extract_creature_data_to_numpy(self)

        for _ in range(200): # 200 steps for stabilization
            # Pass is_stabilizing=True to disable gravity/walls and energy consumption
            temp_simulator._simulate_step_numpy(node_data, muscle_data, 0, self.conf, is_stabilizing=True)

        # After stabilization, reset velocities to zero and update positions
        node_data['vx'].fill(0.0)
        node_data['vy'].fill(0.0)
        
        temp_simulator._update_creature_data_from_numpy(self, node_data, muscle_data)

    def adjust_to_center(self):
        """Adjusts the creature's position so its lowest point is at y=0 (ground) and it's horizontally centered."""
        if not self.nodes:
            return

        node_x = np.array([n.x for n in self.nodes], dtype=np.float64)
        node_y = np.array([n.y for n in self.nodes], dtype=np.float64)
        node_m = np.array([n.m for n in self.nodes], dtype=np.float64) # For node radius (m/2)

        # Calculate horizontal center (average X)
        avx = np.mean(node_x)
        
        # Calculate lowest point (max y + max_radius)
        # Assuming y increases downwards, lowest point would be max y + radius.
        # So to put the lowest point at y=0, we shift by -(max_y + max_radius).
        low_y = np.max(node_y + node_m / 2)
        
        # Apply adjustments directly to the Node objects
        for node in self.nodes:
            node.x -= float(avx)
            node.y -= low_y

    def copy_creature(self, new_id: int = -1) -> 'Creature':
        nodes_copy = [node.copy_node() for node in self.nodes]
        muscles_copy = [muscle.copy_muscle() for muscle in self.muscles]
        
        if new_id == -1:
            new_id = self.id
        
        return Creature(self.conf, new_id, nodes_copy, muscles_copy, self.fitness, self.alive, self.heartbeat, self.mutability)

    def to_dict(self):
        return {
            'id': self.id,
            'nodes': [node.to_dict() for node in self.nodes],
            'muscles': [muscle.to_dict() for muscle in self.muscles],
            'fitness': self.fitness,
            'alive': self.alive,
            'heartbeat': self.heartbeat,
            'mutability': self.mutability
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], conf: Config):
        creature = cls(
            conf=conf,
            tid=data['id'],
            tn=[], # Nodes will be populated
            tm=[], # Muscles will be populated
            td=data['fitness'],
            talive=data['alive'],
            tct=data['heartbeat'],
            tmut=data['mutability']
        )
        creature.nodes = [Node.from_dict(node_data, conf) for node_data in data['nodes']]
        creature.muscles = [Muscle.from_dict(muscle_data, conf) for muscle_data in data['muscles']]
        return creature


class CreatureSimulator:
    def __init__(self, conf: Config):
        self.conf = conf
        self.simulation_timer = 0
        self.energy = 0.0
        self.total_node_nausea = 0.0
        self.average_x = 0.0
        self.average_y = 0.0

    def _extract_creature_data_to_numpy(self, creature: Creature) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Converts lists of Node and Muscle objects into NumPy array dictionaries."""
        num_nodes = len(creature.nodes)
        num_muscles = len(creature.muscles)

        # Node data extraction
        node_x = np.array([n.x for n in creature.nodes], dtype=np.float64)
        node_y = np.array([n.y for n in creature.nodes], dtype=np.float64)
        node_vx = np.array([n.vx for n in creature.nodes], dtype=np.float64)
        node_vy = np.array([n.vy for n in creature.nodes], dtype=np.float64)
        node_m = np.array([n.m for n in creature.nodes], dtype=np.float64)
        node_f = np.array([n.f for n in creature.nodes], dtype=np.float64)
        node_value = np.array([n.value for n in creature.nodes], dtype=np.float64)
        node_value_to_be = np.array([n.value_to_be for n in creature.nodes], dtype=np.float64)
        node_operation = np.array([n.operation for n in creature.nodes], dtype=np.int32)
        node_axon1 = np.array([n.axon1 for n in creature.nodes], dtype=np.int32)
        node_axon2 = np.array([n.axon2 for n in creature.nodes], dtype=np.int32)
        node_pressure = np.array([n.pressure for n in creature.nodes], dtype=np.float64)
        node_prev_x = np.array([n.prev_x for n in creature.nodes], dtype=np.float64)
        node_prev_y = np.array([n.prev_y for n in creature.nodes], dtype=np.float64)
        node_pvx = np.array([n.pvx for n in creature.nodes], dtype=np.float64)
        node_pvy = np.array([n.pvy for n in creature.nodes], dtype=np.float64)

        node_data = {
            'x': node_x, 'y': node_y, 'vx': node_vx, 'vy': node_vy, 'm': node_m, 'f': node_f,
            'value': node_value, 'value_to_be': node_value_to_be,
            'operation': node_operation, 'axon1': node_axon1, 'axon2': node_axon2,
            'pressure': node_pressure,
            'prev_x': node_prev_x, 'prev_y': node_prev_y, 'pvx': node_pvx, 'pvy': node_pvy
        }

        # Muscle data extraction
        muscle_axon = np.array([m.axon for m in creature.muscles], dtype=np.int32)
        muscle_previous_target = np.array([m.previous_target for m in creature.muscles], dtype=np.float64)
        muscle_len = np.array([m.len for m in creature.muscles], dtype=np.float64)
        muscle_c1 = np.array([m.c1 for m in creature.muscles], dtype=np.int32)
        muscle_c2 = np.array([m.c2 for m in creature.muscles], dtype=np.int32)
        muscle_rigidity = np.array([m.rigidity for m in creature.muscles], dtype=np.float64)
        muscle_force = np.array([m.force for m in creature.muscles], dtype=np.float64)

        muscle_data = {
            'axon': muscle_axon, 'previous_target': muscle_previous_target, 'len': muscle_len,
            'c1': muscle_c1, 'c2': muscle_c2, 'rigidity': muscle_rigidity, 'force': muscle_force
        }
        return node_data, muscle_data

    def _update_creature_data_from_numpy(self, creature: Creature, node_data: Dict[str, np.ndarray], muscle_data: Dict[str, np.ndarray]):
        """Copies final state from NumPy array dictionaries back to Node and Muscle objects."""
        for i, node in enumerate(creature.nodes):
            node.x = node_data['x'][i]
            node.y = node_data['y'][i]
            node.vx = node_data['vx'][i]
            node.vy = node_data['vy'][i]
            node.value = node_data['value'][i]
            node.pressure = node_data['pressure'][i]
            node.prev_x = node_data['prev_x'][i]
            node.prev_y = node_data['prev_y'][i]
            node.pvx = node_data['pvx'][i]
            node.pvy = node_data['pvy'][i]

        for i, muscle in enumerate(creature.muscles):
            muscle.previous_target = muscle_data['previous_target'][i]
            muscle.force = muscle_data['force'][i]

    def _simulate_step_numpy(self, node_data: Dict[str, np.ndarray], muscle_data: Dict[str, np.ndarray], current_time: float, conf: Config, is_stabilizing: bool) -> tuple[float, float]:
        """Performs a single simulation step using NumPy arrays for vectorized calculations."""
        energy_consumed_this_step = 0.0
        
        num_nodes = len(node_data['x'])
        
        # --- Muscle Force Application ---
        if len(muscle_data['axon']) > 0:
            c1_idx = muscle_data['c1']
            c2_idx = muscle_data['c2']

            # Mask for muscles with valid connected nodes
            valid_muscle_mask = (c1_idx >= 0) & (c1_idx < num_nodes) & \
                                (c2_idx >= 0) & (c2_idx < num_nodes)
            
            # Apply mask to get valid indices for connected nodes
            c1_valid_idx = c1_idx[valid_muscle_mask]
            c2_valid_idx = c2_idx[valid_muscle_mask]

            # Get positions of connected nodes for valid muscles
            x1_valid = node_data['x'][c1_valid_idx]
            y1_valid = node_data['y'][c1_valid_idx]
            x2_valid = node_data['x'][c2_valid_idx]
            y2_valid = node_data['y'][c2_valid_idx]

            # Calculate distances and angles for valid muscles
            dx_valid = x1_valid - x2_valid
            dy_valid = y1_valid - y2_valid
            distances_valid = np.sqrt(dx_valid**2 + dy_valid**2)
            
            # Handle zero distance to prevent NaNs in angles
            # Where distance is zero, set angle to 0 and force to 0.
            # This avoids division by zero or log(0) type errors.
            angles_valid = np.arctan2(dy_valid, dx_valid)
            
            # Get muscle properties for valid muscles
            muscle_axon_valid = muscle_data['axon'][valid_muscle_mask]
            muscle_len_valid = muscle_data['len'][valid_muscle_mask]
            muscle_prev_target_valid = muscle_data['previous_target'][valid_muscle_mask]
            rigidity_valid = muscle_data['rigidity'][valid_muscle_mask]

            # Calculate target lengths
            node_values = node_data['value'] # Current neuron values for all nodes
            
            # Initialize targets with base length (`muscle_len`)
            targets_valid = muscle_len_valid.copy() 
            
            # Update targets only where axon is valid and energy direction allows
            if conf.ENERGY_DIRECTION == 1 or self.energy >= 0.0001:
                # Mask for muscles whose axon index is valid
                axon_index_in_bounds_mask = (muscle_axon_valid >= 0) & (muscle_axon_valid < num_nodes)
                
                # Get neuron values from the valid axons
                valid_axons_values = node_values[muscle_axon_valid[axon_index_in_bounds_mask]]
                
                # Apply to_muscle_usable logic (vectorized) to only muscles with valid axons
                targets_valid[axon_index_in_bounds_mask] = muscle_len_valid[axon_index_in_bounds_mask] * (valid_axons_values * 1.0 + 0.5)

            # Calculate force. Handle division by zero for distances and targets.
            force_valid = np.zeros_like(distances_valid)
            
            # Mask for muscles where distance is non-zero (to avoid division by zero)
            nonzero_distance_mask = (distances_valid != 0)
            
            # Calculate ratio (current_distance / target_len)
            force_valid[nonzero_distance_mask] = 1 - (distances_valid[nonzero_distance_mask] / targets_valid[nonzero_distance_mask])
            
            # Clip force to [-0.4, 0.4] as in original code
            force_valid = np.clip(force_valid, -0.4, 0.4)
            
            muscle_data['force'][valid_muscle_mask] = force_valid # Update the full force array

            # Apply forces to nodes
            cos_angles = np.cos(angles_valid)
            sin_angles = np.sin(angles_valid)
            
            # Force components for c1 (x1-x2, y1-y2 -> cos(angle), sin(angle))
            force_x_components_c1 = cos_angles * force_valid * rigidity_valid
            force_y_components_c1 = sin_angles * force_valid * rigidity_valid
            
            # Force components for c2 (opposite direction)
            force_x_components_c2 = -force_x_components_c1
            force_y_components_c2 = -force_y_components_c1

            # Accumulate force components using np.add.at for efficiency with duplicate indices
            # Initialize delta_vx/vy arrays to zeros
            delta_vx = np.zeros(num_nodes, dtype=np.float64)
            delta_vy = np.zeros(num_nodes, dtype=np.float64)

            # Handle division by zero mass by masking
            m_c1_valid = node_data['m'][c1_valid_idx]
            m_c2_valid = node_data['m'][c2_valid_idx]
            
            non_zero_m_c1_mask = (m_c1_valid != 0)
            non_zero_m_c2_mask = (m_c2_valid != 0)
            
            # Apply forces to c1 nodes
            np.add.at(delta_vx, c1_valid_idx[non_zero_m_c1_mask], force_x_components_c1[non_zero_m_c1_mask] / m_c1_valid[non_zero_m_c1_mask])
            np.add.at(delta_vy, c1_valid_idx[non_zero_m_c1_mask], force_y_components_c1[non_zero_m_c1_mask] / m_c1_valid[non_zero_m_c1_mask])

            # Apply forces to c2 nodes
            np.add.at(delta_vx, c2_valid_idx[non_zero_m_c2_mask], force_x_components_c2[non_zero_m_c2_mask] / m_c2_valid[non_zero_m_c2_mask])
            np.add.at(delta_vy, c2_valid_idx[non_zero_m_c2_mask], force_y_components_c2[non_zero_m_c2_mask] / m_c2_valid[non_zero_m_c2_mask])
            
            node_data['vx'] += delta_vx
            node_data['vy'] += delta_vy

            # Calculate energy consumed
            energy_consumed_this_step = np.sum(np.abs(muscle_prev_target_valid - targets_valid) * rigidity_valid * conf.ENERGY_UNIT)
            muscle_data['previous_target'][valid_muscle_mask] = targets_valid # Update previous_target for next step
            
        # --- Node Physics Updates ---
        total_node_nausea_this_step = 0.0

        # Store previous velocities for nausea calculation
        node_data['pvx'] = node_data['vx'].copy()
        node_data['pvy'] = node_data['vy'].copy()

        # Apply gravity (if not stabilizing)
        if not is_stabilizing:
            node_data['vy'] += conf.GRAVITY

        # Apply air friction and update positions
        node_data['vx'] *= conf.AIR_FRICTION
        node_data['vy'] *= conf.AIR_FRICTION
        node_data['y'] += node_data['vy']
        node_data['x'] += node_data['vx']
        
        # Calculate nausea
        acc_x = node_data['vx'] - node_data['pvx']
        acc_y = node_data['vy'] - node_data['pvy']
        acc_magnitudes = np.sqrt(acc_x**2 + acc_y**2)
        total_node_nausea_this_step = np.sum(acc_magnitudes**2 * conf.NAUSEA_UNIT)

        # Store previous positions (for potential future use, not directly used in this loop)
        node_data['prev_x'] = node_data['x'].copy()
        node_data['prev_y'] = node_data['y'].copy()

        # --- Wall/Ground Collision ---
        # Reset pressure for this step
        node_data['pressure'].fill(0.0) 

        for i in range(num_nodes):
            # Ground collision
            # The ground is at y=0 in simulation space. Nodes have a radius m/2.
            # So a node touches ground when its center (y) + radius (m/2) >= 0.
            if node_data['y'][i] + node_data['m'][i] / 2 >= 0 and conf.HAVE_GROUND and not is_stabilizing: 
                ground_y = 0.0
                dif = (node_data['y'][i] + node_data['m'][i] / 2) - ground_y # Penetration depth
                node_data['pressure'][i] += dif * conf.PRESSURE_UNIT # Apply pressure based on penetration
                node_data['y'][i] = ground_y - node_data['m'][i] / 2 # Snap to ground
                node_data['vy'][i] = 0.0 # Stop vertical movement
                
                # Apply friction to horizontal movement
                # Original logic: x -= vx * f. This is not standard friction.
                # A more physically accurate approach would apply a force proportional to normal force.
                # However, sticking to the original logic for fidelity:
                
                # Friction only if node is moving horizontally.
                if node_data['vx'][i] > 0:
                    node_data['vx'][i] -= node_data['f'][i] * dif * conf.FRICTION
                    if node_data['vx'][i] < 0:
                        node_data['vx'][i] = 0.0
                else: # vx <= 0
                    node_data['vx'][i] += node_data['f'][i] * dif * conf.FRICTION
                    if node_data['vx'][i] > 0:
                        node_data['vx'][i] = 0.0

            # Rectangle collisions (per node, then per rectangle)
            if not is_stabilizing:
                for r_obj in conf.RECTS:
                    flip = False
                    px, py = 0.0, 0.0
                    
                    # Optimized AABB check before detailed collision
                    if (abs(node_data['x'][i] - (r_obj.x1 + r_obj.x2) / 2) <= (r_obj.x2 - r_obj.x1 + node_data['m'][i]) / 2 and
                        abs(node_data['y'][i] - (r_obj.y1 + r_obj.y2) / 2) <= (r_obj.y2 - r_obj.y1 + node_data['m'][i]) / 2):
                        
                        # Node center is inside the rectangle (penetration case)
                        if r_obj.x1 <= node_data['x'][i] < r_obj.x2 and r_obj.y1 <= node_data['y'][i] < r_obj.y2:
                            d1 = node_data['x'][i] - r_obj.x1
                            d2 = r_obj.x2 - node_data['x'][i]
                            d3 = node_data['y'][i] - r_obj.y1
                            d4 = r_obj.y2 - node_data['y'][i]
                            
                            if d1 < d2 and d1 < d3 and d1 < d4: # Closest to left edge
                                px = r_obj.x1
                                py = node_data['y'][i]
                            elif d2 < d3 and d2 < d4: # Closest to right edge
                                px = r_obj.x2
                                py = node_data['y'][i]
                            elif d3 < d4: # Closest to top edge
                                px = node_data['x'][i]
                                py = r_obj.y1
                            else: # Closest to bottom edge
                                px = node_data['x'][i]
                                py = r_obj.y2
                            flip = True # Indicates node is inside, so repulsion direction should be reversed
                        else: # Node outside, but sphere might be colliding with rectangle
                            px = max(r_obj.x1, min(node_data['x'][i], r_obj.x2))
                            py = max(r_obj.y1, min(node_data['y'][i], r_obj.y2))
                                
                        distance = np.sqrt(((node_data['x'][i] - px) ** 2) + ((node_data['y'][i] - py) ** 2))
                        rad = node_data['m'][i] / 2
                        
                        if distance < rad or flip: # Actual collision
                            if distance == 0: 
                                distance = 0.0001 # Avoid division by zero
                            
                            wall_angle = np.arctan2(py - node_data['y'][i], px - node_data['x'][i])
                            if flip: 
                                wall_angle += np.pi # Reverse angle if penetrating from inside
                            
                            dif = rad - distance # Penetration depth
                            node_data['pressure'][i] += dif * conf.PRESSURE_UNIT # Apply pressure
                            
                            multi = rad / distance
                            if flip:
                                multi = -multi # Reverse multiplier direction if inside
                            
                            # Adjust node position to resolve overlap
                            node_data['x'][i] = (node_data['x'][i] - px) * multi + px
                            node_data['y'][i] = (node_data['y'][i] - py) * multi + py
                            
                            # Apply velocity change/reflection (based on original code's logic)
                            # This reflection is not a standard physics bounce, it's unique to the source.
                            velo_angle = np.arctan2(node_data['vy'][i], node_data['vx'][i])
                            velo_mag = np.sqrt(node_data['vx'][i]**2 + node_data['vy'][i]**2)
                            rel_angle = velo_angle - wall_angle
                            
                            # Original code's unusual "reflection" logic - literally translated
                            # It appears to calculate a y-component of relative velocity, then uses it to set new vx/vy
                            original_rel_y = np.sin(rel_angle) * velo_mag * dif * conf.FRICTION
                            node_data['vx'][i] = -np.sin(rel_angle) * original_rel_y
                            node_data['vy'][i] = np.cos(rel_angle) * original_rel_y

        # --- Node Neuron Math ---
        # Initialize value_to_be_arr with current values (for no-op or default)
        node_data['value_to_be'][:] = node_data['value'] 
        
        # Pre-fetch axon values using advanced indexing with boundary checks
        axon_value1 = np.zeros(num_nodes, dtype=np.float64)
        axon_value2 = np.zeros(num_nodes, dtype=np.float64)
        
        # Mask for valid axon1 indices and get values
        valid_axon1_mask = (node_data['axon1'] >= 0) & (node_data['axon1'] < num_nodes)
        axon_value1[valid_axon1_mask] = node_data['value'][node_data['axon1'][valid_axon1_mask]]

        # Mask for valid axon2 indices and get values
        valid_axon2_mask = (node_data['axon2'] >= 0) & (node_data['axon2'] < num_nodes)
        axon_value2[valid_axon2_mask] = node_data['value'][node_data['axon2'][valid_axon2_mask]]

        # Apply operations using boolean masks for vectorization
        # op == 0: No op / Constant - value_to_be is already defaulted to current value
        
        op_mask = (node_data['operation'] == 1) # Time
        if np.any(op_mask):
            node_data['value_to_be'][op_mask] = current_time / 60.0 # Normalize time

        op_mask = (node_data['operation'] == 2) # X position
        if np.any(op_mask):
            node_data['value_to_be'][op_mask] = node_data['x'][op_mask] * 0.2 # Scale X

        op_mask = (node_data['operation'] == 3) # Y position
        if np.any(op_mask):
            node_data['value_to_be'][op_mask] = -node_data['y'][op_mask] * 0.2 # Scale and invert Y

        op_mask = (node_data['operation'] == 4) # Addition
        if np.any(op_mask):
            node_data['value_to_be'][op_mask] = axon_value1[op_mask] + axon_value2[op_mask]

        op_mask = (node_data['operation'] == 5) # Subtraction
        if np.any(op_mask):
            node_data['value_to_be'][op_mask] = axon_value1[op_mask] - axon_value2[op_mask]

        op_mask = (node_data['operation'] == 6) # Multiplication
        if np.any(op_mask):
            node_data['value_to_be'][op_mask] = axon_value1[op_mask] * axon_value2[op_mask]

        op_mask = (node_data['operation'] == 7) # Division
        if np.any(op_mask):
            denominator = axon_value2[op_mask]
            node_data['value_to_be'][op_mask] = np.where(denominator != 0, axon_value1[op_mask] / denominator, 0.0) # Handle division by zero

        op_mask = (node_data['operation'] == 8) # Modulo
        if np.any(op_mask):
            axon1_values = axon_value1[op_mask]
            axon2_values = axon_value2[op_mask]
            # Vectorized modulo operation with zero division handling
            result = np.zeros_like(axon1_values)
            nonzero_mask = (axon2_values != 0)
            result[nonzero_mask] = np.fmod(axon1_values[nonzero_mask], axon2_values[nonzero_mask])
            node_data['value_to_be'][op_mask] = result

        op_mask = (node_data['operation'] == 9) # Sine
        if np.any(op_mask):
            node_data['value_to_be'][op_mask] = np.sin(axon_value1[op_mask])

        op_mask = (node_data['operation'] == 10) # Sigmoid
        if np.any(op_mask):
            node_data['value_to_be'][op_mask] = 1 / (1 + np.exp(-axon_value1[op_mask]))

        op_mask = (node_data['operation'] == 11) # Pressure
        if np.any(op_mask):
            node_data['value_to_be'][op_mask] = node_data['pressure'][op_mask]

        # --- Realize Math Values (Double Buffering) ---
        node_data['value'][:] = node_data['value_to_be']

        return energy_consumed_this_step, total_node_nausea_this_step

    def simulate_creature_run(self, creature: Creature) -> float:
        """Simulates a creature's movement for a fixed number of steps and returns its fitness."""
        self.simulation_timer = 0
        self.energy = self.conf.BASELINE_ENERGY
        self.total_node_nausea = 0.0

        # Extract creature data into NumPy arrays once before simulation loop
        node_data, muscle_data = self._extract_creature_data_to_numpy(creature)

        for _ in range(self.conf.SIMULATION_STEPS_PER_CREATURE):
            energy_step, nausea_step = self._simulate_step_numpy(node_data, muscle_data, self.simulation_timer, self.conf, is_stabilizing=False)
            
            # Update overall energy and nausea for creature evaluation
            self.energy = max(self.energy + self.conf.ENERGY_DIRECTION * energy_step, 0.0) # Energy never goes below 0
            self.total_node_nausea += nausea_step
            
            self.simulation_timer += 1
        
        # Update creature objects with final NumPy array states
        self._update_creature_data_from_numpy(creature, node_data, muscle_data)

        self._set_averages_numpy(node_data) # Calculate average X/Y using NumPy
        # Fitness is based on average X position after simulation. Scale by 0.2.
        return self.average_x * 0.2

    def _set_averages_numpy(self, node_data: Dict[str, np.ndarray]):
        """Calculates average X and Y positions using NumPy."""
        if len(node_data['x']) == 0:
            self.average_x = 0.0
            self.average_y = 0.0
            return 

        self.average_x = np.mean(a=node_data['x'], dtype=float)
        self.average_y = np.mean(node_data['y'], dtype=float)

class EvolutionManager:
    def __init__(self, conf: Config):
        self.conf = conf
        self.creature_simulator = CreatureSimulator(conf)

        # Global data structures for tracking evolution
        self.percentile: List[List[float]] = [] # Stores lists of floats for each generation's percentiles
        self.bar_counts: List[List[int]] = [] # Stores histogram bar counts for each generation (fitness distribution)
        self.species_raw_counts: List[List[int]] = [] # Stores raw histogram of species counts for each generation
        self.species_cumulative_counts: List[List[int]] = [] # Not directly used in UI, but kept for completeness
        self.top_species_counts: List[int] = [] # Stores the index of the most prevalent species for each generation
        self.creature_database: List[Creature] = [] # Stores the best, median, and worst creature from each generation
        self.creatures_array: List[Creature] = [] # Main population array
        self.current_gen = -1 # Current generation number (starts at 0 after initialization)

    @staticmethod
    def quick_sort(creatures_list: List[Creature]) -> List[Creature]:
        """Sorts a list of creatures by their fitness (self.fitness) in descending order."""
        return sorted(creatures_list, key=lambda c: c.fitness, reverse=True)

    def initialize_population(self) -> List[Creature]:
        print("Initializing 1000 random creatures...")
        self.current_gen = 0
        creatures = []
        
        for idx in range(1000):
            node_num = random.randint(self.conf.INITIAL_NODE_RANGE[0], self.conf.INITIAL_NODE_RANGE[1])
            muscle_num = random.randint(max(0, node_num - 1), max(0, node_num * 3 - 7)) 
            muscle_num = max(1, muscle_num) # At least one muscle for basic connectivity

            temp_n_list: List[Node] = []
            temp_m_list: List[Muscle] = []

            for i in range(node_num):
                temp_n_list.append(Node(self.conf, random.uniform(-1, 1), random.uniform(-1, 1), 0, 0, 0.4, random.uniform(0, 1), random.uniform(0,1),
                                        math.floor(random.uniform(0,self.conf.OPERATION_COUNT)),
                                        math.floor(random.uniform(0,node_num)),        
                                        math.floor(random.uniform(0,node_num))))       
            
            for i in range(muscle_num):
                tc1 = 0
                tc2 = 0
                axon = Muscle.get_new_muscle_axon(node_num) 
                
                if i < node_num - 1: # Try to connect sequentially for first few muscles
                    tc1 = i
                    tc2 = i + 1
                else: # Otherwise, random connections
                    tc1 = math.floor(random.uniform(0, node_num))
                    tc2 = tc1
                    while tc2 == tc1 and node_num > 1:
                        tc2 = math.floor(random.uniform(0, node_num))
                
                len_val = random.uniform(0.5, 1.5)
                tc1 = max(0, min(tc1, node_num - 1))
                tc2 = max(0, min(tc2, node_num - 1))

                if node_num > 1 and tc1 < len(temp_n_list) and tc2 < len(temp_n_list):
                    distance_between_nodes = np.sqrt(((temp_n_list[tc1].x - temp_n_list[tc2].x) ** 2) + ((temp_n_list[tc1].y - temp_n_list[tc2].y) ** 2))
                    if distance_between_nodes > 0:
                        len_val = distance_between_nodes
                    else:
                        len_val = 0.5 

                temp_m_list.append(Muscle(self.conf, axon, tc1, tc2, len_val, random.uniform(0.02, 0.08)))

            creature = Creature(self.conf, self.current_gen * 1000 + idx + 1, temp_n_list, temp_m_list, 0.0, True, random.uniform(40, 80), 1.0) 
            creature.stabilize_configuration()
            creature.adjust_to_center()
            creature.check_for_overlap()
            creature.check_for_lone_nodes()
            creature.check_for_bad_axons()

            creatures.append(creature)
        
        self.creatures_array = creatures
        print("Initial creatures created and stabilized.")
        return creatures

    def run_generation_simulation(self):
        print(f"\n--- Running Generation {self.current_gen} Simulation ---")
        
        for i in range(1000):
            creature: Creature = self.creatures_array[i]
            if creature:
                creature.fitness = self.creature_simulator.simulate_creature_run(creature)
            
            if (i + 1) % 100 == 0 or (i + 1) == 1000:
                print(f"  Simulated {i + 1}/1000 creatures. Last fitness: {creature.fitness:.3f}")

        print(f"Generation {self.current_gen} simulation complete.")

    def sort_and_reproduce(self):
        print(f"\n--- Processing Generation {self.current_gen} Results and Reproducing ---")

        # Filter out None values and sort
        c_sorted = self.quick_sort([c for c in self.creatures_array if c is not None])

        # Store percentile data
        percentile_data_current_gen = [0.0] * len(self.conf.PERCENTILE_INDICES)
        for i, p_idx in enumerate(self.conf.PERCENTILE_INDICES):
            percentile_data_current_gen[i] = c_sorted[p_idx].fitness if p_idx < len(c_sorted) else 0.0
        self.percentile.append(percentile_data_current_gen)

        # Store creature database (worst, median, best)
        if len(c_sorted) >= 1000:
            self.creature_database.append(c_sorted[999].copy_creature()) # Worst
            self.creature_database.append(c_sorted[499].copy_creature()) # Median
            self.creature_database.append(c_sorted[0].copy_creature())   # Best
        elif len(c_sorted) > 0:
            self.creature_database.append(c_sorted[-1].copy_creature()) # Worst
            self.creature_database.append(c_sorted[len(c_sorted)//2].copy_creature()) # Median
            self.creature_database.append(c_sorted[0].copy_creature()) # Best
        else: # Handle case where no creatures exist (shouldn't happen after init)
             self.creature_database.extend([Creature(self.conf, 0, [], [], 0.0, False, 0.0, 0.0)] * 3)


        # Histogram bar counts for fitness
        begin_bar = [0] * self.conf.BAR_LEN
        for creature in c_sorted:
            if math.isnan(creature.fitness): creature.fitness = 0.0
            bar = math.floor(creature.fitness * self.conf.HIST_BARS_PER_METER - self.conf.MIN_BAR)
            if 0 <= bar < self.conf.BAR_LEN:
                begin_bar[bar] += 1
        self.bar_counts.append(begin_bar)

        # Species counts (raw distribution)
        begin_species = np.zeros(101, dtype=np.int32) # Max possible species index is 99 (node_mod_10 * 10 + muscle_mod_10)
        for creature in c_sorted:
            species = (len(creature.nodes) % 10) * 10 + (len(creature.muscles) % 10)
            if 0 <= species < 101: # Ensure species index is within bounds
                begin_species[species] += 1
        
        self.species_raw_counts.append(begin_species.tolist()) # Store as list for consistency with other lists

        # Cumulative species counts (if needed, otherwise can remove)
        species_counts_current_gen = [0] * 101
        cum = 0
        record_max_species_count = 0
        holder_top_species_idx = 0
        for i in range(100):
            cum += begin_species[i]
            species_counts_current_gen[i+1] = cum
            if begin_species[i] > record_max_species_count:
                record_max_species_count = begin_species[i]
                holder_top_species_idx = i
        self.species_cumulative_counts.append(species_counts_current_gen)
        self.top_species_counts.append(holder_top_species_idx)

        new_creatures_for_next_gen: List[Creature] = [None] * 1000 # Pre-allocate 1000 slots

        # Pre-calculate log(1000) for efficiency
        log_1000 = math.log(1000) 

        for j in range(500): # Create 500 copies of survivors and 500 mutated children
            # Probabilistic survival based on rank. Higher rank = higher chance to survive.
            # Ranks are 0 to 999.
            # Logarithmic decay: 1 - log(rank+1)/log(total_population)
            # This makes lower ranks have significantly higher survival probability.
            prob_of_survival_for_j = 1.0 - (math.log(j + 1) / log_1000) if log_1000 != 0 else (1.0 if j == 0 else 0.0)
            
            survives_based_on_randomness = (random.uniform(0, 1) < prob_of_survival_for_j)

            # The survivor for copying is either from the top half or the bottom half
            # based on the survival probability. This encourages diversity.
            survivor_creature = c_sorted[j] if survives_based_on_randomness else c_sorted[999-j]
            
            new_creatures_for_next_gen[j] = survivor_creature.copy_creature(new_id=(self.current_gen+1)*1000 + j + 1)
            
            # The other slot gets a mutated child of the survivor
            child_creature = survivor_creature.modified(new_id=(self.current_gen+1)*1000 + (999-j) + 1)
            child_creature.stabilize_configuration() # Stabilize mutated child
            child_creature.adjust_to_center() # Center mutated child
            new_creatures_for_next_gen[999-j] = child_creature

        self.creatures_array = new_creatures_for_next_gen 
        self.current_gen += 1
        print(f"Generation {self.current_gen-1} processed. New generation {self.current_gen} created.")

class SimulationUI:
    def __init__(self):
        self.config = Config()
        self.evolution_manager = EvolutionManager(self.config)
        self.simulation_thread = None
        self.simulation_running = False
        self.total_generations_run = 0
        
        # Data storage for plots
        self.fitness_history = []
        self.species_history = []
        self.percentile_history = []
        
        # Creature rendering specific (for the main simulation tab)
        self.rendering_simulator = CreatureSimulator(self.config) # Dedicated simulator for rendering
        self.current_rendered_creature: Optional[Creature] = None # The creature to be rendered
        self.rendering_node_data: Optional[Dict[str, np.ndarray]] = None # numpy data for the creature being rendered
        self.rendering_muscle_data: Optional[Dict[str, np.ndarray]] = None # numpy data for the creature being rendered
        self.rendering_max_steps = self.config.SIMULATION_STEPS_PER_CREATURE
        self.rendering_step_counter = 0
        self.render_lock = Lock() # To protect access to rendering_creature data
        self.play_creature_run_flag = False # Flag to control if a creature's run is being replayed

        # Gallery creatures
        self.gallery_creatures: List[Creature] = []
        self.selected_gallery_creature: Optional[Creature] = None

        # Visualization scale for creature rendering (pixels per meter)
        self.creature_view_scale = 50.0 

        # Initialize DearPyGUI
        dpg.create_context()
        self.setup_ui()
        self.load_creatures() # Load creatures on startup

    def setup_ui(self):
        # Main window
        with dpg.window(tag="Primary Window"):
            with dpg.tab_bar():
                # --- Simulation Tab ---
                with dpg.tab(label="Simulation"):
                    with dpg.group(horizontal=True):
                        # Control panel
                        with dpg.child_window(width=300):
                            dpg.add_text("Evolutionary Creature Simulation", color=(0, 200, 255))
                            dpg.add_spacer(height=10)
                            
                            # Simulation controls
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Create 1000 Creatures", callback=self.create_initial_population)
                                dpg.add_button(label="Run Simulation", callback=self.start_simulation)
                                dpg.add_button(label="Stop", callback=self.stop_simulation)
                            
                            dpg.add_spacer(height=20)
                            dpg.add_text("Current Status:")
                            dpg.add_text("Not running", tag="status_text")
                            dpg.add_text("Generations Run: 0", tag="generation_text")
                            
                            dpg.add_spacer(height=20)
                            dpg.add_text("Configuration Parameters:")
                            with dpg.collapsing_header(label="Physics Settings", default_open=True):
                                dpg.add_checkbox(label="Enable Ground", default_value=self.config.HAVE_GROUND, 
                                                tag="ground_checkbox", callback=self.update_config)
                                dpg.add_slider_float(label="Gravity", min_value=0.0, max_value=0.1, 
                                                   default_value=self.config.GRAVITY, format="%.4f",
                                                   tag="gravity_slider", callback=self.update_config)
                                dpg.add_slider_float(label="Air Friction", min_value=0.8, max_value=1.0, 
                                                   default_value=self.config.AIR_FRICTION, format="%.2f",
                                                   tag="air_friction_slider", callback=self.update_config)
                            
                            with dpg.collapsing_header(label="Evolution Settings", default_open=True):
                                dpg.add_slider_float(label="Mutation Chance", min_value=0.01, max_value=0.2, 
                                                   default_value=self.config.BIG_MUTATION_CHANCE, format="%.3f",
                                                   tag="mutation_slider", callback=self.update_config)
                                dpg.add_slider_int(label="Simulation Steps", min_value=100, max_value=2000, 
                                                  default_value=self.config.SIMULATION_STEPS_PER_CREATURE,
                                                  tag="steps_slider", callback=self.update_config)
                        
                        # Visualization area
                        with dpg.child_window():
                            with dpg.tab_bar():
                                # Fitness plot tab
                                with dpg.tab(label="Fitness Progress"):
                                    with dpg.plot(label="Fitness Over Generations", height=300, width=-1):
                                        dpg.add_plot_legend()
                                        dpg.add_plot_axis(dpg.mvXAxis, label="Generation", tag="fitness_x_axis")
                                        dpg.add_plot_axis(dpg.mvYAxis, label="Fitness (m)", tag="fitness_y_axis")
                                        dpg.add_line_series([], [], label="Best Fitness", parent="fitness_y_axis", tag="best_series")
                                        dpg.add_line_series([], [], label="Median Fitness", parent="fitness_y_axis", tag="median_series")
                                        dpg.add_line_series([], [], label="Worst Fitness", parent="fitness_y_axis", tag="worst_series")
                                
                                # Species distribution tab
                                with dpg.tab(label="Species Distribution"):
                                    with dpg.plot(label="Species Distribution", height=300, width=-1):
                                        dpg.add_plot_legend()
                                        dpg.add_plot_axis(dpg.mvXAxis, label="Species Index", tag="species_x_axis")
                                        dpg.add_plot_axis(dpg.mvYAxis, label="Count", tag="species_y_axis")
                                        dpg.add_bar_series([], [], label="Current Species", parent="species_y_axis", tag="species_bar_series")
                                
                                # Percentile plot tab
                                with dpg.tab(label="Percentile Distribution"):
                                    with dpg.plot(label="Percentile Distribution", height=300, width=-1):
                                        dpg.add_plot_legend()
                                        dpg.add_plot_axis(dpg.mvXAxis, label="Percentile Index", tag="percentile_x_axis")
                                        dpg.add_plot_axis(dpg.mvYAxis, label="Fitness (m)", tag="percentile_y_axis")
                                        dpg.add_bar_series([], [], label="Fitness by Percentile", parent="percentile_y_axis", tag="percentile_bar_series")
                            
                                # Creature Visualization Tab (main simulation tab display)
                                with dpg.tab(label="Creature Visualization"):
                                    dpg.add_spacer(height=5)
                                    dpg.add_text("Rendering: No creature selected", tag="rendering_status_text")
                                    dpg.add_progress_bar(width=-1, tag="rendering_progress", overlay="0%", default_value=0.0)
                                    dpg.add_spacer(height=5)
                                    # Use a child window with a drawlist for the creature
                                    with dpg.child_window(width=-1, height=400, border=True, tag="creature_display_window"):
                                        with dpg.drawlist(width=dpg.get_item_width("creature_display_window"), 
                                                          height=dpg.get_item_height("creature_display_window"), 
                                                          tag="creature_drawlist"):
                                            pass # Initial placeholder
                                        # dpg.set_viewport_resize_callback( self.on_creature_window_resize) # Handled by child_window resizing behavior
                            
                            # Creature information for current generation's best/median/worst
                            dpg.add_spacer(height=20)
                            with dpg.group(horizontal=True):
                                with dpg.child_window(width=300):
                                    dpg.add_text("Best Creature Info", color=(0, 255, 0))
                                    dpg.add_text("ID: ", tag="best_id")
                                    dpg.add_text("Fitness: ", tag="best_fitness")
                                    dpg.add_text("Nodes: ", tag="best_nodes")
                                    dpg.add_text("Muscles: ", tag="best_muscles")
                                
                                with dpg.child_window(width=300):
                                    dpg.add_text("Median Creature Info", color=(255, 255, 0))
                                    dpg.add_text("ID: ", tag="median_id")
                                    dpg.add_text("Fitness: ", tag="median_fitness")
                                    dpg.add_text("Nodes: ", tag="median_nodes")
                                    dpg.add_text("Muscles: ", tag="median_muscles")
                                
                                with dpg.child_window(width=300):
                                    dpg.add_text("Worst Creature Info", color=(255, 0, 0))
                                    dpg.add_text("ID: ", tag="worst_id")
                                    dpg.add_text("Fitness: ", tag="worst_fitness")
                                    dpg.add_text("Nodes: ", tag="worst_nodes")
                                    dpg.add_text("Muscles: ", tag="worst_muscles")
                                    
                # --- Creature Gallery Tab ---
                with dpg.tab(label="Creature Gallery"):
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=300, tag="creature_list_panel"):
                            dpg.add_text("Saved Creatures", color=(0, 200, 255))
                            dpg.add_spacer(height=10)
                            with dpg.group(tag="creature_list_container"):
                                self.creature_list_box = dpg.add_listbox(
                                    items=[],
                                    width=-1,
                                    num_items=10,
                                    callback=self.select_gallery_creature
                                )
                            dpg.add_spacer(height=10)
                            dpg.add_button(label="Add to Simulation Replay", callback=self.add_selected_to_replay)
                            dpg.add_button(label="Delete Selected Creature", callback=self.delete_selected_gallery_creature)
                            dpg.add_button(label="Save All Creatures", callback=self.save_creatures)
                            dpg.add_button(label="Load Creatures", callback=self.load_creatures)
                            dpg.add_button(label="Add Random Initial Creature", callback=self.add_random_initial_creature_to_gallery)
                        
                        with dpg.child_window(tag="gallery_creature_view"):
                            # The drawlist here will contain the grid of creature thumbnails
                            with dpg.drawlist(width=dpg.get_item_width("gallery_creature_view"), 
                                              height=dpg.get_item_height("gallery_creature_view"), 
                                              tag="gallery_drawlist"):
                                pass
        
        dpg.set_primary_window("Primary Window", True)
        
    def _sim_to_draw_coords(self, x_sim: float, y_sim: float,
                           draw_offset_x: float, draw_offset_y: float, # Where sim (0,0) maps to in DPG
                           scale: float) -> tuple[float, float]:
        """
        Converts simulation coordinates (x, y) to DearPyGUI drawlist coordinates
        given an offset and scale.

        Args:
            x_sim: Simulation X coordinate of the point.
            y_sim: Simulation Y coordinate of the point.
            draw_offset_x: The DPG X coordinate that maps to simulation X=0.
            draw_offset_y: The DPG Y coordinate that maps to simulation Y=0.
            scale: Pixels per simulation unit.
        """
        # Apply scale and then offset
        draw_x = draw_offset_x + x_sim * scale
        draw_y = draw_offset_y + y_sim * scale # Y-axis direction is consistent (downwards)

        return draw_x, draw_y

    def _draw_creature(self, node_data: Optional[Dict[str, np.ndarray]], muscle_data: Optional[Dict[str, np.ndarray]],
                       drawlist_tag: str,
                       render_offset_x_dpg: float, render_offset_y_dpg: float,
                       scale: float,
                       draw_rects: bool = False):
        """
        Draws the state of a creature from its NumPy data onto a DearPyGUI drawlist.
        Coordinates are relative to `render_offset_x_dpg`, `render_offset_y_dpg` which represent sim (0,0).
        This function should only *add* drawings, not clear the drawlist.
        """
        if node_data is None or muscle_data is None:
            return

        # Draw Rectangles (obstacles) - only for main simulation view
        if draw_rects and self.config.RECTS:
            for r_obj in self.config.RECTS:
                x1_draw, y1_draw = self._sim_to_draw_coords(r_obj.x1, r_obj.y1, render_offset_x_dpg, render_offset_y_dpg, scale)
                x2_draw, y2_draw = self._sim_to_draw_coords(r_obj.x2, r_obj.y2, render_offset_x_dpg, render_offset_y_dpg, scale)
                
                # Ensure proper top-left, bottom-right order for DPG rectangle
                draw_rect_x1 = min(x1_draw, x2_draw)
                draw_rect_y1 = min(y1_draw, y2_draw)
                draw_rect_x2 = max(x1_draw, x2_draw)
                draw_rect_y2 = max(y1_draw, y2_draw)
                
                dpg.draw_rectangle((draw_rect_x1, draw_rect_y1), (draw_rect_x2, draw_rect_y2), color=(200, 200, 0, 255), fill=(200, 200, 0, 100), parent=drawlist_tag)

        # Draw muscles first (behind nodes)
        for i in range(len(muscle_data['axon'])):
            c1_idx = muscle_data['c1'][i]
            c2_idx = muscle_data['c2'][i]

            # Ensure node indices are valid
            if c1_idx < 0 or c1_idx >= len(node_data['x']) or \
               c2_idx < 0 or c2_idx >= len(node_data['x']):
                continue # Skip invalid muscle connections

            x1_sim = node_data['x'][c1_idx]
            y1_sim = node_data['y'][c1_idx]
            x2_sim = node_data['x'][c2_idx]
            y2_sim = node_data['y'][c2_idx]

            x1_draw, y1_draw = self._sim_to_draw_coords(x1_sim, y1_sim, render_offset_x_dpg, render_offset_y_dpg, scale)
            x2_draw, y2_draw = self._sim_to_draw_coords(x2_sim, y2_sim, render_offset_x_dpg, render_offset_y_dpg, scale)

            # Safely get force, default to 0 for static rendering (e.g., gallery)
            force = muscle_data.get('force', np.zeros_like(muscle_data['axon'], dtype=np.float64))[i] 
            color = (100, 100, 100, 255) # Default grey
            if force > 0.05: # Pushing (expanding)
                color = (0, 200, 0, 255)
            elif force < -0.05: # Pulling (contracting)
                color = (200, 0, 0, 255)

            dpg.draw_line((x1_draw, y1_draw), (x2_draw, y2_draw), color=color, thickness=2, parent=drawlist_tag)

        # Draw nodes
        for i in range(len(node_data['x'])):
            x_sim = node_data['x'][i]
            y_sim = node_data['y'][i]
            m_sim = node_data['m'][i] # Mass acts as diameter in simulation

            radius_draw = m_sim / 2 * scale # Node radius is m/2
            x_draw, y_draw = self._sim_to_draw_coords(x_sim, y_sim, render_offset_x_dpg, render_offset_y_dpg, scale)

            dpg.draw_circle((x_draw, y_draw), radius_draw, color=(150, 150, 150, 255), fill=(150, 150, 150, 200), parent=drawlist_tag)
    
    def create_initial_population(self):
        """Create initial population and add to gallery"""
        if self.evolution_manager.current_gen == -1:
            initial_creatures = self.evolution_manager.initialize_population()
            self.gallery_creatures.extend(initial_creatures)
            self.update_gallery_list()
            dpg.set_value("status_text", "1000 creatures created and added to gallery")
        else:
            dpg.set_value("status_text", "Population already initialized")

    def update_config(self):
        """Update configuration parameters from UI controls"""
        self.config.HAVE_GROUND = dpg.get_value("ground_checkbox")
        self.config.GRAVITY = dpg.get_value("gravity_slider")
        self.config.AIR_FRICTION = dpg.get_value("air_friction_slider")
        self.config.BIG_MUTATION_CHANCE = dpg.get_value("mutation_slider")
        self.config.SIMULATION_STEPS_PER_CREATURE = dpg.get_value("steps_slider")
    
    def start_simulation(self):
        """Start the simulation in a separate thread"""
        if self.simulation_running:
            return
            
        if self.evolution_manager.current_gen == -1:
            dpg.set_value("status_text", "Please create initial population first")
            return
            
        self.simulation_running = True
        dpg.set_value("status_text", "Running...")
        
        # Start simulation thread
        self.simulation_thread = Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()
    
    def stop_simulation(self):
        """Stop the running simulation"""
        self.simulation_running = False
        dpg.set_value("status_text", "Stopped")
    
    def run_simulation(self):
        """Run the simulation continuously until stopped"""
        while self.simulation_running:
            # Run one generation
            self.evolution_manager.run_generation_simulation()
            self.evolution_manager.sort_and_reproduce()
            self.total_generations_run += 1
            
            # Update data for UI
            self.update_simulation_data()
            
            # Small delay to allow UI updates from main thread, though render_callback helps
            time.sleep(0.01) # Reduced sleep as render_callback handles continuous drawing
        
        dpg.set_value("status_text", "Stopped")
    
    def update_simulation_data(self):
        """Update data structures for UI visualization and prepare next creature for rendering."""
        if self.evolution_manager.current_gen <= 0:
            return
        
        # Get data from the last completed generation (current_gen is already incremented)
        gen_idx = self.evolution_manager.current_gen - 1
        
        # Fitness data (best, median, worst)
        if len(self.evolution_manager.creature_database) >= (gen_idx + 1) * 3:
            best = self.evolution_manager.creature_database[gen_idx * 3 + 2].fitness
            median = self.evolution_manager.creature_database[gen_idx * 3 + 1].fitness
            worst = self.evolution_manager.creature_database[gen_idx * 3].fitness
            self.fitness_history.append((best, median, worst))
        
        # Species distribution
        if len(self.evolution_manager.species_raw_counts) > gen_idx:
            species_dist_data = self.evolution_manager.species_raw_counts[gen_idx]
            self.species_history.append(np.array(species_dist_data, dtype=np.int32))
        
        # Percentile data
        if len(self.evolution_manager.percentile) > gen_idx:
            percentiles = self.evolution_manager.percentile[gen_idx]
            self.percentile_history.append(percentiles)
            
        # Prepare the best creature of this completed generation for rendering
        with self.render_lock:
            if len(self.evolution_manager.creature_database) >= (gen_idx + 1) * 3:
                best_creature_for_rendering = self.evolution_manager.creature_database[gen_idx * 3 + 2].copy_creature()
                self.set_creature_for_replay(best_creature_for_rendering)
            self.play_creature_run_flag = True # Replay the newly set creature

    def set_creature_for_replay(self, creature: Creature):
        """Prepares a creature for replay visualization in the main simulation tab."""
        with self.render_lock:
            if creature is None:
                return
                
            self.current_rendered_creature = creature.copy_creature()
            
            # Reset the creature's state for the start of a replay
            for node in self.current_rendered_creature.nodes:
                node.vx, node.vy = 0.0, 0.0
                node.prev_x, node.prev_y = node.x, node.y
                node.pvx, node.pvy = 0.0, 0.0
                node.pressure = 0.0
                # Reset neuron values for a consistent replay start
                if self.config.OPERATION_AXONS[node.operation] == 0: # If constant/input type
                    if node.operation == 1: node.value = 0.0 # Time
                    elif node.operation == 2: node.value = node.x * 0.2
                    elif node.operation == 3: node.value = -node.y * 0.2
                    elif node.operation == 11: node.value = 0.0 # Pressure
                    else: node.value = random.uniform(0, 1) # Default constant
                else: # For operations that take inputs, initialize to small value
                    node.value = 0.0 
                node.value_to_be = node.value

            for muscle in self.current_rendered_creature.muscles:
                muscle.previous_target = muscle.len * 0.5 # Start at a mid-point tension
                muscle.force = 0.0

            # Re-center it for display
            self.current_rendered_creature.adjust_to_center()
            self.rendering_node_data, self.rendering_muscle_data = self.rendering_simulator._extract_creature_data_to_numpy(self.current_rendered_creature)
            self.rendering_step_counter = 0
            self.rendering_max_steps = self.config.SIMULATION_STEPS_PER_CREATURE
            dpg.set_value("rendering_status_text", f"Rendering Creature ID: {self.current_rendered_creature.id}")
            dpg.set_value("rendering_progress", 0.0)
            self.play_creature_run_flag = True  # Start playing immediately

    def update_ui(self):
        """Update UI elements with current simulation data and perform rendering steps."""
        # Only update plot data if simulation is running or has run at least one gen
        if not self.simulation_running and self.total_generations_run == 0:
            dpg.set_value("status_text", "Stopped")
        else:
            dpg.set_value("status_text", "Running...")
        
        # Update generation counter
        dpg.set_value("generation_text", f"Generations Run: {self.total_generations_run}")
        
        # Update fitness plot
        if self.fitness_history:
            generations = list(range(1, len(self.fitness_history) + 1))
            best = [x[0] for x in self.fitness_history]
            median = [x[1] for x in self.fitness_history]
            worst = [x[2] for x in self.fitness_history]
            
            dpg.set_value("best_series", [generations, best])
            dpg.set_value("median_series", [generations, median])
            dpg.set_value("worst_series", [generations, worst])
            
            dpg.set_axis_limits("fitness_x_axis", 0, len(generations) + 1)
            max_fitness = max(best + median + worst) if (best or median or worst) else 1.0 
            min_fitness = min(0, min(worst) * 1.1 if worst else 0)
            dpg.set_axis_limits("fitness_y_axis", min_fitness, max(1.0, max_fitness * 1.1))
        
        # Update species plot (bar chart of raw counts)
        if self.species_history:
            current_species_data = self.species_history[-1] # Get raw counts for last generation
            species_indices = np.arange(len(current_species_data)) # X-coordinates for bars
            
            # Filter out species with 0 count for cleaner plot
            non_zero_indices = species_indices[current_species_data != 0]
            non_zero_counts = current_species_data[current_species_data != 0]

            if len(non_zero_indices) > 0:
                dpg.set_value("species_bar_series", [non_zero_indices, non_zero_counts])
                dpg.set_axis_limits("species_x_axis", np.min(non_zero_indices) - 1, np.max(non_zero_indices) + 1)
                dpg.set_axis_limits("species_y_axis", 0, max(10, np.max(non_zero_counts) * 1.2))
            else: # Clear bars if no data
                dpg.set_value("species_bar_series", [[], []]) 
                dpg.set_axis_limits("species_x_axis", 0, 100)
                dpg.set_axis_limits("species_y_axis", 0, 10)
        
        # Update percentile plot (bar chart of fitness at specific percentiles)
        if self.percentile_history:
            current_percentiles = self.percentile_history[-1]
            percentile_x_values = list(range(len(self.config.PERCENTILE_INDICES)))
            
            dpg.set_value("percentile_bar_series", [percentile_x_values, current_percentiles])
            dpg.set_axis_limits("percentile_x_axis", -1, len(percentile_x_values))
            dpg.set_axis_limits("percentile_y_axis", 0, max(0.1, max(current_percentiles) * 1.1))
        
        # Update creature info for best/median/worst
        if self.evolution_manager.creature_database and len(self.evolution_manager.creature_database) >= (self.evolution_manager.current_gen) * 3:
            last_gen_idx = self.evolution_manager.current_gen - 1
            if last_gen_idx >= 0: # Ensure we have at least one completed generation
                best_idx = last_gen_idx * 3 + 2
                if best_idx < len(self.evolution_manager.creature_database):
                    best = self.evolution_manager.creature_database[best_idx]
                    dpg.set_value("best_id", f"ID: {best.id}")
                    dpg.set_value("best_fitness", f"Fitness: {best.fitness:.3f} m")
                    dpg.set_value("best_nodes", f"Nodes: {len(best.nodes)}")
                    dpg.set_value("best_muscles", f"Muscles: {len(best.muscles)}")
                
                median_idx = last_gen_idx * 3 + 1
                if median_idx < len(self.evolution_manager.creature_database):
                    median = self.evolution_manager.creature_database[median_idx]
                    dpg.set_value("median_id", f"ID: {median.id}")
                    dpg.set_value("median_fitness", f"Fitness: {median.fitness:.3f} m")
                    dpg.set_value("median_nodes", f"Nodes: {len(median.nodes)}")
                    dpg.set_value("median_muscles", f"Muscles: {len(median.muscles)}")
                
                worst_idx = last_gen_idx * 3
                if worst_idx < len(self.evolution_manager.creature_database):
                    worst = self.evolution_manager.creature_database[worst_idx]
                    dpg.set_value("worst_id", f"ID: {worst.id}")
                    dpg.set_value("worst_fitness", f"Fitness: {worst.fitness:.3f} m")
                    dpg.set_value("worst_nodes", f"Nodes: {len(worst.nodes)}")
                    dpg.set_value("worst_muscles", f"Muscles: {len(worst.muscles)}")
        
        # --- Creature Rendering Logic for Main Simulation Tab ---
        with self.render_lock:
            dl_width = dpg.get_item_width("creature_drawlist")
            dl_height = dpg.get_item_height("creature_drawlist")
            dpg.delete_item("creature_drawlist", children_only=True) # Clear for new frame

            if self.play_creature_run_flag and self.current_rendered_creature:
                if self.rendering_step_counter < self.rendering_max_steps:
                    # Only update if we have data
                    if self.rendering_node_data is not None and self.rendering_muscle_data is not None:
                        self.rendering_simulator._simulate_step_numpy(
                            self.rendering_node_data, self.rendering_muscle_data,
                            self.rendering_step_counter, self.config, is_stabilizing=False
                        )
                        self.rendering_step_counter += 1
                        
                        progress = self.rendering_step_counter / self.rendering_max_steps
                        dpg.set_value("rendering_progress", progress)
                        dpg.configure_item("rendering_progress", overlay=f"{int(progress*100)}%")

                        # Update the creature's average position for camera centering
                        self.rendering_simulator._set_averages_numpy(self.rendering_node_data)
                        
                        # Drawing the ground for the main view
                        ground_line_from_bottom = 50 # DPG pixels from bottom of drawlist
                        ground_draw_y = dl_height - ground_line_from_bottom
                        if self.config.HAVE_GROUND:
                            dpg.draw_line((0, ground_draw_y), (dl_width, ground_draw_y), color=(100, 100, 100, 255), thickness=2, parent="creature_drawlist")

                        # Calculate offset for drawing based on creature's average position and desired ground
                        # sim (0,0) maps to: (center_of_drawlist_x - avg_x*scale, ground_y_draw - avg_y*scale)
                        render_offset_x_dpg = dl_width / 2 - self.rendering_simulator.average_x * self.creature_view_scale
                        # The sim_y=0 point (ground) should map to ground_draw_y.
                        # So, sim_y=current_avg_y should map to `ground_draw_y + current_avg_y * scale`.
                        # Thus, the offset (where sim y=0 maps) is `ground_draw_y`.
                        render_offset_y_dpg = ground_draw_y 
                        
                        self._draw_creature(self.rendering_node_data, self.rendering_muscle_data, "creature_drawlist", 
                                          render_offset_x_dpg, render_offset_y_dpg,
                                          self.creature_view_scale, draw_rects=True)
                else:
                    self.play_creature_run_flag = False # Stop replay when finished
                    dpg.set_value("rendering_status_text", f"Replay Complete for ID: {self.current_rendered_creature.id}.")
                    dpg.set_value("rendering_progress", 1.0)
            else:
                # If not replaying, show status
                if self.current_rendered_creature:
                    dpg.set_value("rendering_status_text", f"Replay Paused for ID: {self.current_rendered_creature.id}.")
                else:
                    dpg.set_value("rendering_status_text", "No creature selected for replay.")
                
        # --- Gallery Rendering Logic ---
        self.draw_gallery()
            
    def run(self):
        """Run the application"""
        dpg.create_viewport(title='Evolutionary Creature Simulation', width=self.config.SIM_WIDTH, height=self.config.SIM_HEIGHT)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    # --- Gallery Callbacks and Logic ---
    def update_gallery_list(self):
        """Updates the listbox with names/IDs of creatures in the gallery."""
        creature_names = [f"ID:{c.id} (F:{c.fitness:.2f})" for c in self.gallery_creatures]
        dpg.configure_item(self.creature_list_box, items=creature_names)

    def select_gallery_creature(self, sender, app_data):
        """Callback for when a creature is selected in the listbox."""
        if isinstance(app_data, str): # DPG sometimes returns value as string if listbox uses values
            try:
                selected_index = dpg.get_item_configuration(sender)["items"].index(app_data)
            except ValueError:
                return
        else: # Otherwise, it's the index
            selected_index = app_data
            
        if 0 <= selected_index < len(self.gallery_creatures):
            self.selected_gallery_creature = self.gallery_creatures[selected_index]
        else:
            self.selected_gallery_creature = None
        
        # Force a redraw of the gallery to highlight the selected creature
        self.draw_gallery()

    def add_selected_to_replay(self):
        """Copies the selected creature to the main simulation replay view."""
        if self.selected_gallery_creature:
            self.set_creature_for_replay(self.selected_gallery_creature)
            self.play_creature_run_flag = True
            # Switch to simulation tab
            dpg.set_value(dpg.get_item_parent("creature_display_window"), dpg.get_item_alias("Creature Visualization"))

    def delete_selected_gallery_creature(self):
        """Removes the selected creature from the gallery."""
        if self.selected_gallery_creature:
            # Ensure the selected creature is actually in the list before trying to remove by ID.
            # Otherwise, if selected_gallery_creature is an old object no longer in gallery_creatures, this could fail.
            initial_count = len(self.gallery_creatures)
            self.gallery_creatures = [c for c in self.gallery_creatures if c.id != self.selected_gallery_creature.id]
            if len(self.gallery_creatures) < initial_count: # Only clear selection if something was actually removed
                self.selected_gallery_creature = None
            self.update_gallery_list()
            self.draw_gallery()

    def add_random_initial_creature_to_gallery(self):
        """Creates a new random creature (like at the start of evolution) and adds it to the gallery."""
        node_num = random.randint(self.config.INITIAL_NODE_RANGE[0], self.config.INITIAL_NODE_RANGE[1])
        muscle_num = random.randint(max(0, node_num - 1), max(0, node_num * 3 - 7)) 
        muscle_num = max(1, muscle_num) 

        temp_n_list: List[Node] = []
        temp_m_list: List[Muscle] = []

        for i in range(node_num):
            temp_n_list.append(Node(self.config, random.uniform(-1, 1), random.uniform(-1, 1), 0, 0, 0.4, random.uniform(0, 1), random.uniform(0,1),
                                    math.floor(random.uniform(0,self.config.OPERATION_COUNT)),
                                    math.floor(random.uniform(0,node_num)),        
                                    math.floor(random.uniform(0,node_num))))       
        
        for i in range(muscle_num):
            tc1 = 0
            tc2 = 0
            axon = Muscle.get_new_muscle_axon(node_num) 
            
            if i < node_num - 1: 
                tc1 = i
                tc2 = i + 1
            else:
                tc1 = math.floor(random.uniform(0, node_num))
                tc2 = tc1
                while tc2 == tc1 and node_num > 1:
                    tc2 = math.floor(random.uniform(0, node_num))
            
            len_val = random.uniform(0.5, 1.5)
            tc1 = max(0, min(tc1, node_num - 1))
            tc2 = max(0, min(tc2, node_num - 1))

            if node_num > 1 and tc1 < len(temp_n_list) and tc2 < len(temp_n_list):
                distance_between_nodes = np.sqrt(((temp_n_list[tc1].x - temp_n_list[tc2].x) ** 2) + ((temp_n_list[tc1].y - temp_n_list[tc2].y) ** 2))
                if distance_between_nodes > 0:
                    len_val = distance_between_nodes
                else:
                    len_val = 0.5 

            temp_m_list.append(Muscle(self.config, axon, tc1, tc2, len_val, random.uniform(0.02, 0.08)))

        # Assign a unique ID for gallery creatures
        new_id = 1
        if self.gallery_creatures:
            # Find max ID from gallery creatures, then check creatures_array, then creature_database
            max_gallery_id = max(c.id for c in self.gallery_creatures)
            max_population_id = 0
            if self.evolution_manager.creatures_array:
                max_population_id = max(c.id for c in self.evolution_manager.creatures_array if c is not None)
            max_db_id = 0
            if self.evolution_manager.creature_database:
                max_db_id = max(c.id for c in self.evolution_manager.creature_database)
            
            new_id = max(max_gallery_id, max_population_id, max_db_id) + 1
        
        new_creature = Creature(self.config, new_id, temp_n_list, temp_m_list, 0.0, True, random.uniform(40, 80), 1.0) 
        new_creature.stabilize_configuration()
        new_creature.adjust_to_center()
        new_creature.check_for_overlap()
        new_creature.check_for_lone_nodes()
        new_creature.check_for_bad_axons()

        self.gallery_creatures.append(new_creature)
        self.update_gallery_list()
        self.draw_gallery()
        print(f"Added random creature ID: {new_creature.id} to gallery.")

    def save_creatures(self):
        """Save all creatures in the gallery to JSON file."""
        try:
            creatures_data = [c.to_dict() for c in self.gallery_creatures]
            with open(self.config.CREATURE_DB_FILE, 'w') as f:
                json.dump(creatures_data, f, indent=2)
            print(f"Saved {len(creatures_data)} creatures to {self.config.CREATURE_DB_FILE}")
        except Exception as e:
            print(f"Error saving creatures: {e}")

    def load_creatures(self):
        """Load creatures from JSON file into the gallery."""
        try:
            if os.path.exists(self.config.CREATURE_DB_FILE):
                with open(self.config.CREATURE_DB_FILE, 'r') as f:
                    creatures_data = json.load(f)
                
                self.gallery_creatures = []
                for creature_data in creatures_data:
                    self.gallery_creatures.append(Creature.from_dict(creature_data, self.config))
                self.update_gallery_list()
                self.draw_gallery()
                print(f"Loaded {len(self.gallery_creatures)} creatures from {self.config.CREATURE_DB_FILE}")
            else:
                print(f"No creature database found at {self.config.CREATURE_DB_FILE}.")
        except Exception as e:
            print(f"Error loading creatures: {e}")

    def draw_gallery(self):
        """Draws all creatures in the gallery grid with the selected one highlighted."""
        dpg.delete_item("gallery_drawlist", children_only=True)
        
        dl_width = dpg.get_item_width("gallery_drawlist")
        dl_height = dpg.get_item_height("gallery_drawlist")

        # Draw a neutral background for the entire gallery drawlist
        dpg.draw_rectangle((0, 0), (dl_width, dl_height), color=(50, 50, 50, 255), fill=(50, 50, 50, 255), parent="gallery_drawlist")
        
        # Define the size of each gallery display slot
        GALLERY_SLOT_WIDTH = 120
        GALLERY_SLOT_HEIGHT = 120  # Enough space for creature and text

        # Define margins and text offsets within each slot
        SLOT_PADDING = 10  # Padding between slots
        VISUAL_MARGIN = 5  # Margin around creature visual within its rendering area
        TEXT_HEIGHT = 15  # Approximate height for one line of text
        TEXT_OFFSET_Y = 5  # Offset between visual and text and from text to text

        # Calculate columns based on current drawlist width
        cols = max(1, (dl_width - SLOT_PADDING) // (GALLERY_SLOT_WIDTH + SLOT_PADDING))
        
        for i, creature in enumerate(self.gallery_creatures):
            row = i // cols
            col = i % cols
            
            slot_x_tl = col * (GALLERY_SLOT_WIDTH + SLOT_PADDING) + SLOT_PADDING
            slot_y_tl = row * (GALLERY_SLOT_HEIGHT + SLOT_PADDING) + SLOT_PADDING
            
            # --- Draw the creature ---
            if creature.nodes:
                # Extract numpy data for static rendering
                static_node_data, static_muscle_data = self.rendering_simulator._extract_creature_data_to_numpy(creature)
                
                # Calculate bounds and scale
                node_xs = np.array([n.x for n in creature.nodes])
                node_ys = np.array([n.y for n in creature.nodes])
                node_ms = np.array([n.m for n in creature.nodes])
                
                min_x_sim = np.min(node_xs - node_ms/2)
                max_x_sim = np.max(node_xs + node_ms/2)
                min_y_sim = np.min(node_ys - node_ms/2)
                max_y_sim = np.max(node_ys + node_ms/2)
                
                creature_width_sim = max(1e-6, max_x_sim - min_x_sim)
                creature_height_sim = max(1e-6, max_y_sim - min_y_sim)
                
                # Available space for creature visualization
                vis_width = GALLERY_SLOT_WIDTH - 2*VISUAL_MARGIN
                vis_height = GALLERY_SLOT_HEIGHT - 2*VISUAL_MARGIN - TEXT_HEIGHT*2 - TEXT_OFFSET_Y
                
                # Calculate scale to fit
                scale_x = vis_width / creature_width_sim
                scale_y = vis_height / creature_height_sim
                display_scale = min(scale_x, scale_y, 20)  # Cap scale for visibility
                
                # Center position in slot
                center_x = slot_x_tl + GALLERY_SLOT_WIDTH/2
                center_y = slot_y_tl + VISUAL_MARGIN + vis_height/2
                
                # Calculate offset where sim (0,0) maps to in DPG coords
                sim_center_x = (min_x_sim + max_x_sim)/2
                sim_center_y = (min_y_sim + max_y_sim)/2
                render_offset_x = center_x - sim_center_x * display_scale
                render_offset_y = center_y - sim_center_y * display_scale
                
                # Draw the creature
                self._draw_creature(
                    static_node_data, static_muscle_data, 
                    "gallery_drawlist",
                    render_offset_x, render_offset_y,
                    display_scale,
                    draw_rects=False
                )

            # --- Draw slot border (highlight if selected) ---
            border_color = (100, 100, 100, 255)  # Default gray
            if self.selected_gallery_creature and self.selected_gallery_creature.id == creature.id:
                border_color = (0, 255, 0, 255)  # Green for selected
                
            dpg.draw_rectangle(
                (slot_x_tl, slot_y_tl),
                (slot_x_tl + GALLERY_SLOT_WIDTH, slot_y_tl + GALLERY_SLOT_HEIGHT),
                color=border_color,
                thickness=2,
                parent="gallery_drawlist"
            )
            
            # --- Draw creature info text ---
            text_y = slot_y_tl + GALLERY_SLOT_HEIGHT - TEXT_HEIGHT*2 - TEXT_OFFSET_Y
            dpg.draw_text(
                (slot_x_tl + VISUAL_MARGIN, text_y),
                f"ID: {creature.id}",
                color=(255, 255, 255, 255),
                size=12,
                parent="gallery_drawlist"
            )
            dpg.draw_text(
                (slot_x_tl + VISUAL_MARGIN, text_y + TEXT_HEIGHT + TEXT_OFFSET_Y),
                f"Fit: {creature.fitness:.2f}",
                color=(255, 255, 255, 255),
                size=12,
                parent="gallery_drawlist"
            )


if __name__ == "__main__":
    ui = SimulationUI()
    # The DPG event loop is started by ui.run(), which internally calls dpg.start_dearpygui()
    # and keeps the main thread alive. The simulation runs on a separate thread.
    dpg.set_exit_callback(lambda: ui.stop_simulation()) # Ensure simulation thread stops on exit
    #dpg.set_frame_callback(ui.update_ui) # Register a callback to update UI every frame
    ui.run()
