from dataclasses import dataclass, field

import math
import random
import time
from typing import Optional
import dearpygui.dearpygui as dpg

AIRFRICTION = 0.95
NAUSEAUNIT = 5
PRESSUREUNIT = 500.0/2.37
ENERGYUNIT = 20
GRAVITY = 0.005
FRICTION = 4
HAVEGROUND = True
hazelStairs = -1
energyDirection = 1
bigMutationChance = 0.06

@dataclass
class Rectangle:
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2
        
    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2
        
    @property
    def width(self) -> float:
        return self.x2 - self.x1
        
    @property
    def height(self) -> float:
        return self.y2 - self.y1
        
    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2

def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    return math.hypot(dx, dy)

rects: list[Rectangle] = []

@dataclass
class node:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    prevx: float = 0.0
    prevy: float = 0.0
    pvx: float = 0.0
    pvy: float = 0.0
    mass: float = 0.0
    value: float = 0.0
    valToBe: float = 0.0
    oper: int = 0
    axon1: int = 0
    axon2: int = 0
    safeInput: bool = False
    pressure: float = 0.0
    friction: float = 0.0
    nausea: float = 0.0
    radius: float = 0.0

    def __post_init__(self):
        self.radius = self.mass / 2 if self.mass else 0

    def applyForces(self):
        self.vx *= AIRFRICTION
        self.vy *= AIRFRICTION
        self.y += self.vy
        self.x += self.vx
        acc: float = distance(self.vx,self.vy,self.pvx,self.pvy)
        self.nausea = acc * acc * NAUSEAUNIT
        self.pvx = self.vx
        self.pvy = self.vy

    def applyGravity(self):
        self.vy += GRAVITY

    def Grounded(self, ground: float):
        pen = (self.y + self.radius) - ground
        if pen < 0: return
        self.pressure = pen * PRESSUREUNIT
        self.y = ground - self.radius
        self.vy = 0
        self.x -= self.vx * self.friction
        if self.vx > 0:
            self.vx -= self.friction * pen * FRICTION
            self.vx = max(self.vx, 0)
        else:
            self.vx += self.friction * pen * FRICTION
            self.vx = max(self.vx, 0)
    
    def Collision(self):
        self.pressure = 0
        if self.y + self.radius >= 0 and HAVEGROUND:
            self.Grounded(0)
        if self.y > self.prevy and hazelStairs >= 0:
            newbottom = self.y + self.radius
            prevBottom = self.prevy +  self.radius
            newLevel = math.ceil(newbottom / hazelStairs)
            prevLevel = math.ceil(prevBottom / hazelStairs)
            if newLevel > prevLevel:
                groundLevel = prevLevel * hazelStairs
                self.Grounded(groundLevel)
        for r in rects:
            flip = False
            if abs(self.x - (r.x1 + r.x2) / 2) <= (r.x2 - r.x1 + self.mass) / 2 and abs(self.y - (r.y1 + r.y2) / 2) <= (r.y2 - r.y1 + self.mass) / 2:
                if r.x2 > self.x >= r.x1 and r.y2 > self.y >= r.y1:
                    d1 = self.x - r.x1
                    d2 = r.x2 - self.x
                    d3 = self.y - r.y1
                    d4 = r.y2 - self.y
                    if d1 < d2 and d1 < d3 and d1 < d4:
                        self.px = r.x1
                        self.py = self.y
                    elif d2 < d3 and d2 < d4:
                        self.px = r.x2
                        self.py = self.y
                    elif d3 < d4:
                        self.px = self.x
                        self.py = r.y2
                else:
                    if self.x < r.x1:
                        self.px = r.x1
                    elif self.x < r.x2:
                        self.px = self.x
                    else:
                        self.px = r.x2
                    if self.y < r.y1:
                        self.py = r.y1
                    elif self.y < r.y2:
                        self.py = self.y
                    else:
                        self.py = r.y2
                dist = distance(self.x,self.y, self.px, self.py)
                wallangle = math.atan2(self.py-self.y, self.px-self.x)
                if flip:
                    wallangle += math.pi
                if dist < self.radius or flip:
                    dif = self.radius - dist
                    self.pressure += dif * PRESSUREUNIT
                    multi = self.radius / dist
                    if flip:
                        multi = -multi
                    self.x = (self.x - self.px) * multi + self.px
                    self.y = (self.y - self.py) * multi + self.py
                    veloAngle = math.atan2(self.vx, self.vy)
                    velMag = distance(0,0,self.vx, self.vy)
                    relAngle = veloAngle - wallangle
                    rely = math.sin(relAngle) * velMag * dif * FRICTION
                    self.vx = -math.sin(relAngle) * rely
                    self.vy = math.cos(relAngle) * rely
        self.prevy = self.y
        self.prevx = self.x

    def doMath(self, simulation_timer: float, nodes: list['node']):
        axon_value1 = nodes[self.axon1].value if self.axon1 < len(nodes) else 0
        axon_value2 = nodes[self.axon2].value if self.axon2 < len(nodes) else 0
        
        if self.oper == 0:  # constant
            pass  # value remains as is
        elif self.oper == 1:  # time
            self.valToBe = simulation_timer / 60.0
        elif self.oper == 2:  # x-coordinate
            self.valToBe = self.x * 0.2
        elif self.oper == 3:  # y-coordinate
            self.valToBe = -self.y * 0.2
        elif self.oper == 4:  # plus
            self.valToBe = axon_value1 + axon_value2
        elif self.oper == 5:  # minus
            self.valToBe = axon_value1 - axon_value2
        elif self.oper == 6:  # times
            self.valToBe = axon_value1 * axon_value2
        elif self.oper == 7:  # divide
            self.valToBe = axon_value1 / axon_value2 if axon_value2 != 0 else 0
        elif self.oper == 8:  # modulus
            self.valToBe = axon_value1 % axon_value2 if axon_value2 != 0 else 0
        elif self.oper == 9:  # sin
            self.valToBe = math.sin(axon_value1)
        elif self.oper == 10:  # sigmoid
            self.valToBe = 1 / (1 + math.exp(-axon_value1))
        elif self.oper == 11:  # pressure
            self.valToBe = self.pressure

    def realize_math_values(self) -> None:
        self.value = self.valToBe

    
    def copy(self):
        return node(
            x=self.x,
            y=self.y,
            vx=0,
            vy=0,
            mass=self.mass,
            friction=self.friction,
            value=self.value,
            oper=self.oper,
            axon1=self.axon1,
            axon2=self.axon2
        )

    def modify(self, mutability: float, node_count: int, big_mutation_chance: float = 0.1, operation_count: int = 12) -> 'node':
        def r():
            return random.uniform(-1, 1)
            
        new_x = self.x + r() * 0.5 * mutability
        new_y = self.y + r() * 0.5 * mutability
        new_mass = self.mass + r() * 0.1 * mutability
        new_mass = max(min(new_mass, 0.5), 0.3)  # Clamp between 0.3 and 0.5
        new_mass = 0.4  # This line overrides the previous calculation - kept as in original
        
        new_value = self.value * (1 + r() * 0.2 * mutability)
        new_oper = self.oper
        new_axon1 = self.axon1
        new_axon2 = self.axon2
        
        # Potential mutations
        if random.random() < big_mutation_chance * mutability:
            new_oper = random.randint(0, operation_count - 1)
        if random.random() < big_mutation_chance * mutability:
            new_axon1 = random.randint(0, node_count - 1)
        if random.random() < big_mutation_chance * mutability:
            new_axon2 = random.randint(0, node_count - 1)
        
        # Special cases for certain operations
        if new_oper == 1:  # time
            new_value = 0
        elif new_oper == 2:  # x-coordinate
            new_value = new_x * 0.2
        elif new_oper == 3:  # y-coordinate
            new_value = -new_y * 0.2
        
        # Create and return the new node
        return node(
            x=new_x,
            y=new_y,
            vx=0,
            vy=0,
            mass=new_mass,
            friction=max(min(self.friction + r() * 0.1 * mutability, 1), 0),
            value=new_value,
            oper=new_oper,
            axon1=new_axon1,
            axon2=new_axon2
        )
    
@dataclass
class Muscle:
    axon: int = 0
    c1: int = 0
    c2: int = 0
    length: float = 0.0
    rigidity: float = 0.0
    previousTarget: float = 0.0
    energy: float = 0.0
    force: float = 0.0


    def __post_init__(self):
        self.previousTarget = self.length
    
    def to_muscle_usable(self, value: float):
        return (math.tanh(value) + 1) / 2
    
    def apply_force(self, nodes: list[node]) -> None:
        target = self.previous_target
        if energyDirection == 1 or self.energy >= 0.0001:
            if 0 <= self.axon < len(nodes):
                target = self.length * self.to_muscle_usable(nodes[self.axon].value)
            else:
                target = self.length
        
        node1 = nodes[self.c1]
        node2 = nodes[self.c2]
        
        dist = distance(node1.x, node1.y, node2.x, node2.y)
        angle = math.atan2(node1.y - node2.y, node1.x - node2.x)
        
        self.force = min(max(1 - (dist / target), -0.4), 0.4)
        
        fx = math.cos(angle) * self.force * self.rigidity / node1.mass
        fy = math.sin(angle) * self.force * self.rigidity / node1.mass
        
        node1.vx += fx
        node1.vy += fy
        node2.vx -= fx
        node2.vy -= fy
        
        energy_change = energyDirection * abs(self.previous_target - target) * self.rigidity * ENERGYUNIT
        self.energy = max(self.energy + energy_change, 0)
        self.previous_target = target

    def copy(self) -> 'Muscle':
        return Muscle(
            axon=self.axon,
            c1=self.c1,
            c2=self.c2,
            length=self.length,
            rigidity=self.rigidity
        )

    def modify(self, node_count: int, mutability: float) -> 'Muscle':
        def r():
            return random.uniform(-1, 1)
        
        new_c1 = self.c1
        new_c2 = self.c2
        new_axon = self.axon
        
        if random.random() < bigMutationChance * mutability:
            new_c1 = random.randint(0, node_count - 1)
        if random.random() < bigMutationChance * mutability:
            new_c2 = random.randint(0, node_count - 1)
        if random.random() < bigMutationChance * mutability:
            new_axon = random.randint(0, node_count - 1)
        
        new_rigidity = min(max(self.rigidity * (1 + r() * 0.9 * mutability), 0.01), 0.08)
        new_length = min(max(self.length + r() * mutability, 0.4), 1.25)
        
        return Muscle(
            axon=new_axon,
            c1=new_c1,
            c2=new_c2,
            length=new_length,
            rigidity=new_rigidity
        )
    
    
@dataclass
class Creature:
    id: int
    nodes: list[node] = field(default_factory=list)
    muscles: list[Muscle] = field(default_factory=list)
    d: float = 0.0
    alive: bool = True
    creature_timer: float = 0.0
    mutability: float = 1.0

    def r(self) -> float:
        return random.uniform(-1, 1)

    def modified(self, new_id: int) -> 'Creature':
        modified_creature = Creature(
            id=new_id,
            nodes=[],
            muscles=[],
            d=0,
            alive=True,
            creature_timer=self.creature_timer + self.r() * 16 * self.mutability,
            mutability=min(self.mutability * random.uniform(0.8, 1.25), 2)
        )
        
        # Modify nodes
        for n in self.nodes:
            modified_creature.nodes.append(n.modify(self.mutability, len(self.nodes)))
        
        # Modify muscles
        for m in self.muscles:
            modified_creature.muscles.append(m.modify(len(self.nodes), self.mutability))
        
        # Potential mutations
        if random.random() < bigMutationChance * self.mutability or len(self.nodes) <= 2:
            modified_creature.add_random_node()
        
        if random.random() < bigMutationChance * self.mutability:
            modified_creature.add_random_muscle(-1, -1)
        
        if random.random() < bigMutationChance * self.mutability and len(modified_creature.nodes) >= 4:
            modified_creature.remove_random_node()
        
        if random.random() < bigMutationChance * self.mutability and len(modified_creature.muscles) >= 2:
            modified_creature.remove_random_muscle()
        
        modified_creature.check_for_overlap()
        modified_creature.check_for_lone_nodes()
        modified_creature.check_for_bad_axons()
        
        return modified_creature

    def check_for_overlap(self) -> None:
        bad_indices = []
        for i in range(len(self.muscles)):
            for j in range(i + 1, len(self.muscles)):
                m1 = self.muscles[i]
                m2 = self.muscles[j]
                
                if (m1.c1 == m2.c1 and m1.c2 == m2.c2) or \
                   (m1.c1 == m2.c2 and m1.c2 == m2.c1) or \
                   (m1.c1 == m1.c2):
                    bad_indices.append(i)
        
        # Remove in reverse order to maintain indices
        for i in sorted(bad_indices, reverse=True):
            if i < len(self.muscles):
                del self.muscles[i]

    def check_for_lone_nodes(self) -> None:
        if len(self.nodes) >= 3:
            for i in range(len(self.nodes)):
                connections = 0
                connected_to = -1
                
                for j in range(len(self.muscles)):
                    m = self.muscles[j]
                    if m.c1 == i or m.c2 == i:
                        connections += 1
                        connected_to = j
                
                if connections <= 1:
                    new_connection_node = random.randint(0, len(self.nodes) - 1)
                    while new_connection_node == i or new_connection_node == connected_to:
                        new_connection_node = random.randint(0, len(self.nodes) - 1)
                    self.add_random_muscle(i, new_connection_node)

    def check_for_bad_axons(self) -> None:
        operation_requirements = {
            0: 0,   # Constant
            1: 0,   # Time
            2: 0,   # X-coord
            3: 0,   # Y-coord
            4: 2,   # Add
            5: 2,   # Subtract
            6: 2,   # Multiply
            7: 2,   # Divide
            8: 2,   # Modulus
            9: 1,   # Sin
            10: 1,  # Sigmoid
            11: 0   # Pressure
        }
        
        # Check node axons
        for n in self.nodes:
            if n.axon1 >= len(self.nodes):
                n.axon1 = random.randint(0, len(self.nodes) - 1)
            if n.axon2 >= len(self.nodes):
                n.axon2 = random.randint(0, len(self.nodes) - 1)
        
        # Check muscle axons
        for m in self.muscles:
            if m.axon >= len(self.nodes):
                m.axon = self.get_new_muscle_axon(len(self.nodes))
        
        # Mark safe inputs (operations that don't depend on other nodes)
        for n in self.nodes:
            n.safeInput = (operation_requirements.get(n.oper, 0) == 0)
        
        # Propagate safe inputs
        iterations = 0
        did_something = True
        
        while iterations < 1000 and did_something:
            did_something = False
            for n in self.nodes:
                if not n.safeInput:
                    req = operation_requirements.get(n.oper, 0)
                    if req == 1 and self.nodes[n.axon1].safeInput:
                        n.safeInput = True
                        did_something = True
                    elif req == 2 and self.nodes[n.axon1].safeInput and self.nodes[n.axon2].safeInput:
                        n.safeInput = True
                        did_something = True
            iterations += 1
        
        # Cleanse unsafe nodes
        for n in self.nodes:
            if not n.safeInput:
                n.oper = 0  # Constant operation
                n.value = random.random()

    def get_new_muscle_axon(self, node_count: int) -> int:
        if random.random() < 0.5:
            return random.randint(0, node_count - 1)
        else:
            return -1

    def add_random_node(self) -> None:
        parent_node = random.randint(0, len(self.nodes) - 1)
        ang1 = random.uniform(0, 2 * math.pi)
        distance = math.sqrt(random.random())
        
        x = self.nodes[parent_node].x + math.cos(ang1) * 0.5 * distance
        y = self.nodes[parent_node].y + math.sin(ang1) * 0.5 * distance
        
        new_node_count = len(self.nodes) + 1
        
        self.nodes.append(node(
            x=x,
            y=y,
            vx=0,
            vy=0,
            mass=0.4,
            value=random.random(),
            oper=random.randint(0, 11),  # Assuming 12 operations (0-11)
            axon1=random.randint(0, new_node_count - 1),
            axon2=random.randint(0, new_node_count - 1)
        ))
        
        # Find closest node
        next_closest_node = 0
        record = float('inf')
        
        for i in range(len(self.nodes) - 1):
            if i != parent_node:
                dx = self.nodes[i].x - x
                dy = self.nodes[i].y - y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < record:
                    record = dist
                    next_closest_node = i
        
        self.add_random_muscle(parent_node, len(self.nodes) - 1)
        self.add_random_muscle(next_closest_node, len(self.nodes) - 1)

    def add_random_muscle(self, tc1: int, tc2: int) -> None:
        axon = self.get_new_muscle_axon(len(self.nodes))
        
        if tc1 == -1:
            tc1 = random.randint(0, len(self.nodes) - 1)
            tc2 = tc1
            while tc2 == tc1 and len(self.nodes) >= 2:
                tc2 = random.randint(0, len(self.nodes) - 1)
        
        length = random.uniform(0.5, 1.5)
        if tc1 != -1:
            n1 = self.nodes[tc1]
            n2 = self.nodes[tc2]
            length = math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        
        self.muscles.append(Muscle(
            axon=axon,
            c1=tc1,
            c2=tc2,
            length=length,
            rigidity=random.uniform(0.02, 0.08)
        ))

    def remove_random_node(self) -> None:
        if not self.nodes:
            return
            
        choice = random.randint(0, len(self.nodes) - 1)
        del self.nodes[choice]
        
        # Remove muscles connected to this node and adjust indices
        i = 0
        while i < len(self.muscles):
            m = self.muscles[i]
            if m.c1 == choice or m.c2 == choice:
                del self.muscles[i]
            else:
                if m.c1 > choice:
                    m.c1 -= 1
                if m.c2 > choice:
                    m.c2 -= 1
                i += 1

    def remove_random_muscle(self) -> None:
        if self.muscles:
            choice = random.randint(0, len(self.muscles) - 1)
            del self.muscles[choice]

    def copy(self, new_id: Optional[int] = None) -> 'Creature':
        if new_id is None:
            new_id = self.id
            
        return Creature(
            id=new_id,
            nodes=[n.copy() for n in self.nodes],
            muscles=[m.copy() for m in self.muscles],
            d=self.d,
            alive=self.alive,
            creature_timer=self.creature_timer,
            mutability=self.mutability
        )

    def get_operation_name(self, oper: int) -> str:
        operations = {
            0: "Constant",
            1: "Time",
            2: "X-coord",
            3: "Y-coord",
            4: "Add",
            5: "Subtract",
            6: "Multiply",
            7: "Divide",
            8: "Modulus",
            9: "Sin",
            10: "Sigmoid",
            11: "Pressure"
        }
        return operations.get(oper, f"Unknown({oper})")

class Simulation:
    def __init__(
        self,
        max_nodes: int = 100,
        min_nodes: int = 0,
        has_stairs: bool = False,
        has_obstacles: bool = False,
        ground_friction: float = 4.0,
        air_friction: float = 0.95,
        gravity: float = 0.005,
        energy_direction: int = 1
    ):
        # Simulation parameters
        self.max_nodes = max(max_nodes, min_nodes)
        self.min_nodes = max(min_nodes, 0)
        self.has_stairs = has_stairs
        self.has_obstacles = has_obstacles
        self.ground_friction = ground_friction
        self.air_friction = air_friction
        self.gravity = gravity
        self.energy_direction = energy_direction
        
        # Simulation state
        self.creatures: list[Creature] = []
        self.rectangles: list[Rectangle] = []
        self.simulation_time = 0.0
        self.paused = False
        
        # Environment setup
        self.setup_environment()
        
    def setup_environment(self) -> None:
        global HAVEGROUND, hazelStairs, FRICTION, AIRFRICTION, energyDirection
        
        # Set global parameters
        HAVEGROUND = True
        hazelStairs = 0.5 if self.has_stairs else -1
        FRICTION = self.ground_friction
        AIRFRICTION = self.air_friction
        energyDirection = self.energy_direction
        
        # Clear existing rectangles
        self.rectangles.clear()
        
        # Add random obstacles if enabled
        if self.has_obstacles:
            for _ in range(random.randint(2, 5)):
                x = random.uniform(-3, 3)
                width = random.uniform(0.5, 2)
                height = random.uniform(0.2, 1)
                y = random.uniform(-1, -0.2)
                
                self.rectangles.append(Rectangle(
                    x1=x,
                    y1=y,
                    x2=x + width,
                    y2=y + height
                ))
        
        # Update global rectangles list
        global rects
        rects = self.rectangles
    
    def create_random_creature(self, creature_id: int) -> Creature:
        num_nodes = random.randint(max(self.min_nodes, 2), self.max_nodes)
        creature = Creature(id=creature_id)
        
        # Create initial nodes
        for i in range(num_nodes):
            creature.nodes.append(node(
                x=random.uniform(-1, 1),
                y=random.uniform(-1, 1),
                vx=0,
                vy=0,
                mass=0.4,
                value=random.random(),
                oper=random.randint(0, 11),
                axon1=random.randint(0, i),  # Connect to existing nodes
                axon2=random.randint(0, i),
                friction=random.uniform(0, 1)
            ))
        
        # Create muscles to connect nodes
        for i in range(num_nodes * 2):  # Roughly 2 muscles per node
            if len(creature.nodes) >= 2:
                c1 = random.randint(0, len(creature.nodes) - 1)
                c2 = c1
                while c2 == c1:
                    c2 = random.randint(0, len(creature.nodes) - 1)
                
                n1 = creature.nodes[c1]
                n2 = creature.nodes[c2]
                length = math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
                
                creature.muscles.append(Muscle(
                    axon=random.randint(-1, len(creature.nodes) - 1),
                    c1=c1,
                    c2=c2,
                    length=length,
                    rigidity=random.uniform(0.02, 0.08)
                ))
        
        # Validate the creature
        creature.check_for_overlap()
        creature.check_for_lone_nodes()
        creature.check_for_bad_axons()
        
        return creature
    
    def add_random_creatures(self, count: int) -> None:
        start_id = len(self.creatures)
        for i in range(count):
            self.creatures.append(self.create_random_creature(start_id + i))
    
    def update(self, delta_time: float) -> None:
        if self.paused:
            return
            
        self.simulation_time += delta_time
        
        # Update all creatures
        for creature in self.creatures:
            if not creature.alive:
                continue
                
            # Update nodes
            for node in creature.nodes:
                node.applyGravity()
                node.applyForces()
                node.Collision()
                node.doMath(self.simulation_time, creature.nodes)
            
            # Realize math values after all calculations are done
            for node in creature.nodes:
                node.realize_math_values()
            
            # Apply muscle forces
            for muscle in creature.muscles:
                muscle.apply_force(creature.nodes)
            
            # Update creature timer and check for death
            creature.creature_timer += delta_time
            if creature.creature_timer > 60:  # 1 minute lifespan
                creature.alive = False
    
    def reset(self) -> None:
        self.creatures.clear()
        self.simulation_time = 0.0
        self.setup_environment()
    
    def get_alive_creatures(self) -> list[Creature]:
        return [c for c in self.creatures if c.alive]
    
    def get_dead_creatures(self) -> list[Creature]:
        return [c for c in self.creatures if not c.alive]
    
    def get_fitness_scores(self) -> list[float]:
        return [c.d for c in self.creatures]
    
class SimulationVisualizer:
    def __init__(self):
        self.simulation = Simulation(
            max_nodes=20,
            min_nodes=3,
            has_stairs=False,
            has_obstacles=False,
            ground_friction=4.0,
            air_friction=0.95,
            gravity=0.005,
            energy_direction=1
        )
        
        self.generation = 0
        self.best_creatures = []
        self.worst_creatures = []
        self.median_creature = None
        self.viewing_creature = None
        self.viewing_generation = 0
        self.is_running = False
        self.show_full_generation = False
        self.last_update_time = time.time()
        
        # Create GUI
        self.setup_gui()
        
    def setup_gui(self):
        dpg.create_context()
        dpg.create_viewport(title='Creature Simulation', width=1200, height=800)
        
        with dpg.window(label="Main Window", tag="main_window"):
            # Simulation controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Create Starting Creatures", callback=self.create_starting_creatures)
                dpg.add_button(label="Step Generation", callback=self.step_generation)
                dpg.add_button(label="Quick Generation", callback=lambda: self.run_generations(1, quick=True))
                dpg.add_button(label="Run Generations", callback=self.toggle_run_generations)
            
            # Display area
            with dpg.group(horizontal=True):
                # Simulation view
                with dpg.child_window(width=800, height=600, tag="simulation_view"):
                    # Drawlist will be created dynamically
                    pass
                
                # Information panel
                with dpg.child_window(width=380, height=600):
                    dpg.add_text("Generation: 0", tag="generation_text")
                    dpg.add_text("Population: 0", tag="population_text")
                    dpg.add_separator()
                    
                    with dpg.collapsing_header(label="Top Creatures", default_open=True):
                        dpg.add_text("Best:", tag="best_text")
                        dpg.add_text("Median:", tag="median_text")
                        dpg.add_text("Worst:", tag="worst_text")
                    
                    with dpg.collapsing_header(label="View Creature", default_open=True):
                        dpg.add_text("Viewing: None", tag="viewing_text")
                        dpg.add_button(label="View Best", callback=lambda: self.view_creature(0))
                        dpg.add_button(label="View Median", callback=lambda: self.view_creature(1))
                        dpg.add_button(label="View Worst", callback=lambda: self.view_creature(2))
                    
                    with dpg.collapsing_header(label="Simulation Settings", default_open=True):
                        dpg.add_checkbox(label="Show Full Generation", tag="show_full_checkbox", default_value=False)
                        
                        # Make sure min <= max for all sliders
                        dpg.add_slider_int(
                            label="Population Size", 
                            tag="population_size", 
                            default_value=20, 
                            min_value=5, 
                            max_value=100
                        )
                        dpg.add_slider_int(
                            label="Max Nodes", 
                            tag="max_nodes", 
                            default_value=20, 
                            min_value=3, 
                            max_value=50
                        )
                        dpg.add_slider_int(
                            label="Min Nodes", 
                            tag="min_nodes", 
                            default_value=3, 
                            min_value=2, 
                            max_value=10
                        )
                        dpg.add_checkbox(label="Has Stairs", tag="has_stairs", default_value=False)
                        dpg.add_checkbox(label="Has Obstacles", tag="has_obstacles", default_value=False)
                        dpg.add_slider_float(
                            label="Ground Friction", 
                            tag="ground_friction", 
                            default_value=4.0, 
                            min_value=0.1, 
                            max_value=10.0
                        )
                        dpg.add_slider_float(
                            label="Air Friction", 
                            tag="air_friction", 
                            default_value=0.95, 
                            min_value=0.8, 
                            max_value=1.0
                        )
                        dpg.add_slider_float(
                            label="Gravity", 
                            tag="gravity", 
                            default_value=0.005, 
                            min_value=0.001, 
                            max_value=0.02
                        )
                        dpg.add_radio_button(
                            label="Energy Direction", 
                            items=["Gain (1)", "Lose (-1)"], 
                            tag="energy_direction", 
                            default_value="Gain (1)"
                        )

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        
        # Create initial drawlist
        with dpg.drawlist(parent="simulation_view", width=800, height=600, tag="drawing_canvas"):
            pass

    def create_starting_creatures(self):
        """Create a new set of starting creatures based on current settings"""
        self.update_simulation_settings()
        population_size = dpg.get_value("population_size")
        self.simulation.reset()
        self.simulation.add_random_creatures(population_size)
        self.generation = 0
        dpg.set_value("generation_text", f"Generation: {self.generation}")
        dpg.set_value("population_text", f"Population: {len(self.simulation.creatures)}")
        self.draw_simulation()
    
    def update_simulation_settings(self):
        self.simulation.max_nodes = dpg.get_value("max_nodes")
        self.simulation.min_nodes = dpg.get_value("min_nodes")
        self.simulation.has_stairs = dpg.get_value("has_stairs")
        self.simulation.has_obstacles = dpg.get_value("has_obstacles")
        self.simulation.ground_friction = dpg.get_value("ground_friction")
        self.simulation.air_friction = dpg.get_value("air_friction")
        self.simulation.gravity = dpg.get_value("gravity")
        self.simulation.energy_direction = 1 if dpg.get_value("energy_direction") == "Gain (1)" else -1
        self.simulation.setup_environment()
    
    def step_generation(self):
        self.show_full_generation = True
        self.run_generations(1, quick=False)
    
    def toggle_run_generations(self):
        self.is_running = not self.is_running
        self.show_full_generation = dpg.get_value("show_full_checkbox")
    
    def run_generations(self, count=1, quick=False):
        self.update_simulation_settings()
        
        for _ in range(count):
            # Create new generation if we have previous creatures
            if self.simulation.creatures:
                self.evaluate_creatures()
                self.create_next_generation()
            else:
                # Create initial random population
                population_size = dpg.get_value("population_size")
                self.simulation.add_random_creatures(population_size)
            
            self.generation += 1
            dpg.set_value("generation_text", f"Generation: {self.generation}")
            dpg.set_value("population_text", f"Population: {len(self.simulation.creatures)}")
            
            # Run the simulation for this generation
            sim_time = 0
            while sim_time < 10.0:  # Run for 10 seconds
                self.simulation.update(0.016)  # ~60fps
                sim_time += 0.016
                
                if not quick:
                    self.draw_simulation()
                    dpg.render_dearpygui_frame()
                    time.sleep(0.016)
            
            # Evaluate creatures at the end of the generation
            self.evaluate_creatures()
            
            # Update display with best/worst creatures
            self.update_creature_display()
    
    def evaluate_creatures(self):
        # Sort creatures by distance traveled (fitness)
        self.simulation.creatures.sort(key=lambda c: c.d, reverse=True)
        
        population_size = len(self.simulation.creatures)
        if population_size == 0:
            return
        
        # Store top 3, worst 3, and median creature
        self.best_creatures = self.simulation.creatures[:3]
        self.worst_creatures = self.simulation.creatures[-3:]
        self.median_creature = self.simulation.creatures[population_size // 2]
        
        # Update the viewing creature if not set
        if self.viewing_creature is None:
            self.view_creature(0)
    
    def create_next_generation(self):
        population_size = dpg.get_value("population_size")
        new_creatures = []
        
        # Keep top 10% of creatures unchanged
        keep_count = max(1, int(population_size * 0.1))
        for i in range(keep_count):
            if i < len(self.simulation.creatures):
                new_creatures.append(self.simulation.creatures[i].copy(len(new_creatures)))
        
        # Create modified versions of top 50% of creatures
        modify_count = population_size - keep_count
        for i in range(modify_count):
            if i < len(self.simulation.creatures) // 2:
                parent = self.simulation.creatures[i]
                new_creatures.append(parent.modified(len(new_creatures)))
            else:
                # Add some random creatures to maintain diversity
                new_creatures.append(self.simulation.create_random_creature(len(new_creatures)))
        
        self.simulation.creatures = new_creatures
    
    def update_creature_display(self):
        # Update best/worst/median text
        if self.best_creatures:
            best_text = "Best:\n"
            for i, c in enumerate(self.best_creatures):
                best_text += f"  {i+1}. Distance: {c.d:.2f}, Nodes: {len(c.nodes)}, Muscles: {len(c.muscles)}\n"
            dpg.set_value("best_text", best_text)
        
        if self.median_creature:
            dpg.set_value("median_text", f"Median:\n  Distance: {self.median_creature.d:.2f}, Nodes: {len(self.median_creature.nodes)}, Muscles: {len(self.median_creature.muscles)}")
        
        if self.worst_creatures:
            worst_text = "Worst:\n"
            for i, c in enumerate(reversed(self.worst_creatures)):
                worst_text += f"  {i+1}. Distance: {c.d:.2f}, Nodes: {len(c.nodes)}, Muscles: {len(c.muscles)}\n"
            dpg.set_value("worst_text", worst_text)
    
    def view_creature(self, creature_type):
        # 0 = best, 1 = median, 2 = worst
        if creature_type == 0 and self.best_creatures:
            self.viewing_creature = self.best_creatures[0]
            self.viewing_generation = self.generation
        elif creature_type == 1 and self.median_creature:
            self.viewing_creature = self.median_creature
            self.viewing_generation = self.generation
        elif creature_type == 2 and self.worst_creatures:
            self.viewing_creature = self.worst_creatures[-1]
            self.viewing_generation = self.generation
        
        if self.viewing_creature:
            dpg.set_value("viewing_text", f"Viewing: {'Best' if creature_type == 0 else 'Median' if creature_type == 1 else 'Worst'} from Gen {self.viewing_generation}\nDistance: {self.viewing_creature.d:.2f}\nNodes: {len(self.viewing_creature.nodes)}\nMuscles: {len(self.viewing_creature.muscles)}")
    
    def update_simulation(self, delta_time):
        if not self.is_running or self.show_full_generation:
            return
        
        self.simulation.update(delta_time)
    
    def draw_simulation(self):
        # Clear previous drawings
        dpg.delete_item("simulation_view", children_only=True)
        
        # Create new drawlist
        with dpg.drawlist(parent="simulation_view", width=800, height=600, tag="drawing_canvas"):
            # Draw ground
            dpg.draw_line([-10, 0], [10, 0], color=[100, 100, 100, 255])
            
            # Draw stairs if enabled
            if self.simulation.has_stairs:
                for i in range(1, 10):
                    y = i * 0.5
                    x_start = max(-10, i * 0.5 - 10)
                    dpg.draw_line([x_start, y], [10, y], color=[150, 150, 150, 255])
            
            # Draw obstacles
            for rect in self.simulation.rectangles:
                dpg.draw_rectangle(
                    [rect.x1, rect.y1], [rect.x2, rect.y2],
                    fill=[80, 80, 80, 200],
                    color=[120, 120, 120, 255]
                )
            
            # Determine which creature to draw
            creatures_to_draw = []
            if self.viewing_creature and not self.is_running:
                creatures_to_draw = [self.viewing_creature]
            else:
                creatures_to_draw = self.simulation.get_alive_creatures()
            
            # Draw creatures
            for creature in creatures_to_draw:
                # Draw nodes
                for node in creature.nodes:
                    color = [255, 0, 0, 255]  # Red for normal nodes
                    
                    # Highlight special nodes
                    if node.oper == 1:  # Time
                        color = [0, 255, 255, 255]  # Cyan
                    elif node.oper == 2:  # X-coord
                        color = [255, 255, 0, 255]  # Yellow
                    elif node.oper == 3:  # Y-coord
                        color = [0, 255, 0, 255]  # Green
                    elif node.oper == 11:  # Pressure
                        color = [255, 0, 255, 255]  # Magenta
                    
                    dpg.draw_circle(
                        [node.x, node.y], node.radius,
                        fill=color,
                        color=[0, 0, 0, 255]
                    )
                
                # Draw muscles
                for muscle in creature.muscles:
                    if muscle.c1 >= len(creature.nodes) or muscle.c2 >= len(creature.nodes):
                        continue
                    
                    node1 = creature.nodes[muscle.c1]
                    node2 = creature.nodes[muscle.c2]
                    
                    # Calculate color based on muscle activation
                    target_length = muscle.length * ((math.tanh(muscle.previousTarget) + 1) / 2 if muscle.axon != -1 else 1)
                    current_length = distance(node1.x, node1.y, node2.x, node2.y)
                    ratio = current_length / target_length if target_length != 0 else 1
                    
                    # Color gradient from blue (relaxed) to red (contracted)
                    r = min(255, int(255 * (2 - 2 * ratio)))
                    b = min(255, int(255 * (2 * ratio - 1)))
                    g = min(255, int(255 * (1 - abs(1 - ratio))))
                    
                    dpg.draw_line(
                        [node1.x, node1.y], [node2.x, node2.y],
                        color=[r, g, b, 255],
                        thickness=2
                    )

    def run(self):
        """Main simulation loop"""
        while dpg.is_dearpygui_running():
            time.sleep(0.1)
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

if __name__ == "__main__":
    visualizer = SimulationVisualizer()
    visualizer.run()