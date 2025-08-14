from dataclasses import dataclass, field
import random
import math
from part import Part, Limb, LimbType

@dataclass
class Organism:
    x: float
    y: float
    z: float
    oid: int
    direction = random.uniform(0, 2 * math.pi)  # in radians
    energy = 100
    size = 5
    color = (
        random.randint(100, 255),
        random.randint(100, 255),
        random.randint(100, 255),
        255
    )
    memory: dict[tuple[float,float], any] = field(default_factory=dict)
    memory_chance = random.uniform(0.01, 0.1)
    memory_decay = 0.99
    parts: list[Part] = field(default_factory=list)
    
    def __post_init__(self):
        self.path = [(self.x, self.y)]  # Store path for top view
        # Initialize with some basic parts if empty
        if not self.parts:
            self.initialize_default_parts()
        
    
    @property
    def is_plant(self) -> bool:
        if any(isinstance(part, Limb) and part.limb_type == LimbType.ROOT for part in self.parts):
            return True
        else: return False

    
    def initialize_default_parts(self):
        """Initialize with some basic limbs"""
        if random.random() < 0.5:
            # Animal limbs
            self.parts.append(Limb("legs", 1.5, 100, 0.1, 2, 4, LimbType.LEG, 15, 1.0, 0.7, False, 3))
            self.parts.append(Limb("legs", 1.5, 100, 0.1, 2, 4, LimbType.LEG, 15, 1.0, 0.7, False, 3))
            #Limb("fins", 2.0, 80, 0.2, 1.5, 5, LimbType.FIN, 10, 1.2, 0.5, False, 2),
            #Limb("wings", 3.0, 60, 0.3, 3, 2, 5, LimbType.WING, 2.0, 0.3, False, 1),

        else:
            self.parts.append(Limb("roots", 2.0, 150, 0.5, 3, 10, 0, LimbType.ROOT, 0, 0, False, 5))
            self.parts.append(Limb("branches", 1.8, 120, 0.4, 2.5, 6, LimbType.BRANCH, 0, 0, 0, False, 3))

    
    @property
    def speed(self):
        """Calculate speed based on limbs and other factors"""
        base_speed = 0.0
        limb_contributions = 0
        
        # Sum contributions from all limbs
        for part in self.parts:
            if isinstance(part, Limb):
                limb_contributions += part.calculate_movement_contribution()
        
        # Apply diminishing returns to limb contributions
        effective_contributions = math.log(1 + limb_contributions)
        
        # Size affects speed (larger organisms are generally slower)
        size_factor = 1 / (self.size ** 0.33)
        
        return (base_speed + effective_contributions) * size_factor
		
    def move(self, grid_size):
        if self.is_plant:
            return
        dx = math.cos(self.direction) * self.speed
        dy = math.sin(self.direction) * self.speed
        
        # Update position with boundary checking
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Ensure organisms stay within grid boundaries for x and y
        if new_x < 0:
            new_x = 0
            self.direction = math.pi - self.direction
        elif new_x >= grid_size:
            new_x = grid_size - 0.1
            self.direction = math.pi - self.direction
            
        if new_y < 0:
            new_y = 0
            self.direction = -self.direction
        elif new_y >= grid_size:
            new_y = grid_size - 0.1
            self.direction = -self.direction
            
        # Update positions
        self.x = new_x
        self.y = new_y
        self.path.append((self.x, self.y))
        
        # Randomly remember this position based on memory chance
        if random.random() < self.memory_chance:
            self.remember_position((int(self.x), int(self.y)))
        
        # Random direction changes
        if random.random() < 0.05:
            self.direction += random.uniform(-0.5, 0.5)
            
        # Random z-level changes occasionally with boundary checking
        if random.random() < 0.01:
            new_z = self.z + random.randint(-1, 1)
            self.z = max(0, min(grid_size - 1, new_z))
    
    def remember_position(self, position: tuple[float,float], weight=1):
        """Store a position in memory with a given weight"""
        if position in self.memory:
            self.memory[position] += weight
        else:
            self.memory[position] = weight
    
    def get_memory_weight(self, position):
        """Get the weight of a position in memory"""
        return self.memory.get(position, 0)
    
    def get_triangle_points(self, cell_size, offset_x, offset_y):
        # Center of the cell
        center_x = offset_x + self.x * cell_size + cell_size / 2
        center_y = offset_y + self.y * cell_size + cell_size / 2
        
        # Calculate triangle points (pointing in direction of movement)
        front_x = center_x + math.cos(self.direction) * self.size
        front_y = center_y + math.sin(self.direction) * self.size
        
        back_left_x = center_x + math.cos(self.direction + math.pi * 0.75) * self.size / 2
        back_left_y = center_y + math.sin(self.direction + math.pi * 0.75) * self.size / 2
        
        back_right_x = center_x + math.cos(self.direction - math.pi * 0.75) * self.size / 2
        back_right_y = center_y + math.sin(self.direction - math.pi * 0.75) * self.size / 2
        
        return [front_x, front_y, back_left_x, back_left_y, back_right_x, back_right_y]