import hashlib
import dearpygui.dearpygui as dpg
import random
import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import torch
import numpy as np

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@dataclass
class Species:
    class MovementType(Enum):
        TERRESTRIAL = 0
        AVIAN = 1
        AQUATIC = 2
    # Core attributes
    name: str = ""
    name_parts: List[str] = field(default_factory=list)
    color: tuple = (0, 0, 0)
    max_energy: float = 10.0
    energy_gain: float = 1.0
    energy_consumption: float = 0.1
    mature_age: int = 50
    reproduction_cost: float = 5.0
    offspring_energy: float = 3.0
    reproduction_frequency: int = 10
    temperature_sensitivity: float = 0.5
    light_sensitivity: float = 0.5
    cold_resistance: float = 0.5
    
    # Dormancy and Seed attributes
    dormancy_threshold: float = 0.3
    seed_lifespan: int = 200
    seed_maturity_age: int = 10
    seed_spread_radius: int = 3
    seed_germination_chance: float = 0.1
    seed_production_frequency: int = 30
    seed_production_chance: float = 0.5

    # Movement and type-related
    can_move: bool = False
    move_speed: int = 0
    move_energy_cost: float = 0.0
    has_roots: bool = False
    root_energy: float = 0.0
    
    # Lifecycle
    max_age: int = 0
    sleep_ratio: float = 0.0
    diurnal: bool = True
    
    # Physical characteristics
    height: float = 1.0
    body_type: str = "generic"
    surface_covering: str = "skin"
    covering_density: float = 0.5
    covering_color: tuple = (0, 0, 0)
    appendages: List[str] = field(default_factory=list)
    
    # Diet preferences
    preferred_food_parts: List[str] = field(default_factory=list)
    toxic_food_parts: List[str] = field(default_factory=list)
    diet_type: float = 0.0
    edible_parts: List[str] = field(default_factory=list)
    
    # Cached type
    _type: Optional[int] = None
    
    # Predefined databases as class variables
    _name_parts_db = {
        'prefix': ["zo", "ra", "fi", "lo", "pa", "ki", "tu", "ve", "no", "xi"],
        'suffix': ["oid", "ian", "us", "ix", "ae", "or", "ite", "oid", "ax", "is"],
        'plant_suffix': ["folia", "phyll", "herba", "flora", "verd", "chloro"]
    }
    
    _body_parts_db = {
        'body_type': ["slim", "bulky", "streamlined", "segmented", "radial"],
        'surface_covering': ["fur", "scales", "feathers", "skin", "bark", "chitin"],
        'appendages': ["wings", "fins", "tentacles", "antennae", "claws", "hooves"]
    }
    
    _food_parts_db = {
        'plant': ["leaf", "fruit", "nut", "root", "stem", "flower"],
        'animal': ["meat", "organ", "bone", "marrow", "fat"]
    }
    size: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # width, depth, height in yards
    occupies_multiple_cells: bool = False
    
    def __post_init__(self):
        if not hasattr(self, 'movement_type'):
            self._determine_movement_type()
        if not self.name and not self.name_parts:
            self._generate_random_name()
        if not self.name:
            self.name = "".join(part.capitalize() for part in self.name_parts)
        
        if not self.appendages:
            self._generate_physical_characteristics()
        
        if not self.preferred_food_parts and self.type != 0:
            self._generate_diet_preferences()
        
        if not self.edible_parts:
            self._generate_edible_parts()

        if not hasattr(self, 'size'):
            self._generate_size()

    def _determine_movement_type(self):
        """Determine movement type based on physical characteristics"""
        if not self.can_move:
            self.movement_type = self.MovementType.TERRESTRIAL
        elif "wings" in self.appendages and not self.has_roots:
            self.movement_type = self.MovementType.AVIAN
        elif "fins" in self.appendages and not self.has_roots:
            self.movement_type = self.MovementType.AQUATIC
        else:
            self.movement_type = self.MovementType.TERRESTRIAL

    def _generate_size(self):
        """Generate size based on organism type"""
        if self.type == 0:  # Plants
            self.size = (
                random.uniform(0.1, 3.0),  # width
                random.uniform(0.1, 3.0),  # depth
                random.uniform(0.1, 10.0)  # height
            )
        else:  # Animals
            base_size = 0.5 if "wings" in self.appendages else 1.0
            scale = 1.0
            if self.body_type == "slim":
                scale *= 0.8
            elif self.body_type == "bulky":
                scale *= 1.5
            
            self.size = (
                random.uniform(0.8, 1.5) * scale,  # width
                random.uniform(0.8, 1.5) * scale,  # depth
                random.uniform(0.5, 2.0) * scale   # height
            )
        self.occupies_multiple_cells = any(dim > 0.8 for dim in self.size)


    @property
    def type(self) -> int:
        if self._type is None:
            if not self.can_move and self.has_roots:
                self._type = 0  # Plant
            elif self.diet_type < 0.3:
                self._type = 1  # Herbivore
            elif self.diet_type > 0.7:
                self._type = 2  # Carnivore
            else:
                self._type = 3  # Omnivore
        return self._type
    
    def _hash_select(self, key: str, options: List[str]) -> str:
        """Deterministically select an option based on key hash"""
        hash_val = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        return options[hash_val % len(options)]
    
    def _generate_random_name(self):
        if self.type == 0:  # Plant
            prefix = self._hash_select("plant_prefix", self._name_parts_db['prefix'])
            suffix = self._hash_select("plant_suffix", self._name_parts_db['plant_suffix'])
            self.name_parts = [prefix, suffix]
        else:
            prefix = self._hash_select("animal_prefix", self._name_parts_db['prefix'])
            suffix = self._hash_select("animal_suffix", self._name_parts_db['suffix'])
            self.name_parts = [prefix, suffix]
    
    def _generate_physical_characteristics(self):
        if self.can_move:
            self.body_type = self._hash_select("body_type", self._body_parts_db['body_type'])
            self.surface_covering = self._hash_select("covering", self._body_parts_db['surface_covering'])
            self.covering_density = random.uniform(0.1, 0.9)
            
            num_appendages = random.randint(0, 3)
            if num_appendages > 0:
                self.appendages = random.sample(self._body_parts_db['appendages'], num_appendages)
            
            self.height = random.uniform(0.8, 2.0) if self.type in [2, 3] else random.uniform(0.5, 1.5)
        else:
            self.body_type = "radial" if random.random() < 0.5 else "segmented"
            self.surface_covering = "bark" if random.random() < 0.5 else "skin"
            self.covering_density = random.uniform(0.7, 1.0)
            self.height = random.uniform(0.1, 3.0)
    
    def _generate_edible_parts(self):
        if self.type == 0:
            num_parts = random.randint(1, 4)
            self.edible_parts = random.sample(self._food_parts_db['plant'], num_parts)
        else: 
            num_parts = random.randint(1, 3)
            self.edible_parts = random.sample(self._food_parts_db['animal'], num_parts)

    def _generate_diet_preferences(self):
        if self.type == 0:  # Plants don't eat
            return
        
        if self.type == 1:  # Herbivore
            num_preferred = random.randint(1, 3)
            self.preferred_food_parts = random.sample(self._food_parts_db['plant'], num_preferred)
        elif self.type == 2:  # Carnivore
            num_preferred = random.randint(1, 2)
            self.preferred_food_parts = random.sample(self._food_parts_db['animal'], num_preferred)
        else:  # Omnivore
            plant_parts = random.sample(self._food_parts_db['plant'], random.randint(1, 2))
            animal_parts = random.sample(self._food_parts_db['animal'], random.randint(1, 2))
            self.preferred_food_parts = plant_parts + animal_parts
    
    def mutate(self) -> 'Species':
        """Create a mutated version of this species"""
        # Helper function for bounded mutation
        def mutate_value(value, min_val, max_val, factor=0.1):
            new_val = value * random.uniform(1 - factor, 1 + factor)
            return max(min_val, min(max_val, new_val))
        
        # Create new species with mutated attributes
        new_species = Species(
            name_parts=self.name_parts.copy(),
            color=(
                mutate_value(self.color[0], 0, 255),
                mutate_value(self.color[1], 0, 255),
                mutate_value(self.color[2], 0, 255)
            ),
            max_energy=mutate_value(self.max_energy, 1, 100),
            energy_gain=mutate_value(self.energy_gain, 0.05, 5),
            energy_consumption=mutate_value(self.energy_consumption, 0.01, 1),
            mature_age=int(mutate_value(self.mature_age, 5, 200)),
            reproduction_cost=mutate_value(self.reproduction_cost, 0.5, 20),
            offspring_energy=mutate_value(self.offspring_energy, 0.5, 10),
            reproduction_frequency=int(mutate_value(self.reproduction_frequency, 1, 30)),
            temperature_sensitivity=mutate_value(self.temperature_sensitivity, 0.1, 1),
            light_sensitivity=mutate_value(self.light_sensitivity, 0.1, 1),
            cold_resistance=mutate_value(self.cold_resistance, 0.1, 1),
            
            # Seed-related attributes
            dormancy_threshold=mutate_value(self.dormancy_threshold, 0.1, 0.6) if self.has_roots else 0,
            seed_lifespan=int(mutate_value(self.seed_lifespan, 10, 500)) if self.has_roots else 0,
            seed_maturity_age=int(mutate_value(self.seed_maturity_age, 1, 30)) if self.has_roots else 0,
            seed_spread_radius=int(mutate_value(self.seed_spread_radius, 1, 10)) if self.has_roots else 0,
            seed_germination_chance=mutate_value(self.seed_germination_chance, 0.01, 0.5) if self.has_roots else 0,

            can_move=self.can_move,
            move_speed=int(mutate_value(self.move_speed, 0, 5)) if self.can_move else 0,
            move_energy_cost=mutate_value(self.move_energy_cost, 0, 1) if self.can_move else 0,
            has_roots=self.has_roots,
            root_energy=mutate_value(self.root_energy, 0, 30) if self.has_roots else 0,
            max_age=int(mutate_value(self.max_age, 0, 2000)) if self.max_age > 0 else 0,
            sleep_ratio=mutate_value(self.sleep_ratio, 0, 0.5) if self.can_move else 0,
            diurnal=self.diurnal if random.random() < 0.9 else not self.diurnal,
            height=mutate_value(self.height, 0.1, 5),
            body_type=self.body_type,
            surface_covering=self.surface_covering,
            covering_density=mutate_value(self.covering_density, 0, 1),
            covering_color=(
                mutate_value(self.covering_color[0], 0, 255),
                mutate_value(self.covering_color[1], 0, 255),
                mutate_value(self.covering_color[2], 0, 255)
            ),
            appendages=self.appendages.copy(),
            preferred_food_parts=self.preferred_food_parts.copy(),
            toxic_food_parts=self.toxic_food_parts.copy(),
            diet_type=mutate_value(self.diet_type, 0, 1),
            edible_parts=self.edible_parts.copy()
        )
        
        # Random mutations
        if random.random() < 0.05 and new_species.can_move:
            if random.random() < 0.5 and new_species.appendages:
                new_species.appendages.pop(random.randint(0, len(new_species.appendages)-1))
            elif len(new_species.appendages) < 3:
                available = [a for a in self._body_parts_db['appendages'] if a not in new_species.appendages]
                if available:
                    new_species.appendages.append(random.choice(available))
        
        if random.random() < 0.1 and new_species.type != 0:
            if random.random() < 0.5 and new_species.preferred_food_parts:
                new_species.preferred_food_parts.pop(random.randint(0, len(new_species.preferred_food_parts)-1))
            else:
                all_parts = self._food_parts_db['plant'] + self._food_parts_db['animal']
                available = [p for p in all_parts if p not in new_species.preferred_food_parts]
                if available:
                    new_species.preferred_food_parts.append(random.choice(available))
        
        # Rare type change
        if random.random() < 0.005:
            if new_species.type == 0:  # Plant -> Animal
                new_species.can_move = True
                new_species.has_roots = False
                new_species.move_speed = random.randint(1, 3)
                new_species.move_energy_cost = random.uniform(0.1, 0.5)
                new_species.max_age = random.randint(100, 500)
                new_species.sleep_ratio = random.uniform(0.1, 0.3)
                new_species._generate_diet_preferences()
                new_species._generate_physical_characteristics()
            else:  # Animal -> Plant
                if random.random() < 0.1:
                    new_species.can_move = False
                    new_species.has_roots = True
                    new_species.move_speed = 0
                    new_species.move_energy_cost = 0
                    new_species.max_age = 0
                    new_species.sleep_ratio = 0
                    new_species.root_energy = random.uniform(5, 20)
                    new_species.preferred_food_parts = []
                    new_species.toxic_food_parts = []
                    new_species._generate_physical_characteristics()

        if random.random() < 0.01:
            if new_species.movement_type == self.MovementType.TERRESTRIAL:
                if "wings" in new_species.appendages:
                    new_species.movement_type = self.MovementType.AVIAN
                elif "fins" in new_species.appendages:
                    new_species.movement_type = self.MovementType.AQUATIC
            elif new_species.movement_type == self.MovementType.AVIAN:
                if random.random() < 0.5:
                    new_species.movement_type = self.MovementType.TERRESTRIAL
                    new_species.appendages.remove("wings")
            elif new_species.movement_type == self.MovementType.AQUATIC:
                if random.random() < 0.5:
                    new_species.movement_type = self.MovementType.TERRESTRIAL
                    new_species.appendages.remove("fins")

        new_species._generate_random_name()
        return new_species

    @classmethod
    def generate_random_species(cls, food_sources: Optional[List['Species']] = None) -> 'Species':
        """Generate a random species with sensible defaults"""
        species = cls()
        
        # Set core attributes
        species.max_energy = random.uniform(5, 50)
        species.energy_gain = random.uniform(0.1, 2.0)
        species.energy_consumption = random.uniform(0.01, 0.2)
        species.mature_age = random.randint(10, 100)
        species.reproduction_cost = random.uniform(1, 10)
        species.offspring_energy = random.uniform(1, 5)
        species.reproduction_frequency = random.randint(1, 20)
        species.temperature_sensitivity = random.uniform(0.1, 1.0)
        species.light_sensitivity = random.uniform(0.1, 1.0)
        species.cold_resistance = random.uniform(0.1, 1.0)
        
        # Determine if plant or animal
        is_plant = random.random() < (0.3 if food_sources else 0.5)
        species.can_move = not is_plant
        species.has_roots = is_plant

        # Set seed attributes for plants
        if is_plant:
            species.dormancy_threshold = random.uniform(0.2, 0.4)
            species.root_energy = random.uniform(5, 20)
            species.color = (random.randint(50, 150), random.randint(100, 200), random.randint(50, 150))
            species.seed_lifespan = random.randint(100, 400)
            species.seed_maturity_age = random.randint(5, 25)
            species.seed_spread_radius = random.randint(2, 6)
            species.seed_germination_chance = random.uniform(0.05, 0.2)
            species.seed_production_frequency = random.randint(20, 40)
            species.seed_production_chance = random.uniform(0.3, 0.7)
        else:
            # Animal attributes
            species.move_speed = random.randint(1, 3)
            species.move_energy_cost = random.uniform(0.1, 0.5)
            species.max_age = random.randint(100, 1000)
            species.sleep_ratio = random.uniform(0.1, 0.3)
            species.diurnal = random.random() < 0.7
            
            # Set diet based on available food sources
            if food_sources:
                plant_parts = list(set(p for s in food_sources if s.type == 0 for p in s.edible_parts))
                animal_parts = list(set(p for s in food_sources if s.type != 0 for p in s.edible_parts))
                
                can_eat_plants = bool(plant_parts)
                can_eat_animals = bool(animal_parts)
                
                if can_eat_plants and can_eat_animals:
                    choice = random.choices(['herbivore', 'carnivore', 'omnivore'], weights=[25, 25, 50], k=1)[0]
                elif can_eat_plants:
                    choice = 'herbivore'
                elif can_eat_animals:
                    choice = 'carnivore'
                else:
                    choice = 'fallback'
                
                if choice == 'herbivore':
                    species.diet_type = random.uniform(0.0, 0.3)
                    num = random.randint(1, min(len(plant_parts), 3))
                    species.preferred_food_parts = random.sample(plant_parts, num)
                elif choice == 'carnivore':
                    species.diet_type = random.uniform(0.7, 1.0)
                    num = random.randint(1, min(len(animal_parts), 2))
                    species.preferred_food_parts = random.sample(animal_parts, num)
                elif choice == 'omnivore':
                    species.diet_type = random.uniform(0.3, 0.7)
                    p_num = random.randint(1, min(len(plant_parts), 2))
                    a_num = random.randint(1, min(len(animal_parts), 2))
                    species.preferred_food_parts = (random.sample(plant_parts, p_num) + 
                                                  random.sample(animal_parts, a_num))
            
            if not species.preferred_food_parts:
                species.diet_type = random.random()
                species._generate_diet_preferences()

            # Set color based on diet
            if species.diet_type < 0.3:  # Herbivore
                species.color = (random.randint(150, 255), random.randint(150, 255), random.randint(100, 200))
            elif species.diet_type > 0.7:  # Carnivore
                species.color = (random.randint(200, 255), random.randint(100, 150), random.randint(100, 150))
            else:  # Omnivore
                species.color = (random.randint(150, 255), random.randint(150, 200), random.randint(100, 150))

        # Finalize species
        species._generate_random_name()
        species._generate_physical_characteristics()
        species._generate_edible_parts()
        all_food = cls._food_parts_db['plant'] + cls._food_parts_db['animal']
        species.toxic_food_parts = [p for p in all_food if random.random() < 0.1 and p not in species.preferred_food_parts]
        
        return species

# Default species examples
GRASS = Species(
    name_parts=["gra", "ss"],
    color=(100, 200, 100), max_energy=15, energy_gain=0.3, energy_consumption=0.05,
    mature_age=20, reproduction_cost=2, offspring_energy=1.5, reproduction_frequency=5,
    temperature_sensitivity=0.8, light_sensitivity=0.9, cold_resistance=0.7,
    dormancy_threshold=0.3, seed_lifespan=365 * 2, seed_maturity_age=20,
    seed_spread_radius=4, seed_germination_chance=0.15,
    can_move=False, has_roots=True, root_energy=8,
    height=0.3, body_type="radial", surface_covering="skin", covering_density=0.8
)
GRASS.__post_init__()

OMNIVORE = Species(
    name_parts=["om", "niv", "ore"],
    color=(200, 150, 100), max_energy=25, energy_gain=2.0, energy_consumption=0.2,
    mature_age=100, reproduction_cost=5, offspring_energy=3, reproduction_frequency=10,
    temperature_sensitivity=0.6, light_sensitivity=0.3, cold_resistance=0.5,
    dormancy_threshold=0.25,
    can_move=True, move_speed=2, move_energy_cost=0.2,
    max_age=500, sleep_ratio=0.2, diurnal=True,
    height=1.2, body_type="slim", surface_covering="fur", covering_density=0.6,
    appendages=["claws"], diet_type=0.5,
    preferred_food_parts=["fruit", "meat"], toxic_food_parts=["root"]
)
OMNIVORE.__post_init__()

@dataclass
class Organism:
    species: Species
    position: Tuple[int, int, int]  # (x, y, z)
    energy: float = 0
    root_energy: float = 0  # Only used for plants
    age: float = 0
    is_dormant: bool = False
    is_dead: bool = False
    injury_level: float = 0.0
    id: int = field(default=-1) 
    occupied_cells: List[Tuple[int, int, int]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.species.type == 0 or (hasattr(self.species, 'movement_type') and 
                                      self.species.movement_type == Species.MovementType.TERRESTRIAL):
            self.position = (self.position[0], self.position[1], 0)
        self._calculate_occupied_cells()
    
    def _calculate_occupied_cells(self):
        """Calculate all cells this organism occupies based on its size"""
        self.occupied_cells = [self.position]  # Always include the primary position
        
        if self.species.occupies_multiple_cells:
            width, depth, height = self.species.size
            x, y, z = self.position
            
            # Calculate how many cells in each dimension
            cells_x = max(1, math.ceil(width))
            cells_y = max(1, math.ceil(depth))
            cells_z = max(1, math.ceil(height))
            
            # Generate all occupied cells
            for dx in range(cells_x):
                for dy in range(cells_y):
                    for dz in range(cells_z):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue  # Skip the primary position already added
                        new_pos = (x + dx, y + dy, z + dz)
                        if new_pos not in self.occupied_cells:
                            self.occupied_cells.append(new_pos)

@dataclass
class Seed:
    species: Species
    position: Tuple[int, int, int]
    age: int = 0
    id: int = field(default=-1)

@dataclass
class SimulationSettings:
    grid_size: int = 50
    axial_tilt: float = 23.44  # in degrees
    latitude: float = 45.0     # in degrees
    initial_organisms: List[Organism] = field(default_factory=list)
    initial_seeds: List[Seed] = field(default_factory=list)

class IDGenerator:
    def __init__(self):
        self.current_id = 0
        self.max_id = 2**63 - 1  # Max 64-bit signed integer
        self.recycled_ids = set()
    
    def get_id(self) -> int:
        if self.recycled_ids:
            return self.recycled_ids.pop()
        self.current_id += 1
        if self.current_id > self.max_id:
            self.current_id = 1  # Wrap around
        return self.current_id
    
    def recycle_id(self, id: int):
        if id > 0:
            self.recycled_ids.add(id)

class WorldGrid:
    def __init__(self, size: int):
        self.size = size
        self.organisms: Dict[Tuple[int, int, int], Organism] = {}
        self.organism_ids: Dict[int, Tuple[int, int, int]] = {}

        # Seeds are stored in lists per tile
        self.seeds: Dict[Tuple[int, int, int], List[Seed]] = {}
        self.seed_ids: Dict[int, Tuple[int, int, int]] = {}
        self.organism_id_gen = IDGenerator()
        self.seed_id_gen = IDGenerator()
        
        
        # Pre-compute neighbor offsets as tensor
        neighbor_list = [(dx, dy, dz) 
                        for dx in range(-1, 2)
                        for dy in range(-1, 2)
                        for dz in range(-1, 2)
                        if not (dx == 0 and dy == 0 and dz == 0)]
        self.neighbor_offsets = torch.tensor(neighbor_list, dtype=torch.int32, device=DEVICE)
        self.cell_occupants: Dict[Tuple[int, int, int], List[int]] = {}
        
    def can_occupy_position(self, organism: Organism, position: Tuple[int, int, int]) -> bool:
        """Check if this organism can occupy the given position based on its movement type"""
        x, y, z = position
        if not self.is_valid_position(position):
            return False
            
        # Terrestrial organisms must stay at z=0
        if organism.species.movement_type == Species.MovementType.TERRESTRIAL:
            return z == 0
            
        # Aquatic organisms must stay below z=0 (if we implement water levels)
        elif organism.species.movement_type == Species.MovementType.AQUATIC:
            return z <= 0  # Assuming negative z is underwater
            
        # Avian can be anywhere but have restrictions when injured
        elif organism.species.movement_type == Species.MovementType.AVIAN:
            if organism.injury_level > 0.5:  # Severely injured birds try to land
                return z == 0
            return True
            
        return True
    
    def add_organism(self, organism: Organism):
        """Add organism, checking all its cells are available"""
        # Enforce terrestrial constraints
        if organism.species.type == 0:  # Plants must be terrestrial
            organism.position = (organism.position[0], organism.position[1], 0)
        
        # Check all cells are valid for this organism's movement type
        for cell in organism.occupied_cells:
            if not self.can_occupy_position(organism, cell):
                raise ValueError(f"Invalid position {cell} for {organism.species.movement_type}")
        # First calculate occupied cells if not done
        if organism.id == -1:
            organism.id = self.organism_id_gen.get_id()
        if not organism.occupied_cells:
            organism._calculate_occupied_cells()
        
        # Check all cells are valid
        for cell in organism.occupied_cells:
            if not self.is_valid_position(cell):
                raise ValueError(f"Invalid position {cell}")
        
        # Check if primary position is available
        if not self.is_position_available(organism.position):
            raise ValueError(f"Primary position {organism.position} occupied")
        
        # Add to tracking structures
        self.organisms[organism.id] = organism
        self.organism_ids[organism.id] = organism.occupied_cells
        
        # Mark all occupied cells
        for cell in organism.occupied_cells:
            if cell not in self.cell_occupants:
                self.cell_occupants[cell] = []
            self.cell_occupants[cell].append(organism.id)

    def add_seed(self, seed: Seed):
        if seed.id == -1:
            seed.id = self.seed_id_gen.get_id()
        pos = seed.position
        if not self.is_valid_position(pos):
            raise ValueError(f"Invalid position for seed {pos}")

        if pos not in self.seeds:
            self.seeds[pos] = []
        self.seeds[pos].append(seed)
        self.seed_ids[seed.id] = pos
        
    def remove_organism(self, organism_id: int):
        """Remove organism from all its cells"""
        if organism_id not in self.organism_ids:
            return
            
        # Remove from all cells
        for cell in self.organism_ids[organism_id]:
            if cell in self.cell_occupants:
                self.cell_occupants[cell] = [oid for oid in self.cell_occupants[cell] if oid != organism_id]
                if not self.cell_occupants[cell]:
                    del self.cell_occupants[cell]
        
        # Remove from other tracking
        if organism_id in self.organisms:
            del self.organisms[organism_id]
        del self.organism_ids[organism_id]
        self.organism_id_gen.recycle_id(organism_id)

    def remove_seed(self, seed_id: int):
        """Remove seed and recycle its ID"""
        if seed_id not in self.seed_positions:
            return
            
        pos = self.seed_positions[seed_id]
        if pos in self.seeds:
            self.seeds[pos] = [s for s in self.seeds[pos] if s.id != seed_id]
            if not self.seeds[pos]:
                del self.seeds[pos]
        
        del self.seed_positions[seed_id]
        self.seed_id_gen.recycle_id(seed_id)

    def get_organisms_at_position(self, position: Tuple[int, int, int]) -> List[Organism]:
        """Get all organisms at a position (could be multiple for small organisms)"""
        if position not in self.cell_occupants:
            return []
        return [self.organisms[oid] for oid in self.cell_occupants[position]]
    
    def is_position_available(self, position: Tuple[int, int, int], ignore_organism: Optional[int] = None) -> bool:
        """Check if position is available, optionally ignoring a specific organism"""
        if not self.is_valid_position(position):
            return False
        if position not in self.cell_occupants:
            return True
        occupants = self.cell_occupants[position]
        if not occupants:
            return True
        if ignore_organism and ignore_organism in occupants:
            return len(occupants) == 1
        return False

    def remove_seed(self, seed_id: int):
        if seed_id not in self.seed_ids:
            return

        pos = self.seed_ids[seed_id]
        if pos in self.seeds:
            self.seeds[pos] = [s for s in self.seeds[pos] if s.id != seed_id]
            if not self.seeds[pos]: # If list is empty, remove the key
                del self.seeds[pos]
        del self.seed_ids[seed_id]
        
    def move_organism(self, organism_id: int, new_position: Tuple[int, int, int]) -> bool:
        """Move organism to new position if all cells are available"""
        if organism_id not in self.organisms:
            return False
            
        organism = self.organisms[organism_id]
        
        # Injured avian organisms try to move downward
        if (organism.species.movement_type == Species.MovementType.AVIAN and 
            organism.injury_level > 0.3):
            current_z = organism.position[2]
            if current_z > 0:
                # Try to find a position at lower z first
                for z in range(current_z - 1, -1, -1):
                    test_pos = (new_position[0], new_position[1], z)
                    if self.is_position_available(test_pos, ignore_organism=organism_id):
                        new_position = test_pos
                        break
        
        # Check if new position is valid for this organism type
        if not self.can_occupy_position(organism,new_position):
            return False
            
        new_cells = self._calculate_new_cells(organism, new_position)
        
        # Check all new cells are available
        for cell in new_cells:
            if not self.is_position_available(cell, ignore_organism=organism_id):
                return False
        
        # Move is valid - update all tracking
        old_cells = organism.occupied_cells.copy()
        
        # Remove from old cells
        for cell in old_cells:
            if cell in self.cell_occupants:
                self.cell_occupants[cell] = [oid for oid in self.cell_occupants[cell] if oid != organism_id]
                if not self.cell_occupants[cell]:
                    del self.cell_occupants[cell]
        
        # Add to new cells
        organism.position = new_position
        organism.occupied_cells = new_cells
        self.organism_ids[organism_id] = new_cells
        
        for cell in new_cells:
            if cell not in self.cell_occupants:
                self.cell_occupants[cell] = []
            self.cell_occupants[cell].append(organism_id)
        
        return True
        
    def _calculate_new_cells(self, organism: Organism, new_position: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Calculate new occupied cells based on size and new position"""
        if not organism.species.occupies_multiple_cells:
            return [new_position]
            
        width, depth, height = organism.species.size
        x, y, z = new_position
        
        cells_x = max(1, math.ceil(width))
        cells_y = max(1, math.ceil(depth))
        cells_z = max(1, math.ceil(height))
        
        new_cells = []
        for dx in range(cells_x):
            for dy in range(cells_y):
                for dz in range(cells_z):
                    new_cells.append((x + dx, y + dy, z + dz))
        return new_cells
    
    def get_organism(self, position: Tuple[int, int, int]) -> Optional[Organism]:
        return self.organisms.get(position)
        
    def is_valid_position(self, position: Tuple[int, int, int]) -> bool:
        x, y, z = position
        return (0 <= x < self.size and 
                0 <= y < self.size and 
                0 <= z < self.size)
    
    def get_empty_spots_in_radius(self, position: Tuple[int, int, int], occupancy_tensor: torch.Tensor, radius: int = 1) -> List[Tuple[int, int, int]]:
        x, y, z = position
        
        # Create ranges for each dimension
        x_r = torch.arange(max(0, x - radius), min(self.size, x + radius + 1), device=DEVICE)
        y_r = torch.arange(max(0, y - radius), min(self.size, y + radius + 1), device=DEVICE)
        z_r = torch.arange(max(0, z - radius), min(self.size, z + radius + 1), device=DEVICE)
        
        # Create grid of positions
        xx, yy, zz = torch.meshgrid(x_r, y_r, z_r, indexing='ij')
        neighbor_positions = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)

        # Remove the center position
        center_mask = ~((neighbor_positions[:, 0] == x) & (neighbor_positions[:, 1] == y) & (neighbor_positions[:, 2] == z))
        valid_positions = neighbor_positions[center_mask]
        
        if valid_positions.shape[0] == 0:
            return []

        # Use the occupancy tensor to find empty spots
        coords = valid_positions.long()
        is_occupied = occupancy_tensor[coords[:, 0], coords[:, 1], coords[:, 2]]
        empty_positions_tensor = valid_positions[~is_occupied]

        return [tuple(pos) for pos in empty_positions_tensor.cpu().numpy()]

    def get_food_in_radius(self, position: Tuple[int, int, int], occupancy_tensor: torch.Tensor, radius: int) -> List[Tuple[Tuple[int, int, int], Organism]]:
        x, y, z = position
        
        # Create ranges for each dimension
        x_range = torch.arange(max(0, x - radius), min(self.size, x + radius + 1), device=DEVICE)
        y_range = torch.arange(max(0, y - radius), min(self.size, y + radius + 1), device=DEVICE)
        z_range = torch.arange(max(0, z - radius), min(self.size, z + radius + 1), device=DEVICE)
        
        # Create grid of positions
        xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
        positions = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        
        # Remove center position
        center_mask = ~((positions[:, 0] == x) & (positions[:, 1] == y) & (positions[:, 2] == z))
        positions = positions[center_mask]

        if positions.shape[0] == 0:
            return []
            
        # Check occupancy
        coords = positions.long()
        is_occupied = occupancy_tensor[coords[:, 0], coords[:, 1], coords[:, 2]]
        food_positions_tensor = positions[is_occupied]

        # Get actual organisms
        food = []
        for pos_np in food_positions_tensor.cpu().numpy():
            pos_tuple = tuple(pos_np)
            if pos_tuple in self.organisms:
                food.append((pos_tuple, self.organisms[pos_tuple]))
                
        return food

class GameOfLife:
    def __init__(self):
        self.grid_size = 50
        self.cell_size = 10
        self.running = False
        
        self.grid = WorldGrid(self.grid_size)
        self.grid_tensor = torch.zeros((self.grid_size, self.grid_size, 3), dtype=torch.float32, device=DEVICE)
        
        # Time and environmental settings
        self.axial_tilt = math.radians(23.44)  # Earth's axial tilt in radians
        self.latitude = math.radians(45)       # Default to mid-latitude (~45°)
        self.steps_per_day = 48
        self.current_step, self.current_day = 0, 0
        self.summer_solstice, self.winter_solstice = 172, 355
        self.year_length = 365
        self.max_daylight = 32
        self.min_daylight = 16
        self.current_daylight = 17
        self.light_level = 1.0
        self.temperature = 1.0
        self.is_day = True
        
        # Precompute angles for performance
        self.day_angles = torch.linspace(0, torch.pi, self.steps_per_day, device=DEVICE)
        self.season_angles = torch.linspace(0, 2*torch.pi, self.year_length, device=DEVICE)


        self.z_view_mode = "slider"  # "slider", "top", "bottom"
        self.current_z_level = 0
        self.max_z_level = 10  # Maximum z-level to display
        
        # Initialize UI
        dpg.create_context()
        self.setup_ui()
        dpg.create_viewport(title="Evolution Simulation", width=850, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def setup_ui(self):
        """Set up the Dear PyGUI interface"""
        with dpg.window(tag="Main Window"):
            with dpg.group(horizontal=True):
                # Control panel
                with dpg.child_window(width=300):
                    dpg.add_text("Evolution Simulation")
                    dpg.add_button(label="Start", callback=lambda: self.set_running(True))
                    dpg.add_button(label="Stop", callback=lambda: self.set_running(False))
                    dpg.add_button(label="Step", callback=self.step)
                    dpg.add_button(label="Clear", callback=self.clear_grid)
                    dpg.add_button(label="Randomize", callback=self.randomize_grid)
                    
                    dpg.add_spacer(height=10)
                    dpg.add_slider_int(label="Grid Size", min_value=10, max_value=100, 
                                     default_value=self.grid_size, callback=self.change_grid_size)
                    
                    # Environment settings
                    dpg.add_spacer(height=10)
                    dpg.add_text("Environment Settings:")
                    self.axial_tilt_slider = dpg.add_slider_float(
                        label="Axial Tilt (degrees)", 
                        min_value=0.0, max_value=45.0, 
                        default_value=23.44, 
                        callback=self.update_environment_settings
                    )
                    self.latitude_slider = dpg.add_slider_float(
                        label="Latitude (degrees)", 
                        min_value=0.0, max_value=90.0, 
                        default_value=45.0, 
                        callback=self.update_environment_settings
                    )

                    # Z-axis viewing controls
                    dpg.add_spacer(height=10)
                    dpg.add_text("Z-Axis View:")
                    with dpg.group(horizontal=True):
                        dpg.add_radio_button(
                            items=["Slider", "Top", "Bottom"],
                            default_value="Slider",
                            callback=self.change_z_view_mode
                        )
                    self.z_slider = dpg.add_slider_int(
                        label="Z Level",
                        min_value=0,
                        max_value=self.max_z_level,
                        default_value=0,
                        callback=self.change_z_level,
                        show=False  # Initially hidden unless in slider mode
                    )
                    
                    # Status displays
                    self.time_text = dpg.add_text("Time: Day 1, 00:00", tag="time_text")
                    self.daylight_text = dpg.add_text(f"Daylight: {self.current_daylight/2}h", tag="daylight_text")
                    self.light_level_text = dpg.add_text(f"Light: {self.light_level:.2f}", tag="light_level_text")
                    self.temp_text = dpg.add_text(f"Temp: {self.temperature:.2f}", tag="temp_text")
                    self.date_text = dpg.add_text("Date: Jan 1", tag="date_text")
                
                with dpg.child_window(tag="game_window"):
                    with dpg.drawlist(width=self.grid_size*self.cell_size, 
                                    height=self.grid_size*self.cell_size, 
                                    tag="drawlist"):
                        pass
        # Setup callbacks
        dpg.set_viewport_resize_callback(self.on_viewport_resize)
        dpg.set_primary_window("Main Window", True)
        self.draw_grid()

    def update_environment_settings(self):
        """Update environment settings from UI"""
        self.axial_tilt = math.radians(dpg.get_value(self.axial_tilt_slider))
        self.latitude = math.radians(dpg.get_value(self.latitude_slider))
        self.update_daylight_duration()  # Recalculate with new settings

    def on_viewport_resize(self):
        """Handle viewport resizing"""
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dpg.set_item_width("Main Window", viewport_width)
        dpg.set_item_height("Main Window", viewport_height)
        
        if viewport_width > 300:
            dpg.set_item_width("game_window", viewport_width - 320)
    
    def get_species_color(self, organism: Organism) -> Tuple[int, int, int]:
        """Calculate display color for an organism based on environment and state"""
        if organism.is_dead:
            return (50, 50, 50)  # Gray for dead organisms
        
        species, energy, is_dormant = organism.species, organism.energy, organism.is_dormant
        base_color = torch.tensor(species.color, dtype=torch.float32, device=DEVICE)
        
        # Apply injury effect (mix with red)
        if organism.injury_level > 0:
            injury_color = torch.tensor([200, 50, 50], dtype=torch.float32, device=DEVICE)
            base_color = base_color * (1 - organism.injury_level) + injury_color * organism.injury_level
        
        if species.type == 0:  # Plant
            color = base_color * torch.tensor([self.temperature, self.light_level, self.temperature], device=DEVICE)
            if not self.is_day: color *= 0.3
        else:  # Animal
            energy_ratio = energy / species.max_energy
            color = base_color * torch.tensor([energy_ratio, energy_ratio * self.temperature, energy_ratio], device=DEVICE)
        
        # Apply state modifiers
        if is_dormant: color *= 0.4
        elif self.is_animal_sleeping(organism): color *= 0.6
        
        return tuple(torch.clamp(color, 0, 255).to(torch.uint8).cpu().numpy())
    
    def change_z_view_mode(self, sender, app_data):
        """Change how we view the z-axis"""
        self.z_view_mode = app_data.lower()
        dpg.configure_item(self.z_slider, show=(self.z_view_mode == "slider"))
        self.draw_grid()

    def change_z_level(self, sender, app_data):
        """Change current z-level when in slider mode"""
        self.current_z_level = app_data
        self.draw_grid()

    def get_species_char(self, organism: Organism) -> str:
        """Get display character for an organism"""
        if organism.is_dead:
            return "✝"
        if organism.injury_level > 0.5:
            return "✚"
        if organism.injury_level > 0.2:
            return "⚠"
        
        if organism.species and organism.species.name:
            if organism.is_dormant: return "z"
            return organism.species.name[0].upper()
        return "?"
    
    def update_time_cycles(self):
        """Update day/night and seasonal cycles with improved temperature modeling"""
        self.current_step += 1
        
        # Handle day rollover
        if self.current_step >= self.steps_per_day:
            self.current_step = 0
            self.current_day = (self.current_day + 1) % self.year_length
            self.update_daylight_duration()
        
        # Update day/night state
        self.is_day = self.current_step < self.current_daylight
        
        # Calculate light level based on time of day
        if self.is_day:
            progress = self.current_step / self.current_daylight
            angle = self.day_angles[int(progress * (len(self.day_angles)-1))]
            self.light_level = torch.sin(angle).item() * 0.9 + 0.1
        else:
            self.light_level = 0.05
        
        season_angle = self.season_angles[self.current_day]
        
        lat_deg = math.degrees(abs(self.latitude))
        base_temp = 1.0 - (lat_deg / 90) * 0.6 
        solar_declination = self.axial_tilt * math.sin(2 * math.pi * (self.current_day - 80) / self.year_length)
        season_factor = math.cos(self.latitude - solar_declination)
        season_variation = (0.3 + (lat_deg/90)*0.4) * math.sin(season_angle) 
        if self.is_day:
            day_progress = self.current_step / self.current_daylight
            day_angle = self.day_angles[int(day_progress * (len(self.day_angles)-1))]
            daily_cycle = 0.15 * torch.sin(day_angle).item()  # Daytime warming
        else:
            night_progress = (self.current_step - self.current_daylight) / (self.steps_per_day - self.current_daylight)
            daily_cycle = -0.1 * (1 - night_progress)  # Gradual nighttime cooling
        
        # Combine all factors
        self.temperature = max(0.1, min(1.0, 
            base_temp + 
            season_variation * season_factor + 
            daily_cycle
        ))
        
        # Special cases:
        # - Polar night (continuous darkness) gets extra cold
        if self.current_daylight == 0:
            self.temperature = max(0.1, self.temperature - 0.2)
        # - Polar day (continuous light) gets extra warm
        elif self.current_daylight == self.steps_per_day:
            self.temperature = min(1.0, self.temperature + 0.1)
        
        # Update UI displays
        hours = self.current_step // 2
        minutes = (self.current_step % 2) * 30
        time_str = f"{hours:02d}:{minutes:02d}"
        month, day = self.day_to_date(self.current_day)
        
        dpg.set_value("time_text", f"Day {self.current_day+1}, {time_str}")
        dpg.set_value("daylight_text", f"Daylight: {self.current_daylight/2:.1f}h")
        dpg.set_value("light_level_text", f"Light: {self.light_level:.2f}")
        dpg.set_value("temp_text", f"Temp: {self.temperature:.2f}")
        dpg.set_value("date_text", f"Date: {month} {day}")
    
    def day_to_date(self, day):
        """Convert day number to month/day"""
        months = [("Jan", 31), ("Feb", 28), ("Mar", 31), ("Apr", 30), 
                 ("May", 31), ("Jun", 30), ("Jul", 31), ("Aug", 31), 
                 ("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31)]
        for month, days in months:
            if day < days: return month, day + 1
            day -= days
        return "Dec", 31
    
    def update_daylight_duration(self):
        """Calculate daylight duration based on axial tilt and latitude"""
        # Calculate solar declination (angle of sun relative to equatorial plane)
        solar_declination = self.axial_tilt * math.sin(2 * math.pi * (self.current_day - 80) / self.year_length)
        
        # Calculate hour angle at sunrise/sunset
        try:
            hour_angle = math.acos(-math.tan(self.latitude) * math.tan(solar_declination))
        except ValueError:  # Handle polar day/night cases
            if math.tan(self.latitude) * math.tan(solar_declination) > 1:
                hour_angle = math.pi  # Polar day (24h daylight)
            else:
                hour_angle = 0       # Polar night (0h daylight)
        
        # Convert hour angle to daylight hours (1 hour angle = 12/π hours)
        daylight_hours = 24 * hour_angle / math.pi
        
        # Ensure reasonable bounds (never less than 0h or more than 24h)
        daylight_hours = max(0, min(24, daylight_hours))
        
        # Convert to our simulation steps (2 steps per hour)
        self.current_daylight = int(round(daylight_hours * 2))
        
        # Store min/max daylight for solstice tracking
        if daylight_hours > self.max_daylight/2:
            self.max_daylight = daylight_hours * 2
            self.summer_solstice = self.current_day
        elif daylight_hours < self.min_daylight/2:
            self.min_daylight = daylight_hours * 2
            self.winter_solstice = self.current_day

    def draw_grid(self):
        """Draw the current state of the grid with z-axis handling"""
        dpg.delete_item("drawlist", children_only=True)
        
        with dpg.draw_node(parent="drawlist"):
            # Determine which z-levels to show based on view mode
            if self.z_view_mode == "top":
                z_levels = [self.max_z_level]
            elif self.z_view_mode == "bottom":
                z_levels = [0]
            else:  # slider mode
                z_levels = [self.current_z_level]
            
            # Draw organisms first
            for org in self.grid.organisms.values():
                if not org.occupied_cells:
                    continue
                    
                # Check if organism is at current z-level(s)
                org_z = org.position[2]
                if org_z not in z_levels:
                    continue
                    
                # Get primary position
                x, y, z = org.position
                
                # Calculate size in cells
                cells_x = len(set(c[0] for c in org.occupied_cells))
                cells_y = len(set(c[1] for c in org.occupied_cells))
                
                color = self.get_species_color(org)
                char = self.get_species_char(org)
                
                # Draw rectangle
                dpg.draw_rectangle(
                    (x * self.cell_size, y * self.cell_size), 
                    ((x + cells_x) * self.cell_size, (y + cells_y) * self.cell_size), 
                    fill=color, 
                    color=(100, 100, 100), 
                    thickness=1
                )
                
                # Draw character
                dpg.draw_text(
                    (x*self.cell_size + (cells_x*self.cell_size)//2 - self.cell_size//4, 
                    y*self.cell_size + (cells_y*self.cell_size)//2 - self.cell_size//4), 
                    char, 
                    color=(0,0,0), 
                    size=int(self.cell_size*0.8)
                )
            
            # Draw grid lines
            for i in range(self.grid_size + 1):
                dpg.draw_line(
                    (i*self.cell_size, 0), 
                    (i*self.cell_size, self.grid_size*self.cell_size), 
                    color=(50, 50, 50, 100), 
                    thickness=1
                )
                dpg.draw_line(
                    (0, i*self.cell_size), 
                    (self.grid_size*self.cell_size, i*self.cell_size), 
                    color=(50, 50, 50, 100), 
                    thickness=1
                )
            
    def set_running(self, running):
        """Set simulation running state"""
        self.running = running
    
    def is_animal_sleeping(self, organism: Organism) -> bool:
        species = organism.species
        if species.type == 0: return False  # Plants don't sleep
        if organism.is_dormant: return True
        
        # Avian can't sleep while in the air
        if (species.movement_type == Species.MovementType.AVIAN and 
            organism.position[2] > 0):
            return False
            
        wrong_time = (species.diurnal and not self.is_day) or (not species.diurnal and self.is_day)
        return wrong_time or random.random() < species.sleep_ratio
        
    def step(self):
        """Perform one simulation step"""
        self.update_time_cycles()
        
        organisms = list(self.grid.organisms.values())
        seeds_list = [seed for pos_list in self.grid.seeds.values() for seed in pos_list]
        num_organisms = len(organisms)
        if num_organisms < 1 and not seeds_list:
            self.set_running(False)
            self.draw_grid()
            return

        organisms_from_seeds = []
        if self.current_step == 0 and self.grid.seeds:
            positions_with_seeds = list(self.grid.seeds.keys())
            
            for pos in positions_with_seeds:
                if pos not in self.grid.seeds:
                    continue

                surviving_seeds_at_pos = []
                seeds_at_pos = self.grid.seeds[pos]

                for seed in seeds_at_pos:
                    seed.age += 1
                    
                    # Remove old seeds
                    if seed.age > seed.species.seed_lifespan:
                        if seed.id in self.grid.seed_ids:  # Check if seed exists before removal
                            del self.grid.seed_ids[seed.id]
                        continue

                    # Check for germination
                    if (seed.age >= seed.species.seed_maturity_age and
                        pos not in self.grid.organisms and 
                        self.temperature > seed.species.dormancy_threshold and 
                        random.random() < seed.species.seed_germination_chance):
                        
                        species = seed.species
                        new_organism = Organism(
                            species=species, 
                            position=pos, 
                            energy=species.offspring_energy,
                            root_energy=species.offspring_energy*0.5, 
                            age=0
                        )
                        organisms_from_seeds.append(new_organism)
                        
                        # Remove the germinated seed
                        if seed.id in self.grid.seed_ids:  # Check if seed exists before removal
                            del self.grid.seed_ids[seed.id]
                        surviving_seeds_at_pos = [s for s in seeds_at_pos if s.id != seed.id]
                        break  # Only one seed can germinate per position
                    else:
                        surviving_seeds_at_pos.append(seed)
                
                # Update the seeds at this position
                if surviving_seeds_at_pos:
                    self.grid.seeds[pos] = surviving_seeds_at_pos
                else:
                    del self.grid.seeds[pos]
        # Update dormancy state for all organisms
        for org in organisms:
            org.is_dormant = self.temperature < org.species.dormancy_threshold

        for org in organisms:
            if org.species.occupies_multiple_cells:
                size_factor = org.species.size[0] * org.species.size[1] * org.species.size[2]
                org.energy *= size_factor  # Larger organisms need more energy
                org.species.energy_consumption *= size_factor
                org.species.max_energy *= size_factor

        species_list = [org.species for org in organisms]
        is_dormant_tensor = torch.tensor([org.is_dormant for org in organisms], dtype=torch.bool, device=DEVICE)
        energies = torch.tensor([org.energy for org in organisms], dtype=torch.float32, device=DEVICE)
        root_energies = torch.tensor([org.root_energy for org in organisms], dtype=torch.float32, device=DEVICE)
        ages = torch.tensor([org.age for org in organisms], dtype=torch.float32, device=DEVICE)
        types = torch.tensor([s.type for s in species_list], dtype=torch.int8, device=DEVICE)
        max_energies = torch.tensor([s.max_energy for s in species_list], dtype=torch.float32, device=DEVICE)
        energy_consumptions = torch.tensor([s.energy_consumption for s in species_list], dtype=torch.float32, device=DEVICE)
        energy_consumptions[is_dormant_tensor] *= 0.1  # Dormant organisms use 10% energy

        is_plant, is_animal = (types == 0), (types != 0)
        to_remove = torch.zeros(num_organisms, dtype=torch.bool, device=DEVICE)
        if self.current_step == 0: ages += 1.0  # Age organisms once per day

        if torch.any(is_plant):
            plant_indices = torch.where(is_plant)[0]
            light_sens = torch.tensor([s.light_sensitivity for s in species_list if s.type==0], device=DEVICE)
            temp_sens = torch.tensor([s.temperature_sensitivity for s in species_list if s.type==0], device=DEVICE)
            s_energy_gain = torch.tensor([s.energy_gain for s in species_list if s.type==0], device=DEVICE)
            s_root_energy_cap = torch.tensor([s.root_energy for s in species_list if s.type==0], device=DEVICE)
            
            # Calculate energy gain for plants
            energy_gain = (self.light_level * light_sens * s_energy_gain * (self.temperature * temp_sens))
            energy_gain[is_dormant_tensor[is_plant]] *= 0.1  # Dormant plants get 10% energy
            
            # Update energy levels
            energies[is_plant] += energy_gain - energy_consumptions[is_plant]
            excess_mask = energies[is_plant] > max_energies[is_plant]
            
            # Store excess energy in roots
            storage_amount = torch.zeros_like(energies[is_plant])
            storage_amount[excess_mask] = torch.minimum(
                energies[plant_indices[excess_mask]] - max_energies[plant_indices[excess_mask]], 
                s_root_energy_cap[excess_mask] - root_energies[plant_indices[excess_mask]]
            )
            energies[is_plant] = torch.min(energies[is_plant], max_energies[is_plant])
            root_energies[is_plant] += storage_amount
            
            # Use root energy if needed
            deficit_mask = energies[is_plant] < 0
            if torch.any(deficit_mask):
                energy_needed = -energies[plant_indices[deficit_mask]]
                root_draw = torch.minimum(energy_needed, root_energies[plant_indices[deficit_mask]])
                energies[plant_indices[deficit_mask]] += root_draw
                root_energies[plant_indices[deficit_mask]] -= root_draw
            
            # Mark plants for removal if they run out of energy
            to_remove[is_plant] = energies[is_plant] < 0

        if torch.any(is_animal):
            # Basic energy consumption
            energies[is_animal] -= energy_consumptions[is_animal]
            animal_indices = torch.where(is_animal)[0]
            
            # Calculate survival chances based on environment
            max_ages = torch.tensor([s.max_age for s in species_list if s.type != 0], dtype=torch.float32, device=DEVICE)
            cold_resistances = torch.tensor([s.cold_resistance for s in species_list if s.type != 0], device=DEVICE)
            coverings = torch.tensor([1.2 if s.surface_covering == "fur" else 1.0 for s in species_list if s.type != 0], device=DEVICE)
            survival_chances = torch.clamp((self.temperature + cold_resistances) * coverings, 0.0, 1.0)
            survival_chances[is_dormant_tensor[is_animal]] = torch.clamp(survival_chances[is_dormant_tensor[is_animal]] + 0.5, 0.0, 1.0)
            
            # Determine which animals die
            dies_from_conditions = torch.rand(len(animal_indices), device=DEVICE) > survival_chances
            to_remove[is_animal] = (
                (energies[is_animal] <= 0) | 
                ((ages[is_animal] > max_ages) & (max_ages > 0)) | 
                dies_from_conditions
            )
        
        organisms_to_remove_ids, survivors = set(), []
        for i, org in enumerate(organisms):
            if to_remove[i]: 
                organisms_to_remove_ids.add(org.id)
            else:
                org.energy = max(0, energies[i].item())
                org.root_energy = max(0, root_energies[i].item())
                org.age = ages[i].item()
                
                # Healing over time for injured organisms
                if org.injury_level > 0 and not org.is_dead:
                    org.injury_level = max(0, org.injury_level - 0.05)  # 5% healing per step
                
                survivors.append(org)
        
        organisms_to_add, seeds_to_add = [], []
        occupancy_tensor = torch.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=torch.bool, device=DEVICE)
        
        # Build occupancy tensor
        if self.grid.organisms:
            positions = [org.position for org in self.grid.organisms.values()]
            pos_tensor = torch.tensor(positions, dtype=torch.long, device=DEVICE)
            occupancy_tensor[pos_tensor[:, 0], pos_tensor[:, 1], pos_tensor[:, 2]] = True

        for organism in survivors:
            if organism.is_dormant: 
                continue
                
            species, pos = organism.species, organism.position

            # Reproduction
            if (organism.energy >= species.reproduction_cost and 
                organism.age >= species.mature_age and 
                self.current_day % species.reproduction_frequency == 0):
                
                if species.type == 0:  # Plant -> Seeds
                    if (random.random() < species.seed_production_chance and
                        self.temperature > species.dormancy_threshold and
                        self.is_day and self.light_level > 0.5):
                        
                        empty_spots = self.grid.get_empty_spots_in_radius(pos, occupancy_tensor, radius=species.seed_spread_radius)
                        if empty_spots:
                            num_seeds = random.randint(1, 3)  # Reduced from 2-5 to 1-3
                            for new_pos in random.sample(empty_spots, min(len(empty_spots), num_seeds)):
                                new_species = species.mutate() if random.random() < 0.01 else species
                                seeds_to_add.append(Seed(species=new_species, position=new_pos))
                            organism.energy -= species.reproduction_cost
                else:  # Animal -> Live offspring
                    empty_neighbors = self.grid.get_empty_spots_in_radius(pos, occupancy_tensor, radius=1)
                    if empty_neighbors:
                        new_pos = random.choice(empty_neighbors)
                        offspring = Organism(
                            species=species.mutate() if random.random()<0.05 else species, 
                            position=new_pos, 
                            energy=species.offspring_energy
                        )
                        organisms_to_add.append(offspring)
                        organism.energy -= species.reproduction_cost
                        occupancy_tensor[new_pos[0], new_pos[1], new_pos[2]] = True

            # Animal Movement and Eating
            if (organism.species.movement_type == Species.MovementType.AVIAN and 
                organism.injury_level < 0.3 and 
                not self.is_animal_sleeping(organism)):
                
                current_z = organism.position[2]
                # Random chance to change altitude
                if random.random() < 0.2:
                    new_z = current_z
                    if random.random() < 0.5 and current_z > 0:
                        new_z = current_z - 1  # Descend
                    elif current_z < self.max_z_level:
                        new_z = current_z + 1  # Ascend
                    
                    if new_z != current_z:
                        new_pos = (organism.position[0], organism.position[1], new_z)
                        if self.grid.move_organism(organism.id, new_pos):
                            organism.energy -= organism.species.move_energy_cost

            if (species.type != 0 and 
                not self.is_animal_sleeping(organism) and 
                organism.energy < species.max_energy * 0.9):
                if species.occupies_multiple_cells:
                    move_cost = species.move_energy_cost * size_factor
                    move_speed = max(1, int(species.move_speed / size_factor))

    
                food_options = self.grid.get_food_in_radius(pos, occupancy_tensor, species.move_speed)
                best_food = max(
                    food_options, 
                    key=lambda f: f[1].energy if f[1].id not in organisms_to_remove_ids else -1, 
                    default=None
                )
                
                if best_food:
                    food_pos, food_org = best_food
                    
                    # Calculate how much the animal can actually eat
                    max_consumable = min(
                        species.energy_gain,  # Can't eat more than their capacity
                        food_org.energy,  # Can't eat more than what's available
                        species.max_energy - organism.energy  # Can't eat beyond their max
                    )
                    
                    # Apply partial consumption
                    consumption_ratio = random.uniform(0.2, 0.8)  # Eat 20-80% of available
                    energy_gain = min(max_consumable * consumption_ratio, food_org.energy)
                    
                    # Dormancy effect
                    dormancy_mult = 0.2 if food_org.is_dormant and food_org.species.type == 0 else 1.0
                    energy_gain *= dormancy_mult
                    
                    # Try to move to food position
                    moved = False
                    if self.grid.move_organism(organism.id, food_pos):
                        organism.energy += energy_gain - species.move_energy_cost
                        occupancy_tensor[pos] = False
                        occupancy_tensor[food_pos] = True
                        moved = True
                    else:
                        organism.energy += energy_gain
                    
                    # Apply damage to food source
                    if food_org.species.type == 0:  # Plants
                        # Plants lose energy and gain injury
                        food_org.energy -= energy_gain
                        food_org.injury_level += consumption_ratio * 0.5  # Plants heal over time
                        if food_org.energy <= 0:
                            organisms_to_remove_ids.add(food_org.id)
                    else:  # Animals
                        # Animals take more severe damage
                        food_org.injury_level += consumption_ratio
                        food_org.energy -= energy_gain * 1.5  # More energy loss than gain due to inefficiency
                        
                        # Check if food organism dies from injury
                        if (food_org.injury_level >= 1.0 or 
                            food_org.energy <= 0 or 
                            (food_org.injury_level > 0.7 and random.random() < 0.3)):
                            food_org.is_dead = True
                            # Dead organisms become food sources but can't move/reproduce
                            food_org.species.can_move = False
                            food_org.species.has_roots = False
                else:  # No food, move randomly
                    empty_pos = self.grid.get_empty_spots_in_radius(pos, occupancy_tensor, species.move_speed)
                    if empty_pos:
                        new_pos = random.choice(empty_pos)
                        if self.grid.move_organism(organism.id, new_pos):
                            organism.energy -= species.move_energy_cost
                            occupancy_tensor[pos] = False
                            occupancy_tensor[new_pos] = True
        
        for oid in organisms_to_remove_ids: 
            self.grid.remove_organism(oid)
        
        # Add new seeds and organisms
        for seed in seeds_to_add: 
            self.grid.add_seed(seed)
        
        for org in organisms_from_seeds + organisms_to_add:
            try: 
                self.grid.add_organism(org)
            except ValueError: 
                pass
            
        if org.is_dead:
            # Dead organisms decay over time
            decay_rate = 0.1 + (0.4 * (self.temperature if self.is_day else self.temperature * 0.5))
            org.energy = max(0, org.energy - decay_rate)
            
            # When energy reaches 0, remove completely
            if org.energy <= 0:
                organisms_to_remove_ids.add(org.id)
                
        self.draw_grid()
    
    def clear_grid(self):
        """Clear the simulation grid"""
        self.grid = WorldGrid(self.grid_size)
        self.grid_tensor.zero_()
        self.current_step, self.current_day, self.is_day = 0, 0, True
        self.light_level, self.temperature = 1.0, 1.0
        dpg.set_value("time_text", "Day 1, 00:00")
        dpg.set_value("date_text", "Date: Jan 1")
        self.draw_grid()

    def randomize_grid(self):
        """Modified to properly place organisms based on movement type"""
        self.clear_grid()
        num_species = random.randint(5, 10)
        created_species = [GRASS]
        
        # Generate random species
        for _ in range(num_species - 1):
            new_species = Species.generate_random_species(food_sources=created_species)
            created_species.append(new_species)
            
        plant_species = [s for s in created_species if s.type == 0]
        animal_species = [s for s in created_species if s.type != 0]

        # Populate grid with proper z-levels
        for _ in range(self.grid_size * self.grid_size // 4):
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            
            # Choose z-level based on species type
            species = None
            if plant_species and (not animal_species or random.random() < 0.75):
                species = random.choice(plant_species)
                z = 0  # Plants always at ground level
            elif animal_species:
                species = random.choice(animal_species)
                if species.movement_type == Species.MovementType.TERRESTRIAL:
                    z = 0
                elif species.movement_type == Species.MovementType.AQUATIC:
                    z = random.randint(-3, 0)  # Underwater
                else:  # Avian
                    z = random.randint(0, 3)  # Can be in air
                    
            if species and not self.grid.get_organisms_at_position((x, y, z)):
                root_energy = random.uniform(0, species.root_energy) if species.type == 0 else 0
                organism = Organism(
                    species=species,
                    position=(x, y, z),
                    energy=random.uniform(species.offspring_energy, species.max_energy * 0.5),
                    root_energy=root_energy,
                    age=random.randint(0, species.mature_age * 2)
                )
                try:
                    self.grid.add_organism(organism)
                except ValueError as e:
                    print(f"Failed to add organism: {e}")
        
        self.draw_grid()
    
    def change_grid_size(self, sender, app_data):
        """Change the grid size"""
        new_size = app_data
        self.grid_size = new_size
        self.grid = WorldGrid(new_size)
        self.grid_tensor = torch.zeros((new_size, new_size, 3), dtype=torch.float32, device=DEVICE)
        dpg.configure_item("drawlist", width=self.grid_size*self.cell_size, height=self.grid_size*self.cell_size)
        self.draw_grid()
    
    def run(self):
        """Main simulation loop"""
        while dpg.is_dearpygui_running():
            if self.running:
                self.step()
                #time.sleep(0.1)  # Uncomment to slow down visualization
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

if __name__ == "__main__":
    game = GameOfLife()
    game.run()