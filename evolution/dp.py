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
    
    # --- NEW: Dormancy and Seed attributes ---
    dormancy_threshold: float = 0.3  # Temperature below which dormancy can occur
    seed_lifespan: int = 200          # How many days a seed can survive
    seed_maturity_age: int = 10       # Days until a seed can germinate
    seed_spread_radius: int = 3
    seed_germination_chance: float = 0.1

    # Movement and type-related
    can_move: bool = False
    move_speed: int = 0
    move_energy_cost: float = 0.0
    has_roots: bool = False
    root_energy: float = 0.0
    
    # Lifecycle
    max_age: int = 0  # 0 means no max age (for plants)
    sleep_ratio: float = 0.0
    diurnal: bool = True
    
    # Physical characteristics
    height: float = 1.0  # Relative height
    body_type: str = "generic"  # slim, bulky, etc.
    surface_covering: str = "skin"  # fur, scales, feathers, etc.
    covering_density: float = 0.5  # 0-1
    covering_color: tuple = (0, 0, 0)
    appendages: List[str] = field(default_factory=list)  # wings, fins, etc.
    
    # Diet preferences
    preferred_food_parts: List[str] = field(default_factory=list)
    toxic_food_parts: List[str] = field(default_factory=list)
    diet_type: float = 0.0  # 0=plant, 0.5=omnivore, 1=carnivore
    edible_parts: List[str] = field(default_factory=list) 
    
    # Cached type
    _type: Optional[int] = None
    
    # Name part databases
    _name_parts_db = {
        'prefix': ["zo", "ra", "fi", "lo", "pa", "ki", "tu", "ve", "no", "xi"],
        'suffix': ["oid", "ian", "us", "ix", "ae", "or", "ite", "oid", "ax", "is"],
        'plant_suffix': ["folia", "phyll", "herba", "flora", "verd", "chloro"]
    }
    
    # Physical part databases
    _body_parts_db = {
        'body_type': ["slim", "bulky", "streamlined", "segmented", "radial"],
        'surface_covering': ["fur", "scales", "feathers", "skin", "bark", "chitin"],
        'appendages': ["wings", "fins", "tentacles", "antennae", "claws", "hooves"]
    }
    
    # Food part databases
    _food_parts_db = {
        'plant': ["leaf", "fruit", "nut", "root", "stem", "flower"],
        'animal': ["meat", "organ", "bone", "marrow", "fat"]
    }
    
    def __post_init__(self):
        if not self.name and not self.name_parts:
            self._generate_random_name()
        if not self.name:
            self.name = "".join(part.capitalize() for part in self.name_parts)
        
        if not self.name:
            self.name = "Unnamed"
        
        if not self.appendages:
            self._generate_physical_characteristics()
        
        if not self.preferred_food_parts and self.type != 0:
            self._generate_diet_preferences()
        
        if not self.edible_parts:
            self._generate_edible_parts()

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
            
            if self.type in [2, 3]:  # Carnivore/Omnivore
                self.height = random.uniform(0.8, 2.0)
            else:  # Herbivore
                self.height = random.uniform(0.5, 1.5)
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
        new_species = Species(
            name_parts=self.name_parts.copy(),
            color=(
                max(0, min(255, int(self.color[0] * random.uniform(0.9, 1.1)))),
                max(0, min(255, int(self.color[1] * random.uniform(0.9, 1.1)))),
                max(0, min(255, int(self.color[2] * random.uniform(0.9, 1.1))))
            ),
            max_energy=max(1, self.max_energy * random.uniform(0.95, 1.05)),
            energy_gain=max(0.05, self.energy_gain * random.uniform(0.95, 1.05)),
            energy_consumption=max(0.01, self.energy_consumption * random.uniform(0.95, 1.05)),
            mature_age=max(5, int(self.mature_age * random.uniform(0.9, 1.1))),
            reproduction_cost=max(0.5, self.reproduction_cost * random.uniform(0.9, 1.1)),
            offspring_energy=max(0.5, self.offspring_energy * random.uniform(0.9, 1.1)),
            reproduction_frequency=max(1, int(self.reproduction_frequency * random.uniform(0.9, 1.1))),
            temperature_sensitivity=max(0.1, min(1.0, self.temperature_sensitivity * random.uniform(0.9, 1.1))),
            light_sensitivity=max(0.1, min(1.0, self.light_sensitivity * random.uniform(0.9, 1.1))),
            cold_resistance=max(0.1, min(1.0, self.cold_resistance * random.uniform(0.9, 1.1))),
            
            # --- MODIFIED: Mutate new attributes ---
            dormancy_threshold=max(0.1, min(0.6, self.dormancy_threshold * random.uniform(0.95, 1.05))),
            seed_lifespan=max(10, int(self.seed_lifespan * random.uniform(0.9, 1.1))) if self.has_roots else 0,
            seed_maturity_age=max(1, int(self.seed_maturity_age * random.uniform(0.9, 1.1))) if self.has_roots else 0,
            seed_spread_radius=max(1, min(10, int(self.seed_spread_radius * random.uniform(0.9, 1.1)))) if self.has_roots else 0,
            seed_germination_chance=max(0.01, min(0.5, self.seed_germination_chance * random.uniform(0.95, 1.05))) if self.has_roots else 0,

            can_move=self.can_move,
            move_speed=max(0, int(self.move_speed * random.uniform(0.9, 1.1)) if self.can_move else 0),
            move_energy_cost=max(0, self.move_energy_cost * random.uniform(0.9, 1.1) if self.can_move else 0),
            has_roots=self.has_roots,
            root_energy=max(0, self.root_energy * random.uniform(0.9, 1.1)) if self.has_roots else 0,
            max_age=max(0, int(self.max_age * random.uniform(0.9, 1.1)) if self.max_age > 0 else 0),
            sleep_ratio=max(0, min(0.5, self.sleep_ratio * random.uniform(0.9, 1.1))) if self.can_move else 0,
            diurnal=self.diurnal if random.random() < 0.9 else not self.diurnal,
            height=max(0.1, self.height * random.uniform(0.95, 1.05)),
            body_type=self.body_type,
            surface_covering=self.surface_covering,
            covering_density=max(0, min(1, self.covering_density * random.uniform(0.95, 1.05))),
            covering_color=(
                max(0, min(255, int(self.covering_color[0] * random.uniform(0.9, 1.1)))),
                max(0, min(255, int(self.covering_color[1] * random.uniform(0.9, 1.1)))),
                max(0, min(255, int(self.covering_color[2] * random.uniform(0.9, 1.1))))
            ),
            appendages=self.appendages.copy(),
            preferred_food_parts=self.preferred_food_parts.copy(),
            toxic_food_parts=self.toxic_food_parts.copy(),
            diet_type=max(0, min(1, self.diet_type * random.uniform(0.95, 1.05))),
            edible_parts=self.edible_parts.copy()
        )
        
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
        
        new_species._generate_random_name()
        return new_species

    @classmethod
    def generate_random_species(cls, food_sources: Optional[List['Species']] = None) -> 'Species':
        species = cls()
        
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
        
        is_plant = random.random() < (0.3 if food_sources else 0.5)
        species.can_move = not is_plant
        species.has_roots = is_plant

        # --- MODIFIED: Set new attributes during generation ---
        species.dormancy_threshold = random.uniform(0.2, 0.4)
        
        if is_plant:
            species.root_energy = random.uniform(5, 20)
            species.color = (random.randint(50, 150), random.randint(100, 200), random.randint(50, 150))
            species.seed_lifespan = random.randint(100, 400)
            species.seed_maturity_age = random.randint(5, 25)
            species.seed_spread_radius = random.randint(2, 6)
            species.seed_germination_chance = random.uniform(0.05, 0.2)
        else:
            species.move_speed = random.randint(1, 3)
            species.move_energy_cost = random.uniform(0.1, 0.5)
            species.max_age = random.randint(100, 1000)
            species.sleep_ratio = random.uniform(0.1, 0.3)
            species.diurnal = random.random() < 0.7
            species.seed_lifespan = 0
            species.seed_maturity_age = 0
            species.seed_spread_radius = 0
            species.seed_germination_chance = 0.0

            if food_sources:
                plant_parts = list(set(p for s in food_sources if s.type == 0 for p in s.edible_parts))
                animal_parts = list(set(p for s in food_sources if s.type != 0 for p in s.edible_parts))
                
                can_eat_plants = bool(plant_parts)
                can_eat_animals = bool(animal_parts)
                
                choice = 'fallback'
                if can_eat_plants and can_eat_animals:
                    choice = random.choices(['herbivore', 'carnivore', 'omnivore'], weights=[25, 25, 50], k=1)[0]
                elif can_eat_plants:
                    choice = 'herbivore'
                elif can_eat_animals:
                    choice = 'carnivore'
                
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
                    species.preferred_food_parts = (random.sample(plant_parts, p_num) + random.sample(animal_parts, a_num))
            
            if not species.preferred_food_parts:
                species.diet_type = random.random()
                species._generate_diet_preferences()

            if species.diet_type < 0.3:  # Herbivore
                species.color = (random.randint(150, 255), random.randint(150, 255), random.randint(100, 200))
            elif species.diet_type > 0.7:  # Carnivore
                species.color = (random.randint(200, 255), random.randint(100, 150), random.randint(100, 150))
            else:  # Omnivore
                species.color = (random.randint(150, 255), random.randint(150, 200), random.randint(100, 150))

        # --- 4. Finalize with helper methods ---
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

# --- MODIFIED: Organism has dormancy state, new Seed class ---
@dataclass
class Organism:
    species: Species
    position: Tuple[int, int, int]  # (x, y, z)
    energy: float = 0
    root_energy: float = 0  # Only used for plants
    age: float = 0
    is_dormant: bool = False
    id: int = field(default_factory=lambda: random.getrandbits(64))

@dataclass
class Seed:
    species: Species
    position: Tuple[int, int, int]
    age: int = 0
    id: int = field(default_factory=lambda: random.getrandbits(64))


class WorldGrid:
    def __init__(self, size: int):
        self.size = size
        self.organisms: Dict[Tuple[int, int, int], Organism] = {}
        self.organism_ids: Dict[int, Tuple[int, int, int]] = {}

        # --- MODIFIED: Seeds are now a list per tile and don't block tiles ---
        self.seeds: Dict[Tuple[int, int, int], List[Seed]] = {}
        self.seed_ids: Dict[int, Tuple[int, int, int]] = {}
        
        # Pre-compute neighbor offsets as torch tensor
        neighbor_list = [(dx, dy, dz) 
                        for dx in range(-1, 2)
                        for dy in range(-1, 2)
                        for dz in range(-1, 2)
                        if not (dx == 0 and dy == 0 and dz == 0)]
        self.neighbor_offsets = torch.tensor(neighbor_list, dtype=torch.int32, device=DEVICE)
    
    def add_organism(self, organism: Organism):
        pos = organism.position
        if not self.is_valid_position(pos):
            raise ValueError(f"Invalid position {pos}")
            
        if pos in self.organisms:
            raise ValueError(f"Position {pos} already occupied by an organism")
            
        self.organisms[pos] = organism
        self.organism_ids[organism.id] = pos

    def add_seed(self, seed: Seed):
        pos = seed.position
        if not self.is_valid_position(pos):
            raise ValueError(f"Invalid position for seed {pos}")

        if pos not in self.seeds:
            self.seeds[pos] = []
        self.seeds[pos].append(seed)
        self.seed_ids[seed.id] = pos
        
    def remove_organism(self, organism_id: int):
        if organism_id not in self.organism_ids:
            return
            
        pos = self.organism_ids[organism_id]
        if pos in self.organisms:
            del self.organisms[pos]
        del self.organism_ids[organism_id]

    def remove_seed(self, seed_id: int):
        if seed_id not in self.seed_ids:
            return

        pos = self.seed_ids[seed_id]
        if pos in self.seeds:
            self.seeds[pos] = [s for s in self.seeds[pos] if s.id != seed_id]
            if not self.seeds[pos]: # If list is empty, remove the key
                del self.seeds[pos]
        del self.seed_ids[seed_id]
        
    def move_organism(self, organism_id: int, new_position: Tuple[int, int, int]):
        if organism_id not in self.organism_ids:
            return False
            
        if not self.is_valid_position(new_position):
            return False
            
        if new_position in self.organisms: # Only check for other organisms
            return False
            
        old_pos = self.organism_ids[organism_id]
        organism = self.organisms[old_pos]
        
        del self.organisms[old_pos]
        organism.position = new_position
        self.organisms[new_position] = organism
        self.organism_ids[organism.id] = new_position
        return True
        
    def get_organism(self, position: Tuple[int, int, int]) -> Optional[Organism]:
        return self.organisms.get(position)
        
    def is_valid_position(self, position: Tuple[int, int, int]) -> bool:
        x, y, z = position
        return (0 <= x < self.size and 
                0 <= y < self.size and 
                0 <= z < self.size)
    
    # --- MODIFIED: Renamed and now finds empty spots in a variable radius ---
    def get_empty_spots_in_radius(self, position: Tuple[int, int, int], occupancy_tensor: torch.Tensor, radius: int = 1) -> List[Tuple[int, int, int]]:
        x, y, z = position
        
        x_r = torch.arange(max(0, x - radius), min(self.size, x + radius + 1), device=DEVICE)
        y_r = torch.arange(max(0, y - radius), min(self.size, y + radius + 1), device=DEVICE)
        z_r = torch.arange(max(0, z - radius), min(self.size, z + radius + 1), device=DEVICE)
        
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
        
        x_range = torch.arange(max(0, x - radius), min(self.size, x + radius + 1), device=DEVICE)
        y_range = torch.arange(max(0, y - radius), min(self.size, y + radius + 1), device=DEVICE)
        z_range = torch.arange(max(0, z - radius), min(self.size, z + radius + 1), device=DEVICE)
        
        xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
        positions = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        
        center_mask = ~((positions[:, 0] == x) & (positions[:, 1] == y) & (positions[:, 2] == z))
        positions = positions[center_mask]

        if positions.shape[0] == 0:
            return []
            
        coords = positions.long()
        is_occupied = occupancy_tensor[coords[:, 0], coords[:, 1], coords[:, 2]]
        food_positions_tensor = positions[is_occupied]

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
        
        self.steps_per_day = 48
        self.current_step, self.current_day = 0, 0
        self.summer_solstice, self.winter_solstice = 172, 355
        self.year_length = 365
        self.max_daylight, self.min_daylight, self.current_daylight = 32, 16, 17
        self.light_level, self.temperature, self.is_day = 1.0, 1.0, True
        
        self.day_angles = torch.linspace(0, torch.pi, self.steps_per_day, device=DEVICE)
        self.season_angles = torch.linspace(0, 2*torch.pi, self.year_length, device=DEVICE)
        
        dpg.create_context()
        self.setup_ui()
        dpg.create_viewport(title="Evolution Simulation", width=850, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def setup_ui(self):
        with dpg.window(tag="Main Window"):
            with dpg.group(horizontal=True):
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
                
        dpg.set_viewport_resize_callback(self.on_viewport_resize)
        dpg.set_primary_window("Main Window", True)
        self.draw_grid()
    
    def on_viewport_resize(self):
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dpg.set_item_width("Main Window", viewport_width)
        dpg.set_item_height("Main Window", viewport_height)
        
        if viewport_width > 300:
            dpg.set_item_width("game_window", viewport_width - 320)
    
    def update_time_cycles(self):
        self.current_step += 1
        
        if self.current_step >= self.steps_per_day:
            self.current_step = 0
            self.current_day = (self.current_day + 1) % self.year_length
            self.update_daylight_duration()
        
        self.is_day = self.current_step < self.current_daylight
        
        if self.is_day:
            progress = self.current_step / self.current_daylight
            angle = self.day_angles[int(progress * (len(self.day_angles)-1))]
            self.light_level = torch.sin(angle).item() * 0.9 + 0.1
        else:
            self.light_level = 0.05
        
        season_angle = self.season_angles[self.current_day]
        season_temp = 0.5 + 0.4 * torch.sin(season_angle).item()
        
        if self.is_day:
            day_progress = self.current_step / self.current_daylight
            day_angle = self.day_angles[int(day_progress * (len(self.day_angles)-1))]
            daily_temp = 0.8 + 0.2 * torch.sin(day_angle).item()
        else:
            daily_temp = 0.6
        
        self.temperature = season_temp * daily_temp
        
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
        months = [("Jan", 31), ("Feb", 28), ("Mar", 31), ("Apr", 30), ("May", 31), ("Jun", 30), ("Jul", 31), ("Aug", 31), ("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31)]
        for month, days in months:
            if day < days: return month, day + 1
            day -= days
        return "Dec", 31
    
    def update_daylight_duration(self):
        days_since_winter = (self.current_day - self.winter_solstice) % self.year_length
        year_progress = days_since_winter / self.year_length
        daylight_hours = 12 + 4 * torch.cos(torch.tensor(year_progress * math.pi, device=DEVICE)).item()
        self.current_daylight = int(round(daylight_hours * 2))

    def get_species_color(self, organism: Organism) -> Tuple[int, int, int]:
        species, energy, is_dormant = organism.species, organism.energy, organism.is_dormant
        base_color = torch.tensor(species.color, dtype=torch.float32, device=DEVICE)
        
        if species.type == 0:
            color = base_color * torch.tensor([self.temperature, self.light_level, self.temperature], device=DEVICE)
            if not self.is_day: color *= 0.3
        else:
            energy_ratio = energy / species.max_energy
            color = base_color * torch.tensor([energy_ratio, energy_ratio * self.temperature, energy_ratio], device=DEVICE)
        
        if is_dormant: color *= 0.4
        elif self.is_animal_sleeping(organism): color *= 0.6
        
        return tuple(torch.clamp(color, 0, 255).to(torch.uint8).cpu().numpy())
    
    def get_species_char(self, organism: Organism) -> str:
        if organism.species and organism.species.name:
            if organism.is_dormant: return "z"
            return organism.species.name[0].upper()
        return "?"
    
    def draw_grid(self):
        dpg.delete_item("drawlist", children_only=True)
        self.grid_tensor.zero_()
        
        with dpg.draw_node(parent="drawlist"):
            for pos, organism in self.grid.organisms.items():
                x, y, z = pos
                if z == 0:
                    color = self.get_species_color(organism)
                    self.grid_tensor[x, y] = torch.tensor(color, device=DEVICE) / 255.0
            
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    color = self.grid_tensor[x, y] * 255
                    organism = self.grid.get_organism((x, y, 0))
                    char = self.get_species_char(organism) if organism else " "
                    dpg.draw_rectangle((x * self.cell_size, y * self.cell_size), ((x + 1) * self.cell_size, (y + 1) * self.cell_size), fill=(int(color[0]), int(color[1]), int(color[2])), color=(100, 100, 100), thickness=1)
                    dpg.draw_text((x*self.cell_size + self.cell_size//4, y*self.cell_size), char, color=(0,0,0), size=int(self.cell_size*0.8))
            
            # --- MODIFIED: Draw seeds on top ---
            for pos, seed_list in self.grid.seeds.items():
                if seed_list:
                    x, y, z = pos
                    if z == 0:
                        dpg.draw_circle((x*self.cell_size + self.cell_size/2, y*self.cell_size + self.cell_size/2), radius=max(1, self.cell_size/5), color=(139, 69, 19), fill=(139, 69, 19))

            for i in range(self.grid_size + 1):
                dpg.draw_line((i*self.cell_size, 0), (i*self.cell_size, self.grid_size*self.cell_size), color=(50, 50, 50, 100), thickness=1)
                dpg.draw_line((0, i*self.cell_size), (self.grid_size*self.cell_size, i*self.cell_size), color=(50, 50, 50, 100), thickness=1)
    
    def set_running(self, running):
        self.running = running
    
    def is_animal_sleeping(self, organism: Organism) -> bool:
        species = organism.species
        if species.type == 0: return False
        if organism.is_dormant: return True # Dormant animals are always "sleeping"
        
        wrong_time = (species.diurnal and not self.is_day) or (not species.diurnal and self.is_day)
        return wrong_time or random.random() < species.sleep_ratio
        
    def step(self):
        self.update_time_cycles()
        
        organisms = list(self.grid.organisms.values())
        seeds_list = [seed for pos_list in self.grid.seeds.values() for seed in pos_list]
        num_organisms = len(organisms)
        if num_organisms < 1 and not seeds_list:
            self.set_running(False); self.draw_grid(); return

        # --- 0. Seed Processing (once per day) ---
        organisms_from_seeds = []
        if self.current_step == 0 and self.grid.seeds:
            # Use a copy of the keys to allow safe modification during iteration
            positions_with_seeds = list(self.grid.seeds.keys())
            
            for pos in positions_with_seeds:
                # If the position somehow got cleared, skip
                if pos not in self.grid.seeds:
                    continue

                surviving_seeds_at_pos = []
                seeds_at_pos = self.grid.seeds[pos]

                for seed in seeds_at_pos:
                    seed.age += 1
                    
                    # Check for removal by old age
                    if seed.age > seed.species.seed_lifespan:
                        # This seed dies, don't add it to the surviving list
                        del self.grid.seed_ids[seed.id]
                        continue

                    # Check for germination
                    # Note: We check grid.organisms directly, not the potentially stale occupancy_tensor
                    if (seed.age >= seed.species.seed_maturity_age and
                        pos not in self.grid.organisms and 
                        self.temperature > seed.species.dormancy_threshold and 
                        random.random() < seed.species.seed_germination_chance):
                        
                        species = seed.species
                        new_organism = Organism(species=species, position=pos, energy=species.offspring_energy,
                                                root_energy=species.offspring_energy*0.5, age=0)
                        organisms_from_seeds.append(new_organism)
                        
                        # The seed is used up, so remove it and break the loop for this tile
                        # as an organism will now occupy it.
                        del self.grid.seed_ids[seed.id]
                        # Any other seeds at this location this turn can't germinate.
                        # We can keep them for next turn.
                        surviving_seeds_at_pos.extend(s for s in seeds_at_pos if s.id != seed.id)
                        break # Exit the inner for-loop for this position

                    else:
                        # Seed survives and doesn't germinate
                        surviving_seeds_at_pos.append(seed)
                else: # This 'else' belongs to the 'for seed in ...' loop, runs if no 'break'
                    # Update the list of seeds for the position
                    if surviving_seeds_at_pos:
                        self.grid.seeds[pos] = surviving_seeds_at_pos
                    else:
                        # If no seeds survived, remove the entry from the dictionary
                        del self.grid.seeds[pos]
        for org in organisms:
            org.is_dormant = self.temperature < org.species.dormancy_threshold

        species_list = [org.species for org in organisms]
        is_dormant_tensor = torch.tensor([org.is_dormant for org in organisms], dtype=torch.bool, device=DEVICE)
        energies = torch.tensor([org.energy for org in organisms], dtype=torch.float32, device=DEVICE)
        root_energies = torch.tensor([org.root_energy for org in organisms], dtype=torch.float32, device=DEVICE)
        ages = torch.tensor([org.age for org in organisms], dtype=torch.float32, device=DEVICE)
        types = torch.tensor([s.type for s in species_list], dtype=torch.int8, device=DEVICE)
        max_energies = torch.tensor([s.max_energy for s in species_list], dtype=torch.float32, device=DEVICE)
        energy_consumptions = torch.tensor([s.energy_consumption for s in species_list], dtype=torch.float32, device=DEVICE)
        energy_consumptions[is_dormant_tensor] *= 0.1 # Dormant organisms use 10% energy

        is_plant, is_animal = (types == 0), (types != 0)
        to_remove = torch.zeros(num_organisms, dtype=torch.bool, device=DEVICE)
        if self.current_step == 0: ages += 1.0

        # --- 2. Plant Processing ---
        if torch.any(is_plant):
            plant_indices = torch.where(is_plant)[0]
            light_sens = torch.tensor([s.light_sensitivity for s in species_list if s.type==0], device=DEVICE)
            temp_sens = torch.tensor([s.temperature_sensitivity for s in species_list if s.type==0], device=DEVICE)
            s_energy_gain = torch.tensor([s.energy_gain for s in species_list if s.type==0], device=DEVICE)
            s_root_energy_cap = torch.tensor([s.root_energy for s in species_list if s.type==0], device=DEVICE)
            
            energy_gain = (self.light_level * light_sens * s_energy_gain * (self.temperature * temp_sens))
            energy_gain[is_dormant_tensor[is_plant]] *= 0.1 # Dormant plants get 10% energy
            
            energies[is_plant] += energy_gain - energy_consumptions[is_plant]
            excess_mask = energies[is_plant] > max_energies[is_plant]
            storage_amount = torch.zeros_like(energies[is_plant])
            storage_amount[excess_mask] = torch.minimum(energies[plant_indices[excess_mask]] - max_energies[plant_indices[excess_mask]], s_root_energy_cap[excess_mask] - root_energies[plant_indices[excess_mask]])
            energies[is_plant] = torch.min(energies[is_plant], max_energies[is_plant])
            root_energies[is_plant] += storage_amount
            
            deficit_mask = energies[is_plant] < 0
            if torch.any(deficit_mask):
                energy_needed = -energies[plant_indices[deficit_mask]]
                root_draw = torch.minimum(energy_needed, root_energies[plant_indices[deficit_mask]])
                energies[plant_indices[deficit_mask]] += root_draw
                root_energies[plant_indices[deficit_mask]] -= root_draw
            to_remove[is_plant] = energies[is_plant] < 0

        # --- 3. Animal Processing ---
        if torch.any(is_animal):
            energies[is_animal] -= energy_consumptions[is_animal]
            animal_indices = torch.where(is_animal)[0]
            max_ages = torch.tensor([s.max_age for s in species_list if s.type != 0], dtype=torch.float32, device=DEVICE)
            cold_resistances = torch.tensor([s.cold_resistance for s in species_list if s.type != 0], device=DEVICE)
            coverings = torch.tensor([1.2 if s.surface_covering == "fur" else 1.0 for s in species_list if s.type != 0], device=DEVICE)
            survival_chances = torch.clamp((self.temperature + cold_resistances) * coverings, 0.0, 1.0)
            survival_chances[is_dormant_tensor[is_animal]] = torch.clamp(survival_chances[is_dormant_tensor[is_animal]] + 0.5, 0.0, 1.0)
            dies_from_conditions = torch.rand(len(animal_indices), device=DEVICE) > survival_chances
            to_remove[is_animal] = (energies[is_animal] <= 0) | ((ages[is_animal] > max_ages) & (max_ages > 0)) | dies_from_conditions
        
        # --- 4. Apply State Changes ---
        organisms_to_remove_ids, survivors = set(), []
        for i, org in enumerate(organisms):
            if to_remove[i]: organisms_to_remove_ids.add(org.id)
            else:
                org.energy, org.root_energy, org.age = energies[i].item(), root_energies[i].item(), ages[i].item()
                survivors.append(org)
        
        # --- 5. Actions Loop ---
        organisms_to_add, seeds_to_add = [], []
        occupancy_tensor = torch.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=torch.bool, device=DEVICE)
        if self.grid.organisms:
            pos_tensor = torch.tensor(list(self.grid.organisms.keys()), dtype=torch.long, device=DEVICE)
            occupancy_tensor[pos_tensor[:, 0], pos_tensor[:, 1], pos_tensor[:, 2]] = True

        for organism in survivors:
            if organism.is_dormant: continue
            species, pos = organism.species, organism.position

            # Reproduction
            if (organism.energy >= species.reproduction_cost and organism.age >= species.mature_age and self.current_day % species.reproduction_frequency == 0):
                if species.type == 0: # Plant -> Seeds
                    empty_spots = self.grid.get_empty_spots_in_radius(pos, occupancy_tensor, radius=species.seed_spread_radius)
                    if empty_spots:
                        num_seeds = random.randint(2, 5)
                        for new_pos in random.sample(empty_spots, min(len(empty_spots), num_seeds)):
                            new_species = species.mutate() if random.random() < 0.01 else species
                            seeds_to_add.append(Seed(species=new_species, position=new_pos))
                        organism.energy -= species.reproduction_cost
                else: # Animal -> Live offspring
                    empty_neighbors = self.grid.get_empty_spots_in_radius(pos, occupancy_tensor, radius=1)
                    if empty_neighbors:
                        new_pos = random.choice(empty_neighbors)
                        offspring = Organism(species=species.mutate() if random.random()<0.05 else species, position=new_pos, energy=species.offspring_energy)
                        organisms_to_add.append(offspring); organism.energy -= species.reproduction_cost
                        occupancy_tensor[new_pos[0], new_pos[1], new_pos[2]] = True

            # Animal Movement and Eating
            if species.type != 0 and not self.is_animal_sleeping(organism) and organism.energy < species.max_energy * 0.9:
                food_options = self.grid.get_food_in_radius(pos, occupancy_tensor, species.move_speed)
                best_food = max(food_options, key=lambda f: f[1].energy if f[1].id not in organisms_to_remove_ids else -1, default=None)
                if best_food:
                    food_pos, food_org = best_food
                    dormancy_mult = 0.2 if food_org.is_dormant and food_org.species.type == 0 else 1.0
                    energy_gain = min(species.energy_gain, food_org.energy * dormancy_mult)
                    self.grid.remove_organism(food_org.id)
                    if self.grid.move_organism(organism.id, food_pos):
                        organism.energy += energy_gain - species.move_energy_cost
                        occupancy_tensor[pos] = False; occupancy_tensor[food_pos] = True
                    else:
                        organism.energy += energy_gain; organisms_to_remove_ids.add(food_org.id)
                else: # No food, move randomly
                    empty_pos = self.grid.get_empty_spots_in_radius(pos, occupancy_tensor, species.move_speed)
                    if empty_pos:
                        new_pos = random.choice(empty_pos)
                        if self.grid.move_organism(organism.id, new_pos):
                            organism.energy -= species.move_energy_cost
                            occupancy_tensor[pos] = False; occupancy_tensor[new_pos] = True
        
        # --- 6. Apply Grid Changes ---
        for oid in organisms_to_remove_ids: self.grid.remove_organism(oid)
        for seed in seeds_to_add: self.grid.add_seed(seed)
        for org in organisms_from_seeds + organisms_to_add:
            try: self.grid.add_organism(org)
            except ValueError: pass
        self.draw_grid()
    
    def clear_grid(self):
        self.grid = WorldGrid(self.grid_size)
        self.grid_tensor.zero_()
        self.current_step, self.current_day, self.is_day = 0, 0, True
        self.light_level, self.temperature = 1.0, 1.0
        dpg.set_value("time_text", "Day 1, 00:00")
        dpg.set_value("date_text", "Date: Jan 1")
        self.draw_grid()

    def randomize_grid(self):
        self.clear_grid()
        num_species = random.randint(5, 10)
        created_species = [GRASS]
        
        for _ in range(num_species - 1):
            new_species = Species.generate_random_species(food_sources=created_species)
            created_species.append(new_species)
            
        plant_species = [s for s in created_species if s.type == 0]
        animal_species = [s for s in created_species if s.type != 0]

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in [0]:
                    if random.random() < 0.5:
                        
                        species = None
                        if plant_species and (not animal_species or random.random() < 0.75):
                            species = random.choice(plant_species)
                        elif animal_species:
                            species = random.choice(animal_species)

                        if species:
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
                            except ValueError:
                                pass
        self.draw_grid()
    
    def change_grid_size(self, sender, app_data):
        new_size = app_data
        self.grid_size = new_size
        self.grid = WorldGrid(new_size)
        self.grid_tensor = torch.zeros((new_size, new_size, 3), dtype=torch.float32, device=DEVICE)
        dpg.configure_item("drawlist", width=self.grid_size*self.cell_size, height=self.grid_size*self.cell_size)
        self.draw_grid()
    
    def run(self):
        while dpg.is_dearpygui_running():
            if self.running:
                self.step()
                #time.sleep(0.1) for visualization. do not remove
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

if __name__ == "__main__":
    game = GameOfLife()
    game.run()