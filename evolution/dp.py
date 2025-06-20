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
        
        if not self.preferred_food_parts or not self.toxic_food_parts:
            self._generate_diet_preferences()
    
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
    
    def _generate_diet_preferences(self):
        if self.type == 0:  # Plants don't eat
            return
        
        if self.type == 1:  # Herbivore
            self.diet_type = random.uniform(0.0, 0.3)
            num_preferred = random.randint(1, 3)
            self.preferred_food_parts = random.sample(self._food_parts_db['plant'], num_preferred)
            
        elif self.type == 2:  # Carnivore
            self.diet_type = random.uniform(0.7, 1.0)
            num_preferred = random.randint(1, 2)
            self.preferred_food_parts = random.sample(self._food_parts_db['animal'], num_preferred)
            
        else:  # Omnivore
            self.diet_type = random.uniform(0.3, 0.7)
            plant_parts = random.sample(self._food_parts_db['plant'], random.randint(1, 2))
            animal_parts = random.sample(self._food_parts_db['animal'], random.randint(1, 2))
            self.preferred_food_parts = plant_parts + animal_parts
        
        all_parts = self._food_parts_db['plant'] + self._food_parts_db['animal']
        self.toxic_food_parts = [part for part in all_parts if random.random() < 0.2]
    
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
            diet_type=max(0, min(1, self.diet_type * random.uniform(0.95, 1.05)))
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
    def generate_random_species(cls) -> 'Species':
        is_plant = random.random() < 0.5
        
        species = cls()
        species.can_move = not is_plant
        species.has_roots = is_plant
        
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
        
        if is_plant:
            species.root_energy = random.uniform(5, 20)
            species.color = (random.randint(50, 150), random.randint(100, 200), random.randint(50, 150))
        else:
            species.move_speed = random.randint(1, 3)
            species.move_energy_cost = random.uniform(0.1, 0.5)
            species.max_age = random.randint(100, 1000)
            species.sleep_ratio = random.uniform(0.1, 0.3)
            species.diurnal = random.random() < 0.7
            
            species.diet_type = random.random()
            if species.diet_type < 0.3:  # Herbivore
                species.color = (random.randint(150, 255), random.randint(150, 255), random.randint(100, 200))
            elif species.diet_type > 0.7:  # Carnivore
                species.color = (random.randint(200, 255), random.randint(100, 150), random.randint(100, 150))
            else:  # Omnivore
                species.color = (random.randint(150, 255), random.randint(150, 200), random.randint(100, 150))
        
        species._generate_random_name()
        species._generate_physical_characteristics()
        species._generate_diet_preferences()
        
        return species

# Default species examples
GRASS = Species(
    name_parts=["gra", "ss"],
    color=(100, 200, 100),
    max_energy=15,
    energy_gain=0.3,
    energy_consumption=0.05,
    mature_age=20,
    reproduction_cost=2,
    offspring_energy=1.5,
    reproduction_frequency=5,
    temperature_sensitivity=0.8,
    light_sensitivity=0.9,
    cold_resistance=0.7,
    can_move=False,
    has_roots=True,
    root_energy=8,
    height=0.3,
    body_type="radial",
    surface_covering="skin",
    covering_density=0.8
)

OMNIVORE = Species(
    name_parts=["om", "niv", "ore"],
    color=(200, 150, 100),
    max_energy=25,
    energy_gain=2.0,
    energy_consumption=0.2,
    mature_age=100,
    reproduction_cost=5,
    offspring_energy=3,
    reproduction_frequency=10,
    temperature_sensitivity=0.6,
    light_sensitivity=0.3,
    cold_resistance=0.5,
    can_move=True,
    move_speed=2,
    move_energy_cost=0.2,
    max_age=500,
    sleep_ratio=0.2,
    diurnal=True,
    height=1.2,
    body_type="slim",
    surface_covering="fur",
    covering_density=0.6,
    appendages=["claws"],
    diet_type=0.5,
    preferred_food_parts=["fruit", "meat"],
    toxic_food_parts=["root"]
)

@dataclass
class Organism:
    species: Species
    position: Tuple[int, int, int]  # (x, y, z)
    energy: float = 0
    root_energy: float = 0  # Only used for plants
    age: float = 0
    id: int = field(default_factory=lambda: random.getrandbits(64))

class WorldGrid:
    def __init__(self, size: int):
        self.size = size
        self.organisms: Dict[Tuple[int, int, int], Organism] = {}
        self.organism_ids: Dict[int, Tuple[int, int, int]] = {}
        
        # Pre-compute neighbor offsets for faster access
        self.neighbor_offsets = np.array([(dx, dy, dz) 
                                        for dx in range(-1, 2)
                                        for dy in range(-1, 2)
                                        for dz in range(-1, 2)
                                        if not (dx == 0 and dy == 0 and dz == 0)], dtype=np.int32)
        
    def add_organism(self, organism: Organism):
        pos = organism.position
        if not self.is_valid_position(pos):
            raise ValueError(f"Invalid position {pos}")
            
        if pos in self.organisms:
            raise ValueError(f"Position {pos} already occupied")
            
        self.organisms[pos] = organism
        self.organism_ids[organism.id] = pos
        
    def remove_organism(self, organism_id: int):
        if organism_id not in self.organism_ids:
            return
            
        pos = self.organism_ids[organism_id]
        del self.organisms[pos]
        del self.organism_ids[organism_id]
        
    def move_organism(self, organism_id: int, new_position: Tuple[int, int, int]):
        if organism_id not in self.organism_ids:
            return False
            
        if not self.is_valid_position(new_position):
            return False
            
        if new_position in self.organisms:
            return False
            
        old_pos = self.organism_ids[organism_id]
        organism = self.organisms[old_pos]
        
        del self.organisms[old_pos]
        organism.position = new_position
        self.organisms[new_position] = organism
        self.organism_ids[organism_id] = new_position
        return True
        
    def get_organism(self, position: Tuple[int, int, int]) -> Optional[Organism]:
        return self.organisms.get(position)
        
    def is_valid_position(self, position: Tuple[int, int, int]) -> bool:
        x, y, z = position
        return (0 <= x < self.size and 
                0 <= y < self.size and 
                0 <= z < self.size)
    
    def get_empty_neighbors(self, position: Tuple[int, int, int], radius: int = 1) -> List[Tuple[int, int, int]]:
        x, y, z = position
        neighbors = []
        
        # Vectorized neighbor checking
        offsets = self.neighbor_offsets * radius
        neighbor_positions = np.array([x, y, z]) + offsets
        
        # Filter valid positions
        valid_mask = (
            (neighbor_positions[:, 0] >= 0) & (neighbor_positions[:, 0] < self.size) &
            (neighbor_positions[:, 1] >= 0) & (neighbor_positions[:, 1] < self.size) &
            (neighbor_positions[:, 2] >= 0) & (neighbor_positions[:, 2] < self.size)
        )
        
        valid_positions = neighbor_positions[valid_mask]
        
        # Convert to tuples and check if empty
        for pos in valid_positions:
            pos_tuple = tuple(pos)
            if pos_tuple not in self.organisms:
                neighbors.append(pos_tuple)
                
        return neighbors
    
    def get_food_in_radius(self, position: Tuple[int, int, int], radius: int) -> List[Tuple[Tuple[int, int, int], Organism]]:
        x, y, z = position
        food = []
        
        # Create a grid of positions to check
        x_range = np.arange(max(0, x - radius), min(self.size, x + radius + 1))
        y_range = np.arange(max(0, y - radius), min(self.size, y + radius + 1))
        z_range = np.arange(max(0, z - radius), min(self.size, z + radius + 1))
        
        # Create meshgrid of positions
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        positions = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)
        
        # Remove the center position
        center_mask = ~((positions[:, 0] == x) & (positions[:, 1] == y) & (positions[:, 2] == z))
        positions = positions[center_mask]
        
        # Check each position
        for pos in positions:
            pos_tuple = tuple(pos)
            if pos_tuple in self.organisms:
                food.append((pos_tuple, self.organisms[pos_tuple]))
                
        return food

class GameOfLife:
    def __init__(self):
        self.grid_size = 50
        self.cell_size = 10
        self.running = False
        
        # Initialize grid with separate organism tracking
        self.grid = WorldGrid(self.grid_size)
        self.grid_tensor = torch.zeros((self.grid_size, self.grid_size, 3), dtype=torch.float32, device=DEVICE)
        
        # Time parameters
        self.steps_per_day = 48
        self.current_step = 0
        self.current_day = 0
        self.summer_solstice = 172
        self.winter_solstice = 355
        self.year_length = 365
        self.max_daylight = 32
        self.min_daylight = 16
        self.current_daylight = 17
        self.light_level = 1.0
        self.temperature = 1.0
        self.is_day = True
        
        # Precompute trigonometric values for performance
        self.day_angles = np.linspace(0, np.pi, self.steps_per_day)
        self.season_angles = np.linspace(0, 2*np.pi, self.year_length)
        
        # Initialize DPG
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
        
        if viewport_width > 200:
            dpg.set_item_width("game_window", viewport_width - 200)
    
    def update_time_cycles(self):
        self.current_step += 1
        
        if self.current_step >= self.steps_per_day:
            self.current_step = 0
            self.current_day = (self.current_day + 1) % self.year_length
            self.update_daylight_duration()
            
            # Batch age updates using numpy
            positions = np.array(list(self.grid.organisms.keys()))
            organisms = np.array(list(self.grid.organisms.values()), dtype=object)
            
            # Vectorized age update
            for org in organisms:
                org.age += 1
        
        self.is_day = self.current_step < self.current_daylight
        
        # Use precomputed angles for light/temperature calculations
        if self.is_day:
            progress = self.current_step / self.current_daylight
            angle = self.day_angles[int(progress * (len(self.day_angles)-1))]
            self.light_level = np.sin(angle) * 0.9 + 0.1
        else:
            night_progress = (self.current_step - self.current_daylight) / (self.steps_per_day - self.current_daylight)
            if night_progress < 0.1:
                self.light_level = max(0.05, 1.0 - night_progress * 10)
            elif night_progress > 0.9:
                self.light_level = max(0.05, (night_progress - 0.9) * 10)
            else:
                self.light_level = 0.05
        
        # Vectorized season temperature calculation
        season_angle = self.season_angles[self.current_day]
        season_temp = 0.5 + 0.4 * np.sin(season_angle)
        
        if self.is_day:
            day_progress = self.current_step / self.current_daylight
            day_angle = self.day_angles[int(day_progress * (len(self.day_angles)-1))]
            daily_temp = 0.8 + 0.2 * np.sin(day_angle)
        else:
            night_progress = (self.current_step - self.current_daylight) / (self.steps_per_day - self.current_daylight)
            night_angle = self.day_angles[int(night_progress * (len(self.day_angles)-1))]
            daily_temp = 0.6 - 0.1 * np.sin(night_angle)
        
        self.temperature = season_temp * daily_temp
        
        # Update UI
        hours = self.current_step // 2
        minutes = (self.current_step % 2) * 30
        time_str = f"{hours:02d}:{minutes:02d}"
        
        month, day = self.day_to_date(self.current_day)
        date_str = f"{month} {day}"
        
        dpg.set_value("time_text", f"Day {self.current_day+1}, {time_str}")
        dpg.set_value("daylight_text", f"Daylight: {self.current_daylight/2:.1f}h")
        dpg.set_value("light_level_text", f"Light: {self.light_level:.2f}")
        dpg.set_value("temp_text", f"Temp: {self.temperature:.2f}")
        dpg.set_value("date_text", f"Date: {date_str}")
    
    def day_to_date(self, day):
        months = [
            ("Jan", 31), ("Feb", 28), ("Mar", 31), ("Apr", 30),
            ("May", 31), ("Jun", 30), ("Jul", 31), ("Aug", 31),
            ("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31)
        ]
        
        for month, days in months:
            if day < days:
                return month, day + 1
            day -= days
        return "Dec", 31
    
    def update_daylight_duration(self):
        days_since_winter = (self.current_day - self.winter_solstice) % (self.year_length * 48)
        year_progress = days_since_winter / self.year_length
        daylight_hours = 12 + 4 * np.cos(year_progress * math.pi)
        self.current_daylight = int(round(daylight_hours * 2))

    def get_species_color(self, organism: Organism) -> Tuple[int, int, int]:
        species = organism.species
        energy = organism.energy
        
        # Convert to numpy arrays for vectorized operations
        base_color = np.array(species.color, dtype=np.float32)
        
        if species.type == 0:  # Plant
            # Vectorized color calculation
            color = base_color * np.array([
                self.temperature * species.temperature_sensitivity,
                self.light_level * species.light_sensitivity,
                self.temperature * species.temperature_sensitivity
            ])
            
            if not self.is_day:
                color *= 0.3
        else:  # Animal
            energy_ratio = energy / species.max_energy
            color = base_color * np.array([
                energy_ratio,
                energy_ratio * self.temperature,
                energy_ratio
            ])
            
            if self.is_animal_sleeping(organism):
                color *= 0.5
        
        # Clip and convert to integers
        color = np.clip(color, 0, 255).astype(np.uint8)
        return tuple(color)
    
    def get_species_char(self, organism: Organism) -> str:
        if organism.species and organism.species.name:
            return organism.species.name[0].upper()
        return "?"
    
    def draw_grid(self):
        dpg.delete_item("drawlist", children_only=True)
        
        # Clear the grid tensor
        self.grid_tensor.zero_()
        
        with dpg.draw_node(parent="drawlist"):
            # Update grid tensor with organism colors
            for pos, organism in self.grid.organisms.items():
                x, y, z = pos
                if z == 0:  # Only draw the bottom layer for now
                    color = self.get_species_color(organism)
                    self.grid_tensor[x, y] = torch.tensor(color, device=DEVICE) / 255.0
            
            # Draw cells using tensor data
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    color = self.grid_tensor[x, y] * 255
                    organism = self.grid.get_organism((x, y, 0))
                    char = self.get_species_char(organism) if organism else " "
                        
                    dpg.draw_rectangle(
                        (x * self.cell_size, y * self.cell_size),
                        ((x + 1) * self.cell_size, (y + 1) * self.cell_size),
                        fill=(int(color[0]), int(color[1]), int(color[2])),
                        color=(100, 100, 100),
                        thickness=1
                    )
                    
                    dpg.draw_text(
                        (x * self.cell_size + self.cell_size//3, y * self.cell_size + self.cell_size//4),
                        char,
                        color=(0, 0, 0),
                        size=self.cell_size
                    )
            
            # Draw grid lines
            for i in range(self.grid_size + 1):
                dpg.draw_line(
                    (i * self.cell_size, 0),
                    (i * self.cell_size, self.grid_size * self.cell_size),
                    color=(50, 50, 50, 100),
                    thickness=1
                )
                dpg.draw_line(
                    (0, i * self.cell_size),
                    (self.grid_size * self.cell_size, i * self.cell_size),
                    color=(50, 50, 50, 100),
                    thickness=1
                )
    
    def set_running(self, running):
        self.running = running
    
    def is_animal_sleeping(self, organism: Organism) -> bool:
        species = organism.species
        if species.type == 0:  # Plants don't sleep
            return False
        
        wrong_time = (species.diurnal and not self.is_day) or (not species.diurnal and self.is_day)
        
        if wrong_time:
            return True
        else:
            return random.random() < species.sleep_ratio
    
    def step(self):
        self.update_time_cycles()
        
        # Convert to numpy arrays for batch processing
        organisms = np.array(list(self.grid.organisms.values()), dtype=object)
        positions = np.array(list(self.grid.organisms.keys()))
        
        # Precompute masks for different organism types
        is_plant = np.array([org.species.type == 0 for org in organisms])
        is_animal = ~is_plant
        
        # Track changes
        organisms_to_add = []
        organisms_to_remove = set()
        
        # Process plants and animals separately for vectorization
        if np.any(is_plant):
            plant_indices = np.where(is_plant)[0]
            plant_orgs = organisms[plant_indices]
            
            # Vectorized plant energy calculations
            plant_species = np.array([org.species for org in plant_orgs], dtype=object)
            light_sens = np.array([s.light_sensitivity for s in plant_species])
            temp_sens = np.array([s.temperature_sensitivity for s in plant_species])
            max_energies = np.array([s.max_energy for s in plant_species])
            root_energies = np.array([s.root_energy for s in plant_species])
            
            current_energies = np.array([org.energy for org in plant_orgs])
            current_root_energies = np.array([org.root_energy for org in plant_orgs])
            
            # Vectorized energy gain calculation
            energy_gains = (self.light_level * light_sens * 
                           np.array([s.energy_gain for s in plant_species]) * 
                           (self.temperature * temp_sens))
            energy_losses = np.array([s.energy_consumption for s in plant_species])
            
            net_energies = current_energies + energy_gains - energy_losses
            
            # Vectorized energy distribution
            excess_mask = net_energies > max_energies
            storage_amounts = np.where(excess_mask, 
                                     np.minimum(net_energies - max_energies, 
                                               root_energies - current_root_energies),
                                     0)
            
            new_energies = np.where(excess_mask, max_energies, net_energies)
            new_root_energies = current_root_energies + storage_amounts
            
            # Handle energy deficit
            deficit_mask = net_energies < 0
            energy_needed = -net_energies[deficit_mask]
            can_survive = current_root_energies[deficit_mask] >= energy_needed
            
            # Update organisms
            for i, idx in enumerate(plant_indices):
                organism = plant_orgs[i]
                if deficit_mask[i] and not can_survive[i]:
                    organisms_to_remove.add(organism.id)
                    continue
                
                organism.energy = new_energies[i]
                if deficit_mask[i]:
                    organism.energy = 0
                    organism.root_energy = current_root_energies[i] - energy_needed[i]
                else:
                    organism.root_energy = new_root_energies[i]
        
        # Process animals
        if np.any(is_animal):
            animal_indices = np.where(is_animal)[0]
            animal_orgs = organisms[animal_indices]
            
            # Vectorized animal calculations
            animal_species = np.array([org.species for org in animal_orgs], dtype=object)
            max_energies = np.array([s.max_energy for s in animal_species])
            energy_losses = np.array([s.energy_consumption for s in animal_species])
            max_ages = np.array([s.max_age for s in animal_species])
            current_ages = np.array([org.age for org in animal_orgs])
            current_energies = np.array([org.energy for org in animal_orgs])
            
            net_energies = current_energies - energy_losses
            
            # Survival checks
            too_old = current_ages >= max_ages
            no_energy = net_energies <= 0
            
            # Vectorized survival chance
            cold_resistances = np.array([s.cold_resistance for s in animal_species])
            coverings = np.array([1.2 if s.surface_covering == "fur" else 
                                0.9 if s.surface_covering == "scales" else 
                                1.0 for s in animal_species])
            survival_chances = (self.temperature + cold_resistances) * coverings
            survival_rolls = np.random.random(len(animal_orgs))
            dies_from_conditions = survival_rolls > survival_chances
            
            # Mark organisms for removal
            for i, idx in enumerate(animal_indices):
                organism = animal_orgs[i]
                if too_old[i] or no_energy[i] or dies_from_conditions[i]:
                    organisms_to_remove.add(organism.id)
                    continue
                
                organism.energy = net_energies[i]
        
        # Reproduction and movement logic (less vectorizable due to dependencies)
        for organism in organisms:
            if organism.id in organisms_to_remove:
                continue
                
            species = organism.species
            pos = organism.position
            
            # Reproduction
            if (organism.energy >= species.reproduction_cost and 
                organism.age >= species.mature_age and 
                self.current_day % species.reproduction_frequency == 0):
                
                empty_neighbors = self.grid.get_empty_neighbors(pos)
                if empty_neighbors:
                    new_pos = random.choice(empty_neighbors)
                    
                    if random.random() < 0.05:  # Mutation chance
                        new_species = species.mutate()
                    else:
                        new_species = species
                    
                    offspring = Organism(
                        species=new_species,
                        position=new_pos,
                        energy=species.offspring_energy,
                        root_energy=species.offspring_energy * 0.5 if new_species.type == 0 else 0,
                        age=0
                    )
                    organisms_to_add.append(offspring)
                    organism.energy -= species.reproduction_cost
            
            # Animal movement and eating
            if (species.type != 0 and 
                not self.is_animal_sleeping(organism) and 
                organism.energy < species.max_energy * 0.9):
                
                best_food = None
                best_food_value = 0
                best_position = pos
                
                # Check for food in movement radius
                food_options = self.grid.get_food_in_radius(pos, species.move_speed)
                for food_pos, food_organism in food_options:
                    target_species = food_organism.species
                    
                    if (any(part in target_species.name_parts or 
                            part in target_species.surface_covering or 
                            part in target_species.body_type 
                            for part in species.preferred_food_parts)):
                        food_value = species.energy_gain * 1.5
                    elif (species.toxic_food_parts and 
                        any(part in target_species.name_parts or 
                            part in target_species.surface_covering or 
                            part in target_species.body_type 
                            for part in species.toxic_food_parts)):
                        food_value = -species.energy_gain
                    else:
                        food_value = species.energy_gain * 0.5
                    
                    if food_value > best_food_value:
                        best_food_value = food_value
                        best_food = (food_pos, food_organism)
                
                # Find empty positions if no good food found
                if best_food_value <= 0:
                    empty_positions = self.grid.get_empty_neighbors(pos, species.move_speed)
                    if empty_positions:
                        best_position = random.choice(empty_positions)
                
                if best_food is not None and best_food_value > 0:
                    food_pos, food_organism = best_food
                    food_energy = food_organism.energy
                    energy_gain = min(species.energy_gain, food_energy)
                    
                    # Move and eat
                    self.grid.remove_organism(food_organism.id)
                    if self.grid.move_organism(organism.id, food_pos):
                        organism.energy += energy_gain - species.move_energy_cost
                    else:
                        # If move failed, just eat
                        organism.energy += energy_gain - species.move_energy_cost
                        organisms_to_remove.add(food_organism.id)
                elif best_position != pos:
                    # Just move
                    if self.grid.move_organism(organism.id, best_position):
                        organism.energy -= species.move_energy_cost
        
        # Apply changes
        for org_id in organisms_to_remove:
            self.grid.remove_organism(org_id)
            
        for organism in organisms_to_add:
            try:
                self.grid.add_organism(organism)
            except ValueError:
                pass
        
        self.draw_grid()
    
    def clear_grid(self):
        self.grid = WorldGrid(self.grid_size)
        self.grid_tensor.zero_()
        self.current_step = 0
        self.current_day = 0
        self.is_day = True
        self.light_level = 1.0
        self.temperature = 1.0
        
        dpg.set_value("time_text", "Day 1, 00:00")
        dpg.set_value("daylight_text", f"Daylight: {self.current_daylight/2}h")
        dpg.set_value("light_level_text", f"Light: {self.light_level:.2f}")
        dpg.set_value("temp_text", f"Temp: {self.temperature:.2f}")
        dpg.set_value("date_text", "Date: Jan 1")
        
        self.draw_grid()

    def randomize_grid(self):
        self.clear_grid()
        num_species = random.randint(1, 10)
        random_species = [GRASS, OMNIVORE]
        for _ in range(num_species):
            random_species.append(Species.generate_random_species())
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in [0]:  # Only populate bottom layer for now
                    if random.random() < 0.3:
                        species = random.choice(random_species)
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
                            pass  # Skip if position is already taken
        
        self.draw_grid()
    
    def change_grid_size(self, sender, app_data):
        new_size = app_data
        self.grid_size = new_size
        self.grid = WorldGrid(new_size)
        self.grid_tensor = torch.zeros((new_size, new_size, 3), dtype=torch.float32, device=DEVICE)
        
        dpg.configure_item("drawlist", 
                          width=self.grid_size * self.cell_size,
                          height=self.grid_size * self.cell_size)
        
        self.draw_grid()
    
    def run(self):
        while dpg.is_dearpygui_running():
            if self.running:
                self.step()
                time.sleep(0.1)
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()

if __name__ == "__main__":
    game = GameOfLife()
    game.run()