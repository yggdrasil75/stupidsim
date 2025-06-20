import hashlib
import dearpygui.dearpygui as dpg
import random
import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import torch

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
class CellOrganism:
    species: Species
    energy: float = 0
    root_energy: float = 0  # Only used for plants
    age: float = 0

@dataclass
class TotalCell:
    solarRay: float = 0
    organism: CellOrganism = None

class GameOfLife:
    def __init__(self):
        self.grid_size = 50
        self.cell_size = 10
        self.running = False
        
        # Initialize grid as a torch tensor for faster operations
        self.totalgrid = [[[TotalCell() for _ in range(self.grid_size)] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
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
            
            # Update age using tensor operations
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    for z in range(self.grid_size):
                        if self.totalgrid[x][y][z].organism is not None:
                            self.totalgrid[x][y][z].organism.age += 1
        
        self.is_day = self.current_step < self.current_daylight
        
        if self.is_day:
            progress = self.current_step / self.current_daylight
            self.light_level = math.sin(progress * math.pi) * 0.9 + 0.1
        else:
            night_progress = (self.current_step - self.current_daylight) / (self.steps_per_day - self.current_daylight)
            if night_progress < 0.1:
                self.light_level = max(0.05, 1.0 - night_progress * 10)
            elif night_progress > 0.9:
                self.light_level = max(0.05, (night_progress - 0.9) * 10)
            else:
                self.light_level = 0.05
        
        # Calculate temperature using PyTorch for vectorized operations
        season_progress = (self.current_day - self.winter_solstice) % self.year_length / self.year_length
        season_temp = 0.5 + 0.4 * torch.sin(torch.tensor(season_progress * 2 * math.pi, device=DEVICE)).item()
        
        if self.is_day:
            day_progress = self.current_step / self.current_daylight
            daily_temp = 0.8 + 0.2 * torch.sin(torch.tensor(day_progress * math.pi, device=DEVICE)).item()
        else:
            night_progress = (self.current_step - self.current_daylight) / (self.steps_per_day - self.current_daylight)
            daily_temp = 0.6 - 0.1 * torch.sin(torch.tensor(night_progress * math.pi, device=DEVICE)).item()
        
        self.temperature = season_temp * daily_temp
        
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
        daylight_hours = 12 + 4 * torch.cos(torch.tensor(year_progress * math.pi)).item()
        self.current_daylight = int(round(daylight_hours * 2))

    def get_species_color(self, x, y, z):
        if self.totalgrid[x][y][z].organism is None:
            return (0, 0, 0)
            
        species = self.totalgrid[x][y][z].organism.species
        energy = self.totalgrid[x][y][z].organism.energy
        
        if species.type == 0:
            # Use PyTorch for vectorized color calculations
            color = torch.tensor(species.color, dtype=torch.float32, device=DEVICE)
            r = color[0] * self.temperature * species.temperature_sensitivity
            g = color[1] * self.light_level * species.light_sensitivity
            b = color[2] * self.temperature * species.temperature_sensitivity
            
            if not self.is_day:
                r *= 0.3
                g *= 0.3
                b *= 0.3
            
            r = torch.clamp(r, 0, 255).int().item()
            g = torch.clamp(g, 0, 255).int().item()
            b = torch.clamp(b, 0, 255).int().item()
            
            return (r, g, b)
        else:
            energy_ratio = energy / species.max_energy
            color = torch.tensor(species.color, dtype=torch.float32, device=DEVICE)
            r = color[0] * energy_ratio
            g = color[1] * energy_ratio * self.temperature
            b = color[2] * energy_ratio
            
            if self.is_animal_sleeping(x, y, z):
                r *= 0.5
                g *= 0.5
                b *= 0.5
            
            r = torch.clamp(r, 0, 255).int().item()
            g = torch.clamp(g, 0, 255).int().item()
            b = torch.clamp(b, 0, 255).int().item()
            
            return (r, g, b)
    
    def get_species_char(self, x, y, z):
        if self.totalgrid[x][y][z].organism is None:
            return " "
            
        species = self.totalgrid[x][y][z].organism.species
        if species and species.name:
            return species.name[0].upper()
        return "?"
    
    def draw_grid(self):
        dpg.delete_item("drawlist", children_only=True)
        
        with dpg.draw_node(parent="drawlist"):
            # Update grid tensor with current colors
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if self.totalgrid[x][y][0].organism is not None:
                        color = self.get_species_color(x, y, 0)
                        self.grid_tensor[x, y] = torch.tensor(color, device=DEVICE) / 255.0
                    else:
                        self.grid_tensor[x, y] = torch.zeros(3, device=DEVICE)
            
            # Draw cells using tensor data
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    color = self.grid_tensor[x, y] * 255
                    char = self.get_species_char(x, y, 0)
                        
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
    
    def is_animal_sleeping(self, x, y, z):
        if self.totalgrid[x][y][z].organism is None:
            return False
            
        species = self.totalgrid[x][y][z].organism.species
        if species is None or species.type == 0:
            return False
        
        wrong_time = (species.diurnal and not self.is_day) or (not species.diurnal and self.is_day)
        
        if wrong_time:
            return True
        else:
            return random.random() < species.sleep_ratio
        
    def step(self):
        self.update_time_cycles()
        
        # Create new grids for the next generation
        new_grid = [[[TotalCell() for _ in range(self.grid_size)] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        moved_or_eaten = [[[False for _ in range(self.grid_size)] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Process each cell
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in range(self.grid_size):
                    if moved_or_eaten[x][y][z]:
                        continue
                        
                    if self.totalgrid[x][y][z].organism is None:
                        continue
                        
                    current_organism = self.totalgrid[x][y][z].organism
                    current_species = current_organism.species
                    current_energy = current_organism.energy
                    current_root_energy = current_organism.root_energy
                    current_age = current_organism.age
                    
                    energy_loss = current_species.energy_consumption
                    
                    if current_species.type == 0:
                        energy_gain = ((self.light_level * current_species.light_sensitivity) * 
                                      current_species.energy_gain * (self.temperature * current_species.temperature_sensitivity))
                        net_energy = current_energy + energy_gain - energy_loss
                        
                        if net_energy > current_species.max_energy:
                            storage = min(net_energy - current_species.max_energy, 
                                        current_species.root_energy - current_root_energy)
                            new_grid[x][y][z].organism = CellOrganism(
                                species=current_species,
                                energy=current_species.max_energy,
                                root_energy=current_root_energy + storage,
                                age=current_age
                            )
                        elif net_energy > 0:
                            new_grid[x][y][z].organism = CellOrganism(
                                species=current_species,
                                energy=net_energy,
                                root_energy=current_root_energy,
                                age=current_age
                            )
                        else:
                            energy_needed = -net_energy
                            if current_root_energy >= energy_needed:
                                new_grid[x][y][z].organism = CellOrganism(
                                    species=current_species,
                                    energy=0,
                                    root_energy=current_root_energy - energy_needed,
                                    age=current_age
                                )
                            else:
                                new_grid[x][y][z].organism = None
                                continue
                    else:
                        net_energy = current_energy - energy_loss
                        
                        if current_age >= current_species.max_age:
                            new_grid[x][y][z].organism = None
                            continue
                        
                        if net_energy <= 0:
                            new_grid[x][y][z].organism = None
                            continue
                        
                        new_grid[x][y][z].organism = CellOrganism(
                            species=current_species,
                            energy=net_energy,
                            root_energy=0,
                            age=current_age
                        )
                    
                    # Calculate survival chance using PyTorch
                    survival_chance = (self.temperature + current_species.cold_resistance)
                    if current_species.surface_covering == "fur":
                        survival_chance *= 1.2
                    elif current_species.surface_covering == "scales":
                        survival_chance *= 0.9
                    
                    if torch.rand(1).item() > survival_chance:
                        new_grid[x][y][z].organism = None
                        continue
                    
                    # Reproduction
                    if (new_grid[x][y][z].organism is not None and 
                        new_grid[x][y][z].organism.energy >= current_species.reproduction_cost and 
                        current_age >= current_species.mature_age and 
                        self.current_day % current_species.reproduction_frequency == 0):
                        
                        empty_neighbors = []
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue
                                    nx, ny, nz = x + dx, y + dy, z + dz
                                    if (0 <= nx < self.grid_size and 
                                        0 <= ny < self.grid_size and 
                                        0 <= nz < self.grid_size):
                                        if (self.totalgrid[nx][ny][nz].organism is None and 
                                            not moved_or_eaten[nx][ny][nz]):
                                            empty_neighbors.append((nx, ny, nz))
                        
                        if empty_neighbors:
                            nx, ny, nz = random.choice(empty_neighbors)
                            
                            if torch.rand(1).item() < 0.05:  # Mutation chance
                                mutated_species = current_species.mutate()
                            else:
                                mutated_species = current_species
                            
                            new_grid[nx][ny][nz].organism = CellOrganism(
                                species=mutated_species,
                                energy=current_species.offspring_energy,
                                root_energy=current_species.offspring_energy * 0.5 if mutated_species.type == 0 else 0,
                                age=0
                            )
                            new_grid[x][y][z].organism.energy -= current_species.reproduction_cost
                            moved_or_eaten[nx][ny][nz] = True
                    
                    # Animal movement and eating
                    if (current_species.type != 0 and 
                        not self.is_animal_sleeping(x, y, z) and 
                        new_grid[x][y][z].organism.energy < current_species.max_energy * 0.9):
                        
                        best_food = None
                        best_food_value = 0
                        best_position = (x, y, z)
                        
                        for dx in range(-current_species.move_speed, current_species.move_speed + 1):
                            for dy in range(-current_species.move_speed, current_species.move_speed + 1):
                                for dz in range(-current_species.move_speed, current_species.move_speed + 1):
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue
                                    
                                    nx, ny, nz = x + dx, y + dy, z + dz
                                    if (0 <= nx < self.grid_size and 
                                        0 <= ny < self.grid_size and 
                                        0 <= nz < self.grid_size):
                                        
                                        if self.totalgrid[nx][ny][nz].organism is not None:
                                            target_species = self.totalgrid[nx][ny][nz].organism.species
                                            
                                            if (any(part in target_species.name_parts or 
                                                    part in target_species.surface_covering or 
                                                    part in target_species.body_type 
                                                    for part in current_species.preferred_food_parts)):
                                                food_value = current_species.energy_gain * 1.5
                                            elif (current_species.toxic_food_parts and 
                                                any(part in target_species.name_parts or 
                                                    part in target_species.surface_covering or 
                                                    part in target_species.body_type 
                                                    for part in current_species.toxic_food_parts)):
                                                food_value = -current_species.energy_gain
                                            else:
                                                food_value = current_species.energy_gain * 0.5
                                            
                                            if food_value > best_food_value:
                                                best_food_value = food_value
                                                best_food = (nx, ny, nz)
                                        
                                        elif self.totalgrid[nx][ny][nz].organism is None:
                                            if best_food_value <= 0:
                                                best_position = (nx, ny, nz)
                        
                        if best_food is not None and best_food_value > 0:
                            nx, ny, nz = best_food
                            food_energy = self.totalgrid[nx][ny][nz].organism.energy
                            energy_gain = min(current_species.energy_gain, food_energy)
                            
                            new_grid[nx][ny][nz].organism = CellOrganism(
                                species=current_species,
                                energy=new_grid[x][y][z].organism.energy + energy_gain - current_species.move_energy_cost,
                                root_energy=0,
                                age=current_age
                            )
                            new_grid[x][y][z].organism = None
                            moved_or_eaten[nx][ny][nz] = True
                        elif best_position != (x, y, z):
                            nx, ny, nz = best_position
                            new_grid[nx][ny][nz].organism = CellOrganism(
                                species=current_species,
                                energy=new_grid[x][y][z].organism.energy - current_species.move_energy_cost,
                                root_energy=0,
                                age=current_age
                            )
                            new_grid[x][y][z].organism = None
                            moved_or_eaten[nx][ny][nz] = True
        
        self.totalgrid = new_grid
        self.draw_grid()
    
    def clear_grid(self):
        self.totalgrid = [[[TotalCell() for _ in range(self.grid_size)] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.grid_tensor = torch.zeros((self.grid_size, self.grid_size, 3), dtype=torch.float32, device=DEVICE)
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
        num_species = random.randint(1, 10)
        random_species = [GRASS, OMNIVORE]
        for _ in range(num_species):
            random_species.append(Species.generate_random_species())
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in range(self.grid_size):
                    self.totalgrid[x][y][z] = TotalCell()
                    rand = random.random()
                    if rand < 0.3:
                        species = random.choice(random_species)
                        root_energy = random.uniform(0, species.root_energy) if species.type == 0 else 0
                        self.totalgrid[x][y][z].organism = CellOrganism(
                            species=species,
                            energy=random.uniform(species.offspring_energy, species.max_energy * 0.5),
                            root_energy=root_energy,
                            age=random.randint(0, species.mature_age * 2)
                        )
                    else:
                        self.totalgrid[x][y][z].organism = None
        
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
    
    def change_grid_size(self, sender, app_data):
        new_size = app_data
        self.grid_size = new_size
        self.totalgrid = [[[TotalCell() for _ in range(new_size)] for _ in range(new_size)] for _ in range(new_size)]
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