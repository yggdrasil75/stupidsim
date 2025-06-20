import dearpygui.dearpygui as dpg
import random
import time
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class Species(Enum):
    PLANT = 0
    ANIMAL = 1

@dataclass
class PlantSpecies:
    name: str
    color: tuple  # Base color (will be modified by environmental factors)
    max_energy: float
    root_energy: float  # Energy stored in roots for winter/night
    energy_gain: float  # Per step in full light
    energy_consumption: float  # Per step always
    mature_age: int  # Days to reach maturity
    reproduction_cost: float
    offspring_energy: float
    reproduction_frequency: int  # Days between reproduction attempts when mature
    temperature_sensitivity: float  # 0-1, how much temperature affects growth
    light_sensitivity: float  # 0-1, how much light affects growth
    cold_resistance: float  # 0-1, how well it survives cold (higher = better)
    type: Species = Species.PLANT
    
    def mutate(self) -> 'PlantSpecies':
        """Create a slightly mutated version of this plant species"""
        mutation_factor = random.uniform(0.9, 1.1)  # Small variation
        
        # Create new name by adding a number
        new_name = f"{self.name}-{random.randint(1, 1000)}"
        
        return PlantSpecies(
            name=new_name,
            color=(
                max(0, min(255, int(self.color[0] * random.uniform(0.95, 1.05)))),
                max(0, min(255, int(self.color[1] * random.uniform(0.95, 1.05)))),
                max(0, min(255, int(self.color[2] * random.uniform(0.95, 1.05))))
            ),
            max_energy=max(1, self.max_energy * random.uniform(0.9, 1.1)),
            root_energy=max(0.5, self.root_energy * random.uniform(0.8, 1.2)),
            energy_gain=max(0.05, self.energy_gain * random.uniform(0.8, 1.2)),
            energy_consumption=max(0.01, self.energy_consumption * random.uniform(0.8, 1.2)),
            mature_age=max(5, int(self.mature_age * random.uniform(0.8, 1.2))),
            reproduction_cost=max(0.5, self.reproduction_cost * random.uniform(0.8, 1.2)),
            offspring_energy=max(0.5, self.offspring_energy * random.uniform(0.8, 1.2)),
            reproduction_frequency=max(1, int(self.reproduction_frequency * random.uniform(0.8, 1.2))),
            temperature_sensitivity=max(0.1, min(1.0, self.temperature_sensitivity * random.uniform(0.9, 1.1))),
            light_sensitivity=max(0.1, min(1.0, self.light_sensitivity * random.uniform(0.9, 1.1))),
            cold_resistance=max(0.1, min(1.0, self.cold_resistance * random.uniform(0.9, 1.1)))
        )

@dataclass
class AnimalSpecies:
    name: str
    color: tuple  # Base color
    max_energy: float
    energy_gain_plants: float  # Per plant eaten
    energy_gain_meat: float    # Per animal eaten
    idle_energy_consumption: float  # Per step
    move_energy_consumption: float  # Per tile moved
    move_speed: int  # Max tiles per move
    mature_age: int  # Days to reach maturity
    max_age: int  # Days until death from old age
    offspring_energy: float  # Energy given to offspring
    cold_resistance: float  # 0-1, how well it survives cold
    sleep_ratio: float  # 0-1, ratio of time spent sleeping
    diurnal: bool  # True for day-active, False for night-active
    preferred_plants: List[str]  # Names of preferred plant species
    toxic_plants: List[str]     # Names of toxic plant species
    can_eat_meat: bool          # Whether this animal can eat other animals
    preferred_prey: List[str]   # Names of preferred prey species
    type: Species = Species.ANIMAL
    
    def mutate(self) -> 'AnimalSpecies':
        """Create a slightly mutated version of this animal species"""
        # Create new name by combining syllables
        syllables = ["zo", "ra", "fi", "lo", "pa", "ki", "tu", "ve", "no", "xi"]
        new_name = "".join(random.choices(syllables, k=random.randint(2, 3))).capitalize()
        
        return AnimalSpecies(
            name=new_name,
            color=(
                max(0, min(255, int(self.color[0] * random.uniform(0.9, 1.1)))),
                max(0, min(255, int(self.color[1] * random.uniform(0.9, 1.1)))),
                max(0, min(255, int(self.color[2] * random.uniform(0.9, 1.1))))
            ),
            max_energy=max(1, self.max_energy * random.uniform(0.8, 1.2)),
            energy_gain_plants=max(0.1, self.energy_gain_plants * random.uniform(0.8, 1.2)),
            energy_gain_meat=max(0.1, self.energy_gain_meat * random.uniform(0.8, 1.2)),
            idle_energy_consumption=max(0.05, self.idle_energy_consumption * random.uniform(0.8, 1.2)),
            move_energy_consumption=max(0.05, self.move_energy_consumption * random.uniform(0.8, 1.2)),
            move_speed=max(1, min(5, int(self.move_speed * random.uniform(0.8, 1.2)))),
            mature_age=max(30, int(self.mature_age * random.uniform(0.8, 1.2))),
            max_age=max(60, int(self.max_age * random.uniform(0.8, 1.2))),
            offspring_energy=max(0.5, self.offspring_energy * random.uniform(0.8, 1.2)),
            cold_resistance=max(0.1, min(1.0, self.cold_resistance * random.uniform(0.9, 1.1))),
            sleep_ratio=max(0.05, min(0.5, self.sleep_ratio * random.uniform(0.8, 1.2))),
            diurnal=random.random() < 0.7 if random.random() < 0.2 else self.diurnal,  # 20% chance to flip
            preferred_plants=self.preferred_plants.copy(),
            toxic_plants=self.toxic_plants.copy(),
            can_eat_meat=self.can_eat_meat if random.random() < 0.8 else not self.can_eat_meat,  # 20% chance to flip
            preferred_prey=self.preferred_prey.copy()
        )

# Initialize default plant species with more balanced values
PlantSpecies.GRASS = PlantSpecies(
    name="Grass",
    color=(100, 200, 100),
    max_energy=12,
    root_energy=8,  # Grass has substantial root energy
    energy_gain=0.3,
    energy_consumption=0.03,
    mature_age=15,
    reproduction_cost=1.5,
    offspring_energy=1.0,
    reproduction_frequency=5,
    temperature_sensitivity=0.8,
    light_sensitivity=0.9,
    cold_resistance=0.7  # Grass is fairly cold resistant
)

PlantSpecies.BUSH = PlantSpecies(
    name="Bush",
    color=(50, 150, 50),
    max_energy=30,
    root_energy=15,  # Bushes have moderate root energy
    energy_gain=0.2,
    energy_consumption=0.05,
    mature_age=50,
    reproduction_cost=4,
    offspring_energy=2.5,
    reproduction_frequency=15,
    temperature_sensitivity=0.6,
    light_sensitivity=0.7,
    cold_resistance=0.87
)

PlantSpecies.TREE = PlantSpecies(
    name="Tree",
    color=(0, 100, 0),
    max_energy=100,
    root_energy=40,  # Trees have massive root energy storage
    energy_gain=0.15,
    energy_consumption=0.12,
    mature_age=100,
    reproduction_cost=8,
    offspring_energy=4,
    reproduction_frequency=40,
    temperature_sensitivity=0.4,
    light_sensitivity=0.6,
    cold_resistance=0.97
)

# Initialize default herbivore species
AnimalSpecies.ZORAFI = AnimalSpecies(
    name="Zorafi",
    color=(255, 220, 100),  # Light yellow
    max_energy=15,
    energy_gain_plants=3.0,  # Per plant eaten
    energy_gain_meat=0.0,    # Doesn't eat meat
    idle_energy_consumption=0.4,
    move_energy_consumption=0.2,
    move_speed=2,
    mature_age=90,  # ~3 months
    max_age=365*2,  # ~2 years
    offspring_energy=4,
    cold_resistance=0.6,
    sleep_ratio=0.2,  # Sleeps 20% of time
    diurnal=True,  # Day-active
    preferred_plants=["Grass", "Bush"],
    toxic_plants=["Poison Ivy"],  # Example toxic plant
    can_eat_meat=False,
    preferred_prey=[]
)

AnimalSpecies.LOPAKI = AnimalSpecies(
    name="Lopaki",
    color=(255, 200, 50),  # Orange-yellow
    max_energy=20,
    energy_gain_plants=4.0,
    energy_gain_meat=2.0,  # Can eat some meat but prefers plants
    idle_energy_consumption=0.6,
    move_energy_consumption=0.3,
    move_speed=1,
    mature_age=180,  # ~6 months
    max_age=365*3,  # ~3 years
    offspring_energy=6,
    cold_resistance=0.8,
    sleep_ratio=0.15,
    diurnal=True,
    preferred_plants=["Bush", "Tree"],
    toxic_plants=[],
    can_eat_meat=True,
    preferred_prey=["Tuvexi"]  # Prefers to eat Tuvexi
)

AnimalSpecies.TUVEXI = AnimalSpecies(
    name="Tuvexi",
    color=(220, 180, 30),  # Darker yellow
    max_energy=12,
    energy_gain_plants=2.5,
    energy_gain_meat=5.0,  # Primarily carnivorous
    idle_energy_consumption=0.3,
    move_energy_consumption=0.15,
    move_speed=3,
    mature_age=60,  # ~2 months
    max_age=365,  # ~1 year
    offspring_energy=3,
    cold_resistance=0.4,
    sleep_ratio=0.3,
    diurnal=False,  # Night-active
    preferred_plants=[],  # Doesn't eat plants
    toxic_plants=[],
    can_eat_meat=True,
    preferred_prey=["Zorafi"]  # Prefers to eat Zorafi
)

@dataclass
class CellOrganism:
    species: AnimalSpecies | PlantSpecies
    energy: float = 0
    root_energy: float = 0
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
        self.totalgrid = [[[TotalCell() for _ in range(self.grid_size)] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Time parameters (each step = 1/2 hour)
        self.steps_per_day = 48  # 24 hours * 2 steps per hour
        self.current_step = 0     # Current step (0-47)
        self.current_day = 0      # Current day of year (0-364)
        self.summer_solstice = 172  # June 21 (day 172)
        self.winter_solstice = 355  # December 21 (day 355)
        self.year_length = 365      # Days in a year
        self.max_daylight = 32      # Steps of daylight at summer solstice (16 hours)
        self.min_daylight = 16      # Steps of daylight at winter solstice (8 hours)
        self.current_daylight = 17  # Current day's daylight steps (12 hours)
        self.light_level = 1.0      # Current light level (0-1)
        self.temperature = 1.0      # Temperature factor (0-1)
        self.is_day = True
        
        # Population history tracking - now tracking by species
        self.population_history = {}  # Will be populated with species names
        self.history_length = 10000  # Keep last 200 steps of history
        
        # Initialize DPG
        dpg.create_context()
        self.setup_ui()
        dpg.create_viewport(title="Evolution Simulation", width=850, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
    def setup_ui(self):
        # Main window
        with dpg.window(tag="Main Window"):
            # Game of Life visualization
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
                    
                    # Time indicators
                    self.time_text = dpg.add_text("Time: Day 1, 00:00", tag="time_text")
                    self.daylight_text = dpg.add_text(f"Daylight: {self.current_daylight/2}h", tag="daylight_text")
                    self.light_level_text = dpg.add_text(f"Light: {self.light_level:.2f}", tag="light_level_text")
                    self.temp_text = dpg.add_text(f"Temp: {self.temperature:.2f}", tag="temp_text")
                    self.date_text = dpg.add_text("Date: Jan 1", tag="date_text")
                    
                
                # Game display
                with dpg.child_window(tag="game_window"):
                    with dpg.drawlist(width=self.grid_size*self.cell_size, 
                                    height=self.grid_size*self.cell_size, 
                                    tag="drawlist"):
                        pass
                
        # Set up viewport resizing
        dpg.set_viewport_resize_callback(self.on_viewport_resize)
        dpg.set_primary_window("Main Window", True)
        
        # Initial render
        self.draw_grid()
    
    def on_viewport_resize(self):
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dpg.set_item_width("Main Window", viewport_width)
        dpg.set_item_height("Main Window", viewport_height)
        
        # Keep controls panel at fixed width and adjust game window
        if viewport_width > 200:
            dpg.set_item_width("game_window", viewport_width - 200)
    
    def update_time_cycles(self):
        self.current_step += 1
        
        # Check if day has advanced
        if self.current_step >= self.steps_per_day:
            self.current_step = 0
            self.current_day = (self.current_day + 1) % self.year_length
            self.update_daylight_duration()
            
            # Age all living organisms by one day
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    for z in range(self.grid_size):
                        if self.totalgrid[y][x][z].organism is not None:
                            self.totalgrid[y][x][z].organism.age += 1
        
        # Update time of day
        self.is_day = self.current_step < self.current_daylight
        
        # Update light level based on time of day
        if self.is_day:
            # Daytime - light follows a sine curve peaking at noon
            progress = self.current_step / self.current_daylight  # 0-1 through the day
            self.light_level = math.sin(progress * math.pi) * 0.9 + 0.1  # Range 0.1-1.0
        else:
            # Nighttime - minimal light with smooth transitions
            night_progress = (self.current_step - self.current_daylight) / (self.steps_per_day - self.current_daylight)
            if night_progress < 0.1:  # Evening twilight
                self.light_level = max(0.05, 1.0 - night_progress * 10)
            elif night_progress > 0.9:  # Morning twilight
                self.light_level = max(0.05, (night_progress - 0.9) * 10)
            else:  # Middle of night
                self.light_level = 0.05
        
        # Update temperature based on season and time of day
        # Seasonal temperature variation follows a sine curve
        season_progress = (self.current_day - self.winter_solstice) % self.year_length / self.year_length
        season_temp = 0.5 + 0.4 * math.sin(season_progress * 2 * math.pi)
        
        # Daily temperature variation - warmer in day
        if self.is_day:
            day_progress = self.current_step / self.current_daylight
            daily_temp = 0.8 + 0.2 * math.sin(day_progress * math.pi)
        else:
            night_progress = (self.current_step - self.current_daylight) / (self.steps_per_day - self.current_daylight)
            daily_temp = 0.6 - 0.1 * math.sin(night_progress * math.pi)
        
        self.temperature = season_temp * daily_temp
        
        # Update UI elements
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
        """Convert day of year (0-364) to month and day"""
        months = [
            ("Jan", 31), ("Feb", 28), ("Mar", 31), ("Apr", 30),
            ("May", 31), ("Jun", 30), ("Jul", 31), ("Aug", 31),
            ("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31)
        ]
        
        for month, days in months:
            if day < days:
                return month, day + 1
            day -= days
        return "Dec", 31  # Shouldn't happen
    
    def update_daylight_duration(self):
        """Calculate daylight duration based on time of year"""
        # Calculate days since winter solstice (handling year wrap-around)
        days_since_winter = (self.current_day - self.winter_solstice) % (self.year_length * 48)
        
        # Calculate progress through the year (0-1) where 0 is winter solstice, 0.5 is summer solstice
        year_progress = days_since_winter / self.year_length
        
        # Calculate daylight hours using cosine (peaks at summer solstice)
        # Base 12 hours + variation of ±4 hours
        daylight_hours = 12 + 4 * math.cos(year_progress * math.pi)
        
        # Convert hours to steps (2 steps per hour) and round
        self.current_daylight = int(round(daylight_hours * 2))

    def get_species_color(self, x, y, z):
        """Get color for a cell based on its species, energy, and environmental factors"""
        if self.totalgrid[y][x][z].organism is None:
            return (0, 0, 0)
            
        species = self.totalgrid[y][x][z].organism.species
        energy = self.totalgrid[y][x][z].organism.energy
        
        if species.type == Species.PLANT:
            r = species.color[0] * self.temperature * species.temperature_sensitivity
            g = species.color[1] * self.light_level * species.light_sensitivity
            b = species.color[2] * self.temperature * species.temperature_sensitivity
            
            if not self.is_day:
                r *= 0.3
                g *= 0.3
                b *= 0.3
            
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            
            return (int(r), int(g), int(b))
        elif species.type == Species.ANIMAL:
            # Base color modified by energy and temperature
            energy_ratio = energy / species.max_energy
            r = species.color[0] * energy_ratio
            g = species.color[1] * energy_ratio * self.temperature
            b = species.color[2] * energy_ratio
            
            # Darken if sleeping
            if self.is_animal_sleeping(x, y, z):
                r *= 0.5
                g *= 0.5
                b *= 0.5
            
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            
            return (int(r), int(g), int(b))
        return (0, 0, 0)
    
    def get_species_char(self, x, y, z):
        """Get character representation for a cell"""
        if self.totalgrid[y][x][z].organism is None:
            return " "
            
        species = self.totalgrid[y][x][z].organism.species
        if species: 
            return species.name[0].upper()
        else: 
            return " "  # Empty space
    
    def draw_grid(self):
        dpg.delete_item("drawlist", children_only=True)
        
        with dpg.draw_node(parent="drawlist"):
            # Draw cells
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if self.totalgrid[y][x][0].organism is not None:  # Only draw top layer for now
                        color = self.get_species_color(x, y, 0)
                        char = self.get_species_char(x, y, 0)
                            
                        # Draw cell background
                        dpg.draw_rectangle(
                            (x * self.cell_size, y * self.cell_size),
                            ((x + 1) * self.cell_size, (y + 1) * self.cell_size),
                            fill=color,
                            color=(100, 100, 100),
                            thickness=1
                        )
                        
                        # Draw character
                        dpg.draw_text(
                            (x * self.cell_size + self.cell_size//3, y * self.cell_size + self.cell_size//4),
                            char,
                            color=(0, 0, 0),
                            size=self.cell_size
                        )
            
            # Draw grid lines
            for i in range(self.grid_size + 1):
                # Vertical lines
                dpg.draw_line(
                    (i * self.cell_size, 0),
                    (i * self.cell_size, self.grid_size * self.cell_size),
                    color=(50, 50, 50, 100),
                    thickness=1
                )
                # Horizontal lines
                dpg.draw_line(
                    (0, i * self.cell_size),
                    (self.grid_size * self.cell_size, i * self.cell_size),
                    color=(50, 50, 50, 100),
                    thickness=1
                )
                
    def set_running(self, running):
        self.running = running
    
    def is_animal_sleeping(self, x, y, z):
        """Check if animal should be sleeping based on its species and time"""
        if self.totalgrid[y][x][z].organism is None:
            return False
            
        species = self.totalgrid[y][x][z].organism.species
        if species is None or species.type != Species.ANIMAL:
            return False
        
        # Check if it's the wrong time of day for this species
        wrong_time = (species.diurnal and not self.is_day) or (not species.diurnal and self.is_day)
        
        # Determine if sleeping based on sleep ratio and time
        if wrong_time:
            return True  # Always sleep during opposite time
        else:
            # During active time, sleep based on sleep ratio
            return random.random() < species.sleep_ratio
        
    def step(self):
        # Update time cycles first
        self.update_time_cycles()
        
        # Create new grids for the next generation
        new_grid = [[[TotalCell() for _ in range(self.grid_size)] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Track which cells have moved or eaten
        moved_or_eaten = [[[False for _ in range(self.grid_size)] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Process each cell
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                for z in range(self.grid_size):
                    if moved_or_eaten[y][x][z]:
                        continue  # Skip cells that have already been processed
                        
                    if self.totalgrid[y][x][z].organism is None:
                        continue
                        
                    current_organism = self.totalgrid[y][x][z].organism
                    current_species = current_organism.species
                    current_energy = current_organism.energy
                    current_root_energy = current_organism.root_energy
                    current_age = current_organism.age
                    
                    if current_species.type == Species.PLANT:
                        # Plants always consume energy
                        energy_consumption = current_species.energy_consumption
                        
                        # During day, plants gain energy from light (modified by environmental factors)
                        energy_gain = ((self.light_level * current_species.light_sensitivity) * 
                                        current_species.energy_gain * (self.temperature * current_species.temperature_sensitivity))
                        net_energy = current_energy + energy_gain - energy_consumption
                        
                        # Handle energy storage and usage
                        if net_energy > current_species.max_energy:
                            # Store excess in roots (up to root capacity)
                            storage = min(net_energy - current_species.max_energy, 
                                        current_species.root_energy - current_root_energy)
                            new_grid[y][x][z].organism = CellOrganism(
                                species=current_species,
                                energy=current_species.max_energy,
                                root_energy=current_root_energy + storage,
                                age=current_age
                            )
                        elif net_energy > 0:
                            new_grid[y][x][z].organism = CellOrganism(
                                species=current_species,
                                energy=net_energy,
                                root_energy=current_root_energy,
                                age=current_age
                            )
                        else:
                            # Use root energy if available
                            energy_needed = -net_energy
                            if current_root_energy >= energy_needed:
                                new_grid[y][x][z].organism = CellOrganism(
                                    species=current_species,
                                    energy=0,
                                    root_energy=current_root_energy - energy_needed,
                                    age=current_age
                                )
                            else:
                                # Not enough energy - plant dies
                                new_grid[y][x][z].organism = None
                                continue
                        
                        # Check if plant can survive current conditions
                        survival_chance = 1.0
                        survival_chance = (self.temperature + current_species.cold_resistance) * survival_chance
                        
                        if random.random() > survival_chance:
                            new_grid[y][x][z].organism = None
                            continue
                        
                        # If plant has enough energy and is mature, it can reproduce
                        if (new_grid[y][x][z].organism is not None and 
                            new_grid[y][x][z].organism.energy >= current_species.reproduction_cost and 
                            current_age >= current_species.mature_age and 
                            self.current_day % current_species.reproduction_frequency == 0):
                            
                            # Find empty neighboring cells
                            empty_neighbors = []
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    for dz in [-1, 0, 1]:
                                        if dx == 0 and dy == 0 and dz == 0:
                                            continue
                                        nx, ny, nz = x + dx, y + dy, z + dz
                                        if (0 <= nx < self.grid_size and 
                                            0 <= ny < self.grid_size and 
                                            0 <= nz < self.grid_size):
                                            if (self.totalgrid[ny][nx][nz].organism is None and 
                                                not moved_or_eaten[ny][nx][nz]):
                                                empty_neighbors.append((nx, ny, nz))
                            
                            # Reproduce to a random empty neighbor
                            if empty_neighbors:
                                nx, ny, nz = random.choice(empty_neighbors)
                                
                                # Small chance of mutation
                                if random.random() < 0.05:  # 5% mutation chance
                                    mutated_species = current_species.mutate()
                                else:
                                    mutated_species = current_species
                                
                                new_grid[ny][nx][nz].organism = CellOrganism(
                                    species=mutated_species,
                                    energy=current_species.offspring_energy,
                                    root_energy=current_species.offspring_energy * 0.5,
                                    age=0
                                )
                                new_grid[y][x][z].organism.energy -= current_species.reproduction_cost
                                moved_or_eaten[ny][nx][nz] = True
                    
                    elif current_species.type == Species.ANIMAL:
                        # Check for death from old age
                        if current_age >= current_species.max_age:
                            new_grid[y][x][z].organism = None
                            continue
                        
                        # Check if animal is sleeping
                        is_sleeping = self.is_animal_sleeping(x, y, z)
                        
                        # Energy consumption (less when sleeping)
                        if is_sleeping:
                            energy_loss = current_species.idle_energy_consumption * 0.5
                        else:
                            energy_loss = current_species.idle_energy_consumption
                        
                        new_energy = current_energy - energy_loss
                        
                        if new_energy <= 0:
                            new_grid[y][x][z].organism = None
                            continue
                        
                        new_grid[y][x][z].organism = CellOrganism(
                            species=current_species,
                            energy=new_energy,
                            root_energy=0,
                            age=current_age
                        )
                        
                        # Check if overstuffed (won't eat if already full)
                        is_overstuffed = current_energy > current_species.max_energy
                        
                        if not is_sleeping and not is_overstuffed:
                            # Try to eat preferred food first
                            ate = False
                            
                            # Try to eat plants if this animal eats plants
                            if current_species.preferred_plants:
                                plant_neighbors = []
                                preferred_neighbors = []
                                
                                for dy in [-1, 0, 1]:
                                    for dx in [-1, 0, 1]:
                                        for dz in [-1, 0, 1]:
                                            if dx == 0 and dy == 0 and dz == 0:
                                                continue
                                            nx, ny, nz = x + dx, y + dy, z + dz
                                            if (0 <= nx < self.grid_size and 
                                                0 <= ny < self.grid_size and 
                                                0 <= nz < self.grid_size):
                                                if (self.totalgrid[ny][nx][nz].organism is not None and
                                                    self.totalgrid[ny][nx][nz].organism.species.type == Species.PLANT and
                                                    not moved_or_eaten[ny][nx][nz]):
                                                    
                                                    plant_species = self.totalgrid[ny][nx][nz].organism.species
                                                    if plant_species.name in current_species.toxic_plants:
                                                        continue  # Skip toxic plants
                                                        
                                                    plant_neighbors.append((nx, ny, nz))
                                                    if plant_species.name in current_species.preferred_plants:
                                                        preferred_neighbors.append((nx, ny, nz))
                                
                                # Try to eat preferred plants first
                                eat_targets = preferred_neighbors if preferred_neighbors else plant_neighbors
                                
                                if eat_targets and new_grid[y][x][z].organism.energy < current_species.max_energy:
                                    nx, ny, nz = random.choice(eat_targets)
                                    plant_species = self.totalgrid[ny][nx][nz].organism.species
                                    
                                    # More energy from preferred plants
                                    if plant_species.name in current_species.preferred_plants:
                                        energy_gain = current_species.energy_gain_plants * 1.2
                                    else:
                                        energy_gain = current_species.energy_gain_plants * 0.8
                                    
                                    new_grid[y][x][z].organism.energy += min(energy_gain, self.totalgrid[ny][nx][nz].organism.energy)
                                    moved_or_eaten[ny][nx][nz] = True
                                    moved_or_eaten[y][x][z] = True
                                    ate = True
                            
                            # If didn't eat plants (or can't), try to eat animals if this animal eats meat
                            if not ate and current_species.can_eat_meat and current_species.preferred_prey:
                                animal_neighbors = []
                                preferred_prey_neighbors = []
                                
                                for dy in [-1, 0, 1]:
                                    for dx in [-1, 0, 1]:
                                        for dz in [-1, 0, 1]:
                                            if dx == 0 and dy == 0 and dz == 0:
                                                continue
                                            nx, ny, nz = x + dx, y + dy, z + dz
                                            if (0 <= nx < self.grid_size and 
                                                0 <= ny < self.grid_size and 
                                                0 <= nz < self.grid_size):
                                                if (self.totalgrid[ny][nx][nz].organism is not None and
                                                    self.totalgrid[ny][nx][nz].organism.species.type == Species.ANIMAL and
                                                    not moved_or_eaten[ny][nx][nz]):
                                                    
                                                    prey_species = self.totalgrid[ny][nx][nz].organism.species
                                                    animal_neighbors.append((nx, ny, nz))
                                                    if prey_species.name in current_species.preferred_prey:
                                                        preferred_prey_neighbors.append((nx, ny, nz))
                                
                                # Try to eat preferred prey first
                                eat_targets = preferred_prey_neighbors if preferred_prey_neighbors else animal_neighbors
                                
                                if eat_targets and new_grid[y][x][z].organism.energy < current_species.max_energy:
                                    nx, ny, nz = random.choice(eat_targets)
                                    prey_energy = self.totalgrid[ny][nx][nz].organism.energy
                                    
                                    # More energy from preferred prey
                                    prey_species = self.totalgrid[ny][nx][nz].organism.species
                                    if prey_species.name in current_species.preferred_prey:
                                        energy_gain = current_species.energy_gain_meat * 1.2
                                    else:
                                        energy_gain = current_species.energy_gain_meat * 0.8
                                    
                                    new_grid[y][x][z].organism.energy += min(energy_gain, prey_energy)
                                    moved_or_eaten[ny][nx][nz] = True
                                    moved_or_eaten[y][x][z] = True
                                    ate = True
        
        # Update the grid
        self.totalgrid = new_grid
        self.draw_grid()
    
    def clear_grid(self):
        self.totalgrid = [[[TotalCell() for _ in range(self.grid_size)] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_step = 0
        self.current_day = 0
        self.is_day = True
        self.light_level = 1.0
        self.temperature = 1.0
        
        # Update UI
        dpg.set_value("time_text", "Day 1, 00:00")
        dpg.set_value("daylight_text", f"Daylight: {self.current_daylight/2}h")
        dpg.set_value("light_level_text", f"Light: {self.light_level:.2f}")
        dpg.set_value("temp_text", f"Temp: {self.temperature:.2f}")
        dpg.set_value("date_text", "Date: Jan 1")
        
        self.draw_grid()

    def randomize_grid(self):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                for z in range(self.grid_size):
                    self.totalgrid[y][x][z] = TotalCell()
                    rand = random.random()
                    if rand < 0.4:  # 40% chance of being a plant
                        plant_type = random.choice([PlantSpecies.GRASS, PlantSpecies.BUSH, PlantSpecies.TREE])
                        self.totalgrid[y][x][z].organism = CellOrganism(
                            species=plant_type,
                            energy=random.uniform(2, plant_type.max_energy * 0.5),
                            root_energy=random.uniform(2, plant_type.root_energy * 0.5),
                            age=random.randint(0, plant_type.mature_age * 2)
                        )
                    elif rand < 0.425:  # 2.5% chance of being an animal
                        animal_type = random.choice([AnimalSpecies.ZORAFI, AnimalSpecies.LOPAKI, AnimalSpecies.TUVEXI])
                        self.totalgrid[y][x][z].organism = CellOrganism(
                            species=animal_type,
                            energy=random.uniform(2, animal_type.max_energy * 0.5),
                            root_energy=0,
                            age=random.randint(0, animal_type.mature_age * 2)
                        )
                    else:
                        self.totalgrid[y][x][z].organism = None
        
        self.current_step = 0
        self.current_day = 0
        self.is_day = True
        self.light_level = 1.0
        self.temperature = 1.0
        
        # Update UI
        dpg.set_value("time_text", "Day 1, 00:00")
        dpg.set_value("daylight_text", f"Daylight: {self.current_daylight/2}h")
        dpg.set_value("light_level_text", f"Light: {self.light_level:.2f}")
        dpg.set_value("temp_text", f"Temp: {self.temperature:.2f}")
        dpg.set_value("date_text", "Date: Jan 1")
        
        self.draw_grid()
    
    def change_grid_size(self, sender, app_data):
        # Store the new grid size
        new_size = app_data
        
        # Create new grid with the updated size
        new_grid = [[[TotalCell() for _ in range(new_size)] for _ in range(new_size)] for _ in range(new_size)]
        
        # Update all grid references
        self.grid_size = new_size
        self.totalgrid = new_grid
        
        # Update drawlist size
        dpg.configure_item("drawlist", 
                        width=self.grid_size * self.cell_size,
                        height=self.grid_size * self.cell_size)
        
        # Redraw the grid
        self.draw_grid()
    
    def run(self):
        while dpg.is_dearpygui_running():
            if self.running:
                self.step()
                time.sleep(0.1)  # Slow down the simulation
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()

if __name__ == "__main__":
    game = GameOfLife()
    game.run()