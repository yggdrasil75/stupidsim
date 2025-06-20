import dearpygui.dearpygui as dpg
import random
import time
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class SpeciesType(Enum):
    PLANT = 0
    HERBIVORE = 1
    CARNIVORE = 2
    OMNIVORE = 3

@dataclass
class Species:
    name: str
    type: SpeciesType
    color: tuple
    max_energy: float
    energy_gain: float  # For plants: per step in full light, for animals: per food eaten
    energy_consumption: float  # Per step always
    mature_age: int
    reproduction_cost: float
    offspring_energy: float
    reproduction_frequency: int
    temperature_sensitivity: float
    light_sensitivity: float
    cold_resistance: float
    # Plant-specific traits
    root_energy: float = 0  # Energy stored in roots (plants only)
    # Animal-specific traits
    move_speed: int = 0
    move_energy_cost: float = 0  # Energy cost per movement
    max_age: int = 0
    sleep_ratio: float = 0
    diurnal: bool = True
    preferred_food: List[str] = None
    toxic_food: List[str] = None
    
    def mutate(self) -> 'Species':
        """Create a slightly mutated version of this species"""
        mutation_factor = random.uniform(0.8, 1.2)
        
        # Create new name by combining syllables
        syllables = ["zo", "ra", "fi", "lo", "pa", "ki", "tu", "ve", "no", "xi"]
        new_name = "".join(random.choices(syllables, k=random.randint(2, 3))).capitalize()
        
        new_species = Species(
            name=new_name,
            type=self.type,
            color=(
                max(0, min(255, int(self.color[0] * random.uniform(0.8, 1.2)))),
                max(0, min(255, int(self.color[1] * random.uniform(0.8, 1.2)))),
                max(0, min(255, int(self.color[2] * random.uniform(0.8, 1.2))))
            ),
            max_energy=max(1, self.max_energy * mutation_factor),
            energy_gain=max(0.05, self.energy_gain * mutation_factor),
            energy_consumption=max(0.01, self.energy_consumption * mutation_factor),
            mature_age=max(5, int(self.mature_age * random.uniform(0.8, 1.2))),
            reproduction_cost=max(0.5, self.reproduction_cost * mutation_factor),
            offspring_energy=max(0.5, self.offspring_energy * mutation_factor),
            reproduction_frequency=max(1, int(self.reproduction_frequency * random.uniform(0.8, 1.2))),
            temperature_sensitivity=max(0.1, min(1.0, self.temperature_sensitivity * random.uniform(0.8, 1.2))),
            light_sensitivity=max(0.1, min(1.0, self.light_sensitivity * random.uniform(0.8, 1.2))),
            cold_resistance=max(0.1, min(1.0, self.cold_resistance * random.uniform(0.8, 1.2))),
            root_energy=max(0, self.root_energy * random.uniform(0.8, 1.2)) if self.type == SpeciesType.PLANT else 0,
            move_speed=max(0, int(self.move_speed * random.uniform(0.8, 1.2))) if self.type != SpeciesType.PLANT else 0,
            move_energy_cost=max(0, self.move_energy_cost * random.uniform(0.8, 1.2)) if self.type != SpeciesType.PLANT else 0,
            max_age=max(0, int(self.max_age * random.uniform(0.8, 1.2))) if self.type != SpeciesType.PLANT else 0,
            sleep_ratio=max(0, min(0.5, self.sleep_ratio * random.uniform(0.8, 1.2))) if self.type != SpeciesType.PLANT else 0,
            diurnal=self.diurnal if random.random() < 0.8 else not self.diurnal,
            preferred_food=self.preferred_food.copy() if self.preferred_food else None,
            toxic_food=self.toxic_food.copy() if self.toxic_food else None
        )
        
        # Small chance to change type
        if random.random() < 0.005:
            if new_species.type == SpeciesType.PLANT:
                new_species.type = random.choice([SpeciesType.HERBIVORE, SpeciesType.OMNIVORE])
                new_species.move_speed = random.randint(1, 3)
                new_species.move_energy_cost = random.uniform(0.1, 0.5)
                new_species.max_age = random.randint(100, 500)
            else:
                if random.random() < 0.1:
                    new_species.type = SpeciesType.PLANT
                    new_species.move_speed = 0
                    new_species.move_energy_cost = 0
                    new_species.max_age = 0
                    new_species.root_energy = random.uniform(5, 20)
        
        return new_species

def generate_random_species():
    """Generate a completely random species"""
    species_type = random.choice(list(SpeciesType))
    
    # Generate a random color based on type
    if species_type == SpeciesType.PLANT:
        color = (random.randint(50, 150), random.randint(100, 200), random.randint(50, 150))
    elif species_type == SpeciesType.HERBIVORE:
        color = (random.randint(150, 255), random.randint(150, 255), random.randint(100, 200))
    elif species_type == SpeciesType.CARNIVORE:
        color = (random.randint(200, 255), random.randint(100, 150), random.randint(100, 150))
    else:  # OMNIVORE
        color = (random.randint(150, 255), random.randint(150, 200), random.randint(100, 150))
    
    # Generate a name
    syllables = ["zo", "ra", "fi", "lo", "pa", "ki", "tu", "ve", "no", "xi"]
    name = "".join(random.choices(syllables, k=random.randint(2, 3))).capitalize()
    
    # Base traits
    max_energy = random.uniform(5, 50)
    energy_gain = random.uniform(0.1, 2.0)
    energy_consumption = random.uniform(0.01, 0.2)
    mature_age = random.randint(10, 100)
    reproduction_cost = random.uniform(1, 10)
    offspring_energy = random.uniform(1, 5)
    reproduction_frequency = random.randint(1, 20)
    temperature_sensitivity = random.uniform(0.1, 1.0)
    light_sensitivity = random.uniform(0.1, 1.0)
    cold_resistance = random.uniform(0.1, 1.0)
    
    # Plant-specific traits
    root_energy = random.uniform(5, 20) if species_type == SpeciesType.PLANT else 0
    
    # Animal-specific traits
    move_speed = 0
    move_energy_cost = 0
    max_age = 0
    sleep_ratio = 0
    diurnal = True
    preferred_food = []
    toxic_food = []
    
    if species_type != SpeciesType.PLANT:
        move_speed = random.randint(1, 3)
        move_energy_cost = random.uniform(0.1, 0.5)
        max_age = random.randint(100, 1000)
        sleep_ratio = random.uniform(0.1, 0.3)
        diurnal = random.random() < 0.7
        
        # Generate food preferences
        if species_type in [SpeciesType.HERBIVORE, SpeciesType.OMNIVORE]:
            preferred_food = random.sample(["Grass", "Bush", "Algae", "Fruit"], random.randint(1, 2))
            toxic_food = random.sample(["Poison Ivy", "Toxic Shroom"], random.randint(0, 1))
        
        if species_type in [SpeciesType.CARNIVORE, SpeciesType.OMNIVORE]:
            if not preferred_food:
                preferred_food = []
            preferred_food.extend(random.sample(["Herbivore", "Small Carnivore"], random.randint(1, 2)))
    
    return Species(
        name=name,
        type=species_type,
        color=color,
        max_energy=max_energy,
        energy_gain=energy_gain,
        energy_consumption=energy_consumption,
        mature_age=mature_age,
        reproduction_cost=reproduction_cost,
        offspring_energy=offspring_energy,
        reproduction_frequency=reproduction_frequency,
        temperature_sensitivity=temperature_sensitivity,
        light_sensitivity=light_sensitivity,
        cold_resistance=cold_resistance,
        root_energy=root_energy,
        move_speed=move_speed,
        move_energy_cost=move_energy_cost,
        max_age=max_age,
        sleep_ratio=sleep_ratio,
        diurnal=diurnal,
        preferred_food=preferred_food,
        toxic_food=toxic_food
    )

# Default species
Species.GRASS = Species(
    name="Grass",
    type=SpeciesType.PLANT,
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
    root_energy=8  # Grass has substantial root energy
)

Species.OMNIVORE = Species(
    name="Omnivore",
    type=SpeciesType.OMNIVORE,
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
    move_speed=2,
    move_energy_cost=0.2,  # Energy cost per movement
    max_age=500,
    sleep_ratio=0.2,
    diurnal=True,
    preferred_food=["Grass", "Herbivore"],
    toxic_food=["Poison Ivy"]
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
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    for z in range(self.grid_size):
                        if self.totalgrid[x][y][z].organism is not None:
                            self.totalgrid[x][y][z].organism.age += 1
        
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
        if self.totalgrid[x][y][z].organism is None:
            return (0, 0, 0)
            
        species = self.totalgrid[x][y][z].organism.species
        energy = self.totalgrid[x][y][z].organism.energy
        
        if species.type == SpeciesType.PLANT:
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
        else:
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
    
    def get_species_char(self, x, y, z):
        """Get character representation for a cell"""
        if self.totalgrid[x][y][z].organism is None:
            return " "
            
        species = self.totalgrid[x][y][z].organism.species
        if species: 
            return species.name[0].upper()
        else: 
            return " "  # Empty space
    
    def draw_grid(self):
        dpg.delete_item("drawlist", children_only=True)
        
        with dpg.draw_node(parent="drawlist"):
            # Draw cells
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if self.totalgrid[x][y][0].organism is not None:  # Only draw top layer for now
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
        if self.totalgrid[x][y][z].organism is None:
            return False
            
        species = self.totalgrid[x][y][z].organism.species
        if species is None or species.type == SpeciesType.PLANT:
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
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in range(self.grid_size):
                    if moved_or_eaten[x][y][z]:
                        continue  # Skip cells that have already been processed
                        
                    if self.totalgrid[x][y][z].organism is None:
                        continue
                        
                    current_organism = self.totalgrid[x][y][z].organism
                    current_species = current_organism.species
                    current_energy = current_organism.energy
                    current_root_energy = current_organism.root_energy
                    current_age = current_organism.age
                    
                    # All organisms consume energy
                    energy_loss = current_species.energy_consumption
                    
                    # Plants gain energy from light and can use root energy
                    if current_species.type == SpeciesType.PLANT:
                        energy_gain = ((self.light_level * current_species.light_sensitivity) * 
                                      current_species.energy_gain * (self.temperature * current_species.temperature_sensitivity))
                        net_energy = current_energy + energy_gain - energy_loss
                        
                        # Handle energy storage and usage
                        if net_energy > current_species.max_energy:
                            # Store excess in roots (up to root capacity)
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
                            # Use root energy if available
                            energy_needed = -net_energy
                            if current_root_energy >= energy_needed:
                                new_grid[x][y][z].organism = CellOrganism(
                                    species=current_species,
                                    energy=0,
                                    root_energy=current_root_energy - energy_needed,
                                    age=current_age
                                )
                            else:
                                # Not enough energy - plant dies
                                new_grid[x][y][z].organism = None
                                continue
                    else:
                        # Animals just lose energy
                        net_energy = current_energy - energy_loss
                        
                        # Check for death from old age (animals only)
                        if current_age >= current_species.max_age:
                            new_grid[x][y][z].organism = None
                            continue
                        
                        # Check for death from starvation
                        if net_energy <= 0:
                            new_grid[x][y][z].organism = None
                            continue
                        
                        # Create new organism with updated energy
                        new_grid[x][y][z].organism = CellOrganism(
                            species=current_species,
                            energy=net_energy,
                            root_energy=0,  # Animals don't have root energy
                            age=current_age
                        )
                    
                    # Check if organism can survive current conditions
                    survival_chance = (self.temperature + current_species.cold_resistance)
                    if random.random() > survival_chance:
                        new_grid[x][y][z].organism = None
                        continue
                    
                    # Check if organism can reproduce
                    if (new_grid[x][y][z].organism is not None and 
                        new_grid[x][y][z].organism.energy >= current_species.reproduction_cost and 
                        current_age >= current_species.mature_age and 
                        self.current_day % current_species.reproduction_frequency == 0):
                        
                        # Find empty neighboring cells
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
                        
                        # Reproduce to a random empty neighbor
                        if empty_neighbors:
                            nx, ny, nz = random.choice(empty_neighbors)
                            
                            # Small chance of mutation
                            if random.random() < 0.05:  # 5% mutation chance
                                mutated_species = current_species.mutate()
                            else:
                                mutated_species = current_species
                            
                            new_grid[nx][ny][nz].organism = CellOrganism(
                                species=mutated_species,
                                energy=current_species.offspring_energy,
                                root_energy=current_species.offspring_energy * 0.5 if mutated_species.type == SpeciesType.PLANT else 0,
                                age=0
                            )
                            new_grid[x][y][z].organism.energy -= current_species.reproduction_cost
                            moved_or_eaten[nx][ny][nz] = True
                    
                    # Handle animal movement and eating
                    if (current_species.type != SpeciesType.PLANT and 
                        not self.is_animal_sleeping(x, y, z) and 
                        new_grid[x][y][z].organism.energy < current_species.max_energy * 0.9):
                        
                        # Try to find food
                        best_food = None
                        best_food_value = 0
                        best_position = (x, y, z)
                        
                        # Search nearby cells for food
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
                                            
                                            # Check if this is preferred food
                                            if (current_species.preferred_food and 
                                                target_species.name in current_species.preferred_food):
                                                food_value = current_species.energy_gain * 1.5
                                            # Check if this is toxic food
                                            elif (current_species.toxic_food and 
                                                  target_species.name in current_species.toxic_food):
                                                food_value = -current_species.energy_gain  # Negative value
                                            # Neutral food
                                            else:
                                                food_value = current_species.energy_gain * 0.5
                                            
                                            if food_value > best_food_value:
                                                best_food_value = food_value
                                                best_food = (nx, ny, nz)
                                        
                                        # Also consider empty cells as potential movement targets
                                        elif self.totalgrid[nx][ny][nz].organism is None:
                                            if best_food_value <= 0:  # Only move if no food found
                                                best_position = (nx, ny, nz)
                        
                        # If we found food, eat it
                        if best_food is not None and best_food_value > 0:
                            nx, ny, nz = best_food
                            food_energy = self.totalgrid[nx][ny][nz].organism.energy
                            energy_gain = min(current_species.energy_gain, food_energy)
                            
                            # Move to food location and consume it
                            new_grid[nx][ny][nz].organism = CellOrganism(
                                species=current_species,
                                energy=new_grid[x][y][z].organism.energy + energy_gain - current_species.move_energy_cost,
                                root_energy=0,
                                age=current_age
                            )
                            new_grid[x][y][z].organism = None
                            moved_or_eaten[nx][ny][nz] = True
                        # Otherwise, just move randomly if not staying put
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
        # Generate 1-10 random species at startup
        num_species = random.randint(1, 10)
        random_species = [Species.GRASS, Species.OMNIVORE]  # Start with our default species
        for _ in range(num_species):
            random_species.append(generate_random_species())
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in range(self.grid_size):
                    self.totalgrid[x][y][z] = TotalCell()
                    rand = random.random()
                    if rand < 0.3:  # 30% chance of being an organism
                        species = random.choice(random_species)
                        root_energy = random.uniform(0, species.root_energy) if species.type == SpeciesType.PLANT else 0
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