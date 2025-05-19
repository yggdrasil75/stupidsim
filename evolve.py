import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from enum import Enum

class OrganismType(Enum):
    PROKARYOTE = 1
    EUKARYOTE = 2

class EnergySource(Enum):
    CHEMOTROPHY = 1  # Chemical energy
    PHOTOTROPHY = 2  # Light energy
    ORGANOTROPHY = 3  # Organic matter

class MetabolismType(Enum):
    AEROBIC = 1      # Requires oxygen
    ANAEROBIC = 2    # Doesn't require oxygen
    FACULTATIVE = 3  # Can switch between aerobic/anaerobic

class Face:
    def __init__(self):
        # Fixed environmental parameters (won't change except through organism actions)
        self.elevation = random.uniform(0, 5000)  # meters
        self.solar_radiation = 1000  # Fixed solar radiation (W/m²)
        
        # Variable parameters (can be changed by organisms)
        self.water_volume = random.uniform(0, 1000)  # liters
        self.temperature = random.uniform(250, 350)  # Kelvin
        self.groundwater = random.uniform(0, 2000)  # liters
        self.water_depth = random.uniform(0, 1)
        
        # Atmospheric composition (will change due to organism activity)
        self.atmosphere = {
            'N2': random.uniform(70, 80),
            'O2': random.uniform(10, 30),
            'CO2': random.uniform(0.01, 0.5),
            'H2O': random.uniform(0.1, 5),
            'CH4': random.uniform(0.001, 0.1),
            'Ar': random.uniform(0.5, 2)
        }
    
    def get_atmosphere(self):
        return self.atmosphere.copy()
    
    def update_atmosphere(self, changes):
        """Update atmospheric composition based on organism activity"""
        for gas, delta in changes.items():
            if gas in self.atmosphere:
                self.atmosphere[gas] = max(0, self.atmosphere[gas] + delta)
    
    def get_elevation(self):
        return self.elevation
    
    def get_water_depth(self):
        return self.water_depth
    
    def get_water_volume(self):
        return self.water_volume
    
    def get_solar_radiation(self):
        return self.solar_radiation
    
    def get_temperature(self):
        return self.temperature
    
    def get_groundwater(self):
        return self.groundwater

class Environment:
    def __init__(self, face):
        self.face = face
        self.organisms = []
        self.time_elapsed = 0  # years
        self.organic_matter = 0  # Accumulated dead organisms
        
    @property
    def solar_radiation(self):
        return self.face.solar_radiation
    
    @property
    def temperature(self):
        return self.face.temperature
    
    @property
    def groundwater(self):
        return self.face.groundwater
    
    @property
    def surface_water(self):
        return self.face.water_volume
    
    @property
    def atmosphere(self):
        return self.face.get_atmosphere()
    
    @property
    def elevation(self):
        return self.face.elevation
    
    @property
    def pressure(self):
        """Calculate atmospheric pressure based on elevation"""
        return 101325 * math.exp(-self.elevation / 8500)  # Pascals
    
    def update(self, years=1):
        """Update the environment over a number of years"""
        self.time_elapsed += years
        
        # Update all organisms
        for organism in self.organisms[:]:
            organism.update(years, self)
            if not organism.alive:
                self.organic_matter += organism.size * 0.1  # Add to organic matter pool
                self.organisms.remove(organism)
        
        # Reproduction and mutation
        new_organisms = []
        for organism in self.organisms[:]:
            if organism.can_reproduce():
                offspring = organism.reproduce()
                new_organisms.append(offspring)
        self.organisms.extend(new_organisms)
        
        # Random abiogenesis (only early in simulation)
        if random.random() < 0.001 * years and self.time_elapsed < 100:
            # First organisms are always prokaryotes
            self.organisms.append(Microorganism(self, org_type=OrganismType.PROKARYOTE))
        carrying_capacity = 10000000  # Maximum sustainable population
        if len(self.organisms) > carrying_capacity:
            # Reduce reproduction probability when over capacity
            reproduction_scaling = carrying_capacity / len(self.organisms)
        else:
            reproduction_scaling = 1.0
        
        # Reproduction and mutation with population control
        new_organisms = []
        for organism in self.organisms[:]:
            if organism.can_reproduce() and random.random() < reproduction_scaling:
                offspring = organism.reproduce()
                new_organisms.append(offspring)
    
    def get_environmental_fitness(self, organism):
        """Calculate how well suited an organism is to the current environment"""
        fitness = 1.0
        
        # Temperature preference
        temp_diff = abs(organism.optimal_temp - self.temperature)
        fitness *= max(0, 1 - (temp_diff / 50)**2)
        
        # Water availability
        water_pref = organism.water_requirement
        available_water = self.surface_water + self.groundwater
        fitness *= max(0, 1 - abs(water_pref - available_water) / (water_pref + 100))
        
        # Atmospheric requirements
        atmo_fitness = 1.0
        for gas, (min_req, max_req) in organism.atmospheric_needs.items():
            available = self.atmosphere.get(gas, 0)
            if available < min_req or available > max_req:
                atmo_fitness *= 0.5  # Significant penalty for wrong atmosphere
        fitness *= atmo_fitness
        
        # Pressure effects
        pressure_diff = abs(math.log(organism.optimal_pressure/self.pressure))
        fitness *= max(0, 1 - pressure_diff / 2)
        
        return max(0, min(1, fitness))

class Microorganism:
    def __init__(self, environment, parent=None, org_type=None):
        self.environment = environment
        self.alive = True
        self.age = 0  # years
        self.energy = 100
        self.generation = 0
        self.size = random.uniform(0.1, 10)  # micrometers
        self.metabolic_rate = random.uniform(0.1, 2.0)
        self.optimal_temp = random.uniform(
            max(200, environment.temperature - 30), 
            min(400, environment.temperature + 30)
        )
        self.optimal_pressure = random.uniform(
            environment.pressure * 0.5, 
            environment.pressure * 1.5
        )
        self.water_requirement = random.uniform(0.5, 1.5) * (
            environment.surface_water + environment.groundwater
        )
        self.reproduction_threshold = random.uniform(150, 300)
        self.mutation_rate = random.uniform(0.01, 0.02)
        
        # Determine organism type (prokaryote or eukaryote)
        if org_type:
            self.org_type = org_type
        else:
            # Eukaryotes can only evolve from prokaryotes
            self.org_type = random.choices(
                [OrganismType.PROKARYOTE, OrganismType.EUKARYOTE],
                weights=[0.95, 0.05],
                k=1
            )[0]
        
        # Energy source and metabolism
        self.energy_source = self.determine_energy_source()
        self.metabolism = self.determine_metabolism()
        
        # Initialize needs based on type and energy source
        self.atmospheric_needs = self.generate_atmospheric_needs()
        self.resource_needs = self.generate_resource_needs()
        
        if parent:
            self.generation = parent.generation + 1
            self.inherit_traits(parent)
    
    def determine_energy_source(self):
        """Determine how this organism gets energy"""
        if self.org_type == OrganismType.PROKARYOTE:
            # Prokaryotes can use any energy source
            return random.choice(list(EnergySource))
        else:
            # Eukaryotes are more limited (typically phototrophic or organotrophic)
            return random.choice([EnergySource.PHOTOTROPHY, EnergySource.ORGANOTROPHY])
    
    def determine_metabolism(self):
        """Determine metabolic type"""
        if self.org_type == OrganismType.PROKARYOTE:
            # Prokaryotes can have any metabolism
            return random.choice(list(MetabolismType))
        else:
            # Eukaryotes are typically aerobic or facultative
            return random.choice([MetabolismType.AEROBIC, MetabolismType.FACULTATIVE])
    
    def generate_atmospheric_needs(self):
        """Generate atmospheric requirements based on organism type and metabolism"""
        needs = {}
        
        # All organisms need some CO2 and H2O
        needs['CO2'] = (0.001, 10)  # min, max
        needs['H2O'] = (0.1, 20)
        
        # Oxygen requirements depend on metabolism
        if self.metabolism == MetabolismType.AEROBIC:
            needs['O2'] = (5, 30)  # Need significant oxygen
        elif self.metabolism == MetabolismType.ANAEROBIC:
            needs['O2'] = (0, 1)   # Can't tolerate much oxygen
        else:  # FACULTATIVE
            needs['O2'] = (0, 20)  # Can handle a range
        
        return needs
    
    def generate_resource_needs(self):
        """Generate resource needs based on energy source"""
        needs = {}
        
        if self.energy_source == EnergySource.PHOTOTROPHY:
            # Phototrophs need light and certain chemicals
            needs['light'] = 1.0
            needs['CO2'] = 0.5
            if random.random() < 0.5:
                needs['H2S'] = 0.1  # Some phototrophs use hydrogen sulfide
            
        elif self.energy_source == EnergySource.CHEMOTROPHY:
            # Chemotrophs need specific chemicals
            possible_chemicals = ['H2', 'H2S', 'Fe2+', 'NH3', 'CH4']
            num_chemicals = random.randint(1, 2)
            for _ in range(num_chemicals):
                chem = random.choice(possible_chemicals)
                needs[chem] = random.uniform(0.2, 1.0)
            
        else:  # ORGANOTROPHY
            # Organotrophs need organic matter
            needs['organic_matter'] = 0.5
        
        return needs
    
    def inherit_traits(self, parent):
        """Inherit traits from parent with possible mutations"""
        # Basic traits
        self.size = max(0.1, parent.size + random.uniform(-0.5, 0.5))
        self.metabolic_rate = max(0.1, parent.metabolic_rate + random.uniform(-0.1, 0.1))
        self.optimal_temp = parent.optimal_temp + random.uniform(-2, 2)
        self.optimal_pressure = parent.optimal_pressure * random.uniform(0.9, 1.1)
        self.water_requirement = max(0, parent.water_requirement * random.uniform(0.9, 1.1))
        self.reproduction_threshold = max(50, parent.reproduction_threshold * random.uniform(0.9, 1.1))
        self.mutation_rate = max(0.01, min(0.5, parent.mutation_rate * random.uniform(0.9, 1.1)))
        
        # Inherit type (with small chance of prokaryote->eukaryote transition)
        if parent.org_type == OrganismType.PROKARYOTE and random.random() < 0.001:
            self.org_type = OrganismType.EUKARYOTE
        else:
            self.org_type = parent.org_type
        
        # Inherit energy source and metabolism (with possible mutations)
        self.energy_source = parent.energy_source
        self.metabolism = parent.metabolism
        if random.random() < self.mutation_rate:
            self.energy_source = random.choice(list(EnergySource))
        if random.random() < self.mutation_rate and self.org_type == OrganismType.PROKARYOTE:
            self.metabolism = random.choice(list(MetabolismType))
        
        # Inherit needs with possible mutations
        self.atmospheric_needs = {}
        for gas, (min_req, max_req) in parent.atmospheric_needs.items():
            new_min = max(0, min_req * random.uniform(0.9, 1.1))
            new_max = max(new_min + 0.1, max_req * random.uniform(0.9, 1.1))
            self.atmospheric_needs[gas] = (new_min, new_max)
        
        self.resource_needs = {}
        for res, amount in parent.resource_needs.items():
            new_amount = max(0.01, amount * random.uniform(0.8, 1.2))
            self.resource_needs[res] = new_amount
        
        # Small chance to add or remove a resource need
        if random.random() < self.mutation_rate:
            if random.random() < 0.5 and len(self.resource_needs) > 1:
                res_to_remove = random.choice(list(self.resource_needs.keys()))
                del self.resource_needs[res_to_remove]
            else:
                possible_new = ['H2', 'H2S', 'Fe2+', 'NH3', 'CH4', 'organic_matter']
                possible_new = [res for res in possible_new if res not in self.resource_needs]
                if possible_new:
                    new_res = random.choice(possible_new)
                    self.resource_needs[new_res] = random.uniform(0.1, 1.0)
    
    def update(self, years, environment):
        """Update the organism's state over time"""
        self.age += years
        
        # Environmental fitness affects energy gain
        env_fitness = environment.get_environmental_fitness(self)
        
        # Calculate energy gain based on energy source
        energy_gain = 0
        atmospheric_changes = {}
        
        if self.energy_source == EnergySource.PHOTOTROPHY:
            # Photosynthesis: CO2 + H2O + light -> Organic compounds + O2
            if 'light' in self.resource_needs and 'CO2' in self.resource_needs:
                light_available = environment.solar_radiation / 1000  # Normalize
                co2_available = environment.atmosphere.get('CO2', 0)
                
                # Calculate how much we can process
                light_used = min(self.resource_needs['light'], light_available)
                co2_used = min(self.resource_needs['CO2'], co2_available)
                process_amount = min(light_used, co2_used)
                
                energy_gain = process_amount * self.metabolic_rate * years * env_fitness
                
                # Update atmospheric changes
                atmospheric_changes['CO2'] = -process_amount * 0.1  # Consume CO2
                atmospheric_changes['O2'] = process_amount * 0.1    # Produce O2
        
        elif self.energy_source == EnergySource.CHEMOTROPHY:
            # Chemical energy from inorganic compounds
            total_chem_available = 1.0
            for chem, amount in self.resource_needs.items():
                if chem in environment.atmosphere:
                    total_chem_available = min(total_chem_available, 
                                              environment.atmosphere[chem] / amount)
            
            energy_gain = total_chem_available * self.metabolic_rate * years * env_fitness
            
            # Update atmospheric changes (simplified)
            for chem in self.resource_needs:
                if chem in environment.atmosphere:
                    atmospheric_changes[chem] = -self.resource_needs[chem] * 0.1
        
        else:  # ORGANOTROPHY
            # Energy from consuming organic matter
            organic_available = environment.organic_matter
            needed = self.resource_needs.get('organic_matter', 0)
            
            if needed > 0 and organic_available > 0:
                consumed = min(needed, organic_available * 0.1)  # Can only consume a fraction
                energy_gain = consumed * self.metabolic_rate * years * env_fitness
                environment.organic_matter -= consumed
                
                # Organic matter breakdown produces CO2
                atmospheric_changes['CO2'] = consumed * 0.2
        
        # Apply atmospheric changes
        environment.face.update_atmosphere(atmospheric_changes)
        
        # Lose energy based on metabolic rate and size
        energy_loss = self.metabolic_rate * years * (1 + self.size/10)
        
        # Adjust for organism type
        if self.org_type == OrganismType.EUKARYOTE:
            # Eukaryotes are more complex but less efficient
            energy_loss *= 1.2
            energy_gain *= 0.9
        
        # Net energy change
        self.energy += energy_gain - energy_loss
        
        # Death conditions
        if self.energy <= 0 or self.age > (10 if self.org_type == OrganismType.PROKARYOTE else 20):
            self.alive = False
    
    def can_reproduce(self):
        """Check if organism can reproduce with more conditions"""
        # Basic requirements
        if (self.energy <= self.reproduction_threshold or 
            self.age < (0.2 if self.org_type == OrganismType.PROKARYOTE else 0.5)):
            return False
        
        # Eukaryotes need more energy to reproduce
        if self.org_type == OrganismType.EUKARYOTE and self.energy < self.reproduction_threshold * 1.5:
            return False
        
        # Random chance based on environmental fitness
        env_fitness = self.environment.get_environmental_fitness(self)
        reproduction_chance = 0.5 * env_fitness
        
        # Prokaryotes reproduce more readily
        if self.org_type == OrganismType.PROKARYOTE:
            reproduction_chance *= 1.5
        
        return random.random() < reproduction_chance
    
    def reproduce(self):
        """Create an offspring organism with more realistic energy costs"""
        # Eukaryotes reproduce more slowly and at higher cost
        if self.org_type == OrganismType.EUKARYOTE:
            reproduction_cost = self.energy * 0.7  # 70% of energy goes to reproduction
            self.energy *= 0.3  # Parent keeps 30%
            offspring_energy = reproduction_cost * 0.8  # 80% efficiency
        else:
            # Prokaryotes reproduce more efficiently
            reproduction_cost = self.energy * 0.5  # 50% of energy goes to reproduction
            self.energy *= 0.5  # Parent keeps 50%
            offspring_energy = reproduction_cost * 0.9  # 90% efficiency
        
        # Create offspring
        offspring = Microorganism(self.environment, parent=self)
        offspring.energy = offspring_energy
        
        # Add small chance of death after reproduction (especially for prokaryotes)
        if random.random() < (0.1 if self.org_type == OrganismType.PROKARYOTE else 0.02):
            self.alive = False
        
        return offspring

def run_simulation(years=1000, initial_organisms=5):
    """Run a complete evolutionary simulation"""
    # Create a face and environment
    face = Face()
    environment = Environment(face)
    
    # Add initial organisms (all prokaryotes)
    for _ in range(initial_organisms):
        environment.organisms.append(Microorganism(
            environment, 
            org_type=OrganismType.PROKARYOTE
        ))
    
    # Prepare data collection
    history = {
        'population': [],
        'prokaryotes': [],
        'eukaryotes': [],
        'avg_size': [],
        'phototrophs': [],
        'chemotrophs': [],
        'organotrophs': [],
        'o2_level': [],
        'co2_level': [],
        'temperature': [],
        'diversity': []
    }
    
    # Run simulation
    for year in range(years):
        environment.update()
        
        # Record data
        history['population'].append(len(environment.organisms))
        history['prokaryotes'].append(sum(
            1 for org in environment.organisms 
            if org.org_type == OrganismType.PROKARYOTE
        ))
        history['eukaryotes'].append(sum(
            1 for org in environment.organisms 
            if org.org_type == OrganismType.EUKARYOTE
        ))
        history['o2_level'].append(environment.atmosphere.get('O2', 0))
        history['co2_level'].append(environment.atmosphere.get('CO2', 0))
        history['temperature'].append(environment.temperature)
        
        if environment.organisms:
            # Count energy sources
            history['phototrophs'].append(sum(
                1 for org in environment.organisms 
                if org.energy_source == EnergySource.PHOTOTROPHY
            ))
            history['chemotrophs'].append(sum(
                1 for org in environment.organisms 
                if org.energy_source == EnergySource.CHEMOTROPHY
            ))
            history['organotrophs'].append(sum(
                1 for org in environment.organisms 
                if org.energy_source == EnergySource.ORGANOTROPHY
            ))
            
            # Average size
            history['avg_size'].append(sum(
                o.size for o in environment.organisms
            )/len(environment.organisms))
            
            # Measure diversity by counting unique resource need combinations
            resource_combos = set()
            for org in environment.organisms:
                combo = frozenset(org.resource_needs.items())
                resource_combos.add(combo)
            history['diversity'].append(len(resource_combos))
        else:
            history['phototrophs'].append(0)
            history['chemotrophs'].append(0)
            history['organotrophs'].append(0)
            history['avg_size'].append(0)
            history['diversity'].append(0)
        
        if year % 100 == 0:
            print(f"Year {year}: Population {history['population'][-1]} "
                  f"(Pro: {history['prokaryotes'][-1]}, Eu: {history['eukaryotes'][-1]})")
    
    return environment, history

def plot_results(history):
    """Visualize the simulation results"""
    plt.figure(figsize=(18, 12))
    
    # Population plot
    plt.subplot(3, 3, 1)
    plt.plot(history['population'], label='Total')
    plt.plot(history['prokaryotes'], label='Prokaryotes')
    plt.plot(history['eukaryotes'], label='Eukaryotes')
    plt.title('Population Over Time')
    plt.xlabel('Years')
    plt.ylabel('Number of Organisms')
    plt.legend()
    
    # Energy sources plot
    plt.subplot(3, 3, 2)
    plt.plot(history['phototrophs'], label='Phototrophs')
    plt.plot(history['chemotrophs'], label='Chemotrophs')
    plt.plot(history['organotrophs'], label='Organotrophs')
    plt.title('Energy Sources Over Time')
    plt.xlabel('Years')
    plt.ylabel('Count')
    plt.legend()
    
    # Atmospheric gases
    plt.subplot(3, 3, 3)
    plt.plot(history['o2_level'], label='O2')
    plt.plot(history['co2_level'], label='CO2')
    plt.title('Atmospheric Composition')
    plt.xlabel('Years')
    plt.ylabel('Concentration')
    plt.legend()
    
    # Average size plot
    plt.subplot(3, 3, 4)
    plt.plot(history['avg_size'])
    plt.title('Average Organism Size')
    plt.xlabel('Years')
    plt.ylabel('Size (µm)')
    
    # Temperature plot
    plt.subplot(3, 3, 5)
    plt.plot(history['temperature'])
    plt.title('Environment Temperature')
    plt.xlabel('Years')
    plt.ylabel('Temperature (K)')
    
    # Diversity plot
    plt.subplot(3, 3, 6)
    plt.plot(history['diversity'])
    plt.title('Genetic Diversity')
    plt.xlabel('Years')
    plt.ylabel('Unique Resource Combinations')
    
    plt.tight_layout()
    plt.show()

def analyze_organisms(environment):
    """Print analysis of final organisms"""
    if not environment.organisms:
        print("No organisms survived!")
        return
    
    print(f"\nFinal Population Analysis ({len(environment.organisms)} organisms):")
    
    # Count by type
    prokaryotes = [org for org in environment.organisms 
                  if org.org_type == OrganismType.PROKARYOTE]
    eukaryotes = [org for org in environment.organisms 
                 if org.org_type == OrganismType.EUKARYOTE]
    
    print(f"\nOrganism Types:")
    print(f"Prokaryotes: {len(prokaryotes)}")
    print(f"Eukaryotes: {len(eukaryotes)}")
    
    # Energy sources
    print("\nEnergy Sources:")
    for source in EnergySource:
        count = sum(1 for org in environment.organisms 
                   if org.energy_source == source)
        print(f"{source.name}: {count}")
    
    # Metabolism types
    print("\nMetabolism Types:")
    for metab in MetabolismType:
        count = sum(1 for org in environment.organisms 
                   if org.metabolism == metab)
        print(f"{metab.name}: {count}")
    
    # Resource needs
    print("\nCommon Resource Needs:")
    resource_counts = defaultdict(int)
    for org in environment.organisms:
        for res in org.resource_needs:
            resource_counts[res] += 1
    
    for res, count in sorted(resource_counts.items(), key=lambda x: -x[1]):
        print(f"{res}: {count} organisms")

if __name__ == "__main__":
    # Run the simulation
    final_env, history = run_simulation(years=2000, initial_organisms=1000)
    
    # Show results
    plot_results(history)
    analyze_organisms(final_env)