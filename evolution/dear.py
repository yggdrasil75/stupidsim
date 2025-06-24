from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import random
from typing import Optional
import dearpygui.dearpygui as dpg
import string

ALPHABET = string.ascii_letters
RANDOMCHAR = string.printable

@dataclass
class part(ABC): # each part is baked into the species stats, separted here to allow for organisms to lose a limb
    name: str = ""
    size: float = 1.0  # percent of parent part
    vital: bool = False  # whether losing this part is fatal
    
    # General part properties
    health: float = 10.0  # base health of the part
    regrowth_rate: float = 0.0  # how quickly the part regrows (0 for no regrowth)
    energyCost: float = 0.1  # energy cost to maintain this part
    energyProducer: bool = False
    subparts: list['part'] = field(default_factory=list)
    statCache: Optional[dict[str, float|bool]] = None
    minTemp: float = -50.0  
    maxTemp: float = 150.0
    mutationRate: float = 0.0001

    def __post_init__(self):
        if len(self.subparts) == 0:
            self.setup_default_subparts()
        self.calculateStats(True)

    @abstractmethod
    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        if self.statCache is not None and not recalc:
            return self.statCache
        stats = {}
        stats['health'] = self.health
        stats['regrowthrate'] = self.regrowth_rate
        stats['idleE'] = self.energyCost
        if self.vital:
            stats['tempSensCold'] = self.minTemp
            stats['tempSensHot'] = self.maxTemp
        else:
            stats[f'{self.name}.tempSensCold'] = self.minTemp
            stats[f'{self.name}.tempSensHot'] = self.maxTemp
        if self.energyProducer:
            stats['sleepE'] = self.energyCost * 0.5
            stats['dormE'] = self.energyCost * 0.1
        else:
            stats['sleepE'] = self.energyCost * 0.1
            stats['dormE'] = self.energyCost * 0.01

        for subpart in self.subparts:
            substats = subpart.calculateStats(recalc)
            for key, value in substats.items():
                if key in stats:
                    stats[key] += (value * subpart.size)
                else:
                    stats[key] = (value * subpart.size)
        self.statCache = stats
        return self.statCache

@dataclass
class skin(part):
    name: str = ""
    #type: str = random.choice(["fur", "scales", "bark"])
    class Skin(Enum):
        FUR = 0
        SCALES = 1
        BARK = 2
        FEATHERS = 3
        CHITIN = 4
        MEMBRANE = 6
        EXOSKELETON = 7
        CUTICLE = 8
        CORK = 9
        LENTICEL = 10
        THORN = 11
    type: Skin = Skin.FUR
    thickness: float = 0.5  # current thickness (0-max_thickness)
    max_thickness: float = 2.0  # maximum thickness in inches
    pattern: str = "solid"  # pattern type (stripes, spots, etc.)
    camouflage_effectiveness: float = 0.0  # 0-1 scale
    color_change_speed: float = 0.0  # how quickly it can change colors
    insulation_factor: float = 0.5  # how well it retains heat
    cooling_factor: float = 0.5  # how well it dissipates heat
    can_photosynthesize: bool = False
    luminescent: bool = False
    luminense: float = 0.0
    water_retention: float = 0.0  # Important for plants
    gas_exchange: float = 1.0 # For stomata-like structures

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        substats['armor'] = self.thickness * 0.5  # Thicker skin provides more armor
        substats['camouflage'] = self.camouflage_effectiveness
        substats['insulation'] = self.insulation_factor
        substats['cooling'] = self.cooling_factor
        substats['water_retention'] = self.water_retention
        substats['gas_exchange'] = self.gas_exchange
        
        if self.can_photosynthesize:
            substats['photosynthesis'] = 0.5 * self.thickness  # Photosynthetic capability
        if self.luminescent:
            substats['luminescence'] = self.luminense  # Basic luminescence value
            substats['camouflage'] -= self.luminense
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value * self.thickness)
            else:
                stats[stat] = (value * self.thickness)
        return stats

@dataclass
class limb(part):
    class LimbType(Enum):
        ARM = 0
        LEG = 1
        WING = 2
        TENTACLE = 3
        FIN = 4
        TAIL = 5
        PSEUDOPOD = 6
    type: LimbType = LimbType.ARM
    strength: float = 1.0  # physical strength
    dexterity: float = 1.0  # fine motor control
    reach: float = 1.0  # how far it can extend
    prehensile: bool = False  # can grasp objects
    # Movement properties
    movement_speed: float = 1.0  # contributes to overall speed
    movement_cost: float = 0.1  # energy cost per movement
    # Special abilities - assume these mean that this is for the task, if not, then it can still be used to do the task, just at a lower rate.
    can_climb: bool = False
    can_swim: bool = False
    can_dig: bool = False

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats  = {}
        substats['strength'] = self.strength * self.size
        substats['dexterity'] = self.dexterity
        substats['reach'] = self.reach * self.size
        substats['movement_speed'] = self.movement_speed * self.size
        substats['movement_cost'] = self.movement_cost * self.size
        
        if self.prehensile:
            substats['manipulation'] = 0.7 * self.dexterity
        else:
            substats['manipulation'] = 0.1 * self.dexterity
            
        # Movement capabilities
        if self.can_climb:
            substats['climbing'] = (0.5 * self.strength) + (self.dexterity * 0.1)
        else:
            substats['climbing'] = (0.1 * self.strength) + (self.dexterity * 0.1)
        if self.can_swim:
            substats['swimming'] = 0.6 * self.strength + (self.dexterity * 0.1)
        else:
            substats['swimming'] = 0.1 * self.strength + (self.dexterity * 0.1)
        if self.can_dig:
            substats['digging'] = 0.7 * self.strength + (self.dexterity * 0.1)
        else:
            substats['digging'] = 0.05 * self.strength + (self.dexterity * 0.1)
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value * self.size)
            else:
                stats[stat] = (value * self.size)
        return stats
    
    def setup_default_subparts(self):
        skin_type = skin.Skin.FUR if self.type != self.LimbType.FIN else skin.Skin.SCALES
        self.subparts.append(skin(
            name=f"{self.name}_covering",
            type=skin_type,
            size=0.8,
            thickness=0.3,
            energyCost=0.05
        ))
        
        self.subparts.append(internal(
            name="muscles",
            type=internal.InternalType.OTHER,
            efficiency=self.strength,
            capacity=self.dexterity,
            size=0.7,
            energyCost=0.1
        ))
        if self.type in [self.LimbType.ARM, self.LimbType.LEG, self.LimbType.TAIL]:
            self.subparts.append(weapon(
                name="claws",
                type=weapon.WeaponType.CLAW,
                size=0.05,
                damage=0.5,
                energyCost=0.01,
                slash_damage=0.5
            ))

@dataclass
class sensory(part):
    class SensoryType(Enum):
        EYE = 0
        EAR = 1
        NOSE = 2
        ANTENNA = 4
    
    type: SensoryType = SensoryType.EYE
    range: float = 1.0  # detection range
    sensitivity: float = 1.0  # how sensitive it is
    precision: float = 1.0  # how precise the information is
    night_vision: bool = False
    thermal_vision: bool = False
    can_see_colors: bool = True
    underwater_effective: bool = False
    air_effective: bool = True
    active_cost: float = 0.005  # cost when actively used
    passive_cost: float = 0.001  # cost when just present
    angle: float = 90 # angle of perception.

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        substats['sensory_range'] = self.range * self.size
        substats['sensory_sensitivity'] = self.sensitivity
        substats['sensory_precision'] = self.precision
        substats['sensory_angle'] = self.angle
        if self.night_vision:
            substats['night_vision'] = 0.5 * self.sensitivity
        if self.thermal_vision:
            substats['thermal_vision'] = 0.3 * self.sensitivity
        if not self.can_see_colors:
            substats['color_vision'] = 0
        else:
            substats['color_vision'] = 1.0
        if self.underwater_effective:
            substats['underwater_sensing'] = 1.0
        if not self.air_effective:
            substats['air_sensing'] = 0.8
        substats['sensory_active_cost'] = self.active_cost
        substats['sensory_passive_cost'] = self.passive_cost
        substats['visionangle'] = self.angle
        
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value * self.sensitivity)
            else:
                stats[stat] = (value * self.sensitivity)
        return stats

@dataclass
class internal(part):
    class InternalType(Enum):
        HEART = 0
        LUNG = 1
        STOMACH = 2
        BRAIN = 3
        GILL = 4
        LIVER = 5
        KIDNEY = 6
        GLAND = 7
        OTHER = 8
    
    type: InternalType = InternalType.HEART
    efficiency: float = 1.0  # how well it performs its function
    capacity: float = 1.0  # how much it can handle
    can_regenerate: bool = False

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Internal organ stats
        substats[f'{self.type}.efficiency'] = self.efficiency
        substats[f'{self.type}.capacity'] = self.capacity
        
        # Type-specific bonuses
        if self.type == self.InternalType.HEART:
            substats['circulation'] = self.efficiency * 2.0
        elif self.type == self.InternalType.LUNG:
            substats['oxygenation'] = self.efficiency * 1.5
        elif self.type == self.InternalType.GILL:
            substats['water_oxygenation'] = self.efficiency * 2.0
        elif self.type == self.InternalType.BRAIN:
            substats['intelligence'] = self.efficiency * 0.5
            substats['nerve_speed'] = self.efficiency * 1.0
        elif self.type == self.InternalType.STOMACH:
            substats['digestion'] = self.efficiency * 1.5
        elif self.type == self.InternalType.LIVER:
            substats['toxin_processing'] = self.efficiency * 1.0
        elif self.type == self.InternalType.KIDNEY:
            substats['filtration'] = self.efficiency * 1.2
        elif self.type == self.InternalType.GLAND:
            substats['chemical_production'] = self.efficiency * 1.0
            
        if self.can_regenerate:
            substats['regeneration'] = 0.2  # Small regeneration bonus
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value)
            else:
                stats[stat] = (value)
        return stats

@dataclass
class weapon(part):
    class WeaponType(Enum):
        CLAW = 0
        FANG = 1
        HORN = 2
        TUSK = 3
        STINGER = 4
        SPINE = 5
        BEAK = 6
        TAIL_SPIKE = 7
    
    type: WeaponType = WeaponType.CLAW
    damage: float = 1.0  # base damage
    attack_speed: float = 1.0  # how fast it can attack
    reach: float = 0.5  # attack range
    slash_damage: float = 0.0
    pierce_damage: float = 0.0
    blunt_damage: float = 0.0
    venomous: bool = False
    retractable: bool = False
    attack_cost: float = 0.1  # energy per attack

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Base weapon stats
        substats['damage'] = self.damage * self.size
        substats['attack_speed'] = self.attack_speed
        substats['attack_reach'] = self.reach
        substats['attack_cost'] = self.attack_cost
        
        # Damage types
        substats['slash_damage'] = self.slash_damage * self.size
        substats['pierce_damage'] = self.pierce_damage * self.size
        substats['blunt_damage'] = self.blunt_damage * self.size
        
        # Special properties
        if self.venomous:
            substats['venom_potency'] = 0.5  # Base venom potency
        if self.retractable:
            substats['concealment'] = 0.3  # Bonus to hiding the weapon
            
        # Type-specific bonuses
        if self.type == self.WeaponType.CLAW:
            substats['slash_damage'] += 0.5 * self.damage
        elif self.type == self.WeaponType.FANG:
            substats['pierce_damage'] += 0.7 * self.damage
        elif self.type == self.WeaponType.HORN:
            substats['pierce_damage'] += 0.5 * self.damage
            substats['blunt_damage'] += 0.3 * self.damage
        elif self.type == self.WeaponType.STINGER:
            substats['pierce_damage'] += 0.3 * self.damage
            substats['venom_potency'] = substats.get('venom_potency', 0) + 0.5
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value)
            else:
                stats[stat] = (value)
        return stats
    
    def setup_default_subparts(self):
        if self.venomous:
            self.subparts.append(internal(
                name="venom_gland",
                type=internal.InternalType.GLAND,
                efficiency=1.0,
                size=0.3,
                energyCost=0.05
            ))

@dataclass
class appendage(part):
    class AppendageType(Enum):
        SHELL = 0
        SPINE = 1
        PLATE = 2
        FROND = 3
        TUBE = 4
        POD = 5
        SAC = 6
        TENDRIL = 7
        THRON = 8
    
    type: AppendageType = AppendageType.SHELL
    coverage: float = 0.5  # how much of the body it covers (0-1)
    hardness: float = 1.0  # resistance to damage
    poison_production: bool = False
    venom_production: bool = False # typically with an 8.
    storesE: bool = False
    stores_water: bool = False
    stores_air: bool = False

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Base appendage stats
        substats['coverage'] = self.coverage
        substats['hardness'] = self.hardness
        
        # Storage capabilities
        if self.storesE:
            substats['energy_storage'] = 0.5 * self.size
        if self.stores_water:
            substats['water_storage'] = 0.3 * self.size
        if self.stores_air:
            substats['air_storage'] = 0.2 * self.size
            
        # Special properties
        if self.poison_production:
            substats['poison_production'] = 0.3
        if self.venom_production:
            substats['venom_production'] = 0.4
            
        # Type-specific bonuses
        if self.type == self.AppendageType.SHELL:
            substats['armor'] = 1.0 * self.hardness * self.coverage
        elif self.type == self.AppendageType.SPINE:
            substats['defense'] = 0.5 * self.hardness
        elif self.type == self.AppendageType.PLATE:
            substats['armor'] = 0.7 * self.hardness * self.coverage
        elif self.type == self.AppendageType.TENDRIL:
            substats['manipulation'] = 0.3 * self.coverage
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value)
            else:
                stats[stat] = (value)
        return stats

@dataclass
class root(part):
    class RootType(Enum):
        FIBROUS = 0
        TAP = 1
        AERIAL = 2
        PROP = 3
        STORAGE = 4
    
    type: RootType = RootType.FIBROUS
    depth: float = 1.0 # How deep the roots go
    spread: float = 1.0 # How wide the roots spread
    absorption_rate: float = 1.0 # Nutrient/water absorption efficiency
    nitrogen_fixing: bool = False # Can fix nitrogen from air
    storage_capacity: float = 0.0 # For storing water/nutrients
    can_propagate: bool = False  # Can grow new plants from roots

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Root system stats
        substats['root_depth'] = self.depth
        substats['root_spread'] = self.spread
        substats['absorption_rate'] = self.absorption_rate
        
        # Special capabilities
        if self.nitrogen_fixing:
            substats['nitrogen_fixing'] = 0.5
        if self.storage_capacity > 0:
            substats['root_storage'] = self.storage_capacity
        if self.can_propagate:
            substats['vegetative_propagation'] = 0.3
            
        # Type-specific bonuses
        if self.type == self.RootType.TAP:
            substats['root_depth'] *= 1.5
            substats['absorption_rate'] *= 1.2
        elif self.type == self.RootType.AERIAL:
            substats['air_absorption'] = 0.5
        elif self.type == self.RootType.STORAGE:
            substats['root_storage'] = substats.get('root_storage', 0) + 1.0
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value)
            else:
                stats[stat] = (value)
        return stats

@dataclass
class stem(part):
    class StemType(Enum):
        WOODY = 0 # Tree trunks
        HERBACEOUS = 1 # Soft, green stems
        VINE = 2 # Climbing stems
        RHIZOME = 3 # Underground horizontal stems
        TUBER = 4 # Swollen underground stems (potatoes)
        CORM = 5 # Short, vertical underground stems
        BULB = 6 # Underground storage (onions)
    
    type: StemType = StemType.HERBACEOUS
    height: float = 1.0 # Stem height
    flexibility: float = 0.5 # How much it can bend
    structural_strength: float = 1.0 # Support capability
    storage_capacity: float = 0.0 # For storing nutrients
    photosynthetic: bool = True  # Can perform photosynthesis
    annual_growth_rings: bool = False # Shows yearly growth

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Stem structure stats
        substats['height'] = self.height
        substats['flexibility'] = self.flexibility
        substats['structural_strength'] = self.structural_strength
        
        # Special capabilities
        if self.photosynthetic:
            substats['photosynthesis'] = 0.3 * self.height
        if self.storage_capacity > 0:
            substats['stem_storage'] = self.storage_capacity
        if self.annual_growth_rings:
            substats['age_recording'] = 1.0  # Can determine age
            
        # Type-specific bonuses
        if self.type == self.StemType.WOODY:
            substats['structural_strength'] *= 2.0
            substats['durability'] = 1.0
        elif self.type == self.StemType.VINE:
            substats['climbing'] = 0.7 * self.flexibility
        elif self.type == self.StemType.TUBER:
            substats['stem_storage'] = substats.get('stem_storage', 0) + 1.5
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value)
            else:
                stats[stat] = (value)
        return stats
    
    def setup_default_subparts(self):
        if self.type == self.StemType.WOODY:
            self.subparts.append(skin(
                name="bark",
                type=skin.Skin.BARK,
                size=0.9,
                thickness=0.5,
                energyCost=0.02
            ))
        else:
            self.subparts.append(skin(
                name="epidermis",
                type=skin.Skin.CUTICLE,
                size=0.9,
                thickness=0.1,
                energyCost=0.01
            ))

@dataclass
class leaf(part):
    class LeafType(Enum):
        NEEDLE = 0 # Conifer needles
        BROAD = 1 # Standard broad leaves
        SUCCULENT = 2 # Water-storing leaves
        SCALE = 3 # Small, scale-like leaves
        TENDRILL = 4 # Modified for climbing
        SPINE = 5 # Modified for defense
        TRAP = 6 # Carnivorous plant traps
    
    type: LeafType = LeafType.BROAD
    surface_area: float = 1.0 # Photosynthetic surface
    thickness: float = 0.1 # Leaf thickness
    photosynthetic_rate: float = 1.0 # Photosynthesis efficiency
    water_loss_rate: float = 0.5 # Transpiration rate
    seasonal: bool = False # Sheds seasonally
    defense_rating: float = 0.0  # Protection against herbivores

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Leaf properties
        substats['surface_area'] = self.surface_area
        substats['photosynthetic_rate'] = self.photosynthetic_rate
        substats['water_loss_rate'] = self.water_loss_rate
        
        # Special properties
        if self.seasonal:
            substats['seasonal_adaptation'] = 0.5
        if self.defense_rating > 0:
            substats['defense'] = self.defense_rating
            
        # Type-specific bonuses
        if self.type == self.LeafType.NEEDLE:
            substats['water_loss_rate'] *= 0.3  # Reduced water loss
            substats['cold_resistance'] = 0.5
        elif self.type == self.LeafType.SUCCULENT:
            substats['water_storage'] = 0.8 * self.thickness
        elif self.type == self.LeafType.TRAP:
            substats['carnivorous'] = 0.7  # Carnivorous capability
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value)
            else:
                stats[stat] = (value)
        return stats

@dataclass
class flower(part):
    class FlowerType(Enum):
        SIMPLE = 0 # Single flower
        COMPOSITE = 1 # Cluster of flowers (daisies)
        SPIKE = 2 # Spike arrangement
        UMBEL = 3 # Umbrella-like arrangement
        BELL = 4 # Bell-shaped
        
    type: FlowerType = FlowerType.SIMPLE
    pollination_method: str = "wind"  # wind, insect, bird, etc.
    nectar_production: float = 0.0 # Amount of nectar
    scent_strength: float = 0.0 # Attractiveness to pollinators
    color_variety: int = 1 # Number of colors
    blooming_period: float = 1.0 # How long flowers last
    seed_production: float = 1.0 # Seed yield per flower

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Flower properties
        substats['pollination_efficiency'] = 1.0
        substats['nectar_production'] = self.nectar_production
        substats['scent_strength'] = self.scent_strength
        substats['color_variety'] = self.color_variety
        substats['bloom_duration'] = self.blooming_period
        substats['seed_production'] = self.seed_production
        
        # Pollination method bonuses
        if self.pollination_method == "wind":
            substats['wind_pollination'] = 1.0
        elif self.pollination_method == "insect":
            substats['insect_attraction'] = 0.8 + (self.scent_strength * 0.2)
        elif self.pollination_method == "bird":
            substats['bird_attraction'] = 0.7 + (self.color_variety * 0.1)
        
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value)
            else:
                stats[stat] = (value)    
        return stats

@dataclass
class fruit(part):
    class FruitType(Enum):
        BERRY = 0 # Fleshy, many seeds
        DRUPE = 1 # Fleshy, single seed (peach)
        NUT = 2 # Hard shell
        POD = 3 # Dry, splits open
        AGGREGATE = 4 # From multiple flowers (raspberry)
        MULTIPLE = 5 # From cluster of flowers (pineapple)
    
    type: FruitType = FruitType.BERRY
    seed_count: int = 1 # Number of seeds
    dispersal_method: str = "animal" # animal, wind, water, explosive
    nutritional_value: float = 1.0 # Food value for animals
    ripening_time: float = 1.0 # Time to mature
    toxicity: float = 0.0 # Defense against herbivores

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Fruit properties
        substats['seed_count'] = self.seed_count
        substats['nutritional_value'] = self.nutritional_value
        substats['ripening_speed'] = self.ripening_time
        substats['toxicity'] = self.toxicity
        
        # Dispersal method bonuses
        if self.dispersal_method == "animal":
            substats['animal_dispersal'] = 0.8 + (self.nutritional_value * 0.2)
        elif self.dispersal_method == "wind":
            substats['wind_dispersal'] = 1.0
        elif self.dispersal_method == "water":
            substats['water_dispersal'] = 1.0
        elif self.dispersal_method == "explosive":
            substats['explosive_dispersal'] = 1.0
            
        # Type-specific bonuses
        if self.type == self.FruitType.NUT:
            substats['durability'] = 1.0
            substats['longevity'] = 1.5
        elif self.type == self.FruitType.BERRY:
            substats['animal_attraction'] = 0.7 + (self.nutritional_value * 0.3)
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value)
            else:
                stats[stat] = (value)
        return stats

@dataclass
class reproductive(part):
    class ReproductiveType(Enum):
        GONAD = 0         # General reproductive organ
        OVARY = 1         # Egg production
        TESTIS = 2        # Sperm production
        UTERUS = 3        # Gestation chamber
        SPORE_SAC = 4     # Fungal/plant spore production
        FLOWER_BUD = 5    # Plant flower precursor
        CONE = 6          # Gymnosperm reproductive structure
        
    type: ReproductiveType = ReproductiveType.GONAD
    fertility: float = 1.0         # Reproductive success rate
    gestation_period: float = 0.0  # Time for development (if applicable)
    offspring_count: int = 1       # Typical number of offspring
    mating_frequency: float = 1.0  # How often reproduction can occur
    resource_cost: float = 0.5     # Energy/nutrient cost per reproduction
    seasonal: bool = False         # Only functions in certain seasons

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Base reproductive stats
        substats['fertility'] = self.fertility
        substats['offspring_count'] = self.offspring_count
        substats['reproduction_cost'] = self.resource_cost
        
        # Special capabilities
        if self.gestation_period > 0:
            substats['gestation_period'] = self.gestation_period
        if self.seasonal:
            substats['seasonal_breeding'] = 1.0
            
        # Type-specific bonuses
        if self.type == self.ReproductiveType.OVARY:
            substats['egg_quality'] = 0.5 * self.fertility
        elif self.type == self.ReproductiveType.TESTIS:
            substats['sperm_count'] = 2.0 * self.fertility
        elif self.type == self.ReproductiveType.UTERUS:
            substats['gestation_efficiency'] = 1.5
        elif self.type == self.ReproductiveType.SPORE_SAC:
            substats['spore_production'] = 10.0 * self.fertility
        elif self.type == self.ReproductiveType.CONE:
            substats['wind_pollination'] = 1.0
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value * self.size)
            else:
                stats[stat] = (value * self.size)
        return stats

@dataclass
class egg_sac(part):
    class EggType(Enum):
        SOFT = 0          # Amphibian-style eggs
        HARD_SHELL = 1    # Bird/reptile eggs
        GELATINOUS = 2    # Fish/insect eggs
        RESISTANT = 3     # Extremophile eggs (tardigrades)
        
    type: EggType = EggType.SOFT
    egg_count: int = 1
    protection: float = 0.5    # Physical protection
    nutrient_store: float = 1.0 # Yolk/resources
    incubation_time: float = 1.0
    desiccation_resistance: float = 0.0
    camouflage: float = 0.0

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Egg properties
        substats['egg_count'] = self.egg_count
        substats['egg_protection'] = self.protection
        substats['egg_nutrients'] = self.nutrient_store
        substats['incubation_time'] = self.incubation_time
        
        # Special properties
        if self.desiccation_resistance > 0:
            substats['desiccation_resistance'] = self.desiccation_resistance
        if self.camouflage > 0:
            substats['egg_camouflage'] = self.camouflage
            
        # Type-specific bonuses
        if self.type == self.EggType.HARD_SHELL:
            substats['egg_protection'] *= 2.0
            substats['gas_exchange'] = 0.8
        elif self.type == self.EggType.GELATINOUS:
            substats['water_retention'] = 1.0
        elif self.type == self.EggType.RESISTANT:
            substats['environmental_resistance'] = 2.0
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value * self.size)
            else:
                stats[stat] = (value * self.size)
        return stats

@dataclass
class pollen(part):
    class PollenType(Enum):
        LIGHT = 0      # Wind-dispersed
        STICKY = 1     # Animal-dispersed
        EXPLOSIVE = 2  # Self-dispersed
        WATER = 3      # Water-dispersed
        
    type: PollenType = PollenType.LIGHT
    quantity: float = 1.0          # Production amount
    viability: float = 1.0         # Success rate
    dispersal_range: float = 1.0   # How far it spreads
    allergenicity: float = 0.0     # Irritation to others
    nutrient_content: float = 0.0  # For pollinator attraction

    def setup_default_subparts(self):
        pass

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Pollen properties
        substats['pollen_quantity'] = self.quantity
        substats['pollen_viability'] = self.viability
        substats['dispersal_range'] = self.dispersal_range
        
        # Special properties
        if self.allergenicity > 0:
            substats['allergenicity'] = self.allergenicity
        if self.nutrient_content > 0:
            substats['pollinator_reward'] = self.nutrient_content
            
        # Type-specific bonuses
        if self.type == self.PollenType.LIGHT:
            substats['wind_dispersal'] = 1.5 * self.dispersal_range
        elif self.type == self.PollenType.STICKY:
            substats['animal_attachment'] = 1.2
        elif self.type == self.PollenType.EXPLOSIVE:
            substats['dispersal_force'] = 1.0
        elif self.type == self.PollenType.WATER:
            substats['water_dispersal'] = 1.0
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value * self.size)
            else:
                stats[stat] = (value * self.size)
        return stats

@dataclass
class seed:
    class SeedType(Enum):
        NAKED = 0       # Gymnosperm
        ENCLOSED = 1    # Angiosperm
        SPORE = 2       # Fungal/fern
        TUBER = 3       # Underground storage
        BULBIL = 4      # Aerial propagation
        
    type: SeedType = SeedType.ENCLOSED
    viability: float = 1.0          # Chance to germinate
    dormancy: float = 0.0           # Can remain dormant
    dispersal: float = 1.0          # Spread effectiveness
    nutrient_store: float = 1.0     # Endosperm/resources
    defense: float = 0.0            # Anti-predation
    subparts: list['part'] = field(default_factory=list)
    statCache: dict[str, float] = {}


    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        if self.statCache is not None and not recalc:
            return self.statCache
        stats = {}
        substats = {}
        # Seed properties
        substats['seed_viability'] = self.viability
        substats['dispersal_efficiency'] = self.dispersal
        substats['seed_nutrients'] = self.nutrient_store
        
        # Special properties
        if self.dormancy > 0:
            substats['dormancy_period'] = self.dormancy
        if self.defense > 0:
            substats['seed_defense'] = self.defense
            
        # Type-specific bonuses
        if self.type == self.SeedType.NAKED:
            substats['germination_speed'] = 1.2
        elif self.type == self.SeedType.SPORE:
            substats['quantity'] = 10.0  # Spores are numerous
        elif self.type == self.SeedType.TUBER:
            substats['vegetative_growth'] = 1.5
        elif self.type == self.SeedType.BULBIL:
            substats['aerial_propagation'] = 1.0
            
        for subpart in self.subparts:
            substats = subpart.calculateStats(recalc)
            for key, value in substats.items():
                if key in stats:
                    stats[key] += (value * subpart.size)
                else:
                    stats[key] = (value * subpart.size)
        self.statCache = stats
        return substats

@dataclass
class Species:
    name: str = "" # generate from something stupid
    color: tuple = (0,0,0) # generate from something stupid as well
    symbol: str = random.choice(RANDOMCHAR) # random ascii character to distinguish similar color species

    #energy
    maxE: float = 10.0 # maximum energy, organisms will average a normal curve of σ = 5% for plants at same age, but 500% overall, 30% for animals at maturity
    rootE: float = 10.0 # if rooted, this is extra energy stored in roots. not valid for non-rooted species. σ of 500% but based on terrain
    idleE: float = 10.0 # how much energy is consumed when just standing around
    sleepE: float = 5.0 # energy usage during sleep
    dormE: float = 1.0 # energy usage while dormant

    #sleep
    dormancy: bool = False # in extreme conditions, this can be ignored
    dormanThresh: float = 0.3 # when to activate for a normal dormancy. 1% is "extreme", 30% (default) is "normal" for something like a bear
    sleepRatio: float = 0.5 # percent of time spent asleep
    class habits(Enum):
        NONE = 0
        NOCTURNAL = 1
        CREPUSCULAR = 2
        CATHERMERAL = 3
        ULTRADIAN = 4
        UNICAMERAL = 5
        DIURNAL = 6
    sleepHabits: habits = habits.NONE #sleep method. plants use "none"

    #reproduction
    maturity: int = 4320 # number of steps to be considered mature. σ of 9%. divide by 48 for days.
    reproCost: float = 5.0 # how much energy it costs to either produce an egg or to produce a new organism. σ of 10%
    reproFrequency: int = 480 #maximum frequency of reproduction. σ of 15%
    seedAge: int = 100 # seed includes external eggs. This is in steps. after this time seed will die. σ of 10%
    seedMaturity: int = 90 # minimum age to attempt to hatch. certain things may prevent this from occurring. σ of 10%
    pMS: int = 5000 # age at which reproduction is no longer possible. σ of 5%
    class ReproMethod(Enum):
        EXPANSIVE_GROWTH = 0
        OVIPARITY = 1
        VIVIPARITY = 2
        OVOVIPARITY = 3
        BROADCAST = 4
        BUDDING = 5
        SPORES = 6
        PARTHENOGENESIS = 7
    reproMethod: ReproMethod = ReproMethod.EXPANSIVE_GROWTH
    flowering: bool = False # another state to track for reproduction cause why not.
    mutationRate: float = 0.001
    speciesParents: list['Species'] = []
    
    #movement
    canMove: bool = False # plants will almost always be false. but this is a fictional world.
    moveSpeed: float = 0.0 # in yards, each cell is 1 cubic yard. σ of 5% at maturity.
    moveCost: float = 0.0 # energy cost for each movement. σ of 5% at maturity.
    Rooted: bool = True # if it has underground roots. This should set the movement to 0 and canmove to false.
    canFly: bool = False # requires some vertical lift option.

    #eating
    preferredFood: dict[str, float] = {} # list of types of parts that this species prefers. float shows how much preference with 1.0 being always goes for, and 0.01 being barely ever
    inedibleFood: dict[str, float]  = {} # list of parts that are inedible, greater than 1.0 means deadly, 0.01 means minor issues. item can be in both preferred and inedible if both are less than 0.5, though should be rare (think spicy food)
    
    #survival
    tempSensCold: float = 85.0 # minimum temperature for reasonable health. in fahrenheit
    tempSensHot: float = 100.0 # maximum temperature for reasonable health. 
    tempResis: float = 1.0 # resistance of changes to temperature
    tempProduction: float = 15.0 # automatic production of heat above normal
    tempShed: float = 1.0 # shedding of heat rate

    #violence
    natArmor: float = 0 
    natAttack: float = 0

    #behavior
    aggression: float = 0.0
    curious: float = 0.0
    social: float = 0.0
    adapative: float = 0.0 # how much behavior changes while alive. nothing related to physical attributes.

    parts: list[part] = [] # part factory.
    statcache: Optional[dict] = {}

    def _generate_name(self): 
        return "".join(random.sample(ALPHABET, 10))

    def __post_init__(self):
        self.statcache = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (int, float, bool)):
                self.statcache[key] = float(value)
        for part in self.parts:
            for key, value in part.calculateStats(True).items():
                self.statcache[key] = self.statcache.get(key, 0.0) + value
        return self

TREE = Species(
    name="Tree",
    Rooted=True,
    canMove=False,
    reproMethod=Species.ReproMethod.BROADCAST,
    tempSensCold=-20,
    tempSensHot=75,
    parts=[stem(name="trunk", type=stem.StemType.WOODY, vital=True, size= 1.0,
            height=20.0, structural_strength=2.5, annual_growth_rings=True,
            minTemp=-30, maxTemp=80.0, subparts=[
                skin(name="bark", type=skin.Skin.BARK, size=0.9, thickness=1.5,
                     insulation_factor=1.8, energyCost=0.01),
                root(name="taproot", type=root.RootType.TAP, size=0.4,
                     depth=15.0, absorption_rate=0.8),
                reproductive(name="cones", type=reproductive.ReproductiveType.CONE, size=0.1,
                             fertility=0.8, resource_cost=1.0, seasonal=True),
                leaf(name="needle_cluster_0", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_1", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_2", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_3", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_4", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_5", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_6", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_7", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_8", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_9", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_10", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_11", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_12", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_13", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_14", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_15", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_16", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_17", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(name="needle_cluster_18", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
            ])]   
).__post_init__()


BUSH = Species(
    name="bush",
    Rooted=True,
    canMove=False,
    reproMethod=Species.ReproMethod.BROADCAST
    tempSensCold=50.0
    tempSensHot=130.0

).__post_init__()


        











class SimulationUI:
    def __init__(self):
        dpg.create_context()
        self.game = GameOfLife()
        self.setup_ui()


    def run(self):
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
            

if __name__ == "__main__":
    UI = SimulationUI()
    UI.run()