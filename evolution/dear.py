from dataclasses import dataclass
from enum import Enum
import random
import dearpygui.dearpygui as dpg

@dataclass
class part: # each part is baked into the species stats, separted here to allow for organisms to lose a limb
    name: str = ""
    size: float = 1.0  # percent of parent part
    vital: bool = False  # whether losing this part is fatal
    
    # General part properties
    health: float = 10.0  # base health of the part
    regrowth_rate: float = 0.0  # how quickly the part regrows (0 for no regrowth)
    energyCost: float = 0.1  # energy cost to maintain this part
    energyProducer: bool = False
    subparts: list['part'] = []
    statCache = None

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        if self.statCache is not None and recalc == False:
            return self.statCache
        stats = {}
        stats['health'] = self.health
        stats['regrowthrate'] = self.regrowth_rate
        stats['idleE'] = self.energyCost
        if self.energyProducer:
            stats['sleepE'] = self.energyCost * 0.5
            stats['dormE'] = self.energyCost * 0.1
        else:
            stats['sleepE'] = self.energyCost * 0.1
            stats['dormE'] = self.energyCost * 0.01

        for subpart in self.subparts:
            substats = subpart.calculateStats(recalc)
            for key in substats:
                if key in stats:
                    stats[key] += (substats[key] * subpart.size) + stats[key]
                else:
                    stats[key] += (substats[key] * subpart.size)
        self.statCache = stats
        return self.statCache

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

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        stats['armor'] = self.thickness * 0.5  # Thicker skin provides more armor
        stats['camouflage'] = self.camouflage_effectiveness
        stats['insulation'] = self.insulation_factor
        stats['cooling'] = self.cooling_factor
        stats['water_retention'] = self.water_retention
        stats['gas_exchange'] = self.gas_exchange
        
        if self.can_photosynthesize:
            stats['photosynthesis'] = 0.5 * self.thickness  # Photosynthetic capability
        if self.luminescent:
            stats['luminescence'] = self.luminense  # Basic luminescence value
            stats['camouflage'] -= self.luminense
            
        return stats

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
        stats['strength'] = self.strength * self.size
        stats['dexterity'] = self.dexterity
        stats['reach'] = self.reach * self.size
        stats['movement_speed'] = self.movement_speed * self.size
        stats['movement_cost'] = self.movement_cost * self.size
        
        if self.prehensile:
            stats['manipulation'] = 0.7 * self.dexterity
        else:
            stats['manipulation'] = 0.1 * self.dexterity
            
        # Movement capabilities
        if self.can_climb:
            stats['climbing'] = (0.5 * self.strength) + (self.dexterity * 0.1)
        else:
            stats['climbing'] = (0.1 * self.strength) + (self.dexterity * 0.1)
        if self.can_swim:
            stats['swimming'] = 0.6 * self.strength + (self.dexterity * 0.1)
        else:
            stats['swimming'] = 0.1 * self.strength + (self.dexterity * 0.1)
        if self.can_dig:
            stats['digging'] = 0.7 * self.strength + (self.dexterity * 0.1)
        else:
            stats['digging'] = 0.05 * self.strength + (self.dexterity * 0.1)
            
        return stats

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

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        # Base sensory stats
        stats['sensory_range'] = self.range * self.size
        stats['sensory_sensitivity'] = self.sensitivity
        stats['sensory_precision'] = self.precision
        stats['sensory_angle'] = self.angle
        if self.night_vision:
            stats['night_vision'] = 0.5 * self.sensitivity
        if self.thermal_vision:
            stats['thermal_vision'] = 0.3 * self.sensitivity
        if not self.can_see_colors:
            stats['color_vision'] = 0
        else:
            stats['color_vision'] = 1.0
        if self.underwater_effective:
            stats['underwater_sensing'] = 1.0
        if not self.air_effective:
            stats['air_sensing'] = 0.8
        stats['sensory_active_cost'] = self.active_cost
        stats['sensory_passive_cost'] = self.passive_cost
        
        return stats

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

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        # Internal organ stats
        stats['efficiency'] = self.efficiency
        stats['capacity'] = self.capacity
        
        # Type-specific bonuses
        if self.type == self.InternalType.HEART:
            stats['circulation'] = self.efficiency * 2.0
        elif self.type == self.InternalType.LUNG:
            stats['oxygenation'] = self.efficiency * 1.5
        elif self.type == self.InternalType.GILL:
            stats['water_oxygenation'] = self.efficiency * 2.0
        elif self.type == self.InternalType.BRAIN:
            stats['intelligence'] = self.efficiency * 0.5
            stats['nerve_speed'] = self.efficiency * 1.0
        elif self.type == self.InternalType.STOMACH:
            stats['digestion'] = self.efficiency * 1.5
        elif self.type == self.InternalType.LIVER:
            stats['toxin_processing'] = self.efficiency * 1.0
        elif self.type == self.InternalType.KIDNEY:
            stats['filtration'] = self.efficiency * 1.2
        elif self.type == self.InternalType.GLAND:
            stats['chemical_production'] = self.efficiency * 1.0
            
        if self.can_regenerate:
            stats['regeneration'] = 0.2  # Small regeneration bonus
            
        return stats

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
        # Base weapon stats
        stats['damage'] = self.damage * self.size
        stats['attack_speed'] = self.attack_speed
        stats['attack_reach'] = self.reach
        stats['attack_cost'] = self.attack_cost
        
        # Damage types
        stats['slash_damage'] = self.slash_damage * self.size
        stats['pierce_damage'] = self.pierce_damage * self.size
        stats['blunt_damage'] = self.blunt_damage * self.size
        
        # Special properties
        if self.venomous:
            stats['venom_potency'] = 0.5  # Base venom potency
        if self.retractable:
            stats['concealment'] = 0.3  # Bonus to hiding the weapon
            
        # Type-specific bonuses
        if self.type == self.WeaponType.CLAW:
            stats['slash_damage'] += 0.5 * self.damage
        elif self.type == self.WeaponType.FANG:
            stats['pierce_damage'] += 0.7 * self.damage
        elif self.type == self.WeaponType.HORN:
            stats['pierce_damage'] += 0.5 * self.damage
            stats['blunt_damage'] += 0.3 * self.damage
        elif self.type == self.WeaponType.STINGER:
            stats['pierce_damage'] += 0.3 * self.damage
            stats['venom_potency'] = stats.get('venom_potency', 0) + 0.5
            
        return stats

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

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        # Base appendage stats
        stats['coverage'] = self.coverage
        stats['hardness'] = self.hardness
        
        # Storage capabilities
        if self.storesE:
            stats['energy_storage'] = 0.5 * self.size
        if self.stores_water:
            stats['water_storage'] = 0.3 * self.size
        if self.stores_air:
            stats['air_storage'] = 0.2 * self.size
            
        # Special properties
        if self.poison_production:
            stats['poison_production'] = 0.3
        if self.venom_production:
            stats['venom_production'] = 0.4
            
        # Type-specific bonuses
        if self.type == self.AppendageType.SHELL:
            stats['armor'] = 1.0 * self.hardness * self.coverage
        elif self.type == self.AppendageType.SPINE:
            stats['defense'] = 0.5 * self.hardness
        elif self.type == self.AppendageType.PLATE:
            stats['armor'] = 0.7 * self.hardness * self.coverage
        elif self.type == self.AppendageType.TENDRIL:
            stats['manipulation'] = 0.3 * self.coverage
            
        return stats

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

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        # Root system stats
        stats['root_depth'] = self.depth
        stats['root_spread'] = self.spread
        stats['absorption_rate'] = self.absorption_rate
        
        # Special capabilities
        if self.nitrogen_fixing:
            stats['nitrogen_fixing'] = 0.5
        if self.storage_capacity > 0:
            stats['root_storage'] = self.storage_capacity
        if self.can_propagate:
            stats['vegetative_propagation'] = 0.3
            
        # Type-specific bonuses
        if self.type == self.RootType.TAP:
            stats['root_depth'] *= 1.5
            stats['absorption_rate'] *= 1.2
        elif self.type == self.RootType.AERIAL:
            stats['air_absorption'] = 0.5
        elif self.type == self.RootType.STORAGE:
            stats['root_storage'] = stats.get('root_storage', 0) + 1.0
            
        return stats

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
        # Stem structure stats
        stats['height'] = self.height
        stats['flexibility'] = self.flexibility
        stats['structural_strength'] = self.structural_strength
        
        # Special capabilities
        if self.photosynthetic:
            stats['photosynthesis'] = 0.3 * self.height
        if self.storage_capacity > 0:
            stats['stem_storage'] = self.storage_capacity
        if self.annual_growth_rings:
            stats['age_recording'] = 1.0  # Can determine age
            
        # Type-specific bonuses
        if self.type == self.StemType.WOODY:
            stats['structural_strength'] *= 2.0
            stats['durability'] = 1.0
        elif self.type == self.StemType.VINE:
            stats['climbing'] = 0.7 * self.flexibility
        elif self.type == self.StemType.TUBER:
            stats['stem_storage'] = stats.get('stem_storage', 0) + 1.5
            
        return stats
    
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

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        # Leaf properties
        stats['surface_area'] = self.surface_area
        stats['photosynthetic_rate'] = self.photosynthetic_rate
        stats['water_loss_rate'] = self.water_loss_rate
        
        # Special properties
        if self.seasonal:
            stats['seasonal_adaptation'] = 0.5
        if self.defense_rating > 0:
            stats['defense'] = self.defense_rating
            
        # Type-specific bonuses
        if self.type == self.LeafType.NEEDLE:
            stats['water_loss_rate'] *= 0.3  # Reduced water loss
            stats['cold_resistance'] = 0.5
        elif self.type == self.LeafType.SUCCULENT:
            stats['water_storage'] = 0.8 * self.thickness
        elif self.type == self.LeafType.TRAP:
            stats['carnivorous'] = 0.7  # Carnivorous capability
            
        return stats

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

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        # Flower properties
        stats['pollination_efficiency'] = 1.0
        stats['nectar_production'] = self.nectar_production
        stats['scent_strength'] = self.scent_strength
        stats['color_variety'] = self.color_variety
        stats['bloom_duration'] = self.blooming_period
        stats['seed_production'] = self.seed_production
        
        # Pollination method bonuses
        if self.pollination_method == "wind":
            stats['wind_pollination'] = 1.0
        elif self.pollination_method == "insect":
            stats['insect_attraction'] = 0.8 + (self.scent_strength * 0.2)
        elif self.pollination_method == "bird":
            stats['bird_attraction'] = 0.7 + (self.color_variety * 0.1)
            
        return stats

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

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        # Fruit properties
        stats['seed_count'] = self.seed_count
        stats['nutritional_value'] = self.nutritional_value
        stats['ripening_speed'] = self.ripening_time
        stats['toxicity'] = self.toxicity
        
        # Dispersal method bonuses
        if self.dispersal_method == "animal":
            stats['animal_dispersal'] = 0.8 + (self.nutritional_value * 0.2)
        elif self.dispersal_method == "wind":
            stats['wind_dispersal'] = 1.0
        elif self.dispersal_method == "water":
            stats['water_dispersal'] = 1.0
        elif self.dispersal_method == "explosive":
            stats['explosive_dispersal'] = 1.0
            
        # Type-specific bonuses
        if self.type == self.FruitType.NUT:
            stats['durability'] = 1.0
            stats['longevity'] = 1.5
        elif self.type == self.FruitType.BERRY:
            stats['animal_attraction'] = 0.7 + (self.nutritional_value * 0.3)
            
        return stats

class Species:
    def __init__(self):
        self.name: str = "" # generate from something stupid
        self.color: tuple = (0,0,0) # generate from something stupid as well
        self.symbol: str = '' # random ascii character to distinguish similar color species

        #energy
        self.maxE: float = 10.0 # maximum energy, organisms will average a normal curve of σ = 5% for plants at same age, but 500% overall, 30% for animals at maturity
        self.rootE: float = 10.0 # if rooted, this is extra energy stored in roots. not valid for non-rooted species. σ of 500% but based on terrain
        self.idleE: float = 10.0 # how much energy is consumed when just standing around
        self.sleepE: float = 5.0 # energy usage during sleep
        self.dormE: float = 1.0 # energy usage while dormant

        #sleep
        self.dormancy: bool = False # in extreme conditions, this can be ignored
        self.dormanThresh: float = 0.3 # when to activate for a normal dormancy. 1% is "extreme", 30% (default) is "normal" for something like a bear
        self.sleepRatio: float = 0.5 # percent of time spent asleep
        class habits(Enum):
            NONE = 0
            NOCTURNAL = 1
            CREPUSCULAR = 2
            CATHERMERAL = 3
            ULTRADIAN = 4
            UNICAMERAL = 5
            DIURNAL = 6
        self.sleepHabits: habits = habits.NONE #sleep method. plants use "none"

        #reproduction
        self.maturity: int = 4320 # number of steps to be considered mature. σ of 9%. divide by 48 for days.
        self.reproCost: float = 5.0 # how much energy it costs to either produce an egg or to produce a new organism. σ of 10%
        self.reproFrequency: int = 480 #maximum frequency of reproduction. σ of 15%
        self.seedAge: int = 100 # seed includes external eggs. This is in steps. after this time seed will die. σ of 10%
        self.seedMaturity: int = 90 # minimum age to attempt to hatch. certain things may prevent this from occurring. σ of 10%
        self.pMS: int = 5000 # age at which reproduction is no longer possible. σ of 5%
        class ReproMethod(Enum):
            EXPANSIVE_GROWTH = 0
            OVIPARITY = 1
            VIVIPARITY = 2
            OVOVIPARITY = 3
            BROADCAST = 4
            BUDDING = 5
            SPORES = 6
            PARTHENOGENESIS = 7
        self.reproMethod: ReproMethod = ReproMethod.EXPANSIVE_GROWTH
        self.flowering: bool = False # another state to track for reproduction cause why not.
        
        #movement
        self.canMove: bool = False # plants will almost always be false. but this is a fictional world.
        self.moveSpeed: float = 0.0 # in yards, each cell is 1 cubic yard. σ of 5% at maturity.
        self.moveCost: float = 0.0 # energy cost for each movement. σ of 5% at maturity.
        self.Rooted: bool = True # if it has underground roots. This should set the movement to 0 and canmove to false.
        self.canFly: bool = False # requires some vertical lift option.

        #eating
        self.preferredFood: dict[str, float] # list of types of parts that this species prefers. float shows how much preference with 1.0 being always goes for, and 0.01 being barely ever
        self.inedibleFood: dict[str, float] # list of parts that are inedible, greater than 1.0 means deadly, 0.01 means minor issues. item can be in both preferred and inedible if both are less than 0.5, though should be rare (think spicy food)
        
        #survival
        self.tempSensCold: float = 85.0 # minimum temperature for reasonable health. in fahrenheit
        self.tempSensHot: float = 100.0 # maximum temperature for reasonable health. 
        self.tempResis: float = 1.0 # resistance of changes to temperature
        self.tempProduction: float = 15.0 # automatic production of heat above normal
        self.tempShed: float = 1.0 # shedding of heat rate

        #violence
        self.natArmor: float = 0 
        self.natAttack: float = 0

        self.parts: list[part] = [] # part factory.

        def __post_init__(self):
            
        











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