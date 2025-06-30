from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import random
from typing import Optional, Any
import dearpygui.dearpygui as dpg
import string
import numpy as np
import torch
from evolution.parts import part
import evomodel

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ALPHABET = string.ascii_letters
RANDOMCHAR = string.printable

def compare_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> float:
    all_keys = set(dict1.keys()).union(set(dict2.keys()))
    if not all_keys:
        return 1.0
    
    total_score = 0.0
    total_weight = 0.0
    
    for key in all_keys:
        if key not in dict1 or key not in dict2:
            continue
            
        val1 = dict1[key]
        val2 = dict2[key]
        
        weight = 2.0 if isinstance(val1, bool) or isinstance(val2, bool) else 1.0
        if type(val1) != type(val2):
            total_weight += weight
            continue
        if isinstance(val1, bool):
            score = 1.0 if val1 == val2 else 0.0
        elif isinstance(val1, (int, float)):
            if val1 == 0 and val2 == 0:
                score = 1.0
            else:
                max_val = max(abs(val1), abs(val2))
                diff = abs(val1 - val2)
                score = 1.0 - min(diff / max_val, 1.0)
        elif isinstance(val1, dict) and isinstance(val2, dict):
            score = compare_dicts(val1, val2)
        elif isinstance(val1, (list, tuple, set)):
            try:
                score = 1.0 if val1 == val2 else 0.0
            except:
                score = 0.0
        elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
            score = compare_dicts(val1.__dict__, val2.__dict__)
        else:
            score = 1.0 if val1 == val2 else 0.0
            
        total_score += score * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    
    return total_score / total_weight

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
    speciesParents: list['Species'] = field(default_factory=list)
    compatibility_cache: dict[str, float] = field(default_factory=dict)
    reproCache: dict[str, bool] = field(default_factory=dict)
    reprocacheOffspring: dict[str, 'Species'] = field(default_factory=dict)
    
    #movement
    canMove: bool = False # plants will almost always be false. but this is a fictional world.
    moveSpeed: float = 0.0 # in yards, each cell is 1 cubic yard. σ of 5% at maturity.
    moveCost: float = 0.0 # energy cost for each movement. σ of 5% at maturity.
    Rooted: bool = True # if it has underground roots. This should set the movement to 0 and canmove to false.
    canDig: bool = False
    canSwim: bool = False
    
    #eating
    preferredFood: dict[str, float] = field(default_factory=dict)
    inedibleFood: dict[str, float]  = field(default_factory=dict)
    
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
    #adaptive will be used to determine how much the model is trained during the organisms "life"
    intelligence: float = 0.0

    parts: list[part] = field(default_factory=list)
    statcache: Optional[dict] = field(default_factory=dict)

    def __post_init__(self) -> 'Species':
        self.genStats(True)
        return self

    def _generate_name(self) -> str: 
        return "".join(random.sample(ALPHABET, 10))

    def genStats(self, regen=False):
        if self.statcache is not None and len(self.statcache) > 1 and not regen:
            return self.statcache
        self.statcache = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (int, float, bool)):
                self.statcache[key] = float(value)
        for part in self.parts:
            part_stats = part.calculateStats(False)
            for key, value in part_stats.items():
                self.statcache[key] = self.statcache.get(key, 0.0) + value
        return self.statcache

    def mutate(self) -> 'Species':
        offspring = copy.deepcopy(self)
        offspring.speciesParents.append(self)
        offspring.name = self._generate_name()
        offspring.color = (
            max(0, min(255, int(self.color[0] + random.randint(-10, 10)))),
            max(0, min(255, int(self.color[1] + random.randint(-10, 10)))),
            max(0, min(255, int(self.color[2] + random.randint(-10, 10))))
        )
        offspring.mutationRate *= random.uniform(0.8, 1.1) # 0.8-1.1 to encourage settling over time

        for attr_name in offspring.__dataclass_fields__:
            if attr_name in ['name', 'color', 'symbol', 'parts', 'statcache', 'speciesParents', 'preferredFood', 'inedibleFood']:
                continue

            if random.random() < offspring.mutationRate:
                current_val = getattr(offspring, attr_name)

                if isinstance(current_val, bool):
                    setattr(offspring, attr_name, not current_val)
                elif isinstance(current_val, (int, float)):
                    new_val = current_val * random.uniform(0.95, 1.05)
                    setattr(offspring, attr_name, new_val)
                elif isinstance(current_val, Enum):
                    enum_class = type(current_val)
                    choices = [e for e in enum_class if e != current_val]
                    if choices:
                        setattr(offspring, attr_name, random.choice(choices))

        for part_item in offspring.parts:
            part_item.mutate()

        offspring.genStats(True)

        return offspring
    
    def _simkey(self, other: 'Species') -> tuple[str, str]:
        selfkey = f'{self.name}_{other.name}'
        otherkey = f'{other.name}_{self.name}'
        return selfkey, otherkey

    def calcdiff(self, other: 'Species') -> float:
        if self == other:
            return 1.0
        simkey,othersimkey = self._simkey(other)
        if simkey in self.compatibility_cache:
            return self.compatibility_cache[simkey]
        diff = compare_dicts(self.__dict__, other.__dict__)

        self.compatibility_cache[simkey] = diff
        other.compatibility_cache[othersimkey] = diff
        return diff

    def can_reproduce_with(self, other: 'Species') -> bool:
        if self == other:
            return True
        simkey, othersimkey = self._simkey(other)
        can = False
        if simkey in self.reproCache:
            return self.reproCache[simkey]
        if other in self.speciesParents or self in other.speciesParents:
            can = True
        
        common_parents = set(self.speciesParents) & set(other.speciesParents)
        if can == False and common_parents:
            can = True
        
        diff = self.calcdiff(other)
        relationship_degree = self._get_relationship_degree(other)
        if can == False and relationship_degree <= 2:
            can = True
        if can == False and relationship_degree == 3:
            return random.random() < (2 * diff)
        
        
        if can == False:
            can = random.random() < diff
        self.reproCache[simkey] = can
        other.reproCache[othersimkey] = can
        return can

    def _get_relationship_degree(self, other: 'Species') -> int:
        if self == other:
            return 0
        if other in self.speciesParents or self in other.speciesParents:
            return 1
        common_parents = set(self.speciesParents) & set(other.speciesParents)
        if common_parents:
            return 2
        self_parents = set(self.speciesParents)
        other_parents = set(other.speciesParents)
        self_grandparents = set()
        for parent in self_parents:
            self_grandparents.update(parent.speciesParents)
        
        other_grandparents = set()
        for parent in other_parents:
            other_grandparents.update(parent.speciesParents)
        if self_grandparents & other_grandparents:
            return 3
        self_great_grandparents = set()
        for gp in self_grandparents:
            self_great_grandparents.update(gp.speciesParents)
        
        other_great_grandparents = set()
        for gp in other_grandparents:
            other_great_grandparents.update(gp.speciesParents)
        
        if self_great_grandparents & other_great_grandparents:
            return 4
        
        return 999

    def get_hybrid_offspring(self, other: 'Species') -> Optional['Species']:
        if not self.can_reproduce_with(other):
            return None
        
        mother_key = f"{self.name}_mother_{other.name}_father"
        father_key = f"{other.name}_father_{self.name}_mother"
        
        # Check cache first
        if mother_key in self.reproCache:
            return self.reprocacheOffspring[mother_key]
        if father_key in other.reproCache:
            return other.reprocacheOffspring[father_key]
        offspring = copy.deepcopy(self)
        offspring.speciesParents = [self, other]
        offspring.name = self._generate_name()
        offspring.color = (
            (self.color[0] + other.color[0]) // 2,
            (self.color[1] + other.color[1]) // 2,
            (self.color[2] + other.color[2]) // 2
        )
        
        similarity = self.calcdiff(other)
        relationship_degree = self._get_relationship_degree(other)
        
        if relationship_degree <= 1:
            # Parent/child - equal weighting
            self_weight = 0.5 * self.mutationRate
            other_weight = 0.5 * other.mutationRate
        elif relationship_degree == 2:
            # Siblings - slight random variation
            self_weight = random.uniform(0.4, 0.6) * self.mutationRate
            other_weight = 1 - self_weight * other.mutationRate
        else:
            # More distant or unrelated - use similarity
            self_weight = 0.5 + (similarity / 2) * self.mutationRate
            other_weight = 1 - self_weight * other.mutationRate
        for attr_name in offspring.__dataclass_fields__:
            if attr_name in ['name', 'color', 'symbol', 'parts', 'statcache', 'speciesParents', 'preferredFood', 'inedibleFood']:
                continue
            
            self_val = getattr(self, attr_name)
            other_val = getattr(other, attr_name)
            
            if isinstance(self_val, (int, float)):
                new_val = (self_val * self_weight) + (other_val * other_weight)
                setattr(offspring, attr_name, new_val)
            elif isinstance(self_val, Enum):
                if random.random() < self_weight:
                    setattr(offspring, attr_name, self_val)
                else:
                    setattr(offspring, attr_name, other_val)
        offspring.parts = []
        parts_from_self = random.sample(self.parts, max(1, int(len(self.parts) * self_weight)))
        parts_from_other = random.sample(other.parts, max(1, int(len(other.parts) * other_weight)))
        offspring.parts.extend(parts_from_self)
        offspring.parts.extend(parts_from_other)
        
        # Blend mutation rate
        offspring.mutationRate = (self.mutationRate + other.mutationRate) / 2
        offspring = offspring.mutate()
        offspring.genStats(True)
        
        # Cache the result for both parent orders
        self.reprocacheOffspring[mother_key] = offspring
        other.reprocacheOffspring[father_key] = offspring
        
        return offspring
    
    @property
    def canFly(self) -> bool:
        """Calculate flight capability based on wing parts"""
        if not self.canMove:
            return False
        # Need at least one functional wing to fly
        return any(part.can_fly for part in self.parts if isinstance(part, limb))
    
    @classmethod
    def generate_random_species(cls, existing_species: Optional[list['Species']] = None,  # type: ignore
                           organism_type: Optional[str] = None) -> 'Species':
        if organism_type is None:
            organism_type = random.choice(['plant', 'animal'])
        color = (random.randint(0, 255), (random.randint(0, 255)), (random.randint(0, 255)))
        symbol = random.choice(RANDOMCHAR)
        name = "".join(random.sample(ALPHABET, random.randint(5, 12)))
        new_species = Species(
            name=name,
            color=color,
            symbol=symbol,
            mutationRate=random.uniform(0.0005, 0.002)
        )
        if organism_type == 'plant':
            new_species = Species._configure_plant(new_species)
        else:  # animal
            new_species = Species._configure_animal(new_species)
        
        if existing_species:
            new_species = Species._configure_consumer(new_species, existing_species)
        new_species.parts = Species._generate_random_parts(new_species, organism_type)
        
        # Calculate final stats
        new_species.genStats(True)
        
        return new_species

    @classmethod
    def _configure_plant(cls, species: 'Species') -> 'Species':
        species.Rooted = True
        species.canMove = False
        species.moveSpeed = 0.0
        species.moveCost = 0.0
        species.sleepHabits = Species.habits.NONE
        species.sleepRatio = 0.0
        species.dormancy = random.random() < 0.7  # Most plants can go dormant
        species.dormanThresh = random.uniform(0.1, 0.5)
        species.reproMethod = random.choice([
            Species.ReproMethod.EXPANSIVE_GROWTH,
            Species.ReproMethod.BROADCAST,
            Species.ReproMethod.SPORES,
            Species.ReproMethod.BUDDING
        ])
        species.flowering = random.random() < 0.5
        species.tempProduction = random.uniform(0, 5)  # Plants generate little heat
        
        # Plant life cycle parameters
        species.maturity = random.randint(1000, 10000)  # 20-200 days
        species.reproFrequency = random.randint(200, 2000)  # 4-40 days
        species.seedAge = random.randint(50, 500)  # 1-10 days
        species.seedMaturity = random.randint(20, 200)  # 0.5-4 days
        species.pMS = random.randint(5000, 50000)  # 100-1000 days
        
        return species

    @classmethod
    def _configure_animal(cls, species: 'Species') -> 'Species':
        species.Rooted = False
        species.canMove = True
        species.moveSpeed = random.uniform(0.1, 5.0)
        species.moveCost = random.uniform(0.01, 0.5)
        species.sleepHabits = random.choice(list(Species.habits))
        species.sleepRatio = random.uniform(0.2, 0.8)
        species.dormancy = random.random() < 0.3  # Some animals can hibernate
        species.dormanThresh = random.uniform(0.05, 0.3) if species.dormancy else 0.3
        species.reproMethod = random.choice([
            Species.ReproMethod.OVIPARITY,
            Species.ReproMethod.VIVIPARITY,
            Species.ReproMethod.OVOVIPARITY,
            Species.ReproMethod.PARTHENOGENESIS
        ])
        species.flowering = False
        species.tempProduction = random.uniform(10, 30)  # Animals generate more heat
        
        # Animal life cycle parameters
        species.maturity = random.randint(500, 5000)  # 10-100 days
        species.reproFrequency = random.randint(100, 1000)  # 2-20 days
        species.seedAge = random.randint(20, 200)  # 0.5-4 days
        species.seedMaturity = random.randint(10, 100)  # 0.2-2 days
        species.pMS = random.randint(2500, 25000)  # 50-500 days
        
        # Behavioral traits
        species.aggression = random.uniform(0.0, 1.0)
        species.curious = random.uniform(0.0, 1.0)
        species.social = random.uniform(0.0, 1.0)
        species.adapative = random.uniform(0.0, 1.0)
        
        return species

    @classmethod
    def _configure_consumer(cls, species: 'Species', existing_species: list['Species']) -> 'Species':
        """Configure species to consume other species"""
        # Clear any existing food preferences
        species.preferredFood = {}
        species.inedibleFood = {}
        
        # Determine how many species to consume (1 to all)
        num_to_consume = random.randint(1, len(existing_species))
        consumed_species = random.sample(existing_species, num_to_consume)
        
        for prey in consumed_species:
            # Randomly decide if this is preferred or inedible (with small chance)
            if random.random() < 0.9:  # 90% chance to be preferred
                # Value represents preference strength (0.1-1.0)
                species.preferredFood[prey.name] = random.uniform(0.1, 1.0)
            else:
                # Value represents inedibility (0.0 would mean completely inedible)
                species.inedibleFood[prey.name] = random.uniform(0.0, 0.5)
        
        return species

    @classmethod
    def _generate_random_parts(cls, species: 'Species', organism_type: str) -> list[part]:
        """Generate random parts based on organism type"""
        parts = []
        if organism_type == 'plant':
            parts.append(root.randomize())
            if random.random() < 0.9:
                ste = stem.randomize()
            
                lea = leaf.randomize()
                leaves = []
                for i in range(10):
                    leaves.append(lea.mutate())
                ste.subparts.extend(leaves)
                if ste.type in [stem.StemType.HERBACEOUS, stem.StemType.VINE]:
                    stems = []
                    for i in range(10):
                        stems.append(ste.mutate())
                    parts.extend(stems)
                else:
                    parts.append(ste)
            
            # Add reproductive structures
            if species.flowering:
                parts.append(flower.randomize())
                
                parts.append(fruit.randomize())
            else:
                parts.append(reproductive.randomize("plant"))
                
                parts.append(pollen.randomize())
            
            # Add defensive structures
            if random.random() < 0.5:
                parts.append(appendage.randomize("plant"))
        
        else: 
            parts.append(skin.randomize())
            limb_count = random.randint(0, 10)  # Some animals might have no limbs (worms)
            for i in range(limb_count):
                parts.append(limb.randomize("animal"))
            
            sensory_count = random.randint(1, 4)  # Multiple sensory types
            for i in range(sensory_count):
                parts.append(sensory.randomize())
            
            # Add internal organs
            internal_types = list(internal.InternalType)
            for internal_type in internal_types:
                # Not all animals have all organ types
                if random.random() < 0.7:  # 70% chance to have each organ type
                    parts.append(internal(
                        _name=f"{internal_type.name.lower()}",
                        type=internal_type,
                        efficiency=random.uniform(0.5, 1.5),
                        capacity=random.uniform(0.5, 2.0),
                        can_regenerate=random.random() < 0.2
                    ))
            
            # Add weapons (if any)
            if random.random() < 0.7:  # 70% chance to have some weapon
                weapon_count = random.randint(1, 5)
                for i in range(weapon_count):
                    wea = appendage.randomize()
                    if weapon_count % 2 == 0:
                        for j in range(weapon_count):
                            parts.append(wea)
                    else:
                        parts.append(appendage.randomize())
                        for j in range(weapon_count - 1):
                            parts.append(wea)
            
            # Add reproductive organs
            parts.append(reproductive.randomize())
            
            # Add egg sac if oviparous
            if species.reproMethod == Species.ReproMethod.OVIPARITY:
                parts.append(egg_sac(
                    _name="egg_sac",
                    type=random.choice(list(egg_sac.EggType)),
                    egg_count=random.randint(1, 100),
                    protection=random.uniform(0.5, 2.0),
                    nutrient_store=random.uniform(0.5, 3.0),
                    incubation_time=random.uniform(1.0, 30.0),
                    desiccation_resistance=random.uniform(0.0, 1.0),
                    camouflage=random.uniform(0.0, 0.8)
                ))
        
        return parts
    
    def to_torch(self, device=DEVICE) -> dict[str, Any]:
        """Convert species attributes to PyTorch tensors for GPU computation."""
        tensor_dict = {}
        
        # Convert basic attributes
        for field_name, field_value in self.__dict__.items():
            if field_name in ['parts', 'statcache', 'speciesParents', 'name', 'color', 'symbol']:
                continue
                
            if isinstance(field_value, (int, float)):
                tensor_dict[field_name] = torch.tensor([field_value], dtype=torch.float32, device=device)
            elif isinstance(field_value, bool):
                tensor_dict[field_name] = torch.tensor([int(field_value)], dtype=torch.float32, device=device)
            elif isinstance(field_value, Enum):
                tensor_dict[field_name] = torch.tensor([field_value.value], dtype=torch.float32, device=device)
            elif isinstance(field_value, dict):
                # Convert food dictionaries
                if field_name in ['preferredFood', 'inedibleFood']:
                    tensor_dict[field_name] = {
                        k: torch.tensor([v], dtype=torch.float32, device=device) 
                        for k, v in field_value.items()
                    }
        
        # Convert color to tensor
        if hasattr(self, 'color'):
            tensor_dict['color'] = torch.tensor(self.color, dtype=torch.float32, device=device)
        
        # Convert parts recursively
        if hasattr(self, 'parts') and self.parts:
            part_tensors = []
            for part in self.parts:
                part_tensors.append(part.to_torch(device=device))
            tensor_dict['parts'] = part_tensors
            
        return tensor_dict

TREE = Species(
    name="Tree",
    Rooted=True,
    canMove=False,
    reproMethod=Species.ReproMethod.BROADCAST,
    tempSensCold=-20,
    tempSensHot=75,
    parts=[stem(_name="trunk", type=stem.StemType.WOODY, _vital=True, size= 1.0,
            height=20.0, structural_strength=2.5, annual_growth_rings=True,
            minTemp=-30, maxTemp=80.0, subparts=[
                skin(name="bark", type=skin.Skin.BARK, size=0.9, thickness=1.5,
                     insulation_factor=1.8, energyCost=0.01),
                root(_name="taproot", type=root.RootType.TAP, size=0.4,
                     depth=15.0, absorption_rate=0.8),
                reproductive(_name="cones", type=reproductive.ReproductiveType.CONE, size=0.1,
                             fertility=0.8, resource_cost=1.0, seasonal=True),
                leaf(_name="needle_cluster_0", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_1", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_2", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_3", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_4", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_5", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_6", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_7", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_8", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_9", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_10", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_11", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_12", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_13", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_14", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_15", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_16", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_17", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
                leaf(_name="needle_cluster_18", type=leaf.LeafType.NEEDLE, size=0.02, surface_area=0.5,
                     photosynthetic_rate=0.7, water_loss_rate=0.1, seasonal=False),
            ])]   
).__post_init__()

BUSH = Species(
    name="bush",
    Rooted=True,
    canMove=False,
    reproMethod=Species.ReproMethod.BROADCAST,
    tempSensCold=50.0,
    tempSensHot=130.0,
    parts=[
        stem(_name="stem_0", type=stem.StemType.HERBACEOUS, _vital=False, size=1.0, height=1.5,
             flexibility=0.8, structural_strength=0.7, minTemp=40.0, maxTemp=140.0, subparts=[
                skin(name="waxy_cuticle", type=skin.Skin.CUTICLE, size=0.9, thickness=0.2, 
                     water_retention=1.5, cooling_factor=1.2, energyCost=0.01),
                leaf(_name="succulent_leaf_0", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_1", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_2", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_3", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_4", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_5", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                flower(_name="bright_flower", type=flower.FlowerType.SIMPLE, size=0.1, 
                       pollination_method="insect", nectar_production=0.5, scent_strength=0.3, 
                       color_variety=3, seed_production=5)
             ]),
        stem(_name="stem_1", type=stem.StemType.HERBACEOUS, _vital=False, size=1.0, height=1.5,
             flexibility=0.8, structural_strength=0.7, minTemp=40.0, maxTemp=140.0, subparts=[
                skin(name="waxy_cuticle", type=skin.Skin.CUTICLE, size=0.9, thickness=0.2, 
                     water_retention=1.5, cooling_factor=1.2, energyCost=0.01),
                leaf(_name="succulent_leaf_0", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_1", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_2", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_3", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_4", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_5", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                flower(_name="bright_flower", type=flower.FlowerType.SIMPLE, size=0.1, 
                       pollination_method="insect", nectar_production=0.5, scent_strength=0.3, 
                       color_variety=3, seed_production=5)
             ]),
        stem(_name="stem_2", type=stem.StemType.HERBACEOUS, _vital=False, size=1.0, height=1.5,
             flexibility=0.8, structural_strength=0.7, minTemp=40.0, maxTemp=140.0, subparts=[
                skin(name="waxy_cuticle", type=skin.Skin.CUTICLE, size=0.9, thickness=0.2, 
                     water_retention=1.5, cooling_factor=1.2, energyCost=0.01),
                leaf(_name="succulent_leaf_0", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_1", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_2", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_3", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_4", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(_name="succulent_leaf_5", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                flower(_name="bright_flower", type=flower.FlowerType.SIMPLE, size=0.1, 
                       pollination_method="insect", nectar_production=0.5, scent_strength=0.3, 
                       color_variety=3, seed_production=5)
             ]),
        root(_name="fibrous_roots", type=root.RootType.FIBROUS, size=0.5, spread=3.0, absorption_rate=1.2)
    ]
).__post_init__()

@dataclass
class Organism:
    #creature is created from a species, at a position.
    species: Species
    x: float
    y: float
    z: float
    age: int = 0
    facing: float = 0.0
    sex: int = 0 # 0 is female, 1 is male
    injuries: dict[str, float] = field(default_factory=dict)
    parts: list[part] = field(default_factory=list)
    asleep: bool = False
    dormant: bool = False
    current_stats: dict = field(default_factory=dict)

    #the below will be used for basically "world history".
    history: list[str] = field(default_factory=list) # probably shouldnt use this. might cause memory issues
    parents: Optional[list['Organism']] = field(default_factory=list) #optional only for first gen. after that, required. 
    memory: dict[str, Any] = field(default_factory=dict) # this probably wont be used either, but potentially somehow track "known bad" and "known good" areas. 
    
    @property
    def energy(self) -> float:
        return self._energy
    
    @energy.setter
    def energy(self, value: float) -> None:
        self._energy = max(0, min(value, self.species.maxE))

    @property
    def moveSpeed(self) -> float:
        return self._moveSpeed
    
    @moveSpeed.setter
    def moveSpeed(self, value: float) -> None:
        self._moveSpeed = value
    
    @property
    def rootE(self) -> float:
        return self._rootE

    @rootE.setter
    def rootE(self, value: float) -> None:
        self._rootE = max(0, min(value, self.species.rootE))

    @property
    def reproductiveCooldown(self) -> int:
        return getattr(self, '_reproductiveCooldown', 0)

    @reproductiveCooldown.setter
    def reproductiveCooldown(self, value: int) -> None:
        self._reproductiveCooldown = max(0, value)

    @property
    def gestationProg(self) -> float:
        return getattr(self, '_gestationProg', 0.0)

    @gestationProg.setter
    def gestationProg(self, value: float) -> None:
        self._gestationProg = max(0.0, value)

    @property
    def floweringCycle(self) -> int:
        return getattr(self, '_floweringCycle', 0)

    @floweringCycle.setter
    def floweringCycle(self, value: int) -> None:
        self._floweringCycle = max(0, value)

    @property
    def current_temp(self) -> float:
        return getattr(self, '_current_temp', (self.species.tempSensCold + self.species.tempSensHot)/2)

    @current_temp.setter
    def current_temp(self, value: float) -> None:
        self._current_temp = value

    @property
    def can_fly(self) -> bool:
        if not self.species.canMove:
            return False
            
        wing_stats = self._get_part_stats(limb)
        if not wing_stats:
            return False
            
        healthy_wings = sum(1 for wing in wing_stats if wing['health_ratio'] >= 0.7)
        total_wings = len(wing_stats)
        
        if total_wings >= 4:
            return healthy_wings >= 2
        elif total_wings == 2:
            return healthy_wings >= 2
        else:
            return healthy_wings >= 1

    @property
    def flight_efficiency(self) -> float:
        if not self.can_fly:
            return 0.0
            
        wing_stats = self._get_part_stats(limb)
        total_efficiency = sum(
            wing['movement_speed'] * wing['health_ratio'] 
            for wing in wing_stats 
            if wing['health_ratio'] >= 0.7
        )

        healthy_wings = sum(1 for wing in wing_stats if wing['health_ratio'] >= 0.7)
        if healthy_wings >= 4:
            return total_efficiency * 1.5
        elif healthy_wings == 3:
            return total_efficiency * 1.2
        elif healthy_wings == 2:
            return total_efficiency * 1.0
        else:
            return total_efficiency * 0.2

    @property
    def vital_health(self) -> float:
        total_health = 0.0
        vital_count = 0
        
        for part in self.parts:
            if part.vital:
                health_ratio = (part.health - part.injuries.get(part.name, 0.0)) / part.health
                total_health += health_ratio
                vital_count += 1
                
        if vital_count == 0:
            return 1.0
        return total_health / vital_count

    @property
    def is_alive(self) -> bool:
        for part in self.parts:
            if part.vital:
                health_ratio = (part.health - part.injuries.get(part.name, 0.0)) / part.health
                if health_ratio <= 0:
                    return False
        return True

    @property
    def move_cost(self) -> float:
        return self.current_stats.get('moveCost', 0.1)

    @property
    def idle_energy_cost(self) -> float:
        return self.current_stats.get('idleE', 0.1)

    @property
    def sleep_energy_cost(self) -> float:
        return self.current_stats.get('sleepE', 0.05)

    @property
    def dormancy_energy_cost(self) -> float:
        return self.current_stats.get('dormE', 0.01)

    @property
    def can_climb(self) -> bool:
        return self.current_stats.get('climbing', 0) > 0.5

    @property
    def can_swim(self) -> bool:
        return self.current_stats.get('swimming', 0) > 0.5

    @property
    def can_dig(self) -> bool:
        return self.current_stats.get('digging', 0) > 0.5

    @property
    def vision_quality(self) -> float:
        return self.current_stats.get('vision_quality', 0)

    @property
    def is_mature(self) -> bool:
        return self.age >= self.species.maturity and self.age <= self.species.pMS


    def __post_init__(self):
        self.current_stats = copy.deepcopy(self.species.statcache)
        if self.age < self.species.maturity:
            self.energy = self.species.maxE * (self.age + random.uniform(1, self.species.maturity) / self.species.maturity) # start at less than normal energy
            self.rootE = self.species.rootE * (self.age + random.uniform(1, self.species.maturity) / self.species.maturity)
        else:
            self.energy = self.species.maxE
            self.rootE = self.species.rootE
        if self.age > self.species.maturity:
            self.reproductiveCooldown = self.age % self.species.reproFrequency
            if hasattr(self.current_stats, 'gestation_period') and self.current_stats['gestation_period'] != 0 and self.sex == 0:
                self.gestationProg = random.uniform(0, self.current_stats['gestation_period'])
        self.flowering = self.species.flowering
        self.moveSpeed = self.species.moveSpeed
        return self
    
    def ageUp(self, steps: int = 1) -> None:
        self.age += steps
        if self.asleep or self.dormant:
            energy_cost = (self.current_stats.get('sleepE', 0) if self.asleep else (self.current_stats.get('dormE', 0)))
            self.energy -= energy_cost * steps
        else:
            self.energy -= self.current_stats.get('idleE', 0) * steps
        if hasattr(self, 'reproductiveCooldown'):
            self.reproductiveCooldown = max(0, self.reproductiveCooldown - steps)
        if hasattr(self, 'gestationProg'):
            self.gestationProg += steps
            if self.gestationProg >= self.current_stats.get('gestation_period', 0):
                self.gestationProg = 0
        if self.species.flowering and hasattr(self, 'floweringCycle'):
            self.floweringCycle += steps
            if self.floweringCycle >= self.current_stats.get('flowering_frequency', 1000):
                self.floweringCycle = 0
                self.flowering = not self.flowering
 
    def updateTemp(self, env_temp: float) -> float:
        temp_diff = env_temp - self.current_stats.get('tempSensCold', 50)
        resistance = self.current_stats.get('tempResis', 1.0)
        production = self.current_stats.get('tempProduction', 0)
        shedding = self.current_stats.get('tempShed', 1.0)
        temp_change = (temp_diff / resistance) + production - shedding
        
        self.current_temp = env_temp + temp_change
        
        min_temp = self.current_stats.get('tempSensCold', -50)
        max_temp = self.current_stats.get('tempSensHot', 150)
        
        if self.current_temp < min_temp or self.current_temp > max_temp:
            severity = max(
                abs(self.current_temp - min_temp) if self.current_temp < min_temp else 0,
                abs(self.current_temp - max_temp) if self.current_temp > max_temp else 0
            )
            self.injure(severity * 0.01)
        
        return self.current_temp
    
    def updateSleep(self, time_of_day: float) -> bool:
        if self.species.sleepHabits == Species.habits.NONE:
            return False
        
        was_asleep = self.asleep
        
        if self.species.sleepHabits == Species.habits.DIURNAL:
            self.asleep = time_of_day < 0.25 or time_of_day > 0.75
        elif self.species.sleepHabits == Species.habits.NOCTURNAL:
            self.asleep = 0.25 <= time_of_day <= 0.75
        elif self.species.sleepHabits == Species.habits.CREPUSCULAR:
            self.asleep = 0.4 <= time_of_day <= 0.6 or time_of_day <= 0.1 or time_of_day >= 0.9
        else:
            sleep_chance = self.species.sleepRatio
            if random.random() < sleep_chance / 100:
                self.asleep = not self.asleep
        
        return was_asleep != self.asleep
    
    def reproduce(self, partner: Optional['Organism'] = None) -> Optional['Organism']:
        if self.age < self.species.maturity or self.age > self.species.pMS:
            return None
        if hasattr(self, 'reproductiveCooldown') and self.reproductiveCooldown > 0:
            return None
        if self.species.Rooted and (partner is None or partner is self):
            if random.random() < self.species.mutationRate:
                offspring_species = self.species.mutate()
                return Organism(species=offspring_species,x=self.x,y=self.y,z=self.z,parents=[self])
            if random.random() < self.species.reproFrequency:
                return Organism(species=self.species, parents=[self], x=self.x, y=self.y, z=self.z)
            return None
        
        if partner is None:
            return None

        if partner.age < partner.species.maturity or partner.age > partner.species.pMS:
            return None
        
        if hasattr(partner, 'reproductiveCooldown') and partner.reproductiveCooldown > 0:
            return None
        
        if self.species.reproMethod not in [Species.ReproMethod.PARTHENOGENESIS, Species.ReproMethod.BROADCAST,Species.ReproMethod.SPORES] and partner.species.reproMethod not in [Species.ReproMethod.PARTHENOGENESIS, Species.ReproMethod.BROADCAST,Species.ReproMethod.SPORES]:
            if self.sex == partner.sex:
                return None
            
        if partner.species != self.species:
            if self.sex == 0:
                offspring_species = self.species.get_hybrid_offspring(partner.species)
            elif partner.sex == 0:
                offspring_species = partner.species.get_hybrid_offspring(self.species)
            if offspring_species is not None:
                if random.random() < self.species.reproFrequency:    
                    return Organism(species=offspring_species, parents=[self, partner], x=self.x, y=self.y, z=self.z)
        
        if random.random() < self.species.mutationRate:
            offspring_species = self.species.mutate()
        else:
            offspring_species = self.species
        self.reproductiveCooldown = self.species.reproFrequency
        if partner is not self:
            partner.reproductiveCooldown = partner.species.reproFrequency
        
        return Organism(species=offspring_species,x=(self.x + partner.x) / 2,y=(self.y + partner.y) / 2,z=(self.z + partner.z) / 2,parents=[self, partner])
    
    def injure(self, damage: float, part_name: Optional[str] = None) -> bool:
        if part_name:
            part = next((p for p in self.parts if p.name == part_name), None)
            if not part:
                return False
            
            self.injuries[part_name] = self.injuries.get(part_name, 0.0) + damage
            if self.injuries[part_name] >= part.health:
                if part.vital:
                    return True
                self.parts.remove(part)
                del self.injuries[part_name]
            return False
        else:
            total_health = sum(p.health for p in self.parts)
            if total_health <= 0:
                return True  # Organism is dead
            
            remaining_damage = damage
            while remaining_damage > 0 and self.parts:
                weights = [p.health for p in self.parts]
                part = random.choices(self.parts, weights=weights)[0]
                
                part_damage = min(remaining_damage, part.health * 0.5)  # Max 50% damage per part
                self.injuries[part.name] = self.injuries.get(part.name, 0.0) + part_damage
                
                if self.injuries[part.name] >= part.health:
                    if part.vital:
                        return True  # Organism dies
                    self.parts.remove(part)
                    del self.injuries[part.name]
                
                remaining_damage -= part_damage
            
            return False
    
    def heal(self, amount: float, part_name: Optional[str] = None) -> None:
        if part_name:
            #this is to allow for manual healing.
            if part_name in self.injuries:
                self.injuries[part_name] = max(0, self.injuries[part_name] - amount)
                if self.injuries[part_name] == 0:
                    del self.injuries[part_name]
        else:
            if not self.injuries:
                return
                
            heal_per_part = amount / len(self.injuries)
            for part_name in list(self.injuries.keys()):
                self.injuries[part_name] = max(0, self.injuries[part_name] - heal_per_part)
                if self.injuries[part_name] == 0:
                    del self.injuries[part_name]
    
    def Move(self, dx: float, dy: float, dz: float):
        if self.species.Rooted or not self.species.canMove:
            return (self.x, self.y, self.z)
        
        requested_distance = math.sqrt(dx**2 + dy**2 + dz**2)
        move_speed = self.current_stats.get('moveSpeed', 1.0)
        
        if requested_distance > move_speed:
            scale_factor = move_speed / requested_distance
            dx *= scale_factor
            dy *= scale_factor
            dz *= scale_factor
            requested_distance = move_speed
        
        # Calculate movement cost
        move_cost = self.current_stats.get('moveCost', 0.1) * requested_distance
        
        # Check energy
        if self.energy < move_cost:
            return (self.x, self.y, self.z)
        
        # Apply horizontal movement
        self.x += dx
        self.y += dy
        
        # Apply vertical movement with checks
        if dz != 0:
            if self.current_stats['canFly']:
                self.z += dz
            elif self.current_stats['climbing']:
                # TODO: Add check for climbable surface here
                self.z += dz
        
        # Deduct energy and update facing direction
        self.energy -= move_cost
        if dx != 0 or dy != 0:
            self.facing = math.atan2(dy, dx)
        
        return (self.x, self.y, self.z)
    
    def tryMove(self, dx: float, dy: float, dz: float) -> bool:
        if not self.species.canMove:
            return False
            
        # Check if trying to move vertically without flight capability
        if dz != 0 and not self.can_fly:
            return False
            
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Calculate maximum possible distance based on move speed and energy
        max_possible_by_speed = self.species.moveSpeed
        max_possible_by_energy = self.energy / self.species.moveCost if self.species.moveCost > 0 else float('inf')
        max_possible_distance = min(distance, max_possible_by_speed, max_possible_by_energy)
        
        # If we can't move at all
        if max_possible_distance <= 0:
            return False
            
        # Scale down the movement vector if needed
        if max_possible_distance < distance:
            scale_factor = max_possible_distance / distance
            dx *= scale_factor
            dy *= scale_factor
            dz *= scale_factor
            distance = max_possible_distance
        
        # Calculate energy cost
        cost = distance * self.species.moveCost
        
        # Update position and energy
        self.energy -= cost
        self.x += dx
        self.y += dy
        self.z += dz
        return True

    def _get_part_stats(self, part_type: type) -> list[dict]:
        stats = []
        for part in self.parts:
            if isinstance(part, part_type):
                health_ratio = (part.health - part.injuries.get(part.name, 0.0)) / part.health
                part_stats = part.calculateStats()
                part_stats['health_ratio'] = health_ratio
                stats.append(part_stats)
        return stats

    def get_vital_health(self) -> float:
        total_health = 0.0
        vital_count = 0
        
        for part in self.parts:
            if part.vital:
                health_ratio = (part.health - part.injuries.get(part.name, 0.0)) / part.health
                total_health += health_ratio
                vital_count += 1
                
        if vital_count == 0:
            return 1.0
        return total_health / vital_count

@dataclass
class simSettings:
    size: tuple[int, int, int] = (50, 50, 10) # x, y, z defaults
    axialTilt: float = 23.44
    latitude: float = 40.0
    initialSpecies: list[Species] = [BUSH, TREE]

@dataclass
class WorldGrid:
    maxX: int = 50
    maxY: int = 50
    maxZ: int = 10
    organisms: list[Organism] = field(default_factory=list)
    species: list[Species] = field(default_factory=list)

    cell_size: float = 5.0  # Size of each grid cell
    spatial_grid: dict[tuple[int, int, int], list[Organism]] = field(default_factory=dict)
    
    def addOrganism(self, organism: Organism):
        self.organisms.append(organism)
        self._add_to_spatial_grid(organism)
    
    def _add_to_spatial_grid(self, organism: Organism):
        cell_x = int(organism.x / self.cell_size)
        cell_y = int(organism.y / self.cell_size)
        cell_z = int(organism.z / self.cell_size)
        cell_key = (cell_x, cell_y, cell_z)
        
        if cell_key not in self.spatial_grid:
            self.spatial_grid[cell_key] = []
        self.spatial_grid[cell_key].append(organism)
    
    def getOrgsInRange(self, center: tuple[float, float, float], radius: float,
                      shape: str = 'circle', facing: Optional[tuple[float, float, float]] = None,
                      angle: Optional[float] = None) -> Optional[list[Organism]]:
        x, y, z = center
        
        # Determine which grid cells to check
        min_cell_x = int((x - radius) / self.cell_size)
        max_cell_x = int((x + radius) / self.cell_size)
        min_cell_y = int((y - radius) / self.cell_size)
        max_cell_y = int((y + radius) / self.cell_size)
        min_cell_z = int((z - radius) / self.cell_size)
        max_cell_z = int((z + radius) / self.cell_size)
        
        orgs_in_range = []
        
        # Check all relevant grid cells
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                for cell_z in range(min_cell_z, max_cell_z + 1):
                    if (cell_x, cell_y, cell_z) in self.spatial_grid:
                        for org in self.spatial_grid[(cell_x, cell_y, cell_z)]:
                            dx = org.x - x
                            dy = org.y - y
                            dz = org.z - z
                            
                            if shape == 'circle':
                                distance_sq = dx*dx + dy*dy + dz*dz
                                if distance_sq <= radius*radius:
                                    orgs_in_range.append(org)
                            elif shape == 'square':
                                if abs(dx) <= radius and abs(dy) <= radius and abs(dz) <= radius:
                                    orgs_in_range.append(org)
                            elif shape == 'cone' and facing is not None and angle is not None:
                                distance_sq = dx*dx + dy*dy + dz*dz
                                if distance_sq <= radius*radius:
                                    # Calculate angle between facing direction and organism direction
                                    dot_product = dx*facing[0] + dy*facing[1] + dz*facing[2]
                                    org_distance = math.sqrt(distance_sq)
                                    facing_distance = math.sqrt(facing[0]**2 + facing[1]**2 + facing[2]**2)
                                    org_angle = math.acos(dot_product / (org_distance * facing_distance))
                                    if org_angle <= angle:
                                        orgs_in_range.append(org)
        
        return orgs_in_range or None
        
    def addSpecies(self, species: Species):
        if species not in self.species:
            self.species.append(species)
    
    def find_food(self, organism: Organism, radius: float = 10.0) -> Optional[list[Organism]]:
        """Find potential food sources for the organism within a given radius."""
        if not organism.species.preferredFood:
            return None
            
        nearby_orgs = self.getOrgsInRange((organism.x, organism.y, organism.z), radius)
        if not nearby_orgs:
            return None
            
        potential_food = []
        for org in nearby_orgs:
            # Check if this organism is edible
            if org.species.name in organism.species.preferredFood:
                # Higher preference food comes first
                preference = organism.species.preferredFood[org.species.name]
                potential_food.append((preference, org))
            # Also check if this is a plant that can be eaten (plants don't have preferredFood lists)
            elif organism.species.preferredFood.get("plants", 0) > 0 and org.species.Rooted:
                potential_food.append((0.5, org))  # Default preference for generic plants
        
        if not potential_food:
            return None
            
        # Sort by preference (highest first)
        potential_food.sort(reverse=True, key=lambda x: x[0])
        return [org for (pref, org) in potential_food]

    def find_danger(self, organism: Organism, radius: float = 10.0) -> Optional[list[Organism]]:
        """Find potential threats to the organism within a given radius."""
        nearby_orgs = self.getOrgsInRange((organism.x, organism.y, organism.z), radius)
        if not nearby_orgs:
            return None
            
        threats = []
        for org in nearby_orgs:
            # Check if this organism considers us as food
            if organism.species.name in org.species.preferredFood:
                threats.append(org)
            # Check if this organism is aggressive towards us
            elif org.species.aggression > 0.7 and org != organism:
                threats.append(org)
        
        return threats or None

    def move_towards_food(self, organism: Organism, food: Organism) -> bool:
        """Move organism towards food source."""
        if not organism.species.canMove:
            return False
            
        dx = food.x - organism.x
        dy = food.y - organism.y
        dz = food.z - organism.z
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        if distance <= 0.5:  # Close enough to eat
            return True
            
        # Normalize direction vector
        dx /= distance
        dy /= distance
        dz /= distance
        
        # Scale by movement speed
        move_dist = min(distance, organism.species.moveSpeed)
        dx *= move_dist
        dy *= move_dist
        dz *= move_dist
        
        return organism.tryMove(dx, dy, dz)

    def move_away_from_danger(self, organism: Organism, danger: Organism) -> bool:
        """Move organism away from danger."""
        if not organism.species.canMove:
            return False
            
        dx = organism.x - danger.x
        dy = organism.y - danger.y
        dz = organism.z - danger.z
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        if distance >= 20.0:  # Far enough to be safe
            return False
            
        # Normalize direction vector
        if distance > 0:
            dx /= distance
            dy /= distance
            dz /= distance
        else:
            # If exactly on top of each other, move randomly
            dx, dy, dz = random.uniform(-1, 1), random.uniform(-1, 1), 0
        
        # Scale by movement speed (flee faster than normal movement)
        move_dist = min(organism.species.moveSpeed * 1.5, 20.0 - distance)
        dx *= move_dist
        dy *= move_dist
        dz *= move_dist
        
        return organism.tryMove(dx, dy, dz)

    def basic_ai_step(self, organism: Organism) -> None:
        """Basic AI decision making for an organism."""
        # Check for danger first
        dangers = self.find_danger(organism)
        if dangers:
            # Flee from the closest danger
            closest_danger = min(dangers, key=lambda d: 
                math.sqrt((d.x-organism.x)**2 + (d.y-organism.y)**2 + (d.z-organism.z)**2))
            self.move_away_from_danger(organism, closest_danger)
            return
            
        # If no danger, look for food
        food_sources = self.find_food(organism)
        if food_sources:
            # Go for the highest preference food
            preferred_food = food_sources[0]
            self.move_towards_food(organism, preferred_food)
            return
            
        # If no food or danger, random wandering
        if organism.species.canMove and random.random() < 0.3:  # 30% chance to move
            dx = random.uniform(-1, 1) * organism.species.moveSpeed
            dy = random.uniform(-1, 1) * organism.species.moveSpeed
            dz = 0  # Keep it simple with 2D movement unless flying
            organism.tryMove(dx, dy, dz)