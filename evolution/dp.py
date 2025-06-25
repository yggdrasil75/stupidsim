from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import random
from typing import Optional, Any
import dearpygui.dearpygui as dpg
import string
import numpy as np
import torch

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
        elif isinstance(val1, dict):
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
    injuries: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.subparts) == 0:
            self.setup_default_subparts()
        self.calculateStats(True)

    @abstractmethod
    def setup_default_subparts(self):
        pass

    def mutate(self):
        for attr_name in self.__dataclass_fields__:
            if attr_name in ['name', 'subparts', 'statCache', 'mutationRate']:
                continue

            if random.random() < self.mutationRate:
                current_val = getattr(self, attr_name)

                if isinstance(current_val, bool): #bools probably should be far less often switched.
                    if random.random() < self.mutationRate:
                        setattr(self, attr_name, not current_val)
                elif isinstance(current_val, (int, float)):
                    new_val = current_val * random.uniform(1-self.mutationRate, 1+self.mutationRate)
                    setattr(self, attr_name, new_val)
                elif isinstance(current_val, Enum):
                    enum_class = type(current_val)
                    choices = [e for e in enum_class]
                    if choices:
                        setattr(self, attr_name, random.choice(choices))
        
        for subpart in self.subparts:
            if random.random() < self.mutationRate and random.random() < self.mutationRate:
                self.subparts.remove(subpart)
            subpart.mutate()
        if random.random() < self.mutationRate and random.random() < self.mutationRate:
            self.subparts.append(self.randomize())
        self.statCache = None
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'part':
        part_type = random.choice([
            skin, limb, sensory, internal, weapon, appendage,
            root, stem, leaf, flower, fruit, reproductive,
            egg_sac, pollen
        ])

        return part_type.randomize(name, organism_type)

    def to_torch(self, device = DEVICE) -> dict[str, Any]:
        tensor_dict = {}
        
        for field_name, field_value in self.__dict__.items():
            if field_name in ['subparts', 'statCache']:
                continue
                
            if isinstance(field_value, (int, float)):
                tensor_dict[field_name] = torch.tensor([field_value], dtype=torch.float32, device=device)
            elif isinstance(field_value, bool):
                tensor_dict[field_name] = torch.tensor([int(field_value)], dtype=torch.float32, device=device)
            elif isinstance(field_value, Enum):
                tensor_dict[field_name] = torch.tensor([field_value.value], dtype=torch.float32, device=device)
        
        if hasattr(self, 'subparts') and self.subparts:
            subpart_tensors = []
            for subpart in self.subparts:
                subpart_tensors.append(subpart.to_torch(device=device))
            tensor_dict['subparts'] = subpart_tensors
            
        return tensor_dict
    
    def is_healthy(self, threshold: float = 0.7) -> bool:
        current_health = self.health - self.injuries.get(self.name, 0.0)
        return (current_health / self.health) >= threshold

    def count_healthy_subparts(self, part_type: type, threshold: float = 0.7) -> tuple[int, int]:
        total = 0
        healthy = 0
        
        for subpart in self.subparts:
            if isinstance(subpart, part_type):
                total += 1
                if subpart.is_healthy(threshold):
                    healthy += 1
                    
        return healthy, total

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        if self.statCache is not None and not recalc:
            return self.statCache
            
        stats = {}
        # Base stats
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

        # Calculate subpart contributions with health checks
        for subpart in self.subparts:
            substats = subpart.calculateStats(recalc)
            
            # Apply health multiplier to all stats from this subpart
            health_multiplier = (subpart.health - subpart.injuries.get(subpart.name, 0.0)) / subpart.health
            health_multiplier = max(0, min(1, health_multiplier))  # Clamp between 0-1
            
            for key, value in substats.items():
                weighted_value = value * subpart.size * health_multiplier
                if key in stats:
                    stats[key] += weighted_value
                else:
                    stats[key] = weighted_value

        # Special part count dependencies
        if isinstance(self, limb) and self.type == limb.LimbType.WING:
            # Flying capability depends on having enough healthy wings
            healthy_wings, total_wings = self.count_healthy_subparts(limb)
            if total_wings > 1:
                # Flying effectiveness scales with ratio of healthy wings
                wing_ratio = healthy_wings / total_wings
                if wing_ratio < 0.5:  # Less than half wings healthy
                    stats['can_fly'] = 0.0
                else:
                    stats['flight_efficiency'] = wing_ratio * self.movement_speed
                    
        elif isinstance(self, sensory) and self.type == sensory.SensoryType.EYE:
            # Vision quality depends on healthy eyes
            healthy_eyes, total_eyes = self.count_healthy_subparts(sensory)
            if total_eyes > 0:
                eye_ratio = healthy_eyes / total_eyes
                stats['vision_quality'] = eye_ratio * self.precision
                if eye_ratio < 0.3:  # Most eyes damaged
                    stats['blindness'] = 1.0 - eye_ratio
                    
        elif isinstance(self, internal) and self.type == internal.InternalType.HEART:
            # Circulatory efficiency drops with heart damage
            healthy_hearts, total_hearts = self.count_healthy_subparts(internal)
            if total_hearts > 0:
                heart_ratio = healthy_hearts / total_hearts
                stats['circulation'] *= heart_ratio
                if heart_ratio < 0.5:
                    stats['circulatory_shock'] = 1.0 - heart_ratio

        self.statCache = stats
        return stats

@dataclass
class skin(part):
    name: str = ""
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

    def mutate(self):
        super().mutate()
        if random.random() < self.mutationRate:
            self.pattern = random.choice(["solid", "stripes", "spots", "marbled", "mottled"])

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

    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'skin':
        skin_type = random.choice(list(skin.Skin))
        
        if organism_type == "plant":
            thickness = random.uniform(0.05, 0.3)
            can_photosynthesize = True
            water_retention = random.uniform(0.5, 2.0)
            camouflage = random.uniform(0.0, 0.3)
        else:
            thickness = random.uniform(0.1, 1.0)
            can_photosynthesize = random.random() < 0.1
            water_retention = random.uniform(0.0, 0.5)
            camouflage = random.uniform(0.0, 0.8)
            
        return cls(
            name=name or "skin",
            type=skin_type,
            thickness=thickness,
            max_thickness=thickness * random.uniform(1.2, 3.0),
            pattern=random.choice(["solid", "stripes", "spots", "marbled", "mottled"]),
            camouflage_effectiveness=camouflage,
            color_change_speed=random.uniform(0.0, 1.0),
            insulation_factor=random.uniform(0.2, 1.5),
            cooling_factor=random.uniform(0.2, 1.0),
            can_photosynthesize=can_photosynthesize,
            luminescent=random.random() < 0.05,
            luminense=random.uniform(0.0, 1.0) if random.random() < 0.05 else 0.0,
            water_retention=water_retention,
            gas_exchange=random.uniform(0.5, 1.5),
            energyCost=random.uniform(0.05, 0.2),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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
    can_climb: bool = False if type == LimbType.PSEUDOPOD else True
    can_swim: bool = True if type == LimbType.FIN else False
    can_dig: bool = False if (type == LimbType.PSEUDOPOD or type == LimbType.TENTACLE or type == LimbType.WING) else True
    can_fly: bool = True if type == LimbType.WING else False

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

    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'limb':
        limb_type = random.choice(list(limb.LimbType))
        
        if not name:
            name = f"{limb_type.name.lower()}_{random.randint(0, 1000)}"
            
        new_limb = cls(
            name=name,
            type=limb_type,
            size=random.uniform(0.2, 1.5),
            strength=random.uniform(0.5, 3.0),
            dexterity=random.uniform(0.1, 2.0),
            reach=random.uniform(0.3, 3.0),
            prehensile=random.random() < 0.3,
            movement_speed=random.uniform(0.5, 3.0),
            movement_cost=random.uniform(0.05, 0.3),
            can_climb=random.random() < 0.4,
            can_swim=random.random() < 0.4,
            can_dig=random.random() < 0.3,
            energyCost=random.uniform(0.1, 0.3),
            mutationRate=random.uniform(0.0001, 0.001)
        )
        
        new_limb.setup_default_subparts()
        return new_limb

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
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'sensory':
        sensory_type = random.choice(list(sensory.SensoryType))
        if not name:
            name = f"{sensory_type.name.lower()}_{random.randint(0, 1000)}"
            
        if sensory_type == sensory.SensoryType.EYE:
            can_see_colors = random.random() < 0.8
            night_vision = random.random() < 0.4
            thermal_vision = random.random() < 0.2
        else:
            can_see_colors = False
            night_vision = False
            thermal_vision = False
            
        return cls(
            name=name,
            type=sensory_type,
            range=random.uniform(0.5, 10.0),
            sensitivity=random.uniform(0.5, 2.0),
            precision=random.uniform(0.5, 1.5),
            night_vision=night_vision,
            thermal_vision=thermal_vision,
            can_see_colors=can_see_colors,
            underwater_effective=random.random() < 0.5,
            air_effective=True,
            angle=random.uniform(30, 180),
            active_cost=random.uniform(0.001, 0.01),
            passive_cost=random.uniform(0.0001, 0.001),
            energyCost=random.uniform(0.01, 0.05),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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

    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'internal':
        internal_type = random.choice(list(internal.InternalType))
        
        if not name:
            name = f"{internal_type.name.lower()}_{random.randint(0, 1000)}"
        if internal_type in [internal.InternalType.HEART, internal.InternalType.LUNG, internal.InternalType.GILL]:
            efficiency = random.uniform(0.8, 1.5)
            capacity = random.uniform(0.8, 2.0)
        else:
            efficiency = random.uniform(0.5, 1.5)
            capacity = random.uniform(0.5, 2.0)
            
        return cls(
            name=name,
            type=internal_type,
            efficiency=efficiency,
            capacity=capacity,
            can_regenerate=random.random() < 0.2,
            energyCost=random.uniform(0.05, 0.2),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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
            
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'weapon':
        weapon_type = random.choice(list(weapon.WeaponType))
        if not name:
            name = f"{weapon_type.name.lower()}_{random.randint(0, 1000)}"
        
        new_weapon = cls(
            name=name,
            type=weapon_type,
            damage=random.uniform(0.5, 3.0),
            attack_speed=random.uniform(0.5, 2.0),
            reach=random.uniform(0.1, 1.5),
            slash_damage=random.uniform(0.0, 2.0),
            pierce_damage=random.uniform(0.0, 2.0),
            blunt_damage=random.uniform(0.0, 1.0),
            venomous=random.random() < 0.3,
            retractable=random.random() < 0.2,
            attack_cost=random.uniform(0.05, 0.3),
            energyCost=random.uniform(0.05, 0.2),
            mutationRate=random.uniform(0.0001, 0.001)
        )
        new_weapon.setup_default_subparts()
        return new_weapon

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
        THORN = 8
    
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
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'appendage':
        appendage_type = random.choice(list(appendage.AppendageType))
        if not name:
            name = f"{appendage_type.name.lower()}_{random.randint(0, 1000)}"
        return cls(
            name=name,
            type=appendage_type,
            coverage=random.uniform(0.1, 0.8),
            hardness=random.uniform(0.5, 3.0),
            poison_production=random.random() < 0.2,
            venom_production=random.random() < 0.05,
            storesE=random.random() < 0.3,
            stores_water=random.random() < 0.4,
            stores_air=random.random() < 0.1,
            energyCost=random.uniform(0.05, 0.3),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "plant") -> 'root':
        root_type = random.choice(list(root.RootType))
        
        return cls(
            name=name or "root_system",
            type=root_type,
            depth=random.uniform(0.5, 5.0),
            spread=random.uniform(0.5, 10.0),
            absorption_rate=random.uniform(0.5, 2.0),
            nitrogen_fixing=random.random() < 0.3,
            storage_capacity=random.uniform(0.0, 3.0),
            can_propagate=random.random() < 0.2,
            energyCost=random.uniform(0.1, 0.5),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "plant") -> 'stem':
        stem_type = random.choice(list(stem.StemType))
        
        new_stem = cls(
            name=name or "stem",
            type=stem_type,
            height=random.uniform(0.1, 20.0),
            flexibility=random.uniform(0.1, 1.0),
            structural_strength=random.uniform(0.5, 3.0),
            storage_capacity=random.uniform(0.0, 2.0),
            photosynthetic=random.random() < 0.8,
            annual_growth_rings=random.random() < 0.5,
            energyCost=random.uniform(0.1, 0.5),
            mutationRate=random.uniform(0.0001, 0.001)
        )
        
        # Let the stem setup its own subparts
        new_stem.setup_default_subparts()
        return new_stem

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
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "plant") -> 'leaf':
        leaf_type = random.choice(list(leaf.LeafType))
        
        return cls(
            name=name or "leaf",
            type=leaf_type,
            surface_area=random.uniform(0.5, 5.0),
            thickness=random.uniform(0.01, 0.5),
            photosynthetic_rate=random.uniform(0.5, 2.0),
            water_loss_rate=random.uniform(0.1, 1.0),
            seasonal=random.random() < 0.5,
            defense_rating=random.uniform(0.0, 1.0),
            energyCost=random.uniform(0.05, 0.2),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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

    def mutate(self):
        super().mutate()
        if random.random() < self.mutationRate:
            self.pollination_method = random.choice(["wind", "insect", "bird", "bat", "water"])
    
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
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "plant") -> 'flower':
        flower_type = random.choice(list(flower.FlowerType))
        pollination_method = random.choice(["wind", "insect", "bird", "bat", "water"])
        
        return cls(
            name=name or "flower",
            type=flower_type,
            pollination_method=pollination_method,
            nectar_production=random.uniform(0.0, 1.0),
            scent_strength=random.uniform(0.0, 1.0),
            color_variety=random.randint(1, 3),
            blooming_period=random.uniform(0.5, 3.0),
            seed_production=random.uniform(0.5, 5.0),
            energyCost=random.uniform(0.1, 0.5),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "plant") -> 'fruit':
        fruit_type = random.choice(list(fruit.FruitType))
        
        return cls(
            name=name or "fruit",
            type=fruit_type,
            seed_count=random.randint(1, 100),
            dispersal_method=random.choice(["animal", "wind", "water", "explosive"]),
            nutritional_value=random.uniform(0.1, 2.0),
            ripening_time=random.uniform(0.5, 3.0),
            toxicity=random.uniform(0.0, 1.0),
            energyCost=random.uniform(0.1, 0.5),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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

    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'reproductive':
        """Create randomized reproductive organ."""
        if organism_type == "plant":
            repro_type = random.choice([
                reproductive.ReproductiveType.SPORE_SAC,
                reproductive.ReproductiveType.CONE,
                reproductive.ReproductiveType.FLOWER_BUD
            ])
            offspring_count = random.randint(10, 1000)
        else:
            repro_type = random.choice([
                reproductive.ReproductiveType.GONAD,
                reproductive.ReproductiveType.OVARY,
                reproductive.ReproductiveType.TESTIS,
                reproductive.ReproductiveType.UTERUS
            ])
            offspring_count = random.randint(1, 20)
            
        return cls(
            name=name or "reproductive",
            type=repro_type,
            fertility=random.uniform(0.5, 2.0),
            gestation_period=random.uniform(0.0, 30.0),
            offspring_count=offspring_count,
            mating_frequency=random.uniform(0.5, 3.0),
            resource_cost=random.uniform(0.1, 1.0),
            seasonal=random.random() < 0.4,
            energyCost=random.uniform(0.1, 0.5),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'egg_sac':
        """Create randomized egg sac."""
        egg_type = random.choice(list(egg_sac.EggType))
        
        return cls(
            name=name or "egg_sac",
            type=egg_type,
            egg_count=random.randint(1, 100),
            protection=random.uniform(0.5, 2.0),
            nutrient_store=random.uniform(0.5, 3.0),
            incubation_time=random.uniform(1.0, 30.0),
            desiccation_resistance=random.uniform(0.0, 1.0),
            camouflage=random.uniform(0.0, 0.8),
            energyCost=random.uniform(0.1, 0.5),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "plant") -> 'pollen':
        """Create randomized pollen."""
        pollen_type = random.choice(list(pollen.PollenType))
        
        return cls(
            name=name or "pollen",
            type=pollen_type,
            quantity=random.uniform(0.5, 5.0),
            viability=random.uniform(0.5, 1.0),
            dispersal_range=random.uniform(0.5, 5.0),
            allergenicity=random.uniform(0.0, 1.0),
            nutrient_content=random.uniform(0.0, 0.5),
            energyCost=random.uniform(0.05, 0.2),
            mutationRate=random.uniform(0.0001, 0.001)
        )

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
    statCache: dict[str, float] = field(default_factory=dict)


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
                        name=f"{internal_type.name.lower()}",
                        type=internal_type,
                        efficiency=random.uniform(0.5, 1.5),
                        capacity=random.uniform(0.5, 2.0),
                        can_regenerate=random.random() < 0.2
                    ))
            
            # Add weapons (if any)
            if random.random() < 0.7:  # 70% chance to have some weapon
                weapon_count = random.randint(1, 5)
                for i in range(weapon_count):
                    wea = weapon.randomize()
                    if weapon_count % 2 == 0:
                        for j in range(weapon_count):
                            parts.append(wea)
                    else:
                        parts.append(weapon.randomize())
                        for j in range(weapon_count - 1):
                            parts.append(wea)
            
            # Add reproductive organs
            parts.append(reproductive.randomize())
            
            # Add egg sac if oviparous
            if species.reproMethod == Species.ReproMethod.OVIPARITY:
                parts.append(egg_sac(
                    name="egg_sac",
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
    reproMethod=Species.ReproMethod.BROADCAST,
    tempSensCold=50.0,
    tempSensHot=130.0,
    parts=[
        stem(name="stem_0", type=stem.StemType.HERBACEOUS, vital=False, size=1.0, height=1.5,
             flexibility=0.8, structural_strength=0.7, minTemp=40.0, maxTemp=140.0, subparts=[
                skin(name="waxy_cuticle", type=skin.Skin.CUTICLE, size=0.9, thickness=0.2, 
                     water_retention=1.5, cooling_factor=1.2, energyCost=0.01),
                leaf(name="succulent_leaf_0", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_1", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_2", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_3", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_4", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_5", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                flower(name="bright_flower", type=flower.FlowerType.SIMPLE, size=0.1, 
                       pollination_method="insect", nectar_production=0.5, scent_strength=0.3, 
                       color_variety=3, seed_production=5)
             ]),
        stem(name="stem_1", type=stem.StemType.HERBACEOUS, vital=False, size=1.0, height=1.5,
             flexibility=0.8, structural_strength=0.7, minTemp=40.0, maxTemp=140.0, subparts=[
                skin(name="waxy_cuticle", type=skin.Skin.CUTICLE, size=0.9, thickness=0.2, 
                     water_retention=1.5, cooling_factor=1.2, energyCost=0.01),
                leaf(name="succulent_leaf_0", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_1", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_2", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_3", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_4", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_5", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                flower(name="bright_flower", type=flower.FlowerType.SIMPLE, size=0.1, 
                       pollination_method="insect", nectar_production=0.5, scent_strength=0.3, 
                       color_variety=3, seed_production=5)
             ]),
        stem(name="stem_2", type=stem.StemType.HERBACEOUS, vital=False, size=1.0, height=1.5,
             flexibility=0.8, structural_strength=0.7, minTemp=40.0, maxTemp=140.0, subparts=[
                skin(name="waxy_cuticle", type=skin.Skin.CUTICLE, size=0.9, thickness=0.2, 
                     water_retention=1.5, cooling_factor=1.2, energyCost=0.01),
                leaf(name="succulent_leaf_0", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_1", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_2", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_3", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_4", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                leaf(name="succulent_leaf_5", type=leaf.LeafType.SUCCULENT, size=0.05, 
                     thickness=0.8, photosynthetic_rate=0.9, water_loss_rate=0.2),
                flower(name="bright_flower", type=flower.FlowerType.SIMPLE, size=0.1, 
                       pollination_method="insect", nectar_production=0.5, scent_strength=0.3, 
                       color_variety=3, seed_production=5)
             ]),
        root(name="fibrous_roots", type=root.RootType.FIBROUS, size=0.5, spread=3.0, absorption_rate=1.2)
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
    current_stats: dict[str, Any] = field(default_factory=dict)

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
            if hasattr(self.species.statcache['gestation_period']) and self.species.statcache['gestation_period'] != 0 and self.sex == 0:
                self.gestationProg = random.uniform(0, self.species.statcache['gestation_period'])
        self.flowering = self.species.flowering
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
        
        move_cost = self.current_stats.get('moveCost', 0.1) * math.sqrt(dx**2 + dy**2 + dz**2)
        if self.energy < move_cost:
            return (self.x, self.y, self.z)
        self.x += dx
        self.y += dy
        if dz != self.z:
            if self.current_stats['canFly']:
                self.z += dz
            elif self.current_stats['climbing']:
                #I need to add a check here for something to climb on. currently it will just climb air.
                self.z += dz
        self.energy -= move_cost
        if dx != 0 or dy != 0:
            self.facing = math.atan2(dy, dx)
        
        return (self.x, self.y, self.z)
    
    def tryMove(self, dx: float, dy: float, dz: float) -> bool:
        if not self.species.canMove:
            return False
            
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        cost = distance * self.species.moveCost
        
        # Flying is more expensive but allows vertical movement
        if dz != 0 and not self.can_fly:
            return False
            
        if self.energy < cost:
            return False
            
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
class SimulationSettings:
    grid_size: int = 50
    axial_tilt: float = 23.44  # in degrees
    latitude: float = 45.0     # in degrees
    initial_organisms: list[Organism] = field(default_factory=list)
    initial_seeds: list[seed] = field(default_factory=list)

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
        self.organisms: dict[tuple[int, int, int], Organism] = {}
        self.organism_ids: dict[int, tuple[int, int, int]] = {}

        # seeds are stored in lists per tile
        self.seeds: dict[tuple[int, int, int], list[seed]] = {}
        self.seed_ids: dict[int, tuple[int, int, int]] = {}
        self.organism_id_gen = IDGenerator()
        self.seed_id_gen = IDGenerator()
        
        
        # Pre-compute neighbor offsets as tensor
        neighbor_list = [(dx, dy, dz) 
                        for dx in range(-1, 2)
                        for dy in range(-1, 2)
                        for dz in range(-1, 2)
                        if not (dx == 0 and dy == 0 and dz == 0)]
        self.neighbor_offsets = torch.tensor(neighbor_list, dtype=torch.int32, device=DEVICE)
        self.cell_occupants: dict[tuple[int, int, int], list[int]] = {}
        
    def can_occupy_position(self, organism: Organism, position: tuple[int, int, int]) -> bool:
        """Check if this organism can occupy the given position based on its movement type"""
        x, y, z = position
        if not self.is_valid_position(position):
            return False
            
        # Terrestrial organisms must stay at z=0
        if not organism.can_fly:
            return z == 0
            
        return True
    
    def add_organism(self, organism: Organism):
        self.organisms[organism] = organism
        
        # Mark all occupied cells
        for cell in organism.occupied_cells:
            if cell not in self.cell_occupants:
                self.cell_occupants[cell] = []
            self.cell_occupants[cell].append(organism.id)

    def add_seed(self, seed: seed):
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

    def get_organisms_at_position(self, position: tuple[int, int, int]) -> list[Organism]:
        """Get all organisms at a position (could be multiple for small organisms)"""
        if position not in self.cell_occupants:
            return []
        return [self.organisms[oid] for oid in self.cell_occupants[position]]
    
    def is_position_available(self, position: tuple[int, int, int], ignore_organism: Optional[int] = None) -> bool:
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
        
    def move_organism(self, organism_id: int, new_position: tuple[int, int, int]) -> bool:
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
        
    def _calculate_new_cells(self, organism: Organism, new_position: tuple[int, int, int]) -> list[tuple[int, int, int]]:
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
    
    def get_organism(self, position: tuple[int, int, int]) -> Optional[Organism]:
        return self.organisms.get(position)
        
    def is_valid_position(self, position: tuple[int, int, int]) -> bool:
        x, y, z = position
        return (0 <= x < self.size and 
                0 <= y < self.size and 
                0 <= z < self.size)
    
    def get_empty_spots_in_radius(self, position: tuple[int, int, int], occupancy_tensor: torch.Tensor, radius: int = 1) -> list[tuple[int, int, int]]:
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

    def get_food_in_radius(self, position: tuple[int, int, int], occupancy_tensor: torch.Tensor, radius: int) -> list[tuple[tuple[int, int, int], Organism]]:
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
        self.max_z_level = self.grid_size  # Maximum z-level to display
        
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
    
    def get_species_color(self, organism: Organism) -> tuple[int, int, int]:
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
                
                if species.type == 0:  # Plant -> seeds
                    if (random.random() < species.seed_production_chance and
                        self.temperature > species.dormancy_threshold and
                        self.is_day and self.light_level > 0.5):
                        
                        empty_spots = self.grid.get_empty_spots_in_radius(pos, occupancy_tensor, radius=species.seed_spread_radius)
                        if empty_spots:
                            num_seeds = random.randint(1, 3)  # Reduced from 2-5 to 1-3
                            for new_pos in random.sample(empty_spots, min(len(empty_spots), num_seeds)):
                                new_species = species.mutate() if random.random() < 0.01 else species
                                seeds_to_add.append(seed(species=new_species, position=new_pos))
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
        created_species: list[Species] = []
        created_species.append(BUSH)
        created_species.append(TREE)
        
        # Generate random species
        for _ in range(num_species - 1):
            new_species = Species.generate_random_species(existing_species=created_species, organism_type="plant" if random.random() < 0.7 else "animal")
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