from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from enum import Enum
import json
import random
import string
from typing import Any, Optional
import dearpygui.dearpygui as dpg
import numpy as np
import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ALPHABET = string.ascii_letters
RANDOMCHAR = string.printable

class Units(Enum):
    INCH = 0
    FOOT = 1
    YARD = 2
    CENTIMETER = 3
    METER = 4

DISPLAYUNIT = Units.INCH

@dataclass
class part(ABC):
    name: str = ""
    vital: bool = False
    energyProducer: bool = False
    separable: bool = False
    _statCache: dict[str, float|bool] = field(default_factory=dict)
    _injuries: dict[str, float] = field(default_factory=dict)
    _size: float = field(default=1.0, metadata={"sigma": 0.1})  # convert to a different unit for display
    sizeUnit: Units = Units.INCH # this is what the creator of the part used when making it
    expandable: bool = False # expandable means that size above is ignored. this is for plant limbs mostly where the limbs never stop growing
    multipart: int = 1 # if this is over 1, then its a grouping of parts. if expandable then it can apply to this or size.
    # multipart is for leaves, fingers, etc. 
    _health: float = field(default=10.0, metadata={"sigma": 2.0})  # base health of the part
    _regrowth_rate: float = field(default=0.0, metadata={"sigma": 0.05})  # regrowth rate
    _energyCost: float = field(default=0.1, metadata={"sigma": 0.02})  # energy cost
    minTemp: float = field(default=-50.0, metadata={"sigma": 5.0, "neg": True})  
    maxTemp: float = field(default=150.0, metadata={"sigma": 5.0, "neg": True})
    _mutationRate: float = field(default=0.0001, metadata={"sigma": 0.00005})
    _subparts: list['part'] = field(default_factory=list)

    def __post_init__(self):
        for fieldname, field_info in self.__dataclass_fields__.items():
            if fieldname.startswith('_') and hasattr(self, fieldname):
                sigma = field_info.metadata.get("sigma", 0.0)
                if sigma > 0 and not isinstance(getattr(self, fieldname), (bool, str)):
                    current_val = getattr(self, fieldname)
                    new_val = random.gauss(current_val, sigma)
                    if isinstance(current_val, float) and field_info.metadata.get("neg", False) is not True:
                        new_val = max(0, new_val)
                    setattr(self, fieldname, new_val)
        
        if len(self._subparts) == 0:
            self.setup_default_subparts()
        self.calculateStats(True)
   
    @property
    def size(self) -> float:
        if DISPLAYUNIT == self.sizeUnit:
            return self._size
        
        if self.sizeUnit == Units.INCH:
            size_in_meters = self._size * 0.0254
        elif self.sizeUnit == Units.FOOT:
            size_in_meters = self._size * 0.3048
        elif self.sizeUnit == Units.YARD:
            size_in_meters = self._size * 0.9144
        elif self.sizeUnit == Units.CENTIMETER:
            size_in_meters = self._size * 0.01
        else: 
            size_in_meters = self._size
        
        if DISPLAYUNIT == Units.INCH:
            return size_in_meters / 0.0254
        elif DISPLAYUNIT == Units.FOOT:
            return size_in_meters / 0.3048
        elif DISPLAYUNIT == Units.YARD:
            return size_in_meters / 0.9144
        elif DISPLAYUNIT == Units.CENTIMETER:
            return size_in_meters / 0.01
        else:
            return size_in_meters
    
    @size.setter
    def size(self, value: float):
        self._size = max(0, value)
        self.sizeUnit = DISPLAYUNIT
        self._statCache = self.calculateStats(True)
        
    @property
    def health(self) -> float:
        return self._health
    
    @health.setter
    def health(self, value: float):
        self._health = max(0, value)
        self._statCache = self.calculateStats(True)
        
    @property
    def regrowth_rate(self) -> float:
        return self._regrowth_rate
    
    @regrowth_rate.setter
    def regrowth_rate(self, value: float):
        self._regrowth_rate = max(0, value)
        self._statCache = self.calculateStats(True)
        
    @property
    def energyCost(self) -> float:
        return self._energyCost
    
    @energyCost.setter
    def energyCost(self, value: float):
        self._energyCost = max(0, value)
        self._statCache = self.calculateStats(True)
    
    @property
    def subparts(self) -> list['part']:
        return self._subparts
    
    @subparts.setter
    def subparts(self, value: list['part']):
        self._subparts = value
        self._statCache = self.calculateStats(True)
        
    @property
    def statCache(self) -> dict[str, float|bool]:
        self._statCache = self.calculateStats(False)
        return self._statCache
    
    @statCache.setter
    def statCache(self, value: dict[str, float|bool]):
        self._statCache = value
        
    @property
    def mutationRate(self) -> float:
        return self._mutationRate
    
    @mutationRate.setter
    def mutationRate(self, value: float):
        self._mutationRate = max(0, value)
        
    @property
    def injuries(self) -> dict[str, float]:
        return self._injuries
    
    @injuries.setter
    def injuries(self, value: dict[str, float]):
        self._injuries = value
        self._statCache = self.calculateStats(True)

    @abstractmethod
    def setup_default_subparts(self):
        pass

    def mutate(self):
        for attrname in self.__dataclass_fields__:
            if attrname in ['name', '_subparts', '_statCache', '_mutationRate', '_vital', '_energyProducer', '_injuries']:
                continue

            if random.random() < self.mutationRate:
                current_val = getattr(self, attrname)

                if isinstance(current_val, bool):
                    if random.random() < self.mutationRate:
                        setattr(self, attrname, not current_val)
                elif isinstance(current_val, (int, float)):
                    new_val = current_val * random.uniform(1-self.mutationRate, 1+self.mutationRate)
                    setattr(self, attrname, new_val)
                elif isinstance(current_val, Enum):
                    enum_class = type(current_val)
                    choices = [e for e in enum_class]
                    if choices:
                        setattr(self, attrname, random.choice(choices))
        
        for subpart in self.subparts:
            if random.random() < self.mutationRate and random.random() < self.mutationRate:
                self.subparts.remove(subpart)
            subpart.mutate()
        if random.random() < self.mutationRate and random.random() < self.mutationRate:
            self.subparts.append(self.randomize())
        self.statCache = self.calculateStats(True)
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'part':
        part_type = random.choice([
            skin, limb, sensory, internal, appendage,
            root, stem, leaf,  reproductive,
            egg_sac
        ])
        return part_type.randomize(name, organism_type)
    
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
        if self._statCache is not None and not recalc:
            return self._statCache
            
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
            healthy_wings, total_wings = self.count_healthy_subparts(limb)
            if total_wings > 1:
                wing_ratio = healthy_wings / total_wings
                if wing_ratio < 0.5:
                    stats['can_fly'] = 0.0
                else:
                    stats['flight_efficiency'] = wing_ratio * self.movement_speed
                    
        elif isinstance(self, sensory) and self.type == sensory.SensoryType.EYE:
            healthy_eyes, total_eyes = self.count_healthy_subparts(sensory)
            if total_eyes > 0:
                eye_ratio = healthy_eyes / total_eyes
                stats['vision_quality'] = eye_ratio * self.precision
                if eye_ratio < 0.3:
                    stats['blindness'] = 1.0 - eye_ratio
                    
        elif isinstance(self, internal) and self.type == internal.InternalType.HEART:
            healthy_hearts, total_hearts = self.count_healthy_subparts(internal)
            if total_hearts > 0:
                heart_ratio = healthy_hearts / total_hearts
                stats['circulation'] *= heart_ratio
                if heart_ratio < 0.5:
                    stats['circulatory_shock'] = 1.0 - heart_ratio

        self.statCache = stats
        return stats
    
    @classmethod
    def validate_config(cls, config: dict) -> bool:
        """Validate a configuration dictionary for this part"""
        required_fields = {
            'name': str,
            '_vital': bool,
            '_energyProducer': bool,
            '_size': float,
            '_health': float,
            '_regrowth_rate': float,
            '_energyCost': float,
            '_minTemp': float,
            '_maxTemp': float,
            '_mutationRate': float
        }
        
        for fieldname, field_type in required_fields.items():
            if fieldname not in config:
                raise ValueError(f"Missing required field: {fieldname}")
            if not isinstance(config[fieldname], field_type):
                raise TypeError(f"Field {fieldname} must be of type {field_type}")
        
        if '_subparts' in config:
            if not isinstance(config['_subparts'], list):
                raise TypeError("_subparts must be a list")
            for subpart in config['_subparts']:
                cls.validate_config(subpart)
        
        return True

    @classmethod
    def from_json(cls, json_str: str) -> 'part':
        """Create a part from JSON string"""
        config = json.loads(json_str)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: dict) -> 'part':
        """Create a part from configuration dictionary"""
        cls.validate_config(config)
        
        # Create subparts first
        subparts = []
        if '_subparts' in config:
            for subpart_config in config['_subparts']:
                # Try to determine the appropriate class
                part_class = cls._determine_part_class(subpart_config)
                subparts.append(part_class.from_config(subpart_config))
        
        # Create the part instance
        new_part = cls(
            name=config['name'],
            vital=config['_vital'],
            energyProducer=config['_energyProducer'],
            _size=config['_size'],
            _health=config['_health'],
            _regrowth_rate=config['_regrowth_rate'],
            _energyCost=config['_energyCost'],
            minTemp=config['_minTemp'],
            maxTemp=config['_maxTemp'],
            _mutationRate=config['_mutationRate'],
            _subparts=subparts
        )
        
        return new_part

    @classmethod
    def _determine_part_class(cls, config: dict) -> 'part':
        """Determine the appropriate part class from config"""
        # This should be implemented based on your actual part hierarchy
        # For now, we'll return the current class as default
        return cls()

    def to_dict(self) -> dict:
        """Convert the part to a dictionary"""
        data = {
            'name': self.name,
            'vital': self.vital,
            'energyProducer': self.energyProducer,
            '_size': self.size,
            '_health': self.health,
            '_regrowth_rate': self.regrowth_rate,
            '_energyCost': self.energyCost,
            'minTemp': self.minTemp,
            'maxTemp': self.maxTemp,
            '_mutationRate': self.mutationRate,
            '_subparts': [subpart.to_dict() for subpart in self.subparts]
        }
        return data

    def to_json(self) -> str:
        """Convert the part to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def create_dpg_editor(self, parent: str) -> str:
        """Create a Dear PyGui editor for this part"""
        if parent is None:
            parent = dpg.add_window(label=f"Part Editor: {self.name}", width=600, height=800)
        
        with dpg.group(parent=parent):
            # Basic properties
            dpg.add_text(f"Editing: {self.name}")
            dpg.add_input_text(label="Name", default_value=self.name, callback=lambda s, d: setattr(self, 'name', d))
            dpg.add_checkbox(label="Vital", default_value=self.vital, callback=lambda s, d: setattr(self, 'vital', d))
            dpg.add_checkbox(label="Energy Producer", default_value=self.energyProducer, 
                            callback=lambda s, d: setattr(self, 'energyProducer', d))
            
            # Numeric properties with sliders
            self._add_dpg_slider("Size", "size", 0.0, 5.0)
            self._add_dpg_slider("Health", "health", 0.0, 100.0)
            self._add_dpg_slider("Regrowth Rate", "regrowth_rate", 0.0, 1.0)
            self._add_dpg_slider("Energy Cost", "energyCost", 0.0, 5.0)
            self._add_dpg_slider("Min Temp", "minTemp", -100.0, 0.0)
            self._add_dpg_slider("Max Temp", "maxTemp", 0.0, 300.0)
            self._add_dpg_slider("Mutation Rate", "mutationRate", 0.0, 0.01)
            
            # Subparts management
            with dpg.collapsing_header(label="Subparts"):
                for i, subpart in enumerate(self.subparts):
                    with dpg.tree_node(label=f"Subpart {i}: {subpart.name}"):
                        subpart.create_dpg_editor(parent)
                
                def add_new_subpart():
                    new_part = self.randomize("New Subpart")
                    self.subparts.append(new_part)
                    dpg.delete_item(parent, children_only=True)
                    self.create_dpg_editor(parent)
                
                dpg.add_button(label="Add Subpart", callback=add_new_subpart)
            
            # Stats display
            with dpg.collapsing_header(label="Stats"):
                stats = self.calculateStats(True)
                for statname, stat_value in stats.items():
                    dpg.add_text(f"{statname}: {stat_value:.2f}")
            
            # Save/Load buttons
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save to JSON", callback=lambda: self._save_to_json())
                dpg.add_button(label="Load from JSON", callback=lambda: self._load_from_json(parent))
        
        return parent

    def _add_dpg_slider(self, label: str, attribute: str, min_val: float, max_val: float):
        """Helper to add a slider for a numeric attribute"""
        current_val = getattr(self, attribute)
        dpg.add_slider_float(
            label=label,
            default_value=current_val,
            min_value=min_val,
            max_value=max_val,
            callback=lambda s, d: self._on_dpg_slider_change(attribute, d)
        )

    def _on_dpg_slider_change(self, attribute: str, value: float):
        """Handle slider changes"""
        setattr(self, attribute, value)
        self._statCache = self.calculateStats(True)

    def _save_to_json(self):
        """Save the part to a JSON file"""
        json_str = self.to_json()
        with open("part_config.json", "w") as f:
            f.write(json_str)
        print("Saved part configuration to part_config.json")

    def _load_from_json(self, parent: str):
        """Load a part from JSON file"""
        try:
            with open("part_config.json", "r") as f:
                json_str = f.read()
            new_part = self.from_json(json_str)
            
            # Update current part with loaded values
            self.__dict__.update(new_part.__dict__)
            
            # Refresh UI
            dpg.delete_item(parent, children_only=True)
            self.create_dpg_editor(parent)
            
            print("Loaded part configuration from part_config.json")
        except Exception as e:
            print(f"Error loading part: {str(e)}")

    def detach(self) -> Optional['part']:
        """Detach this part if it's separable, returning the detached part"""
        if not self.separable:
            return None
        
        # Create a copy of this part to return
        detached_part = copy.deepcopy(self)
        
        # If this part is regrowable, reset its health to begin regrowth
        if self.regrowth_rate > 0.0:
            self.health = 0.1  # Start with minimal health for regrowth
            self._injuries = {'Detached': 1.0}
            
        return detached_part

    def can_detach(self) -> bool:
        """Check if this part can be safely detached"""
        if not self.separable:
            return False
            
        # Check if any vital subparts would be lost
        if any(subpart.vital for subpart in self.subparts):
            return False
            
        return True

@dataclass
class neural(part):
    class Neural(Enum):
        NERVOUSSYSTEM = 0
        #nervous system is generic decision tree.
        # if see food, eat food.
        BRAIN = 1
        # brain is a neural net instead. minimum size of say 1k for insects, max of 5b. 
        # After 5b, will probably be getting too big to reasonably run for the sim
        MULTIBRAIN = 2
        # a "moe" (parallel layer) type model will be used. this is for stuff like starfish. 
        # it will probably be limited to max of 5b total, so not any smarter than brain.
        HIVEBRAIN = 3
        # when a creature has hivebrain, it will also have 1 of the other 3.
        # hivebrain will move the decision tree to another organism or be the host of the brain for others
        # the host requires a neural net, the clients usually have nerous system.
    """trying to decide how to actually handle the networks:
    if I were to have each species contain a base model, then have each organism use a lora that is a merge of the loras of their parents, and every sleep cycle (or year, or something) I tune the lora of the creature based on recent actions and reapply it, then would it work?
    what if I had creatures start with a tiny base model that expands from birth to maturity, each year a new layer is added, and only that new layer is tuned? but when the layer is added, it pulls from the species defaults for the layer. after maturity is reached, then it tunes a lora at a significantly lower learning rate or just pulls from memory somehow instead of training further."""
    type: Neural = Neural.NERVOUSSYSTEM 
    vital = True
    _complexity: int = field(default=1000, metadata={"sigma": 500})  # number of parameters
    _learning_rate: float = field(default=0.01, metadata={"sigma": 0.005})  # learning rate
    _energyCost: float = field(default=5.0, metadata={"sigma": 1.0})
    #memory = embedding
    memoryMax: int = field(default=1000, metadata={"sigma": 100}) # max memories to store.
    memorySpeed: float = field(default=5.0, metadata={"sigma": 1.0}) # frequency of events being turned into memories
    waitTime: float = 5.0 # wait 5 seconds at max for a decision from the neural net before falling back to nervous system. this is per brain, not total.
    hormoneReg: dict = field(default_factory=dict) # I need to define hormones somehow first. this might just be ignored entirely instead. 
    # note: not just male/female, this is intended for general hormones such as growth hormones, adrenaline, melatonin, dopamine


    
    def setup_default_subparts(self):
        # if I wanted to get into it in more detail, I could specify stuff like hippocampus, and that would specify how frequently memories are formed, size of memories, etc.
        # I could also specify how long the program is willing to wait on the ai before falling back to nervous system by way of parietal lobe (or whatever, I am not a brain surgeon)
        # and the max number of memories with the amygdala, and hormones with pituitary gland, etc.
        # easier option is to just include those in the brain directly.
        pass
    
    @property
    def complexity(self) -> int:
        return self._complexity
    
    @complexity.setter
    def complexity(self, value: int):
        self._complexity = max(1000, min(5000000000, value))
        self._statCache = self.calculateStats(True)
        
    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = max(0.001, min(1.0, value))
        self._statCache = self.calculateStats(True)
    
    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        
        # Base neural stats
        stats['neural_complexity'] = self.complexity
        stats['learning_rate'] = self.learning_rate
        
        # Efficiency modifiers for other systems
        if self.type == self.Neural.NERVOUSSYSTEM:
            stats['system_efficiency'] = 1.0  # Baseline
        elif self.type == self.Neural.BRAIN:
            stats['system_efficiency'] = 1.2  # more efficient due to a centralized nervous system
        elif self.type == self.Neural.MULTIBRAIN:
            stats['system_efficiency'] = 1.3  # 30% even more so due to multiple centralized nervous systems
        else:  # HIVEBRAIN
            stats['system_efficiency'] = 0.9  # 10% less efficient because the nervous system is outside the body
            
        # Energy costs scale with complexity
        stats['neural_energy_cost'] = (
            self.energyCost * 
            (self.complexity / 1000) ** 0.7
        )
        
        return stats
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'neural':
        neural_type = random.choice(list(cls.Neural))
        
        # Set reasonable complexity defaults based on type
        if neural_type == cls.Neural.NERVOUSSYSTEM:
            complexity = random.randint(1000, 10000)
        elif neural_type == cls.Neural.BRAIN:
            complexity = random.randint(1000000, 10000000)
        elif neural_type == cls.Neural.MULTIBRAIN:
            complexity = random.randint(500000, 5000000)
        else:  # HIVEBRAIN
            complexity = random.randint(100000, 1000000)
            
        return cls(
            name=name or "Nervous System",
            type=neural_type,
            _complexity=complexity,
            _learning_rate=random.gauss(0.01, 0.005),
            _energyCost=random.gauss(5.0, 1.0),
            _size=random.gauss(3.0, 0.5) if organism_type == "animal" else random.gauss(0.5, 0.1)
        )
    
@dataclass
class torso(part):
    name: str = ""
    vital: bool = True
    class TorsoType(Enum):
        STANDARD = 0        # Typical animal torso
        SEGMENTED = 1       # Like insects or worms
        RADIAL = 2         # Starfish-like symmetry
        HYDROSTATIC = 3    # Fluid-filled like worms
        EXOSKELETAL = 4    # Hard outer shell
        TRUNK = 5          # Plant trunk/stem
        BULB = 6           # Bulbous plant base
        TUBER = 7          # Potato-like storage organ
    
    type: TorsoType = TorsoType.STANDARD
    flexibility: float = 0.5  # 0=rigid, 1=very flexible
    segments: int = 1         # Number of body segments
    _storage_capacity: float = 0.0  # For storing nutrients/water
    _structural_integrity: float = 1.0  # Resistance to damage
    
    def setup_default_subparts(self):
        has_neural = any(isinstance(p, neural) for p in self.subparts)
        if not has_neural:
            self.subparts.append(neural.randomize("Nervous System"))
            
        # Add basic internal organs
        self.subparts.extend([
            internal.randomize("Heart", "animal"),
            internal.randomize("Lungs", "animal"),
            internal.randomize("Stomach", "animal")
        ])

    @property
    def storage_capacity(self) -> float:
        return self._storage_capacity
    
    @storage_capacity.setter
    def storage_capacity(self, value: float):
        self._storage_capacity = max(0, value)
        self._statCache = self.calculateStats(True)
        
    @property
    def structural_integrity(self) -> float:
        return self._structural_integrity
    
    @structural_integrity.setter
    def structural_integrity(self, value: float):
        self._structural_integrity = max(0.1, min(1.0, value))
        self._statCache = self.calculateStats(True)
    
    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        
        # Torso-specific stats
        stats['storage'] = self.storage_capacity * self.size
        stats['structural_integrity'] = self.structural_integrity
        
        # Modify movement stats based on torso type
        if self.type == self.TorsoType.SEGMENTED:
            stats['flexibility'] = self.flexibility * 1.5
            stats['burrow_speed'] = 0.5 * self.segments
        elif self.type == self.TorsoType.RADIAL:
            stats['balance'] = 1.5
            stats['flexibility'] = self.flexibility * 0.8
        elif self.type == self.TorsoType.HYDROSTATIC:
            stats['flexibility'] = self.flexibility * 2.0
            stats['burrow_speed'] = 1.0 * self.segments
            stats['impact_resistance'] = 0.7
        elif self.type == self.TorsoType.EXOSKELETAL:
            stats['armor'] = 2.0 * self.structural_integrity
            stats['flexibility'] = self.flexibility * 0.5
        elif self.type in [self.TorsoType.TRUNK, self.TorsoType.BULB, self.TorsoType.TUBER]:
            stats['photosynthesis_area'] = 0.5 * self.size
            stats['water_storage'] = 2.0 * self.storage_capacity * self.size
            
        return stats

    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'torso':
        if organism_type == "plant":
            torso_type = random.choice([
                cls.TorsoType.TRUNK,
                cls.TorsoType.BULB,
                cls.TorsoType.TUBER
            ])
            storage_capacity = random.uniform(1.0, 5.0)
        else:
            torso_type = random.choice([
                cls.TorsoType.STANDARD,
                cls.TorsoType.SEGMENTED,
                cls.TorsoType.RADIAL,
                cls.TorsoType.HYDROSTATIC,
                cls.TorsoType.EXOSKELETAL
            ])
            storage_capacity = random.uniform(0.1, 2.0)
            
        return cls(
            name=name or "Torso",
            type=torso_type,
            flexibility=random.uniform(0.2, 0.8),
            segments=random.randint(1, 20) if torso_type in [cls.TorsoType.SEGMENTED, cls.TorsoType.HYDROSTATIC] else 1,
            _storage_capacity=storage_capacity,
            _structural_integrity=random.uniform(0.5, 1.0),
            _size=random.uniform(6.0, 24.0) if organism_type == "animal" else random.uniform(12.0, 96.0),
            _health=random.uniform(20.0, 50.0),
            _energyCost=random.uniform(1.0, 3.0),
            _mutationRate=random.uniform(0.00005, 0.0002)
        )
    
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
            _energyCost=random.uniform(0.05, 0.2),
            _mutationRate=random.uniform(0.0001, 0.001)
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
            _size=0.8,
            thickness=0.3,
            _energyCost=0.05
        ))
        
        self.subparts.append(internal(
            name="muscles",
            type=internal.InternalType.OTHER,
            efficiency=self.strength,
            capacity=self.dexterity,
            _size=0.7,
            _energyCost=0.1
        ))
        if self.type in [self.LimbType.ARM, self.LimbType.LEG, self.LimbType.TAIL]:
            self.subparts.append(appendage(
                name="claws",
                type=appendage.AppendageType.CLAW,
                _size=0.05,
                damage=0.5,
                _energyCost=0.01,
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
            _size=random.uniform(0.2, 1.5),
            strength=random.uniform(0.5, 3.0),
            dexterity=random.uniform(0.1, 2.0),
            reach=random.uniform(0.3, 3.0),
            prehensile=random.random() < 0.3,
            movement_speed=random.uniform(0.5, 3.0),
            movement_cost=random.uniform(0.05, 0.3),
            can_climb=random.random() < 0.4,
            can_swim=random.random() < 0.4,
            can_dig=random.random() < 0.3,
            _energyCost=random.uniform(0.1, 0.3),
            _mutationRate=random.uniform(0.0001, 0.001)
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
            _energyCost=random.uniform(0.01, 0.05),
            _mutationRate=random.uniform(0.0001, 0.001)
        )

@dataclass
class internal(part):
    class InternalType(Enum):
        HEART = 0
        LUNG = 1
        STOMACH = 2
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
            _energyCost=random.uniform(0.05, 0.2),
            _mutationRate=random.uniform(0.0001, 0.001)
        )

@dataclass
class appendage(part):
    class AppendageType(Enum):
        CLAW = 0
        FANG = 1
        HORN = 2
        TUSK = 3
        STINGER = 4
        SPINE = 5
        BEAK = 6
        TAIL_SPIKE = 7
        SHELL = 8
        THORN = 9
        PLATE = 10
        FROND = 11
        TUBE = 12
        POD = 13
        SAC = 14
        TENDRIL = 15
        
    
    type: AppendageType = AppendageType.CLAW
    damage: float = 1.0  # base damage
    attack_speed: float = 1.0  # how fast it can attack
    reach: float = 0.5  # attack range
    slash_damage: float = 0.0
    pierce_damage: float = 0.0
    blunt_damage: float = 0.0
    venomous: bool = False
    retractable: bool = False
    attack_cost: float = 0.1  # energy per attack
    coverage: float = 0.5  # how much of the body it covers (0-1)
    hardness: float = 1.0  # resistance to damage
    poison_production: bool = False
    venom_production: bool = False # typically with an 8.
    storesE: bool = False
    stores_water: bool = False
    stores_air: bool = False

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
        if self.type == self.AppendageType.CLAW:
            substats['slash_damage'] += 0.5 * self.damage
        elif self.type == self.AppendageType.FANG:
            substats['pierce_damage'] += 0.7 * self.damage
        elif self.type == self.AppendageType.HORN:
            substats['pierce_damage'] += 0.5 * self.damage
            substats['blunt_damage'] += 0.3 * self.damage
        elif self.type == self.AppendageType.STINGER:
            substats['pierce_damage'] += 0.3 * self.damage
            substats['venom_potency'] = substats.get('venom_potency', 0) + 0.5
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
    
    def setup_default_subparts(self):
        if self.venomous:
            self.subparts.append(internal(
                name="venom_gland",
                type=internal.InternalType.GLAND,
                efficiency=1.0,
                _size=0.3,
                _energyCost=0.05
            ))
            
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'appendage':
        weapon_type = random.choice(list(appendage.AppendageType))
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
            _energyCost=random.uniform(0.05, 0.2),
            _mutationRate=random.uniform(0.0001, 0.001),
            coverage=random.uniform(0.1, 0.8),
            hardness=random.uniform(0.5, 3.0),
            poison_production=random.random() < 0.2,
            venom_production=random.random() < 0.05,
            storesE=random.random() < 0.3,
            stores_water=random.random() < 0.4,
            stores_air=random.random() < 0.1,
        )
        new_weapon.setup_default_subparts()
        return new_weapon

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
            _energyCost=random.uniform(0.1, 0.5),
            _mutationRate=random.uniform(0.0001, 0.001)
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
                _size=0.9,
                thickness=0.5,
                _energyCost=0.02
            ))
        else:
            self.subparts.append(skin(
                name="epidermis",
                type=skin.Skin.CUTICLE,
                _size=0.9,
                thickness=0.1,
                _energyCost=0.01
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
            _energyCost=random.uniform(0.1, 0.5),
            _mutationRate=random.uniform(0.0001, 0.001)
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
            _energyCost=random.uniform(0.05, 0.2),
            _mutationRate=random.uniform(0.0001, 0.001)
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
        POLLEN_SAC = 7    # Pollen production
        
    type: ReproductiveType = ReproductiveType.GONAD
    fertility: float = 1.0         # Reproductive success rate
    gestation_period: float = 0.0  # Time for development (if applicable)
    offspring_count: int = 1       # Typical number of offspring
    mating_frequency: float = 1.0  # How often reproduction can occur
    resource_cost: float = 0.5     # Energy/nutrient cost per reproduction
    seasonal: bool = False         # Only functions in certain seasons
    produces_seeds: bool = False   # For plants
    produces_eggs: bool = False    # For animals
    produces_pollen: bool = False  # For plants

    def __post_init__(self):
        super().__post_init__()
        # Set separable based on type
        self.separable = self.type in [
            self.ReproductiveType.SPORE_SAC,
            self.ReproductiveType.FLOWER_BUD,
            self.ReproductiveType.CONE,
            self.ReproductiveType.POLLEN_SAC
        ]
        self._regrowable = self.separable  # These parts can regrow if separable

    def setup_default_subparts(self):
        if self.produces_eggs:
            self.subparts.append(egg_sac(
                name=f"{self.name}_eggs",
                egg_count=self.offspring_count,
                _size=0.5,
                _energyCost=0.1
            ))
        elif self.produces_seeds:
            self.subparts.append(seed(
                name=f"{self.name}_seeds",
                seed_count=self.offspring_count,
                _size=0.3,
                _energyCost=0.05
            ))
        elif self.produces_pollen:
            self.subparts.append(pollen(
                name=f"{self.name}_pollen",
                quantity=self.offspring_count * 10,
                _size=0.1,
                _energyCost=0.02
            ))

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
        elif self.type == self.ReproductiveType.POLLEN_SAC:
            substats['pollen_production'] = 5.0 * self.fertility
            
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
                cls.ReproductiveType.SPORE_SAC,
                cls.ReproductiveType.CONE,
                cls.ReproductiveType.FLOWER_BUD,
                cls.ReproductiveType.POLLEN_SAC
            ])
            offspring_count = random.randint(10, 1000)
            produces_seeds = random.random() < 0.7
            produces_pollen = not produces_seeds and random.random() < 0.5
        else:
            repro_type = random.choice([
                cls.ReproductiveType.GONAD,
                cls.ReproductiveType.OVARY,
                cls.ReproductiveType.TESTIS,
                cls.ReproductiveType.UTERUS
            ])
            offspring_count = random.randint(1, 20)
            produces_eggs = random.random() < 0.5
            
        return cls(
            name=name or "reproductive",
            type=repro_type,
            fertility=random.uniform(0.5, 2.0),
            gestation_period=random.uniform(0.0, 30.0),
            offspring_count=offspring_count,
            mating_frequency=random.uniform(0.5, 3.0),
            resource_cost=random.uniform(0.1, 1.0),
            seasonal=random.random() < 0.4,
            produces_seeds=produces_seeds,
            produces_eggs=produces_eggs,
            produces_pollen=produces_pollen,
            _energyCost=random.uniform(0.1, 0.5),
            _mutationRate=random.uniform(0.0001, 0.001)
        )

@dataclass
class seed(part):
    class SeedType(Enum):
        NAKED = 0       # Gymnosperm
        ENCLOSED = 1    # Angiosperm
        SPORE = 2       # Fungal/fern
        TUBER = 3       # Underground storage
        BULBIL = 4      # Aerial propagation
        
    type: SeedType = SeedType.ENCLOSED
    seed_count: int = 1
    viability: float = 1.0          # Chance to germinate
    dormancy: float = 0.0           # Can remain dormant
    dispersal: float = 1.0          # Spread effectiveness
    nutrient_store: float = 1.0     # Endosperm/resources
    defense: float = 0.0            # Anti-predation
    requires_pollination: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.separable = True  # Seeds are always separable
        self._regrowable = False  # Seeds don't regrow on the parent

    def setup_default_subparts(self):
        # Seeds can have small nutrient stores or protective coatings
        if random.random() < 0.5:
            self.subparts.append(internal(
                name="nutrient_store",
                type=internal.InternalType.OTHER,
                efficiency=self.nutrient_store,
                capacity=self.nutrient_store * 2,
                _size=0.5,
                _energyCost=0.01
            ))
            
        if self.defense > 0:
            self.subparts.append(skin(
                name="seed_coat",
                type=skin.Skin.BARK if random.random() < 0.5 else skin.Skin.CHITIN,
                _size=0.9,
                thickness=self.defense * 0.2,
                _energyCost=0.01
            ))

    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        substats = {}
        # Seed properties
        substats['seed_count'] = self.seed_count
        substats['seed_viability'] = self.viability
        substats['dispersal_efficiency'] = self.dispersal
        substats['seed_nutrients'] = self.nutrient_store
        
        # Special properties
        if self.dormancy > 0:
            substats['dormancy_period'] = self.dormancy
        if self.defense > 0:
            substats['seed_defense'] = self.defense
        if self.requires_pollination:
            substats['pollination_required'] = 1.0
            
        # Type-specific bonuses
        if self.type == self.SeedType.NAKED:
            substats['germination_speed'] = 1.2
        elif self.type == self.SeedType.SPORE:
            substats['quantity'] = 10.0  # Spores are numerous
        elif self.type == self.SeedType.TUBER:
            substats['vegetative_growth'] = 1.5
        elif self.type == self.SeedType.BULBIL:
            substats['aerial_propagation'] = 1.0
            
        for stat, value in substats.items():
            if stat in stats:
                stats[stat] += (value * self.size)
            else:
                stats[stat] = (value * self.size)
        return stats
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "plant") -> 'seed':
        """Create randomized seed."""
        seed_type = random.choice(list(cls.SeedType))
        
        return cls(
            name=name or "seed",
            type=seed_type,
            seed_count=random.randint(1, 1000) if seed_type == cls.SeedType.SPORE else random.randint(1, 20),
            viability=random.uniform(0.5, 1.0),
            dormancy=random.uniform(0.0, 2.0),
            dispersal=random.uniform(0.5, 3.0),
            nutrient_store=random.uniform(0.5, 3.0),
            defense=random.uniform(0.0, 1.0),
            requires_pollination=random.random() < 0.7,
            _energyCost=random.uniform(0.05, 0.2),
            _mutationRate=random.uniform(0.0001, 0.001)
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
    parental_care_required: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.separable = True  # Egg sacs are separable
        self._regrowable = False  # They don't regrow on the parent

    def setup_default_subparts(self):
        # Egg sacs can have protective coatings
        if random.random() < 0.5:
            self.subparts.append(skin(
                name="sac_covering",
                type=skin.Skin.MEMBRANE if self.type == self.EggType.SOFT else skin.Skin.CHITIN,
                _size=0.9,
                thickness=self.protection * 0.3,
                _energyCost=0.02
            ))
            
        # Nutrient stores for the eggs
        self.subparts.append(internal(
            name="nutrient_store",
            type=internal.InternalType.OTHER,
            efficiency=self.nutrient_store,
            capacity=self.nutrient_store * 2,
            _size=0.7,
            _energyCost=0.05
        ))

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
        if self.parental_care_required:
            substats['parental_care'] = 1.0
            
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
        egg_type = random.choice(list(cls.EggType))
        
        return cls(
            name=name or "egg_sac",
            type=egg_type,
            egg_count=random.randint(1, 100),
            protection=random.uniform(0.5, 2.0),
            nutrient_store=random.uniform(0.5, 3.0),
            incubation_time=random.uniform(1.0, 30.0),
            desiccation_resistance=random.uniform(0.0, 1.0),
            camouflage=random.uniform(0.0, 0.8),
            parental_care_required=random.random() < 0.3,
            _energyCost=random.uniform(0.1, 0.5),
            _mutationRate=random.uniform(0.0001, 0.001)
        )



def main():
    # Initialize Dear PyGui
    dpg.create_context()
    dpg.create_viewport(title='Organism Part Editor', width=1200, height=800)
    
    # Define the directory for JSON files
    import os
    JSON_DIR = "part_configs"
    os.makedirs(JSON_DIR, exist_ok=True)  # Create directory if it doesn't exist
    
    def load_part_from_file(filename: str):
        """Load a part from a JSON file in the config directory"""
        try:
            with open(os.path.join(JSON_DIR, filename), "r") as f:
                json_str = f.read()
            return part.from_json(json_str)
        except Exception as e:
            print(f"Error loading part: {str(e)}")
            return None
    
    def save_part_to_file(part_obj: part, filename: str):
        """Save a part to a JSON file in the config directory"""
        try:
            with open(os.path.join(JSON_DIR, filename), "w") as f:
                f.write(part_obj.to_json())
            print(f"Saved part configuration to {filename}")
        except Exception as e:
            print(f"Error saving part: {str(e)}")
    
    def refresh_file_list():
        """Refresh the list of available JSON files"""
        try:
            files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
            dpg.configure_item("file_list", items=files)
        except Exception as e:
            print(f"Error refreshing file list: {str(e)}")
    
    # Create a window for the file browser
    with dpg.window(label="File Browser", width=300, height=400, pos=(0, 0)):
        dpg.add_text("Available Configurations:")
        dpg.add_listbox(tag="file_list", width=280)
        dpg.add_input_text(tag="filename_input", hint="filename.json", width=200)
        dpg.add_button(label="Refresh", callback=refresh_file_list)
        dpg.add_button(label="Load", callback=lambda: load_button_callback())
        dpg.add_button(label="Save", callback=lambda: save_button_callback())
    
    # Create a window for the part editor (will be populated when a part is loaded)
    editor_window = dpg.add_window(label="Part Editor", width=900, height=800, pos=(300, 0))
    
    # Store the current part
    current_part = None
    
    def load_button_callback():
        nonlocal current_part
        selected_file = dpg.get_value("file_list")
        if selected_file:
            loaded_part = load_part_from_file(selected_file)
            if loaded_part:
                current_part = loaded_part
                # Clear and rebuild the editor
                dpg.delete_item(editor_window, children_only=True)
                current_part.create_dpg_editor(editor_window)
    
    def save_button_callback():
        if current_part:
            filename = dpg.get_value("filename_input")
            if not filename.endswith('.json'):
                filename += '.json'
            save_part_to_file(current_part, filename)
            refresh_file_list()
    
    # Create a default part if none exists
    if current_part is None:
        current_part = part.randomize("root_organism")
        current_part.create_dpg_editor(editor_window)
    
    # Initial file list refresh
    refresh_file_list()
    
    # Setup and start Dear PyGui
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()