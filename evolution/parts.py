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
        part_type = random.choice([neural, limb
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

    def create_dpg_editor(self, parent: str):
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
                
                # Add part type selection combo
                part_types = [
                    "Random", "Neural", "Skin", "Limb", "Sensory", 
                    "Internal", "Appendage", "Root", "Stem", "Leaf",
                    "Reproductive", "Egg Sac"
                ]
                
                with dpg.group(horizontal=True):
                    part_combo = dpg.add_combo(
                        label="Part Type",
                        items=part_types,
                        default_value="Random",
                        width=150
                    )
                    org_type_combo = dpg.add_combo(
                        label="Organism Type",
                        items=["Animal", "Plant"],
                        default_value="Animal",
                        width=100
                    )
                    dpg.add_button(
                        label="Add Subpart",
                        callback=lambda: self._add_subpart_callback(
                            dpg.get_value(part_combo),
                            dpg.get_value(org_type_combo).lower(),
                            parent
                        ))
            
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

    def _add_subpart_callback(self, part_type: str, organism_type: str, parent: str):
        """Callback for adding a new subpart of specific type"""
        part_map = {
            "neural": neural,
            "limb": limb,
            "sensory": sensory,
            "internal": internal,
            "reproductive": reproductive,
            "skin": skin
        }
        
        if part_type.lower() == "random":
            new_part = part.randomize("New Part", organism_type)
        elif part_type.lower() in part_map:
            new_part = part_map[part_type.lower()].randomize(
                f"New {part_type}", 
                organism_type
            )
        else:
            new_part = part.randomize("New Part", organism_type)
        
        self.subparts.append(new_part)
        # Refresh the UI
        dpg.delete_item(parent, children_only=True)
        self.create_dpg_editor(parent)

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
class limb(part):
    class LimbType(Enum):
        ARM = 0
        LEG = 1
        WING = 2
        TAIL = 3
        TENTACLE = 4
        BRANCH = 5  # For plants
        ROOT = 6    # For plants
        FIN = 7
        ANTENNA = 8
        CLAW = 9
        HOOF = 10
        WEBBED = 11
    
    type: LimbType = LimbType.ARM
    _strength: float = field(default=1.0, metadata={"sigma": 0.2})  # Physical strength
    _flexibility: float = field(default=1.0, metadata={"sigma": 0.2})  # Flexibility
    _movement_speed: float = field(default=1.0, metadata={"sigma": 0.2})  # Movement speed
    _grip_strength: float = field(default=0.5, metadata={"sigma": 0.1})  # Grip strength
    _energyCost: float = field(default=0.5, metadata={"sigma": 0.1})
    _regrowth_rate: float = field(default=0.01, metadata={"sigma": 0.005})
    _mutationRate: float = field(default=0.001, metadata={"sigma": 0.0005})
    joints: int = field(default=1, metadata={"sigma": 0.5})  # Number of joints
    digits: int = field(default=0, metadata={"sigma": 0.5})  # Number of digits/fingers
    prehensile: bool = False  # Can grasp objects
    opposable: bool = False  # Has opposable digits
    _subparts: list['part'] = field(default_factory=list)
    
    def setup_default_subparts(self):
        # Add basic components based on limb type
        if self.type in [self.LimbType.ARM, self.LimbType.LEG, self.LimbType.TAIL, 
                        self.LimbType.TENTACLE, self.LimbType.FIN, self.LimbType.WING]:
            # Animal limbs typically have muscles, bones, and skin
            self._subparts.append(
                internal.randomize("Muscle", "animal")
            )
            self._subparts.append(
                internal.randomize("Bone", "animal")
            )
            self._subparts.append(
                skin.randomize("Skin", "animal")
            )
            
            # Add digits if specified
            if self.digits > 0:
                for i in range(self.digits):
                    digit_type = "Finger" if self.type == self.LimbType.ARM else "Toe"
                    self._subparts.append(
                        limb.randomize(f"{digit_type} {i+1}", "animal")
                    )
        
        elif self.type in [self.LimbType.BRANCH, self.LimbType.ROOT]:
            # Plant limbs have bark and possibly leaves/roots
            self._subparts.append(
                skin.randomize("Bark", "plant")
            )
            if self.type == self.LimbType.BRANCH:
                # Add some leaves
                leaf_count = random.randint(1, 5)
                for i in range(leaf_count):
                    self._subparts.append(
                        leaf.randomize(f"Leaf {i+1}")
                    )
            else:  # ROOT
                # Add root hairs
                hair_count = random.randint(3, 10)
                for i in range(hair_count):
                    self._subparts.append(
                        root.randomize(f"Root Hair {i+1}")
                    )
    
    @property
    def strength(self) -> float:
        return self._strength
    
    @strength.setter
    def strength(self, value: float):
        self._strength = max(0, value)
        self._statCache = self.calculateStats(True)
        
    @property
    def flexibility(self) -> float:
        return self._flexibility
    
    @flexibility.setter
    def flexibility(self, value: float):
        self._flexibility = max(0, min(1, value))
        self._statCache = self.calculateStats(True)
        
    @property
    def movement_speed(self) -> float:
        return self._movement_speed
    
    @movement_speed.setter
    def movement_speed(self, value: float):
        self._movement_speed = max(0, value)
        self._statCache = self.calculateStats(True)
        
    @property
    def grip_strength(self) -> float:
        return self._grip_strength
    
    @grip_strength.setter
    def grip_strength(self, value: float):
        self._grip_strength = max(0, value)
        self._statCache = self.calculateStats(True)
    
    def calculateStats(self, recalc: bool = False) -> dict[str, float]:
        stats = super().calculateStats(recalc)
        
        # Base limb stats
        stats['strength'] = self.strength
        stats['flexibility'] = self.flexibility
        stats['movement_speed'] = self.movement_speed
        stats['grip_strength'] = self.grip_strength
        
        # Type-specific modifiers
        if self.type == self.LimbType.ARM:
            stats['manipulation'] = (self.strength + self.flexibility + self.grip_strength) / 3
            if self.opposable:
                stats['manipulation'] *= 1.5
            if self.prehensile:
                stats['manipulation'] *= 1.2
                
        elif self.type == self.LimbType.LEG:
            stats['locomotion'] = (self.strength + self.movement_speed) / 2
            stats['jump_power'] = self.strength * 0.5
            
        elif self.type == self.LimbType.WING:
            stats['flight_power'] = (self.strength + self.movement_speed) / 2
            stats['flight_efficiency'] = self.flexibility * 0.8
            
        elif self.type == self.LimbType.TAIL:
            stats['balance'] = self.strength * 0.7
            stats['swimming'] = self.flexibility * 0.5
            
        elif self.type == self.LimbType.TENTACLE:
            stats['manipulation'] = (self.flexibility + self.grip_strength) / 2
            stats['reach'] = self.size * 1.2
            
        elif self.type == self.LimbType.BRANCH:
            stats['photosynthesis'] = self.size * 0.2
            stats['structural_support'] = self.strength
            
        elif self.type == self.LimbType.ROOT:
            stats['water_absorption'] = self.size * 0.3
            stats['nutrient_absorption'] = self.size * 0.2
            stats['anchoring'] = self.strength
            
        elif self.type == self.LimbType.FIN:
            stats['swimming'] = (self.movement_speed + self.flexibility) / 2
            
        elif self.type == self.LimbType.ANTENNA:
            stats['sensory_range'] = self.size * 2.0
            
        # Digit-specific stats
        if self.digits > 0:
            stats['digits'] = self.digits
            if self.opposable:
                stats['opposable_digits'] = 1.0
            if self.prehensile:
                stats['prehensile_digits'] = 1.0
                
        # Joint-specific stats
        if self.joints > 1:
            stats['articulation'] = min(1.0, self.joints / 10.0)
            
        return stats
    
    @classmethod
    def randomize(cls, name: str = "", organism_type: str = "animal") -> 'limb':
        if organism_type == "animal":
            limb_types = [
                cls.LimbType.ARM, cls.LimbType.LEG, cls.LimbType.WING,
                cls.LimbType.TAIL, cls.LimbType.TENTACLE, cls.LimbType.FIN,
                cls.LimbType.ANTENNA, cls.LimbType.CLAW, cls.LimbType.HOOF,
                cls.LimbType.WEBBED
            ]
        else:  # plant
            limb_types = [cls.LimbType.BRANCH, cls.LimbType.ROOT]
            
        limb_type = random.choice(limb_types)
        
        # Set reasonable defaults based on type
        if limb_type == cls.LimbType.ARM:
            size = random.gauss(24.0, 3.0)  # ~2 feet for arms
            digits = random.randint(3, 5)
            prehensile = random.random() > 0.3
            opposable = random.random() > 0.7 if prehensile else False
        elif limb_type == cls.LimbType.LEG:
            size = random.gauss(30.0, 4.0)  # ~2.5 feet for legs
            digits = random.randint(1, 5)
            prehensile = random.random() > 0.8
            opposable = False
        elif limb_type == cls.LimbType.WING:
            size = random.gauss(60.0, 10.0)  # ~5 feet wingspan
            digits = random.randint(2, 4)
            prehensile = False
            opposable = False
        elif limb_type == cls.LimbType.TAIL:
            size = random.gauss(36.0, 6.0)  # ~3 feet
            digits = 0
            prehensile = random.random() > 0.5
            opposable = False
        elif limb_type == cls.LimbType.TENTACLE:
            size = random.gauss(48.0, 8.0)  # ~4 feet
            digits = 0
            prehensile = True
            opposable = False
        elif limb_type == cls.LimbType.BRANCH:
            size = random.gauss(72.0, 12.0)  # ~6 feet
            digits = 0
            prehensile = False
            opposable = False
        elif limb_type == cls.LimbType.ROOT:
            size = random.gauss(36.0, 6.0)  # ~3 feet
            digits = 0
            prehensile = False
            opposable = False
        elif limb_type == cls.LimbType.FIN:
            size = random.gauss(18.0, 3.0)  # ~1.5 feet
            digits = 0
            prehensile = False
            opposable = False
        elif limb_type == cls.LimbType.ANTENNA:
            size = random.gauss(6.0, 1.0)  # ~6 inches
            digits = 0
            prehensile = False
            opposable = False
        else:  # CLAW, HOOF, WEBBED
            size = random.gauss(12.0, 2.0)  # ~1 foot
            digits = random.randint(1, 4)
            prehensile = limb_type == cls.LimbType.CLAW
            opposable = False
            
        return cls(
            name=name or f"{limb_type.name.capitalize()}",
            type=limb_type,
            _strength=random.gauss(1.0, 0.2),
            _flexibility=random.gauss(0.5 if limb_type in [cls.LimbType.TENTACLE, cls.LimbType.WING] else 0.3, 0.1),
            _movement_speed=random.gauss(1.0, 0.2),
            _grip_strength=random.gauss(0.5 if prehensile else 0.1, 0.1),
            _size=size,
            _energyCost=random.gauss(0.5, 0.1),
            _regrowth_rate=random.gauss(0.01, 0.005),
            joints=random.randint(1, 3) if limb_type in [cls.LimbType.ARM, cls.LimbType.LEG] else 1,
            digits=digits,
            prehensile=prehensile,
            opposable=opposable,
            separable=limb_type not in [cls.LimbType.ROOT, cls.LimbType.BRANCH]
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