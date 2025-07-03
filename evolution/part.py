from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class LimbType(Enum):
    ARM = auto()
    LEG = auto()
    BRANCH = auto()
    ROOT = auto()
    WING = auto()
    FIN = auto()

@dataclass
class Part:
    name: str
    expandable: float  # How large this part can get (multiplier of base size)
    health: float
    regrowth_rate: float  # 0 disables regrowth
    size: float  # Size in grid units
    multipart: int  # Number of instances (1 for single, 2 for pairs, etc.)
    
    def grow(self, amount: float):
        """Grow the part by the given amount, up to expandable limit"""
        if self.regrowth_rate > 0:
            self.size = min(self.size + amount, self.size * self.expandable)
    
    def damage(self, amount: float):
        """Apply damage to the part"""
        self.health = max(0, self.health - amount)
        # If health is too low, reduce size
        if self.health < 10:
            self.size = max(1, self.size * 0.9)

@dataclass
class Limb(Part):
    limb_type: LimbType
    strength: float  # Affects carrying capacity and power
    speed: float  # Affects movement speed
    dexterity: float  # Affects fine motor control
    prehensile: bool  # Can grasp/manipulate objects
    joints: int  # Number of joints (affects flexibility)
    
    def calculate_movement_contribution(self) -> float:
        """Calculate how much this limb contributes to movement speed"""
        # Roots and branches don't contribute to movement
        if self.limb_type in [LimbType.ROOT, LimbType.BRANCH]:
            return 0
            
        base = self.speed * (self.size / 10)
        joint_bonus = 1 + (self.joints * 0.1)
        return base * joint_bonus * (self.health / 100)