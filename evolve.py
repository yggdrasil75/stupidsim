from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class EnergySource(Enum):
    LIGHT = "light"
    ORGANIC = "organic"
    INORGANIC = "inorganic"
    METHANE = "methane"
    SULFATE = "sulfate"
    OXYGEN = "oxygen"
    OTHER_ELECTRON_ACCEPTORS = "other_electron_acceptors"

class ReproMethod(Enum):
    SEXUAL = 'sexual'
    ASEXUAL = 'asexual'
    MIXED = 'genetic mix'
    EXCHANGED = 'gene exchange'
    UPTAKE = 'gene uptake'

@dataclass
class EnergyMode:
    name: str
    type: str
    energy_source: EnergySource
    efficiency: float
    oxygen_requirement: str
    rules: Dict
    maintenance_cost: int
    _is_phototrophic: torch.Tensor = field(init=False)
    _is_chemotrophic: torch.Tensor = field(init=False)

    def __post_init__(self):
        self._is_phototrophic = torch.tensor(self.energy_source == EnergySource.LIGHT, dtype=torch.bool, device=DEVICE)
        self._is_chemotrophic = torch.tensor(self.energy_source in {EnergySource.ORGANIC, EnergySource.INORGANIC}, dtype=torch.bool, device=DEVICE)

    @property
    def is_phototrophic(self) -> torch.Tensor:
        return self._is_phototrophic
    
    @property
    def is_chemotrophic(self) -> torch.Tensor:
        return self._is_chemotrophic
    
    @property
    def requires_oxygen(self) -> torch.Tensor:
        return torch.tensor(self.oxygen_requirement == 0, dtype=torch.bool)
    
    @property
    def produces_oxygen(self) -> torch.Tensor:
        return torch.tensor(self.oxygen_requirement == 1, dtype=torch.bool)
    
    @property
    def inhibited_by_oxygen(self) -> torch.Tensor:
        return torch.tensor(self.oxygen_requirement == 2, dtype=torch.bool)
    
@dataclass
class ReproductionMode:
    name: str
    type: str
    organisms: List[str]
    initial_rate: float
    color: str
    rules: Dict
    energy_cost: int
    child_energy: Tuple[int, int]
    parent_cost: Tuple[int, int]

    def is_asexual(self) -> torch.Tensor:
        return torch.tensor(self.type.lower() == ReproMethod.ASEXUAL, dtype=torch.bool)
    
    def is_sexual(self) -> torch.Tensor:
        return torch.tensor(self.type.lower() == ReproMethod.SEXUAL, dtype=torch.bool)
    
    def is_genetic_mix(self) -> torch.Tensor:
        return torch.tensor(self.type.lower() == ReproMethod.MIXED, dtype=torch.bool)
    
    def is_gene_exchange(self) -> torch.Tensor:
        return torch.tensor(self.type.lower() in {ReproMethod.EXCHANGED, ReproMethod.UPTAKE}, dtype=torch.bool)