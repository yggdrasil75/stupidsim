import json
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Optional, Tuple, Dict, List, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

class EnergySource(Enum):
    """Enumeration of possible energy sources for microorganisms."""
    LIGHT = "light"
    ORGANIC = "organic"
    INORGANIC = "inorganic"
    METHANE = "methane"
    SULFATE = "sulfate"
    OXYGEN = "oxygen"
    OTHER_ELECTRON_ACCEPTORS = "other_electron_acceptors"

@dataclass
class EnergyMode:
    """
    Represents an energy acquisition mode for microorganisms.
    
    Attributes:
        name: Descriptive name of the mode
        type: General category (e.g., "phototrophic", "chemotrophic")
        energy_source: Primary energy source (from EnergySource enum)
        efficiency: Conversion efficiency (0.0-1.0)
        oxygen_requirement: Relationship with oxygen ('produces', 'requires', 'none', 'inhibited_by')
        organisms: List of organism types that use this mode
        color: Visualization color for this mode
        rules: Dictionary of specific behavioral rules
        maintenance_cost: Base energy consumption per cycle
    """
    name: str
    type: str
    energy_source: EnergySource
    efficiency: float
    oxygen_requirement: str
    organisms: List[str]
    color: str
    rules: Dict
    maintenance_cost: int
    
    @property
    def is_phototrophic(self) -> bool:
        return self.energy_source == EnergySource.LIGHT
    
    @property
    def is_chemotrophic(self) -> bool:
        return self.energy_source in [EnergySource.ORGANIC, EnergySource.INORGANIC]
    
    @property
    def requires_oxygen(self) -> bool:
        return self.oxygen_requirement == 'requires'
    
    @property
    def produces_oxygen(self) -> bool:
        return self.oxygen_requirement == 'produces'
    
    @property
    def inhibited_by_oxygen(self) -> bool:
        return self.oxygen_requirement == 'inhibited_by'

@dataclass
class ReproductionMode:
    """
    Represents a reproduction mode for microorganisms.
    
    Attributes:
        name: Descriptive name of the mode
        type: General category (e.g., "asexual", "sexual")
        organisms: List of organism types that use this mode
        initial_rate: Initial probability of this mode being assigned
        color: Visualization color for this mode
        rules: Dictionary of specific behavioral rules
        energy_cost: Energy required to reproduce
        child_energy: Range of energy given to offspring (min, max)
        parent_cost: Range of energy consumed by parent during reproduction (min, max)
    """
    name: str
    type: str
    organisms: List[str]
    initial_rate: float
    color: str
    rules: Dict
    energy_cost: int
    child_energy: Tuple[int, int]
    parent_cost: Tuple[int, int]
    
    @property
    def is_asexual(self) -> bool:
        return "asexual" in self.type.lower()
    
    @property
    def is_sexual(self) -> bool:
        return "sexual" in self.type.lower()
    
    @property
    def is_genetic_mix(self) -> bool:
        return "genetic mix" in self.type.lower()
    
    @property
    def is_gene_exchange(self) -> bool:
        return "gene exchange" in self.type.lower() or "gene uptake" in self.type.lower()

@dataclass
class Organism:
    """
    Represents an individual organism in the simulation.
    
    Attributes:
        id: Unique identifier
        position: (x, y, z) coordinates in the grid
        energy_mode: Index of the energy acquisition mode
        reproduction_mode: Index of the reproduction mode
        energy: Current energy level
        age: Number of time steps the organism has survived
        parent_ids: IDs of parent organisms (empty for initial organisms)
        children_ids: IDs of child organisms
        alive: Whether the organism is currently alive
    """
    id: str
    position: Tuple[int, int, int]
    energy_mode: int
    reproduction_mode: int
    energy: float
    age: int = 0
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    alive: bool = True
    
    def update_energy(self, delta: float) -> None:
        """Update the organism's energy level."""
        self.energy += delta
        if self.energy <= 0:
            self.alive = False
    
    def increment_age(self) -> None:
        """Increment the organism's age."""
        self.age += 1
    
    def reproduce(self, child_id: str, child_position: Tuple[int, int, int], 
                 child_energy: float, parent_energy_cost: float, game: 'GameOfLife3D') -> 'Organism':
        """
        Create a new child organism and update parent's energy.
        
        Args:
            child_id: ID for the new organism
            child_position: Position for the new organism
            child_energy: Initial energy for the new organism
            parent_energy_cost: Energy to deduct from parent
            
        Returns:
            New Organism instance
        """
        self.energy -= parent_energy_cost
        self.children_ids.append(child_id)
        
        # Create new organism with possible mutations
        new_energy_mode = self._mutate_energy_mode(game)
        new_repro_mode = self._mutate_reproduction_mode(game)
        
        return Organism(
            id=child_id,
            position=child_position,
            energy_mode=new_energy_mode,
            reproduction_mode=new_repro_mode,
            energy=child_energy,
            parent_ids=[self.id]
        )
    
    def _mutate_energy_mode(self, game: 'GameOfLife3D') -> int:
        """Potentially mutate the energy mode (5% chance)."""
        if np.random.random() < 0.95:
            return self.energy_mode
        else:
            return np.random.randint(0, len(game.energy_modes))
    
    def _mutate_reproduction_mode(self, game: 'GameOfLife3D') -> int:
        """Potentially mutate the reproduction mode (5% chance)."""
        if np.random.random() < 0.95:
            return self.reproduction_mode
        else:
            current_mode = game.repro_modes[self.reproduction_mode]
            compatible_modes = [i for i, m in enumerate(game.repro_modes) 
                              if m.type == current_mode.type]
            return np.random.choice(compatible_modes)

class GameOfLife3D:
    """
    3D simulation of microbial life with multiple energy acquisition and reproduction modes.
    
    Features:
    - 3D grid environment with customizable size
    - Multiple energy acquisition strategies
    - Different reproduction modes with genetic inheritance
    - Environmental factors (light, oxygen)
    - GPU acceleration support
    
    Key Components:
    1. Organism tracking with individual properties
    2. Grid management (3D tensor operations)
    3. Energy system (acquisition and consumption)
    4. Reproduction system (mode-specific rules)
    5. Environmental simulation (light, oxygen)
    6. Visualization tools
    """
    
    def __init__(self, 
                grid_size: Tuple[int, int, int] = (50, 50, 50), 
                initial_prob: float = 0.1,
                energy_config: str = 'energy_modes.json',
                repro_config: str = 'reproduction_modes.json',
                device: Optional[str] = None,
                light_intensity: float = 100.0,
                oxygen_level: float = 50.0,
                gui: bool = True):
        """
        Initialize the 3D simulation environment.
        
        Args:
            grid_size: Dimensions of the 3D grid (rows, cols, depth)
            initial_prob: Initial probability of cell existence
            energy_config: Path to JSON file with energy mode configurations
            repro_config: Path to JSON file with reproduction mode configurations
            device: Hardware device ('cuda' or 'cpu')
            light_intensity: Initial light level at grid top
            oxygen_level: Initial oxygen concentration
            gui: Whether to enable GUI visualization
        """
        # Grid dimensions and initialization parameters
        self.ROWS, self.COLS, self.DEPTH = grid_size
        self.INITIAL_PROB = initial_prob
        self.DEVICE = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.LIGHT_INTENSITY = light_intensity
        self.OXYGEN_LEVEL = oxygen_level
        self.GUI = gui
        
        # Load configuration files
        self._load_configurations(energy_config, repro_config)
        
        # Initialize simulation components
        self._initialize_kernel()
        self._initialize_organisms()
        self._initialize_environment()
        
        # Visualization setup
        self.fig = None
        self.ax = None
        self.ani = None
    
    def _load_configurations(self, energy_config: str, repro_config: str) -> None:
        """Load and validate energy and reproduction mode configurations."""
        with open(energy_config, 'r') as f:
            energy_config = json.load(f)
        with open(repro_config, 'r') as f:
            repro_config = json.load(f)
        
        self.energy_modes = [EnergyMode(**mode) for mode in energy_config['modes']]
        self.repro_modes = [ReproductionMode(**mode) for mode in repro_config['modes']]
        
        # Create mode mappings
        self.energy_mode_indices = {i: mode for i, mode in enumerate(self.energy_modes)}
        self.repro_mode_indices = {i: mode for i, mode in enumerate(self.repro_modes)}
        
        # Validate initial reproduction rates
        total_rate = sum(mode.initial_rate for mode in self.repro_modes)
        if total_rate > 1:
            raise ValueError(f"Total initial reproduction rate {total_rate} exceeds 1.0")
    
    def _initialize_kernel(self) -> None:
        """Initialize the 3D convolution kernel for neighbor counting."""
        self.kernel = torch.ones((3, 3, 3), dtype=torch.float32, device=self.DEVICE)
        self.kernel[1, 1, 1] = 0  # Center cell doesn't count
        self.kernel = self.kernel.view(1, 1, 3, 3, 3)  # Shape for conv3d
    
    def _initialize_organisms(self) -> None:
        """Initialize the organism population."""
        self.organisms: Dict[str, Organism] = {}
        self.position_map: Dict[Tuple[int, int, int], str] = {}  # Maps positions to organism IDs
        
        # Create initial organisms
        for x in range(self.ROWS):
            for y in range(self.COLS):
                for z in range(self.DEPTH):
                    if np.random.random() < self.INITIAL_PROB:
                        self._create_organism((x, y, z))
    
    def _create_organism(self, position: Tuple[int, int, int]) -> None:
        """Create a new organism at the specified position."""
        # Randomly assign energy mode
        energy_mode_idx = np.random.randint(0, len(self.energy_modes))
        
        # Assign reproduction mode based on probabilities
        rand_val = np.random.random()
        cum_prob = 0.0
        repro_mode_idx = 0
        
        for i, mode in enumerate(self.repro_modes):
            if rand_val < cum_prob + mode.initial_rate:
                repro_mode_idx = i
                break
            cum_prob += mode.initial_rate
        
        # Create organism
        organism_id = str(uuid.uuid4())
        new_organism = Organism(
            id=organism_id,
            position=position,
            energy_mode=energy_mode_idx,
            reproduction_mode=repro_mode_idx,
            energy=np.random.uniform(70, 150)
        )
        
        self.organisms[organism_id] = new_organism
        self.position_map[position] = organism_id
    
    def _initialize_environment(self) -> None:
        """Initialize environmental factors."""
        self.oxygen_grid = torch.full((self.ROWS, self.COLS, self.DEPTH), 
                                    self.OXYGEN_LEVEL, 
                                    device=self.DEVICE)
    
    def count_neighbors(self) -> torch.Tensor:
        """Count live neighbors for all cells using 3D convolution."""
        # Create a grid of alive cells
        alive_grid = torch.zeros((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        positions = torch.tensor([org.position for org in self.organisms.values() if org.alive], device=self.DEVICE)
        alive_grid[positions[:,0], positions[:,1], positions[:,2]] = 1
        
        alive_grid = alive_grid.unsqueeze(0).unsqueeze(0)
        neighbors = torch.nn.functional.conv3d(alive_grid, self.kernel, padding=1)
        return neighbors.squeeze()
    
    def count_specific_energy_neighbors(self, mode_index: int) -> torch.Tensor:
        """Count neighbors with a specific energy mode."""
        mode_grid = torch.zeros((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        
        for org_id, org in self.organisms.items():
            if org.alive and org.energy_mode == mode_index:
                x, y, z = org.position
                mode_grid[x, y, z] = 1
        
        mode_grid = mode_grid.unsqueeze(0).unsqueeze(0)
        return torch.nn.functional.conv3d(mode_grid, self.kernel, padding=1).squeeze()
    
    def count_specific_repro_neighbors(self, mode_index: int) -> torch.Tensor:
        """Count neighbors with a specific reproduction mode."""
        mode_grid = torch.zeros((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        
        for org_id, org in self.organisms.items():
            if org.alive and org.reproduction_mode == mode_index:
                x, y, z = org.position
                mode_grid[x, y, z] = 1
        
        mode_grid = mode_grid.unsqueeze(0).unsqueeze(0)
        return torch.nn.functional.conv3d(mode_grid, self.kernel, padding=1).squeeze()
    
    def propagate_light(self) -> torch.Tensor:
        """
        Simulate light attenuation through the 3D grid.
        
        Returns:
            Tensor with light intensity at each cell position
        """
        light = torch.full((self.ROWS, self.COLS), self.LIGHT_INTENSITY, device=self.DEVICE)
        light_grid = torch.zeros((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        
        for z in range(self.DEPTH):
            light_grid[:, :, z] = light
            
            # Calculate absorption by phototrophic cells in this layer
            phototrophic_mask = torch.zeros((self.ROWS, self.COLS), 
                                         dtype=torch.bool, 
                                         device=self.DEVICE)
            
            for mode_idx, mode in self.energy_mode_indices.items():
                if mode.is_phototrophic:
                    # Check if any organism at this (x,y,z) has this energy mode
                    for x in range(self.ROWS):
                        for y in range(self.COLS):
                            if (x, y, z) in self.position_map:
                                org_id = self.position_map[(x, y, z)]
                                if self.organisms[org_id].energy_mode == mode_idx:
                                    phototrophic_mask[x, y] = True
            
            # Simple absorption model
            absorption = phototrophic_mask.float() * 0.1  
            light = light * (1 - absorption)
            light = torch.clamp(light, 0.0, self.LIGHT_INTENSITY)
        
        return light_grid
    
    def update_oxygen(self) -> None:
        """Update oxygen levels based on microorganism activity and diffusion."""
        # Track oxygen production and consumption
        oxygen_producers = torch.zeros((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        oxygen_consumers = torch.zeros((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        
        for org_id, org in self.organisms.items():
            if not org.alive:
                continue
                
            x, y, z = org.position
            mode = self.energy_modes[org.energy_mode]
            
            if mode.produces_oxygen:
                oxygen_producers[x, y, z] += 0.1
            elif mode.requires_oxygen:
                oxygen_consumers[x, y, z] += 0.2
        
        # Apply production and consumption
        self.oxygen_grid += oxygen_producers - oxygen_consumers
        
        # Apply diffusion
        self._apply_oxygen_diffusion()
        
        # Maintain reasonable bounds
        self.oxygen_grid = torch.clamp(self.oxygen_grid, 0.0, 100.0)
    
    def _apply_oxygen_diffusion(self) -> None:
        """Simulate oxygen diffusion using 3D convolution."""
        # Diffusion kernel (center is 0.6, faces 0.1)
        diffusion_kernel = torch.zeros((3, 3, 3), device=self.DEVICE)
        diffusion_kernel[1, 1, 1] = 0.6  # center
        diffusion_kernel[1, 1, 0] = diffusion_kernel[1, 1, 2] = 0.1  # front/back
        diffusion_kernel[1, 0, 1] = diffusion_kernel[1, 2, 1] = 0.1  # left/right
        diffusion_kernel[0, 1, 1] = diffusion_kernel[2, 1, 1] = 0.1  # top/bottom
        diffusion_kernel = diffusion_kernel.unsqueeze(0).unsqueeze(0)
        
        # Apply diffusion with boundary replication
        padded_oxygen = torch.nn.functional.pad(
            self.oxygen_grid.unsqueeze(0).unsqueeze(0), 
            (1,1,1,1,1,1), 
            mode='replicate'
        )
        self.oxygen_grid = torch.nn.functional.conv3d(
            padded_oxygen, 
            diffusion_kernel, 
            padding=0
        ).squeeze()
    
    def calculate_energy_gain(self) -> Dict[str, float]:
        """Calculate energy gain for all organisms based on their energy mode."""
        energy_gains = {}
        light_grid = self.propagate_light()
        
        for org_id, org in self.organisms.items():
            if not org.alive:
                continue
                
            x, y, z = org.position
            mode = self.energy_modes[org.energy_mode]
            gain = 0.0
            
            if mode.is_phototrophic:
                # Phototrophs gain energy from light
                gain = light_grid[x, y, z] * mode.efficiency
                
            elif mode.energy_source == EnergySource.ORGANIC:
                # Heterotrophs gain from organic matter
                gain = torch.rand(1).item() * 5 * mode.efficiency
                
            elif mode.energy_source == EnergySource.INORGANIC:
                # Chemolithotrophs gain from inorganic compounds
                gain = torch.rand(1).item() * 3 * mode.efficiency
                
            elif mode.energy_source == EnergySource.OXYGEN and mode.requires_oxygen:
                # Aerobic respirers - oxygen-dependent gain
                gain = self.oxygen_grid[x, y, z] * 0.5 * mode.efficiency
            
            # Apply oxygen inhibition if needed
            if mode.inhibited_by_oxygen:
                inhibition = 1.0 - (self.oxygen_grid[x, y, z] / 100.0)
                gain *= inhibition
            
            energy_gains[org_id] = gain
        
        return energy_gains
    
    def update(self, frameNum: Optional[int] = None, visualize: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Advance the simulation by one time step.
        
        Args:
            frameNum: Current frame number (for visualization)
            visualize: Whether to return data for visualization
            
        Returns:
            If visualize=True, returns tuple of numpy arrays (grid, energy_assignments, repro_assignments)
        """
        # Count neighbors and update environment
        neighbors = self.count_neighbors()
        self.update_oxygen()
        
        # Update energy levels
        energy_gains = self.calculate_energy_gain()
        for org_id, gain in energy_gains.items():
            if org_id in self.organisms and self.organisms[org_id].alive:
                mode = self.energy_modes[self.organisms[org_id].energy_mode]
                net_gain = gain - mode.maintenance_cost
                self.organisms[org_id].update_energy(net_gain)
                self.organisms[org_id].increment_age()
        
        # Process reproduction for each reproduction mode
        for repro_idx, repro_mode in self.repro_mode_indices.items():
            self._process_reproduction_mode(repro_idx, repro_mode, neighbors)
        
        # Remove dead organisms
        self._cleanup_dead_organisms()
        
        if visualize:
            return self._prepare_visualization_data()
        return None
    
    def _process_reproduction_mode(self, repro_idx: int, repro_mode: ReproductionMode,
                                 neighbors: torch.Tensor) -> None:
        """Apply rules for a specific reproduction mode."""
        # Get all organisms with this reproduction mode
        mode_organisms = [org for org in self.organisms.values() 
                         if org.alive and org.reproduction_mode == repro_idx]
        
        if not mode_organisms:
            return
        
        rules = repro_mode.rules
        
        # Determine potential birth positions
        birth_positions = self._find_birth_positions(repro_idx, rules, neighbors)
        
        # For each potential birth position, find nearby parents
        for birth_pos in birth_positions:
            parent_organisms = self._find_nearby_parents(birth_pos, repro_idx, repro_mode.energy_cost)
            
            if parent_organisms:
                # Select a random parent (could implement more sophisticated selection)
                parent = np.random.choice(parent_organisms)
                
                # Calculate child energy and parent cost
                child_energy = (np.random.uniform(*repro_mode.child_energy))
                parent_cost = (np.random.uniform(*repro_mode.parent_cost))
                
                # Create new organism
                self._create_child_organism(parent, birth_pos, child_energy, parent_cost)
    
    def _find_birth_positions(self, repro_idx: int, rules: Dict,
                             neighbors: torch.Tensor) -> List[Tuple[int, int, int]]:
        birth_positions = []
        living_positions = set(org.position for org in self.organisms.values() if org.alive)
        
        # Only check positions adjacent to living organisms
        empty_neighbors = set()
        for x, y, z in living_positions:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if (0 <= nx < self.ROWS and 0 <= ny < self.COLS and 0 <= nz < self.DEPTH):
                            if (nx, ny, nz) not in self.position_map:
                                empty_neighbors.add((nx, ny, nz))
        
        for pos in empty_neighbors:
            x, y, z = pos
            neighbor_count = neighbors[x, y, z].item()
            
            # Check birth conditions
            valid = True
            
            if "min_neighbors" in rules["birth_conditions"]:
                valid &= (neighbor_count >= rules["birth_conditions"]["min_neighbors"])
            if "max_neighbors" in rules["birth_conditions"]:
                valid &= (neighbor_count <= rules["birth_conditions"]["max_neighbors"])
            if "min_same_type" in rules["birth_conditions"]:
                same_type_neighbors = self.count_specific_repro_neighbors(repro_idx)[x, y, z].item()
                valid &= (same_type_neighbors >= rules["birth_conditions"]["min_same_type"])
            if "max_same_type" in rules["birth_conditions"]:
                same_type_neighbors = self.count_specific_repro_neighbors(repro_idx)[x, y, z].item()
                valid &= (same_type_neighbors <= rules["birth_conditions"]["max_same_type"])
            
            if valid:
                birth_positions.append(pos)
        return birth_positions
    
    def _find_nearby_parents(self, birth_pos: Tuple[int, int, int], 
                            repro_idx: int, 
                            required_energy: int) -> List[Organism]:
        """Find nearby parent organisms that can reproduce."""
        x, y, z = birth_pos
        parents = []
        
        # Check all adjacent positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # Skip the birth position itself
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < self.ROWS and 0 <= ny < self.COLS and 0 <= nz < self.DEPTH):
                        if (nx, ny, nz) in self.position_map:
                            org_id = self.position_map[(nx, ny, nz)]
                            org = self.organisms[org_id]
                            
                            if (org.alive and 
                                org.reproduction_mode == repro_idx and 
                                org.energy >= required_energy):
                                parents.append(org)
        
        return parents
    
    def _create_child_organism(self, parent: Organism, 
                              birth_pos: Tuple[int, int, int],
                              child_energy: float,
                              parent_cost: float) -> None:
        """Create a new child organism through reproduction."""
        child_id = str(uuid.uuid4())
        child = parent.reproduce(
            child_id=child_id,
            child_position=birth_pos,
            child_energy=child_energy,
            parent_energy_cost=parent_cost,
            game=self
        )
        
        self.organisms[child_id] = child
        self.position_map[birth_pos] = child_id
    
    def _cleanup_dead_organisms(self) -> None:
        """Remove dead organisms from the simulation."""
        dead_ids = [org_id for org_id, org in self.organisms.items() if not org.alive]
        
        for org_id in dead_ids:
            # Remove from position map
            org = self.organisms[org_id]
            if org.position in self.position_map and self.position_map[org.position] == org_id:
                del self.position_map[org.position]
            
            # Remove from organisms dict
            del self.organisms[org_id]
    
    def _prepare_visualization_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for visualization as numpy arrays."""
        grid = np.zeros((self.ROWS, self.COLS, self.DEPTH))
        energy_assignments = np.zeros((self.ROWS, self.COLS, self.DEPTH))
        repro_assignments = np.zeros((self.ROWS, self.COLS, self.DEPTH))
        
        for org in self.organisms.values():
            if org.alive:
                x, y, z = org.position
                grid[x, y, z] = 1
                energy_assignments[x, y, z] = org.energy_mode
                repro_assignments[x, y, z] = org.reproduction_mode
        
        return grid, energy_assignments, repro_assignments
    
    def visualize_frame(self, grid_data: np.ndarray, 
                       energy_data: np.ndarray, 
                       repro_data: np.ndarray, 
                       frameNum: int = 0) -> None:
        """Render a single frame of the simulation."""
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        
        # Plot each energy mode with its color
        for mode_idx, mode in self.energy_mode_indices.items():
            x, y, z = np.where((grid_data == 1) & (energy_data == mode_idx))
            if len(x) > 0:
                self.ax.scatter(x, y, z, c=mode.color, marker='o', s=1, label=mode.name)
        
        # Configure plot appearance
        self.ax.set_xlim(0, self.ROWS)
        self.ax.set_ylim(0, self.COLS)
        self.ax.set_zlim(0, self.DEPTH)
        self.ax.set_title(f"3D Microbial Life Simulation - Frame {frameNum}")
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def animate(self, frames: int = 50, interval: int = 200) -> animation.FuncAnimation:
        """Create and return an animation of the simulation."""
        if not self.GUI:
            raise RuntimeError("GUI is disabled. Set gui=True when initializing the simulation.")
            
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        def update_wrapper(frameNum, *args):
            grid_data, energy_data, repro_data = self.update(frameNum=frameNum, visualize=True)
            self.visualize_frame(grid_data, energy_data, repro_data, frameNum)
            return self.ax,
        
        # Initial frame
        initial_data = self._prepare_visualization_data()
        self.visualize_frame(*initial_data, 0)
        
        # Create and return animation
        self.ani = animation.FuncAnimation(
            self.fig, update_wrapper,
            frames=frames,
            interval=interval,
            blit=False
        )
        
        plt.tight_layout()
        return self.ani
    
    def run(self, steps: int = 100) -> None:
        """Run the simulation without visualization."""
        for _ in range(steps):
            self.update()
    
    def get_population_density(self) -> np.ndarray:
        """Get 2D population density projection (sum along depth axis)."""
        grid_data, _, _ = self._prepare_visualization_data()
        return grid_data.sum(axis=2)
    
    def get_organism_count(self) -> int:
        """Get the current number of living organisms."""
        return sum(1 for org in self.organisms.values() if org.alive)

if __name__ == '__main__':
    # Example usage with GUI
    simulation = GameOfLife3D(
        grid_size=(50, 50, 50),
        initial_prob=0.1,
        gui=False  # Set to False for non-GUI execution
    )
    
    if simulation.GUI:
        # Run with animation
        ani = simulation.animate(frames=100, interval=200)
        plt.show()
    else:
        # Run without visualization
        simulation.run(steps=100)
        print(f"Final population: {simulation.get_organism_count()}")