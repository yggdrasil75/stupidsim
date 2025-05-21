import json
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum

class EnergySource(Enum):
    LIGHT = "light"
    ORGANIC = "organic"
    INORGANIC = "inorganic"
    METHANE = "methane"
    SULFATE = "sulfate"
    OXYGEN = "oxygen"
    OTHER_ELECTRON_ACCEPTORS = "other_electron_acceptors"

@dataclass
class EnergyMode:
    name: str
    type: str
    energy_source: EnergySource
    efficiency: float  # How efficiently this mode converts source to energy
    oxygen_requirement: str  # 'produces', 'requires', 'none', 'inhibited_by'
    organisms: List[str]
    color: str
    rules: Dict
    maintenance_cost: int  # Base energy consumption per cycle
    
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
    name: str
    type: str
    organisms: List[str]
    initial_rate: float
    color: str
    rules: Dict
    energy_cost: int  # Energy required to reproduce
    child_energy: Tuple[int, int]  # Range of energy given to child
    parent_cost: Tuple[int, int]  # Range of energy consumed by parent
    
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

class GameOfLife3D:
    def __init__(self, 
                 grid_size: Tuple[int, int, int] = (50, 50, 50), 
                 initial_prob: float = 0.1,
                 energy_config: str = 'energy_modes.json',
                 repro_config: str = 'reproduction_modes.json',
                 device: Optional[str] = None,
                 light_intensity: float = 100.0,
                 oxygen_level: float = 50.0):
        """
        3D Game of Life simulator with multiple energy acquisition modes and reproduction modes.
        
        Args:
            grid_size: Tuple of (rows, cols, depth) for the 3D grid
            initial_prob: Probability of a cell being alive initially
            energy_config: Path to JSON file containing energy modes configuration
            repro_config: Path to JSON file containing reproduction modes configuration
            device: 'cuda' or 'cpu' (None for auto-detection)
            light_intensity: Initial light intensity at the top of the grid
            oxygen_level: Initial oxygen concentration in the environment
        """
        self.ROWS, self.COLS, self.DEPTH = grid_size
        self.INITIAL_PROB = initial_prob
        self.DEVICE = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.LIGHT_INTENSITY = light_intensity
        self.OXYGEN_LEVEL = oxygen_level
        
        # Load energy and reproduction modes from JSON
        with open(energy_config, 'r') as f:
            energy_config = json.load(f)
        with open(repro_config, 'r') as f:
            repro_config = json.load(f)
        
        self.energy_modes = [EnergyMode(**mode) for mode in energy_config['modes']]
        self.repro_modes = [ReproductionMode(**mode) for mode in repro_config['modes']]
        
        # Create mappings for modes
        self.energy_mode_indices = {i: mode for i, mode in enumerate(self.energy_modes)}
        self.repro_mode_indices = {i: mode for i, mode in enumerate(self.repro_modes)}
        
        # Validate initial rates sum to <= 1
        total_rate = sum(mode.initial_rate for mode in self.repro_modes)
        if total_rate > 1:
            raise ValueError(f"Total initial reproduction rate {total_rate} exceeds 1.0")
        
        # Create the 3D kernel for neighbor counting
        self.kernel = torch.ones((3, 3, 3), dtype=torch.float32, device=self.DEVICE)
        self.kernel[1, 1, 1] = 0  # Center cell doesn't count
        self.kernel = self.kernel.view(1, 1, 3, 3, 3)  # Shape for conv3d
        
        # Initialize grids
        self.grid = self.initialize_grid()
        self.energy_assignments = self.initialize_energy_assignments()
        self.repro_assignments = self.initialize_repro_assignments()
        self.energy = self.initialize_energy()
        
        # Environmental factors
        self.oxygen_grid = torch.full((self.ROWS, self.COLS, self.DEPTH), 
                                    self.OXYGEN_LEVEL, 
                                    device=self.DEVICE)
        
        # Visualization attributes
        self.fig = None
        self.ax = None
        self.ani = None
    
    def initialize_grid(self) -> torch.Tensor:
        """Create a random initial 3D grid on the chosen device"""
        return (torch.rand((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE) < self.INITIAL_PROB).float()
    
    def initialize_energy_assignments(self) -> torch.Tensor:
        # Assign modes randomly (for now - could implement weighted distribution)
        # Just assign randomly among available modes
        modes = torch.randint(0, len(self.energy_modes), 
                            (self.ROWS, self.COLS, self.DEPTH), 
                            device=self.DEVICE)
        
        # Only keep modes for alive cells
        modes = modes * self.grid.byte()
        
        return modes
    
    def initialize_repro_assignments(self) -> torch.Tensor:
        """Initialize reproduction modes based on the configuration"""
        rand_vals = torch.rand((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        modes = torch.zeros((self.ROWS, self.COLS, self.DEPTH), dtype=torch.int32, device=self.DEVICE)
        cum_prob = 0.0
        for i, mode in enumerate(self.repro_modes):
            mask = (rand_vals >= cum_prob) & (rand_vals < cum_prob + mode.initial_rate)
            modes[mask] = i
            cum_prob += mode.initial_rate
        modes = modes * self.grid.byte()
        return modes
    
    def initialize_energy(self) -> torch.Tensor:
        """Initialize energy levels for cells (70-150 for initial cells)"""
        return (torch.rand((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE) * 80 + 70)
    
    def count_neighbors(self, grid: torch.Tensor) -> torch.Tensor:
        """Count live neighbors for all cells in 3D using PyTorch convolution"""
        grid = grid.unsqueeze(0).unsqueeze(0)
        neighbors = torch.nn.functional.conv3d(grid, self.kernel, padding=1)
        return neighbors.squeeze()
    
    def count_specific_energy_neighbors(self, mode_index: int) -> torch.Tensor:
        """Count neighbors of a specific energy mode type"""
        mode_grid = (self.energy_assignments == mode_index).float()
        mode_grid = mode_grid.unsqueeze(0).unsqueeze(0)
        neighbors = torch.nn.functional.conv3d(mode_grid, self.kernel, padding=1)
        return neighbors.squeeze()
    
    def count_specific_repro_neighbors(self, mode_index: int) -> torch.Tensor:
        """Count neighbors of a specific reproduction mode type"""
        mode_grid = (self.repro_assignments == mode_index).float()
        mode_grid = mode_grid.unsqueeze(0).unsqueeze(0)
        neighbors = torch.nn.functional.conv3d(mode_grid, self.kernel, padding=1)
        return neighbors.squeeze()
    
    def propagate_light(self) -> torch.Tensor:
        """
        Simulate light propagation from top to bottom of the grid.
        Returns the light available at each cell position.
        """
        light = torch.full((self.ROWS, self.COLS), self.LIGHT_INTENSITY, device=self.DEVICE)
        light_grid = torch.zeros((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        
        for z in range(self.DEPTH):
            light_grid[:, :, z] = light
            
            # Get current layer of phototrophic cells
            phototrophic_mask = torch.zeros((self.ROWS, self.COLS), dtype=torch.bool, device=self.DEVICE)
            for mode_idx, mode in self.energy_mode_indices.items():
                if mode.is_phototrophic:
                    phototrophic_mask |= (self.energy_assignments[:, :, z] == mode_idx) & (self.grid[:, :, z] == 1)
            
            # Calculate absorption by phototrophic cells
            absorption = phototrophic_mask.float() * 0.1  # Simple absorption model
            light = light * (1 - absorption)
            
            # Ensure light doesn't go negative
            light = torch.clamp(light, 0.0, self.LIGHT_INTENSITY)
        
        return light_grid
    
    def update_oxygen(self) -> None:
        """Update oxygen levels based on microorganism activity"""
        # Oxygen production by phototrophs
        oxygen_producers = torch.zeros((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        for mode_idx, mode in self.energy_mode_indices.items():
            if mode.produces_oxygen:
                oxygen_producers += (self.energy_assignments == mode_idx).float() * 0.1
        
        # Oxygen consumption by aerobes
        oxygen_consumers = torch.zeros((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        for mode_idx, mode in self.energy_mode_indices.items():
            if mode.requires_oxygen:
                oxygen_consumers += (self.energy_assignments == mode_idx).float() * 0.2
        
        # Diffusion (simple 3D convolution)
        diffusion_kernel = torch.tensor([[[0, 0.1, 0],
                                        [0.1, 0.6, 0.1],
                                        [0, 0.1, 0]]], device=self.DEVICE).view(1, 1, 3, 3, 3) / 1.1
        
        # Apply changes to oxygen grid
        self.oxygen_grid = self.oxygen_grid + oxygen_producers - oxygen_consumers
        self.oxygen_grid = torch.nn.functional.conv3d(
            self.oxygen_grid.unsqueeze(0).unsqueeze(0),
            diffusion_kernel,
            padding=1
        ).squeeze()
        
        # Ensure oxygen stays within reasonable bounds
        self.oxygen_grid = torch.clamp(self.oxygen_grid, 0.0, 100.0)
    
    def calculate_energy_gain(self) -> torch.Tensor:
        """Calculate energy gain for all cells based on their energy mode"""
        energy_gain = torch.zeros((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE)
        light_grid = self.propagate_light()
        
        for mode_idx, mode in self.energy_mode_indices.items():
            cell_mask = (self.energy_assignments == mode_idx) & (self.grid == 1)
            
            if mode.is_phototrophic:
                # Phototrophs gain energy from light
                gain = light_grid * mode.efficiency
                energy_gain[cell_mask] = gain[cell_mask]
                
            elif mode.energy_source == EnergySource.ORGANIC:
                # Heterotrophs gain energy from organic matter (simplified)
                # In a more complex model, we'd track organic matter in the environment
                gain = torch.rand((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE) * 5 * mode.efficiency
                energy_gain[cell_mask] = gain[cell_mask]
                
            elif mode.energy_source == EnergySource.INORGANIC:
                # Chemolithotrophs gain energy from inorganic compounds
                gain = torch.rand((self.ROWS, self.COLS, self.DEPTH), device=self.DEVICE) * 3 * mode.efficiency
                energy_gain[cell_mask] = gain[cell_mask]
                
            elif mode.energy_source == EnergySource.OXYGEN and mode.requires_oxygen:
                # Aerobic respirers - gain depends on oxygen availability
                gain = self.oxygen_grid * 0.5 * mode.efficiency
                energy_gain[cell_mask] = gain[cell_mask]
                
            # Add other energy source types here...
            
            # Apply oxygen inhibition if needed
            if mode.inhibited_by_oxygen:
                inhibition = 1.0 - (self.oxygen_grid / 100.0)  # More oxygen = more inhibition
                energy_gain[cell_mask] = energy_gain[cell_mask] * inhibition[cell_mask]
        
        return energy_gain
    
    def update(self, frameNum: Optional[int] = None, visualize: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Update the grid according to the reproduction mode rules and energy acquisition.
        
        Args:
            frameNum: Optional frame number (for visualization)
            visualize: Whether to prepare data for visualization
            
        Returns:
            If visualize=True, returns tuple of (grid_state, energy_assignments, repro_assignments) as numpy arrays
        """
        neighbors = self.count_neighbors(self.grid)
        
        # Update environmental factors
        self.update_oxygen()
        
        # Calculate energy gain for all cells
        energy_gain = self.calculate_energy_gain()
        self.energy = torch.where(self.grid == 1, self.energy + energy_gain, self.energy)
        
        # Cells consume energy for maintenance based on their energy mode
        for mode_idx, mode in self.energy_mode_indices.items():
            cell_mask = (self.energy_assignments == mode_idx) & (self.grid == 1)
            self.energy[cell_mask] -= mode.maintenance_cost
        
        # Initialize new grid and assignments
        new_grid = torch.zeros_like(self.grid)
        new_energy_assign = torch.zeros_like(self.energy_assignments)
        new_repro_assign = torch.zeros_like(self.repro_assignments)
        new_energy = torch.zeros_like(self.energy)
        
        # Process each reproduction mode with its specific rules
        for repro_idx, repro_mode in self.repro_mode_indices.items():
            mode_mask = (self.repro_assignments == repro_idx) & (self.grid == 1)
            rules = repro_mode.rules
            
            # Initialize birth and survival conditions
            birth = (self.grid == 0)
            survive = mode_mask.clone()
            
            # Handle birth conditions
            if "min_neighbors" in rules["birth_conditions"]:
                birth &= (neighbors >= rules["birth_conditions"]["min_neighbors"])
            if "max_neighbors" in rules["birth_conditions"]:
                birth &= (neighbors <= rules["birth_conditions"]["max_neighbors"])
            if "min_same_type" in rules["birth_conditions"]:
                same_type_neighbors = self.count_specific_repro_neighbors(repro_idx)
                birth &= (same_type_neighbors >= rules["birth_conditions"]["min_same_type"])
            if "max_same_type" in rules["birth_conditions"]:
                same_type_neighbors = self.count_specific_repro_neighbors(repro_idx)
                birth &= (same_type_neighbors <= rules["birth_conditions"]["max_same_type"])
            
            # Handle survival conditions
            if "min_neighbors" in rules["survival_conditions"]:
                survive &= (neighbors >= rules["survival_conditions"]["min_neighbors"])
            if "max_neighbors" in rules["survival_conditions"]:
                survive &= (neighbors <= rules["survival_conditions"]["max_neighbors"])
            if "min_same_type" in rules["survival_conditions"]:
                same_type_neighbors = self.count_specific_repro_neighbors(repro_idx)
                survive &= (same_type_neighbors >= rules["survival_conditions"]["min_same_type"])
            if "max_same_type" in rules["survival_conditions"]:
                same_type_neighbors = self.count_specific_repro_neighbors(repro_idx)
                survive &= (same_type_neighbors <= rules["survival_conditions"]["max_same_type"])
            
            # Apply the rules - only if parent has enough energy for reproduction
            potential_new_cells = birth | survive
            parent_energy_mask = (self.energy >= repro_mode.energy_cost) & mode_mask
            
            # Only allow reproduction if parent has enough energy
            new_grid[potential_new_cells & parent_energy_mask.any()] = 1
            
            # For surviving cells, transfer their energy (minus maintenance cost)
            new_energy[survive] = self.energy[survive]
            
            # For new cells (birth), assign energy and deduct from parent
            birth_cells = birth & parent_energy_mask.any()
            if birth_cells.any():
                # Assign random energy to new cells within specified range
                child_energy = (torch.rand(birth_cells.sum(), device=self.DEVICE) * 
                               (repro_mode.child_energy[1] - repro_mode.child_energy[0]) + 
                               repro_mode.child_energy[0])
                new_energy[birth_cells] = child_energy
                
                # Deduct parent energy (random within specified range)
                parent_cost = (torch.rand(parent_energy_mask.sum(), device=self.DEVICE) * 
                              (repro_mode.parent_cost[1] - repro_mode.parent_cost[0]) + 
                              repro_mode.parent_cost[0])
                new_energy[parent_energy_mask] = self.energy[parent_energy_mask] - parent_cost
            
            # Assign reproduction modes to new cells
            if repro_mode.is_asexual or repro_mode.is_gene_exchange:
                # Asexual or gene exchange modes tend to keep their type
                new_repro_assign[(birth | survive) & (new_repro_assign == 0)] = repro_idx
            elif repro_mode.is_sexual or repro_mode.is_genetic_mix:
                # Sexual and genetic mix modes can sometimes change type
                if torch.rand(1).item() < 0.9:  # 90% chance to keep same type
                    new_repro_assign[(birth | survive) & (new_repro_assign == 0)] = repro_idx
                else:
                    # 10% chance to switch to a random compatible type
                    compatible_modes = [i for i, m in self.repro_mode_indices.items() 
                                      if m.type == repro_mode.type and i != repro_idx]
                    if compatible_modes:
                        new_mode = np.random.choice(compatible_modes)
                        new_repro_assign[(birth | survive) & (new_repro_assign == 0)] = new_mode
            
            # Assign energy modes to new cells (inherit from parent or mutate)
            parent_energy_modes = self.energy_assignments[parent_energy_mask]
            if parent_energy_modes.numel() > 0:
                # Most cells inherit their parent's energy mode
                if torch.rand(1).item() < 0.95:  # 95% chance to inherit
                    # Need to map parent modes to birth cells - this is simplified
                    # In a more accurate model, we'd track which parent produced which offspring
                    new_energy_assign[(birth | survive) & (new_energy_assign == 0)] = parent_energy_modes[0]
                else:
                    # 5% chance to switch to a random energy mode
                    new_mode = np.random.choice(len(self.energy_modes))
                    new_energy_assign[(birth | survive) & (new_energy_assign == 0)] = new_mode
        
        # Cells die if they have no energy left
        new_grid[new_energy <= 0] = 0
        
        self.grid = new_grid
        self.energy_assignments = new_energy_assign * self.grid.byte()  # Only keep modes for alive cells
        self.repro_assignments = new_repro_assign * self.grid.byte()    # Only keep modes for alive cells
        self.energy = new_energy * self.grid  # Only keep energy for alive cells
        
        if visualize:
            return (self.grid.cpu().numpy(), 
                    self.energy_assignments.cpu().numpy(),
                    self.repro_assignments.cpu().numpy())
        return None
    
    def visualize_frame(self, grid_data: np.ndarray, energy_data: np.ndarray, repro_data: np.ndarray, frameNum: int = 0) -> None:
        """Visualize a single frame of the simulation with colored cells"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        
        # Plot each energy mode with its corresponding color
        for mode_idx, mode in self.energy_mode_indices.items():
            x, y, z = np.where((grid_data == 1) & (energy_data == mode_idx))
            if len(x) > 0:
                self.ax.scatter(x, y, z, c=mode.color, marker='o', s=1, label=mode.name)
        
        self.ax.set_xlim(0, self.ROWS)
        self.ax.set_ylim(0, self.COLS)
        self.ax.set_zlim(0, self.DEPTH)
        self.ax.set_title(f"3D Microbial Life Simulation - Frame {frameNum}")
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def animate(self, frames: int = 50, interval: int = 200) -> None:
        """Create and display an animation of the simulation"""
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        def update_wrapper(frameNum, *args):
            grid_data, energy_data, repro_data = self.update(frameNum=frameNum, visualize=True)
            self.visualize_frame(grid_data, energy_data, repro_data, frameNum)
            return self.ax,
        
        # Initial frame
        initial_grid = self.grid.cpu().numpy()
        initial_energy = self.energy_assignments.cpu().numpy()
        initial_repro = self.repro_assignments.cpu().numpy()
        self.visualize_frame(initial_grid, initial_energy, initial_repro, 0)
        
        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, update_wrapper,
            frames=frames,
            interval=interval,
            blit=False
        )
        
        plt.tight_layout()
        plt.show()
    
    def get_population_density(self) -> np.ndarray:
        """Get the population density as a 2D numpy array (projection on XY plane)"""
        return self.grid.sum(dim=2).cpu().numpy()
    
    def get_3d_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the current 3D grid, mode assignments, and energy state as numpy arrays"""
        return (self.grid.cpu().numpy(), 
                self.energy_assignments.cpu().numpy(),
                self.repro_assignments.cpu().numpy(),
                self.energy.cpu().numpy())
    
    def set_3d_grid(self, new_grid: np.ndarray, new_energy_assign: np.ndarray, 
                   new_repro_assign: np.ndarray, new_energy: np.ndarray) -> None:
        """Set the 3D grid, mode assignments, and energy state from numpy arrays"""
        if new_grid.shape != (self.ROWS, self.COLS, self.DEPTH):
            raise ValueError(f"Grid shape must be {(self.ROWS, self.COLS, self.DEPTH)}")
        if new_energy_assign.shape != (self.ROWS, self.COLS, self.DEPTH):
            raise ValueError(f"Energy assignments shape must be {(self.ROWS, self.COLS, self.DEPTH)}")
        if new_repro_assign.shape != (self.ROWS, self.COLS, self.DEPTH):
            raise ValueError(f"Repro assignments shape must be {(self.ROWS, self.COLS, self.DEPTH)}")
        if new_energy.shape != (self.ROWS, self.COLS, self.DEPTH):
            raise ValueError(f"Energy shape must be {(self.ROWS, self.COLS, self.DEPTH)}")
        
        self.grid = torch.from_numpy(new_grid).float().to(self.DEVICE)
        self.energy_assignments = torch.from_numpy(new_energy_assign).int().to(self.DEVICE)
        self.repro_assignments = torch.from_numpy(new_repro_assign).int().to(self.DEVICE)
        self.energy = torch.from_numpy(new_energy).float().to(self.DEVICE)

# Example usage
if __name__ == '__main__':
    # Create simulation with default config
    gol = GameOfLife3D(
        grid_size=(50, 50, 50),
        initial_prob=0.1
    )
    
    # Run with visualization
    gol.animate(frames=100, interval=200)