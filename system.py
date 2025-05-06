from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict

from holder.globals import EARTH_RADIUS_TO_AU
from holder.vertex import Vertex
from world import World

class System:
    def __init__(self, radius_au: float = 2.0, grid_resolution: int = 1000, device: str = 'cpu'):
        """
        A 3D grid system centered at (0,0,0) with given radius in AU.
        
        Args:
            radius_au: System radius in astronomical units
            grid_resolution: Resolution of the spatial grid
            device: 'cpu' or 'cuda' for PyTorch operations
        """
        self.radius_au = radius_au
        self.grid_resolution = grid_resolution
        self.grid_size = np.array([2*radius_au, 2*radius_au, 2*radius_au])
        self.device = device
        
        # Core data structures
        self.vertex_grid = defaultdict(set)  # (x,y,z) -> Set[Vertex]
        self.vertex_positions = {}           # Vertex -> (x,y,z)
        
        # System bodies
        self.primary_star_pos = torch.zeros(3, device=self.device)
        self.worlds: List[World] = []        # List of worlds in the system
        self.world_positions: List[torch.tensor] = []            # Positions of each world (torch tensors)
        self.world_rotations: List[torch.tensor] = []            # Rotation matrices (torch tensors)
        self.world_orbit_params = []         # Orbital parameters for each world

    def add_world(self, world: World, 
                 orbit_distance_au: float = 1.0,
                 orbit_period_days: float = 365.25,
                 initial_rotation: torch.Tensor = None,
                 inclination: float = 0.0):
        """
        Add a world to the system with orbital parameters.
        
        Args:
            world: The World object to add
            orbit_distance_au: Semi-major axis in AU
            orbit_period_days: Orbital period in Earth days
            initial_rotation: Initial rotation matrix (3x3 tensor) or quaternion (4 tensor)
            inclination: Orbital inclination in degrees
        """
        # Initialize orbital parameters
        self.world_orbit_params.append({
            'distance': orbit_distance_au,
            'period': orbit_period_days,
            'inclination': np.radians(inclination),
            'current_angle': 0  # Start at 0 degrees in orbit
        })
        
        # Initialize rotation
        if initial_rotation is None:
            initial_rotation = torch.eye(3, device=self.device)
        elif initial_rotation.shape == (4,):  # Convert quaternion to matrix
            initial_rotation = self.quaternion_to_matrix(initial_rotation)
        self.world_rotations.append(initial_rotation)
        
        # Place world at initial orbital position
        initial_pos = self._calculate_orbit_position(0, len(self.worlds))
        self.world_positions.append(initial_pos.to(self.device))
        self.worlds.append(world)
        
        # Register world vertices
        self._register_world(world, len(self.worlds)-1)

    def quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix"""
        q = q / q.norm()  # Normalize
        x, y, z, w = q[0], q[1], q[2], q[3]
        
        return torch.tensor([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ], device=self.device)

    def axis_angle_to_matrix(self, axis: torch.Tensor, angle: float) -> torch.Tensor:
        """Create rotation matrix from axis-angle representation"""
        axis = axis / axis.norm()
        x, y, z = axis[0], axis[1], axis[2]
        c, s = np.cos(angle), np.sin(angle)
        
        return torch.tensor([
            [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
        ], device=self.device)

    def _calculate_orbit_position(self, time_days: float, world_index: int) -> torch.Tensor:
        """Calculate orbital position at given time."""
        params = self.world_orbit_params[world_index]
        angle = 2 * np.pi * (time_days / params['period']) + params['current_angle']
        
        # Basic circular orbit
        x = params['distance'] * np.cos(angle)
        y = params['distance'] * np.sin(angle) * np.cos(params['inclination'])
        z = params['distance'] * np.sin(angle) * np.sin(params['inclination'])
        
        return torch.tensor([x, y, z], device=self.device)

    def rotate_world(self, world_index: int, rotation: torch.Tensor):
        """
        Rotate a world by the given rotation matrix or quaternion.
        
        Args:
            world_index: Index of the world to rotate
            rotation: 3x3 rotation matrix or 4-element quaternion
        """
        if world_index >= len(self.worlds):
            raise IndexError(f"World index {world_index} out of range")
            
        # Convert quaternion to matrix if needed
        if rotation.shape == (4,):
            rotation = self.quaternion_to_matrix(rotation)
            
        # Update the world's rotation state (matrix multiplication)
        self.world_rotations[world_index] = rotation @ self.world_rotations[world_index]
        
        # Re-register all vertices with new rotation
        self._reregister_world(world_index)

    def move_world_in_orbit(self, world_index: int, time_days: float):
        """
        Move a world along its orbit by the specified time increment.
        """
        if world_index >= len(self.worlds):
            raise IndexError(f"World index {world_index} out of range")
            
        # Update orbital position
        new_pos = self._calculate_orbit_position(time_days, world_index)
        self.world_positions[world_index] = new_pos
        self.world_orbit_params[world_index]['current_angle'] += 2 * np.pi * (time_days / 
                                            self.world_orbit_params[world_index]['period'])
        
        # Re-register all vertices at new position
        self._reregister_world(world_index)

    def _reregister_world(self, world_index: int):
        """Internal method to update a world's vertex positions after movement/rotation."""
        world = self.worlds[world_index]
        old_positions = {v: self.vertex_positions[v] for v in world.vertices 
                        if v in self.vertex_positions}
        
        # First remove all existing vertices
        for vertex in world.vertices:
            if vertex in self.vertex_positions:
                grid_pos = self.vertex_positions[vertex]
                self.vertex_grid[grid_pos].discard(vertex)
                del self.vertex_positions[vertex]
        
        # Then register with new positions
        world_pos = self.world_positions[world_index]
        rotation = self.world_rotations[world_index]
        
        for vertex in world.vertices:
            # Convert vertex position to tensor if needed
            vertex_pos = torch.tensor(vertex.pos, device=self.device, dtype=torch.float32)
            
            # Apply rotation to vertex position (relative to world center)
            rotated_pos = rotation @ vertex_pos
            system_pos = world_pos + rotated_pos * EARTH_RADIUS_TO_AU
            
            # Convert back to numpy for grid storage
            grid_pos = self._pos_to_grid(system_pos.cpu().numpy())
            
            self.vertex_positions[vertex] = grid_pos
            self.vertex_grid[grid_pos].add(vertex)

    def _register_world(self, world: World, world_index: int):
        """Internal method to register all world vertices in the system."""
        world_pos = self.world_positions[world_index]
        rotation = self.world_rotations[world_index]
        world_radius_au = world.radius * EARTH_RADIUS_TO_AU
        
        # Clear area where world will be placed
        self.clear_area_around(world_pos.cpu().numpy(), world_radius_au)

        for vertex in world.vertices:
            # Convert vertex position to tensor
            vertex_pos = torch.tensor(vertex.pos, device=self.device, dtype=torch.float32)
            
            # Apply rotation
            rotated_pos = rotation @ vertex_pos
            system_pos = world_pos + rotated_pos * EARTH_RADIUS_TO_AU
            
            # Convert back to numpy for grid storage
            grid_pos = self._pos_to_grid(system_pos.cpu().numpy())
            
            self.vertex_positions[vertex] = grid_pos
            self.vertex_grid[grid_pos].add(vertex)

    def update_environment(self, time_days: float = 0):
        """Update environmental factors based on positions and time."""
        for i, world in enumerate(self.worlds):
            if time_days > 0:
                self.move_world_in_orbit(i, time_days)
                
            world_pos = self.world_positions[i]
            #print(world_pos)
            
            for vertex in world.vertices:
                # Get vertex position in world coordinates
                vertex_pos = torch.tensor(vertex.pos, device=self.device, dtype=torch.float32)
                
                # Get actual system position in AU
                rotated_pos = self.world_rotations[i] @ vertex_pos
                #print(rotated_pos)
                system_pos = world_pos + rotated_pos
                
                # 1. Solar radiation (inverse square law)
                distance_to_star_au = system_pos.norm().item()
                distance_to_star_au = system_pos.norm()
                vertex.solar_radiation = self._calculate_solar_flux(distance_to_star_au, 
                                                                  vertex, 
                                                                  system_pos.cpu().numpy(),
                                                                  world_pos.cpu().numpy())
                
                # 2. Temperature (depends on solar flux and elevation)
                vertex.temperature = self._calculate_temperature(vertex)
                
                # 3. Tidal forces (depends on water and position)
                vertex.tidal_force = vertex.water_depth * 0.01 * (1.0/distance_to_star_au)**3
                
                # 4. Initialize reflected light
                vertex.reflected_light = 0
            
            # Calculate inter-vertex reflections
            self._calculate_reflections()

    def _calculate_solar_flux(self, distance_au: float, vertex: Vertex, 
                             system_pos: np.ndarray, world_pos: np.ndarray) -> float:
        """Calculate solar flux at given distance (AU), adjusted for vertex orientation."""
        if distance_au < 1e-9:
            distance_au = 1e-9
        base_flux = 1361 * (1.0**2) / (distance_au**2)  # Earth's solar constant at 1 AU
        
        # Vertex normal in world coordinates
        vertex_normal = vertex.pos / np.linalg.norm(vertex.pos)
        
        # Sun direction in world coordinates
        sun_dir_world = -world_pos / np.linalg.norm(world_pos)
        
        # Angle between vertex normal and sun direction
        cos_angle = max(0, np.dot(vertex_normal, sun_dir_world))
        return base_flux * cos_angle


    def _calculate_temperature(self, vertex: Vertex) -> float:
        """Calculate temperature considering solar flux and elevation."""
        base_temp = 20 + (vertex.solar_radiation / 50)  # Solar heating
        elevation_effect = -vertex.elevation * 0.0065    # Lapse rate (6.5°C per km)
        return base_temp + elevation_effect

    def _calculate_reflections(self):
        """Calculate light reflection between nearby vertices."""
        for source_vertex in self.vertex_positions:
            source_energy = source_vertex.solar_radiation * source_vertex.albedo
            if source_energy <= 0:
                continue
                
            source_pos = self.vertex_positions[source_vertex]
            
            # Get all vertices within reflection range (3 grid cells)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_pos = (source_pos[0]+dx, source_pos[1]+dy, source_pos[2]+dz)
                        for neighbor in self.vertex_grid.get(neighbor_pos, set()):
                            if neighbor == source_vertex:
                                continue
                                
                            # Simple reflection model
                            distance = np.linalg.norm(
                                self._grid_to_pos(source_pos) - 
                                self._grid_to_pos(neighbor_pos)
                            )
                            reflection_strength = source_energy * 0.3 / (distance ** 2)
                            neighbor.reflected_light += reflection_strength

    def get_environment(self, vertex: Vertex) -> Dict[str, float]:
        """Get complete environmental data for a vertex."""
        return {
            'solar': vertex.solar_radiation,
            'tidal': vertex.tidal_force,
            'temp': vertex.temperature,
            'reflected': vertex.reflected_light,
            'grid_pos': self.vertex_positions.get(vertex)
        }

    def get_cell_summary(self, x: int, y: int, z: int) -> Dict[str, float]:
        """Get aggregated environmental data for a grid cell."""
        vertices = self.vertex_grid.get((x,y,z), set())
        if not vertices:
            return {'vertex_count': 0}
            
        return {
            'vertex_count': len(vertices),
            'avg_solar': sum(v.solar_radiation for v in vertices) / len(vertices),
            'avg_tidal': sum(v.tidal_force for v in vertices) / len(vertices),
            'avg_temp': sum(v.temperature for v in vertices) / len(vertices),
            'avg_reflected': sum(v.reflected_light for v in vertices) / len(vertices)
        }

    def _remove_vertex(self, vertex: Vertex) -> None:
        """Internal method to completely remove a vertex from the system."""
        if vertex in self.vertex_positions:
            grid_pos = self.vertex_positions[vertex]
            self.vertex_grid[grid_pos].discard(vertex)
            del self.vertex_positions[vertex]

            
    def clear_area_around(self, center_au: np.ndarray, radius_au: float):
        """Clear all vertices within radius_au of center position."""
        min_coords = np.maximum(center_au - radius_au, -self.radius_au)
        max_coords = np.minimum(center_au + radius_au, self.radius_au)
        
        # Convert to grid coordinates
        min_grid = self._pos_to_grid(min_coords)
        max_grid = self._pos_to_grid(max_coords)
        
        # Clear affected cells
        for x in range(min_grid[0], max_grid[0] + 1):
            for y in range(min_grid[1], max_grid[1] + 1):
                for z in range(min_grid[2], max_grid[2] + 1):
                    grid_pos = (x, y, z)
                    cell_center = self._grid_to_pos(grid_pos)
                    if np.linalg.norm(center_au - cell_center) <= radius_au:
                        for vertex in self.vertex_grid.get(grid_pos, set()):
                            self._remove_vertex(vertex)
                        self.vertex_grid.pop(grid_pos, None)

    def _pos_to_grid(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert system coordinates (in AU) to grid indices."""
        # Scale position from [-radius_au, radius_au] to [0, grid_resolution]
        normalized = (pos + self.radius_au) / (2 * self.radius_au)
        grid_pos = (normalized * (self.grid_resolution - 1)).astype(int)
        return tuple(grid_pos)

    def _grid_to_pos(self, grid_pos: Tuple[int, int, int]) -> np.ndarray:
        """Convert grid indices back to system coordinates (in AU)."""
        normalized = np.array(grid_pos) / (self.grid_resolution - 1)
        return normalized * (2 * self.radius_au) - self.radius_au

    def _is_in_grid(self, pos: np.ndarray) -> bool:
        """Check if position (in AU) is within system bounds."""
        return np.linalg.norm(pos) <= self.radius_au

    def plot_system(self, figsize=(10, 8), ax=None):
        """Plot the entire system showing star and world positions."""
        fig = plt.figure(figsize=figsize)
        if not ax:
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot the primary star
        ax.scatter(*self.primary_star_pos, color='yellow', s=500, label='Primary Star')
        
        # Plot all worlds
        for i, (world, pos) in enumerate(zip(self.worlds, self.world_positions)):
            color = plt.cm.tab10(i % 10)  # Different color for each world
            ax.scatter(*pos, color=color, s=300, label=f'World {i}')
            
            # Draw orbit path
            if i < len(self.world_orbit_params):
                params = self.world_orbit_params[i]
                theta = np.linspace(0, 2*np.pi, 100)
                x = params['distance'] * np.cos(theta)
                y = params['distance'] * np.sin(theta) * np.cos(params['inclination'])
                z = params['distance'] * np.sin(theta) * np.sin(params['inclination'])
                ax.plot(x, y, z, '--', color=color, alpha=0.3)
        
        # Set plot limits
        max_dist = self.radius_au * 1.1
        ax.set_xlim([-max_dist, max_dist])
        ax.set_ylim([-max_dist, max_dist])
        ax.set_zlim([-max_dist, max_dist])
        
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        ax.set_title('System Overview')
        ax.legend()
        
        return fig, ax
