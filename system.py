from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict

from holder.globals import EARTH_RADIUS_TO_AU
from holder.vertex import Vertex
from world import World

class System:
    def __init__(self, radius_au: float = 2.0, grid_resolution: int = 1000):
        """
        A 3D grid system centered at (0,0,0) with given radius in AU.
        The grid spans from -radius_au to +radius_au in each dimension.
        """
        self.radius_au = radius_au
        self.grid_resolution = grid_resolution
        self.grid_size = np.array([2*radius_au, 2*radius_au, 2*radius_au])  # Total size in AU
        
        # Core data structures
        self.vertex_grid = defaultdict(set)  # (x,y,z) -> Set[Vertex]
        self.vertex_positions = {}           # Vertex -> (x,y,z)
        
        # System bodies
        self.primary_star_pos = np.zeros(3)  # Center at (0,0,0)
        self.world: List[World] = []
        self.world_pos = None                # Will be placed in habitable zone

    def place_world_in_habitable_zone(self, world: World, distance_from_star_au: float = 1.0):
        """
        Places the world at a habitable distance from the star along the x-axis.
        Distance is in AU (1.0 = Earth's distance).
        """
        self.world_pos = np.array([distance_from_star_au, 0, 0])
        self.world.append(world)
        self._register_world(world)

    def place_world_at_position(self, world: World, position: Tuple[float, float, float]):
        """Place world at specific coordinates in the system (in AU units)."""
        self.world_pos = np.array(position)
        if not self._is_in_grid(self.world_pos):
            raise ValueError(f"Position {position} is outside the system grid")
        self._register_world(world)

    def _register_world(self, world: World):
        """Internal method to register all world vertices in the system."""
        world_radius_au = world.radius * EARTH_RADIUS_TO_AU
        self.clear_area_around(self.world_pos, world_radius_au)

        for vertex in world.vertices:
            # Convert vertex position to system coordinates (in AU)
            system_pos = self.world_pos + vertex.pos
            grid_pos = self._pos_to_grid(system_pos)
            
            self.vertex_positions[vertex] = grid_pos
            self.vertex_grid[grid_pos].add(vertex)
            
            # Initialize vertex environmental factors
            vertex.solar_radiation = 0
            vertex.tidal_force = 0
            vertex.temperature = 0
            vertex.reflected_light = 0

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

    def update_environment(self):
        """Update environmental factors based on position relative to star."""
        if self.world_pos is None:
            return
            
        for vertex in self.vertex_positions:
            # Get actual system position in AU
            system_pos = self.world_pos + vertex.pos
            
            # 1. Solar radiation (inverse square law)
            distance_to_star_au = np.linalg.norm(system_pos)
            vertex.solar_radiation = self._calculate_solar_flux(distance_to_star_au, vertex)
            
            # 2. Temperature (depends on solar flux and elevation)
            vertex.temperature = self._calculate_temperature(vertex)
            
            # 3. Tidal forces (depends on water and position)
            vertex.tidal_force = vertex.water_depth * 0.01 * (1.0/distance_to_star_au)**3
            
            # 4. Initialize reflected light
            vertex.reflected_light = 0
        
        # Calculate inter-vertex reflections
        self._calculate_reflections()

    def _calculate_solar_flux(self, distance_au: float, vertex: Vertex) -> float:
        """Calculate solar flux at given distance (AU), adjusted for vertex orientation."""
        base_flux = 1361 * (1.0**2) / (distance_au**2)  # Earth's solar constant at 1 AU
        normal = vertex.pos / np.linalg.norm(vertex.pos)
        sun_dir = -self.world_pos / np.linalg.norm(self.world_pos)
        cos_angle = max(0, np.dot(normal, sun_dir))
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

            
    def plot_system(self, figsize=(10, 8), ax=None):
        """Plot the entire system showing star and world positions."""
        fig = plt.figure(figsize=figsize)
        if not ax:
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot the primary star
        ax.scatter(*self.primary_star_pos, color='yellow', s=500, label='Primary Star')
        
        # Plot all worlds
        for world in self.world:
            if hasattr(self, 'world_pos') and self.world_pos is not None:
                ax.scatter(*self.world_pos, color='blue', s=300, label='World')
                # Draw orbit circle
                theta = np.linspace(0, 2*np.pi, 100)
                orbit_radius = np.linalg.norm(self.world_pos)
                x = orbit_radius * np.cos(theta)
                y = orbit_radius * np.sin(theta)
                ax.plot(x, y, np.zeros_like(x), 'b--', alpha=0.3)
        
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
    
    def plot_world(self, world_index=0, **kwargs):
        """Plot a specific world in the system."""
        if world_index >= len(self.world):
            raise IndexError(f"World index {world_index} out of range")
        
        world = self.world[world_index]
        return world.plot(**kwargs)
    
    def plot_world_environment(self, world_index=0, figsize=(15, 10)):
        """Plot a world with its environmental factors."""
        if world_index >= len(self.world):
            raise IndexError(f"World index {world_index} out of range")
            
        world = self.world[world_index]
        fig = plt.figure(figsize=figsize)
        
        # Create 4 subplots for different environmental factors
        titles = ['Solar Radiation', 'Tidal Forces', 'Temperature', 'Reflected Light']
        axes = [fig.add_subplot(221, projection='3d'),
                fig.add_subplot(222, projection='3d'),
                fig.add_subplot(223, projection='3d'),
                fig.add_subplot(224, projection='3d')]
        
        # Precompute vertex data
        verts = [v.pos for v in world.vertices]
        x, y, z = zip(*verts)
        
        # Plot each factor
        factors = ['solar_radiation', 'tidal_force', 'temperature', 'reflected_light']
        cmaps = ['YlOrRd', 'Purples', 'coolwarm', 'Blues']
        
        for ax, title, factor, cmap in zip(axes, titles, factors, cmaps):
            values = np.array([getattr(v, factor) for v in world.vertices])
            
            # Normalize values for coloring
            if values.max() > values.min():
                norm = plt.Normalize(values.min(), values.max())
            else:
                norm = plt.Normalize(0, 1)  # handle case where all values are equal
                
            sc = ax.scatter(x, y, z, c=values, cmap=cmap, norm=norm, s=20)
            plt.colorbar(sc, ax=ax, label=title)
            
            ax.set_title(title)
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_zlim([-1.1, 1.1])
            ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, axes