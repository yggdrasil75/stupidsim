import pygame
import numpy as np
import math
from collections import defaultdict

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
BACKGROUND = (10, 10, 20)
LIGHT_COLOR = (255, 255, 200)
CHARACTER_COLOR = (100, 200, 255)
TRAIL_COLOR = (255, 150, 50, 100)
WALL_COLOR = (100, 100, 100)
TRAIL_DECAY = 0.98  # How quickly trails fade
TRAIL_STRENGTH = 10  # How much light adds to trails

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Light Propagation with Walls and Pheromone Trails")
clock = pygame.time.Clock()

class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.start = (x1, y1)
        self.end = (x2, y2)
        self.reflection = np.random.uniform(0.3, 0.9)  # Reflection coefficient
        self.refraction = np.random.uniform(0.7, 1.3)  # Refraction index
        
    def draw(self, surface):
        pygame.draw.line(surface, WALL_COLOR, self.start, self.end, 3)
        
    def get_normal(self, intersection_point):
        # Calculate normal vector to the wall
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return (0, 0)
            
        # Normalize
        dx /= length
        dy /= length
        
        # Return perpendicular vector (normal)
        return (-dy, dx)

class LightSource:
    def __init__(self, x, y, radius, intensity):
        self.x = x
        self.y = y
        self.radius = radius
        self.intensity = intensity
        self.rays = []
        
    def emit_light(self, characters, pheromone_map, walls, num_rays=100):
        # Emit rays in all directions
        for i in range(num_rays):
            angle = 2 * math.pi * i / num_rays
            dx = math.cos(angle)
            dy = math.sin(angle)
            self.cast_ray(self.x, self.y, dx, dy, characters, pheromone_map, walls)
    
    def cast_ray(self, x, y, dx, dy, characters, pheromone_map, walls, intensity=1.0, depth=0, max_depth=10):
        if depth > max_depth or intensity < 0.1:
            return
            
        # Move the ray step by step until we hit something
        step_size = 5
        max_steps = 100
        hit_something = False
        
        for step in range(max_steps):
            # Calculate next position
            next_x = x + dx * step_size
            next_y = y + dy * step_size
            
            # Check if out of bounds
            if next_x < 0 or next_x >= WIDTH or next_y < 0 or next_y >= HEIGHT:
                break
                
            # Check if ray hits any wall
            hit_wall = None
            min_dist = float('inf')
            intersection_point = None
            
            for wall in walls:
                # Line-line intersection between ray and wall
                x1, y1 = wall.start
                x2, y2 = wall.end
                x3, y3 = x, y
                x4, y4 = next_x, next_y
                
                # Calculate denominator
                den = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
                if den == 0:
                    continue  # Lines are parallel
                    
                ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / den
                ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / den
                
                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    # Intersection point
                    ix = x1 + ua*(x2-x1)
                    iy = y1 + ua*(y2-y1)
                    
                    # Distance from current position
                    dist = math.sqrt((ix-x)**2 + (iy-y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        hit_wall = wall
                        intersection_point = (ix, iy)
                        hit_something = True
            
            if hit_something:
                # Add to pheromone map up to the wall
                steps_to_wall = int(min_dist / step_size)
                for i in range(steps_to_wall):
                    px = int(x + dx * i * step_size)
                    py = int(y + dy * i * step_size)
                    if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                        pheromone_map[px, py] += TRAIL_STRENGTH * intensity * (1 - i/steps_to_wall)
                
                # Handle reflection and refraction
                if np.random.random() < hit_wall.reflection:
                    # Reflect the ray
                    normal = hit_wall.get_normal(intersection_point)
                    dot_product = dx * normal[0] + dy * normal[1]
                    rdx = dx - 2 * dot_product * normal[0]
                    rdy = dy - 2 * dot_product * normal[1]
                    
                    # Continue with reflected ray
                    self.cast_ray(intersection_point[0], intersection_point[1], 
                                 rdx, rdy, characters, pheromone_map, walls, 
                                 intensity * hit_wall.reflection, depth + 1)
                
                # Refract the ray (continue through but with some deviation)
                refracted_dx = dx * hit_wall.refraction + np.random.uniform(-0.2, 0.2)
                refracted_dy = dy * hit_wall.refraction + np.random.uniform(-0.2, 0.2)
                
                # Normalize
                length = math.sqrt(refracted_dx**2 + refracted_dy**2)
                if length > 0:
                    refracted_dx /= length
                    refracted_dy /= length
                    
                    self.cast_ray(intersection_point[0], intersection_point[1], 
                                 refracted_dx, refracted_dy, characters, pheromone_map, walls, 
                                 intensity * (1 - hit_wall.reflection), depth + 1)
                
                break
                
            # Check if ray hits any character
            for character in characters:
                dist = math.sqrt((next_x - character.x)**2 + (next_y - character.y)**2)
                if dist < character.radius:
                    # Character sees the light - reinforce the path
                    steps_to_character = step + 1
                    for i in range(steps_to_character):
                        px = int(x + dx * i * step_size)
                        py = int(y + dy * i * step_size)
                        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                            pheromone_map[px, py] += TRAIL_STRENGTH * intensity * (1 - i/steps_to_character)
                    hit_something = True
                    break
                    
            if hit_something:
                break
                
            # Update position
            x, y = next_x, next_y
            
            # Add to pheromone map (weaker for indirect paths)
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                pheromone_map[int(x), int(y)] += 0.1 * TRAIL_STRENGTH * intensity
    
    def draw(self, surface):
        pygame.draw.circle(surface, LIGHT_COLOR, (int(self.x), int(self.y)), self.radius)

class Character:
    def __init__(self, x, y, radius, fov=120, view_distance=300):
        self.x = x
        self.y = y
        self.radius = radius
        self.fov = fov  # Field of view in degrees
        self.view_distance = view_distance
        self.view_direction = np.random.uniform(0, 2 * math.pi)
        self.visible_lights = []
        
    def update(self, light_sources, pheromone_map):
        self.visible_lights = []
        
        # Simple vision - check if light is in field of view
        for light in light_sources:
            # Calculate angle to light
            dx = light.x - self.x
            dy = light.y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > self.view_distance:
                continue
                
            angle_to_light = math.atan2(dy, dx)
            angle_diff = abs((angle_to_light - self.view_direction + math.pi) % (2 * math.pi) - math.pi)
            
            fov_radians = math.radians(self.fov / 2)
            if angle_diff < fov_radians:
                self.visible_lights.append(light)
                
                # Reinforce the path in pheromone map
                steps = int(distance / 5)
                for i in range(steps):
                    px = int(self.x + dx * i/steps)
                    py = int(self.y + dy * i/steps)
                    if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                        pheromone_map[px, py] += TRAIL_STRENGTH * (1 - i/steps)
    
    def draw(self, surface):
        # Draw character
        pygame.draw.circle(surface, CHARACTER_COLOR, (int(self.x), int(self.y)), self.radius)
        
        # Draw field of view
        fov_rad = math.radians(self.fov / 2)
        pygame.draw.line(surface, (200, 200, 200), 
                         (int(self.x), int(self.y)),
                         (int(self.x + math.cos(self.view_direction - fov_rad) * self.view_distance),
                          int(self.y + math.sin(self.view_direction - fov_rad) * self.view_distance)))
        pygame.draw.line(surface, (200, 200, 200), 
                         (int(self.x), int(self.y)),
                         (int(self.x + math.cos(self.view_direction + fov_rad) * self.view_distance),
                          int(self.y + math.sin(self.view_direction + fov_rad) * self.view_distance)))
        
        # Draw view direction
        pygame.draw.line(surface, (255, 255, 255), 
                         (int(self.x), int(self.y)),
                         (int(self.x + math.cos(self.view_direction) * self.radius * 1.5),
                          int(self.y + math.sin(self.view_direction) * self.radius * 1.5)))

# Create light sources
sun = LightSource(WIDTH * 0.2, HEIGHT * 0.3, 30, 1.0)
fire = LightSource(WIDTH * 0.7, HEIGHT * 0.7, 15, 0.7)
light_sources = [sun, fire]

# Create walls
walls = []
for _ in range(15):  # Create 15 random walls
    x1 = np.random.randint(0, WIDTH)
    y1 = np.random.randint(0, HEIGHT)
    x2 = x1 + np.random.randint(-100, 100)
    y2 = y1 + np.random.randint(-100, 100)
    
    # Make sure wall is within screen bounds
    x2 = max(0, min(WIDTH, x2))
    y2 = max(0, min(HEIGHT, y2))
    
    walls.append(Wall(x1, y1, x2, y2))

# Create characters (reduced to 5)
characters = []
for _ in range(5):
    # Make sure characters don't spawn inside walls
    valid_position = False
    while not valid_position:
        x = np.random.randint(50, WIDTH - 50)
        y = np.random.randint(50, HEIGHT - 50)
        
        # Check if too close to any wall
        valid_position = True
        for wall in walls:
            # Simple distance check - could be improved with proper point-line distance
            dist_to_start = math.sqrt((x - wall.start[0])**2 + (y - wall.start[1])**2)
            dist_to_end = math.sqrt((x - wall.end[0])**2 + (y - wall.end[1])**2)
            if dist_to_start < 20 or dist_to_end < 20:
                valid_position = False
                break
                
    characters.append(Character(x, y, 10))

# Initialize pheromone map
pheromone_map = np.zeros((WIDTH, HEIGHT))

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                # Reset walls and characters
                walls = []
                for _ in range(15):
                    x1 = np.random.randint(0, WIDTH)
                    y1 = np.random.randint(0, HEIGHT)
                    x2 = x1 + np.random.randint(-100, 100)
                    y2 = y1 + np.random.randint(-100, 100)
                    x2 = max(0, min(WIDTH, x2))
                    y2 = max(0, min(HEIGHT, y2))
                    walls.append(Wall(x1, y1, x2, y2))
                
                characters = []
                for _ in range(5):
                    valid_position = False
                    while not valid_position:
                        x = np.random.randint(50, WIDTH - 50)
                        y = np.random.randint(50, HEIGHT - 50)
                        valid_position = True
                        for wall in walls:
                            dist_to_start = math.sqrt((x - wall.start[0])**2 + (y - wall.start[1])**2)
                            dist_to_end = math.sqrt((x - wall.end[0])**2 + (y - wall.end[1])**2)
                            if dist_to_start < 20 or dist_to_end < 20:
                                valid_position = False
                                break
                        characters.append(Character(x, y, 10))
                
                pheromone_map = np.zeros((WIDTH, HEIGHT))
    
    # Clear screen
    screen.fill(BACKGROUND)
    
    # Decay pheromone trails
    pheromone_map *= TRAIL_DECAY
    
    # Emit light from sources
    for light in light_sources:
        light.emit_light(characters, pheromone_map, walls, num_rays=300)
    
    # Update characters
    for character in characters:
        character.update(light_sources, pheromone_map)
        
        # Randomly change direction occasionally
        if np.random.random() < 0.2:
            character.view_direction += np.random.uniform(-0.5, 0.5)
    
    # Draw walls
    for wall in walls:
        wall.draw(screen)
    
    # Draw pheromone trails
    trail_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for x in range(0, WIDTH, 4):
        for y in range(0, HEIGHT, 4):
            intensity = min(255, int(pheromone_map[x, y]))
            if intensity > 5:
                color = (TRAIL_COLOR[0], TRAIL_COLOR[1], TRAIL_COLOR[2], intensity)
                pygame.draw.circle(trail_surface, color, (x, y), 2)
    screen.blit(trail_surface, (0, 0))
    
    # Draw light sources
    for light in light_sources:
        light.draw(screen)
    
    # Draw characters
    for character in characters:
        character.draw(screen)
    
    # Display info
    font = pygame.font.SysFont(None, 24)
    #text = font.render("Light Propagation with Walls - Press ESC to exit, R to reset", True, (255, 255, 255))
    #screen.blit(text, (10, 10))
    
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()