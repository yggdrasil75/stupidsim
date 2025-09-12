import dearpygui.dearpygui as dpg
import numpy as np
import math
import random

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
BACKGROUND = (0.04, 0.04, 0.08, 1.0)
LIGHT_COLOR = (1.0, 1.0, 0.78, 1.0)
CHARACTER_COLOR = (0.39, 0.78, 1.0, 1.0)
TRAIL_COLOR = (1.0, 0.59, 0.20, 0.39)
WALL_COLOR = (0.39, 0.39, 0.39, 1.0)
TRAIL_DECAY = 0.95  # Percentage of trails that remain each frame
TRAIL_STRENGTH = 50  # How much light adds to trails
TRAIL_DRAW_THRESHOLD = 5  # Minimum intensity to draw a trail

# Material properties
MATERIALS = {
    "glass": {"reflection": 0.1, "refraction": 1.0, "absorption": 0.1, "color": (1.0, 0.0, 1.0, 0.39)},
    "stone": {"reflection": 0.01, "refraction": 0.0, "absorption": 1.0, "color": (1.0, 0.0, 0.0, 1.0)},
    "metal": {"reflection": 1.0, "refraction": 0.5, "absorption": 0.5, "color": (0.0, 1.0, 0.0, 1.0)}
}

class Wall:
    def __init__(self, x1, y1, x2, y2, material="stone"):
        self.start = (x1, y1)
        self.end = (x2, y2)
        self.material = material
        self.properties = MATERIALS[material]
        self.draw_id = None
        self.thickness = np.random.randint(1, 7)
        self.normal = None
        
    def draw(self):
        dpg.draw_line(self.start, self.end, color=self.properties["color"], thickness=self.thickness, parent="canvas")
    
    def get_normal(self):
        if self.normal is None:
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
            self.normal = (-dy, dx)
        return self.normal

class LightSource:
    def __init__(self, x, y, radius, intensity):
        self.x = x
        self.y = y
        self.radius = radius
        self.intensity = intensity
        self.rays = []
        self.draw_id = None
        
    def emit_light(self, characters, pheromone_map, walls, num_rays=100):
        # Emit rays in all directions
        for i in range(num_rays):
            angle = 2 * math.pi * i / num_rays
            dx = math.cos(angle)
            dy = math.sin(angle)
            self.cast_ray(self.x, self.y, dx, dy, characters, pheromone_map, walls)
    
    def cast_ray(self, x, y, dx, dy, characters, pheromone_map, walls, intensity=1.0, depth=0, max_depth=5):
        if depth > max_depth or intensity < 0.05:
            return
            
        # Move the ray step by step until we hit something
        step_size = 5
        max_steps = 200
        hit_something = False
        
        for step in range(max_steps):
            # Calculate next position
            next_x = x + dx * step_size
            next_y = y + dy * step_size
            
            # Check if out of bounds
            if next_x < 0 or next_x >= WIDTH or next_y < 0 or next_y >= HEIGHT:
                break
                
            # Check if ray hits any character
            for character in characters:
                dist = math.sqrt((next_x - character.x)**2 + (next_y - character.y)**2)
                if dist < character.radius:
                    # Character is hit by the light - reinforce the path
                    steps_to_character = step + 1
                    for i in range(steps_to_character):
                        px = int(x + dx * i * step_size)
                        py = int(y + dy * i * step_size)
                        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                            # Add more intensity closer to the character
                            pheromone_map[px, py] += TRAIL_STRENGTH * intensity * (1 - i/steps_to_character)
                    hit_something = True
                    break
                    
            if hit_something:
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
            
            if hit_something and hit_wall:
                # Add to pheromone map up to the wall
                steps_to_wall = int(min_dist / step_size)
                for i in range(steps_to_wall):
                    px = int(x + dx * i * step_size)
                    py = int(y + dy * i * step_size)
                    if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                        pheromone_map[px, py] += TRAIL_STRENGTH * intensity * (1 - i/steps_to_wall)
                
                # Handle reflection and refraction based on material properties
                material = hit_wall.properties
                
                # Reflection
                reflection_chance = material["reflection"]
                if random.random() < reflection_chance and reflection_chance > 0:
                    # Reflect the ray
                    normal = hit_wall.get_normal()
                    dot_product = dx * normal[0] + dy * normal[1]
                    rdx = dx - 2 * dot_product * normal[0]
                    rdy = dy - 2 * dot_product * normal[1]
                    
                    # Continue with reflected ray
                    self.cast_ray(intersection_point[0], intersection_point[1], 
                                 rdx, rdy, characters, pheromone_map, walls, 
                                 intensity * reflection_chance * (1 - material["absorption"]), depth + 1)
                
                # Refraction (continue through but with some deviation)
                refraction_chance = material["refraction"]
                if refraction_chance > 0 and random.random() < refraction_chance:
                    refracted_dx = dx * 0.9 + random.uniform(-0.1, 0.1)
                    refracted_dy = dy * 0.9 + random.uniform(-0.1, 0.1)
                    
                    # Normalize
                    length = math.sqrt(refracted_dx**2 + refracted_dy**2)
                    if length > 0:
                        refracted_dx /= length
                        refracted_dy /= length
                        
                        self.cast_ray(intersection_point[0], intersection_point[1], 
                                     refracted_dx, refracted_dy, characters, pheromone_map, walls, 
                                     intensity * refraction_chance * (1 - material["absorption"]), depth + 1)
                
                break
                
            # Update position
            x, y = next_x, next_y
            
            # Add to pheromone map (weaker for indirect paths)
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                pheromone_map[int(x), int(y)] += 0.1 * TRAIL_STRENGTH * intensity
    
    def draw(self):
        self.draw_id = dpg.draw_circle((self.x, self.y), self.radius, color=LIGHT_COLOR, fill=LIGHT_COLOR, parent="canvas")

class Character:
    def __init__(self, x, y, radius, fov=120, view_distance=300):
        self.x = x
        self.y = y
        self.radius = radius
        self.fov = fov  # Field of view in degrees
        self.view_distance = view_distance
        self.view_direction = random.uniform(0, 2 * math.pi)
        self.visible_lights = []
        self.draw_ids = []
        
    def update(self, light_sources):
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
    
    def draw(self):
        # Clear previous drawings
        self.draw_ids = []
        
        # Draw character
        self.draw_ids.append(dpg.draw_circle((self.x, self.y), self.radius, color=CHARACTER_COLOR, fill=CHARACTER_COLOR, parent="canvas"))
        
        # Draw field of view
        fov_rad = math.radians(self.fov / 2)
        self.draw_ids.append(dpg.draw_line(
            (self.x, self.y),
            (self.x + math.cos(self.view_direction - fov_rad) * self.view_distance,
             self.y + math.sin(self.view_direction - fov_rad) * self.view_distance),
            color=(0.78, 0.78, 0.78, 1.0), parent="canvas"))
        
        self.draw_ids.append(dpg.draw_line(
            (self.x, self.y),
            (self.x + math.cos(self.view_direction + fov_rad) * self.view_distance,
             self.y + math.sin(self.view_direction + fov_rad) * self.view_distance),
            color=(0.78, 0.78, 0.78, 1.0), parent="canvas"))
        
        # Draw view direction
        self.draw_ids.append(dpg.draw_line(
            (self.x, self.y),
            (self.x + math.cos(self.view_direction) * self.radius * 1.5,
             self.y + math.sin(self.view_direction) * self.radius * 1.5),
            color=(1.0, 1.0, 1.0, 1.0), parent="canvas"))

# Global variables
light_sources = []
walls = []
characters = []
pheromone_map = np.zeros((WIDTH, HEIGHT))
trail_draw_ids = []

def init_simulation():
    global light_sources, walls, characters, pheromone_map, trail_draw_ids
    
    # Clear existing objects
    light_sources.clear()
    walls.clear()
    characters.clear()
    pheromone_map = np.zeros((WIDTH, HEIGHT))
    trail_draw_ids = []
    
    # Create light sources
    sun = LightSource(WIDTH * 0.2, HEIGHT * 0.3, 30, 1.0)
    fire = LightSource(WIDTH * 0.7, HEIGHT * 0.7, 15, 0.7)
    light_sources.extend([sun, fire])
    
    # Create walls with a stone bounding box
    walls.append(Wall(5, 5, WIDTH-5, 5, "stone"))  # Top
    walls.append(Wall(WIDTH-5, 5, WIDTH-5, HEIGHT-5, "stone"))  # Right
    walls.append(Wall(WIDTH-5, HEIGHT-5,5, HEIGHT-5, "stone"))  # Bottom
    walls.append(Wall(5, HEIGHT-5, 5, 5, "stone"))  # Left

    # Add some interior walls with different materials
    for _ in range(10):  # Create 10 random interior walls
        x1 = random.randint(50, WIDTH-50)
        y1 = random.randint(50, HEIGHT-50)
        x2 = x1 + random.randint(-100, 100)
        y2 = y1 + random.randint(-100, 100)
        
        # Make sure wall is within screen bounds
        x2 = max(50, min(WIDTH-50, x2))
        y2 = max(50, min(HEIGHT-50, y2))
        
        # Randomly choose a material
        material = random.choice(["glass", "stone", "metal"])
        walls.append(Wall(x1, y1, x2, y2, material))

    # Create characters (reduced to 5)
    for _ in range(5):
        # Make sure characters don't spawn inside walls
        valid_position = False
        while not valid_position:
            x = random.randint(50, WIDTH - 50)
            y = random.randint(50, HEIGHT - 50)
            
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

def draw_trails():
    global trail_draw_ids, pheromone_map
    
    # Clear previous trail drawings
    trail_draw_ids = []
    
    # Draw trails as points where pheromone intensity is above threshold
    for x in range(0, WIDTH, 2):  # Step by 2 to reduce number of points
        for y in range(0, HEIGHT, 2):
            intensity = pheromone_map[x, y]
            if intensity > TRAIL_DRAW_THRESHOLD:
                # Calculate alpha based on intensity
                alpha = min(1.0, intensity / 255.0)
                color = (TRAIL_COLOR[0], TRAIL_COLOR[1], TRAIL_COLOR[2], alpha * TRAIL_COLOR[3])
                
                # Draw a small circle for the trail point
                size = max(1, min(3, int(intensity / 50)))
                trail_draw_ids.append(
                    dpg.draw_circle((x, y), size, color=color, fill=color, parent="canvas")
                )

def update_simulation():
    global pheromone_map
    
    # Decay pheromone trails
    pheromone_map *= TRAIL_DECAY
    
    # Emit light from sources
    for light in light_sources:
        light.emit_light(characters, pheromone_map, walls, num_rays=1000)
    
    # Update characters
    for character in characters:
        character.update(light_sources)
        
        # Randomly change direction occasionally
        if random.random() < 0.05:
            character.view_direction += random.normalvariate(0.0, 0.25)
    
    # Draw pheromone trails
    draw_trails()
    
    # Redraw everything
    draw_scene()

def draw_scene():
    # Clear canvas (but keep trails as they're drawn separately)
    dpg.delete_item("canvas", children_only=True)
    
    # Draw trails first (as background)
    draw_trails()
    
    # Draw walls
    for wall in walls:
        wall.draw()
    
    # Draw light sources
    for light in light_sources:
        light.draw()
    
    # Draw characters
    for character in characters:
        character.draw()

def reset_simulation():
    init_simulation()
    draw_scene()

def main():
    # Initialize Dear PyGui
    dpg.create_context()
    dpg.create_viewport(title="Light Propagation with Walls and Pheromone Trails", width=WIDTH, height=HEIGHT+100)
    
    # Create main window
    with dpg.window(label="Main", tag="main_window", width=WIDTH+10, height=HEIGHT+100):
        # Create drawlist for the simulation
        with dpg.drawlist(width=WIDTH, height=HEIGHT, tag="canvas"):
            pass
        
    # Initialize simulation
    init_simulation()
    
    # Setup Dear PyGui
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    
    # Main loop
    while dpg.is_dearpygui_running():
        update_simulation()
        dpg.render_dearpygui_frame()
    
    # Cleanup
    dpg.destroy_context()

if __name__ == "__main__":
    main()