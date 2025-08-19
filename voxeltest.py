import numpy as np
from numba import types, njit, prange, typeof
from numba.typed import List, Dict
from numba.experimental import jitclass
import numba as nb
from PIL import Image
import math
import time
import random
import dearpygui.dearpygui as dpg

# Define the spec for the voxel class
voxel_spec = [
    ('position', types.float64[:]),
    ('color', types.uint8[:]),
    ('size', types.float64),
    ('plate_id', types.int64),  # Add plate ID
    ('elevation', types.float64),  # Add elevation
]

@jitclass(voxel_spec)
class Voxel:
    def __init__(self, position, color, size=1.0, plate_id=-1, elevation=0.0):
        self.position = position
        self.color = color
        self.size = size
        self.plate_id = plate_id
        self.elevation = elevation

# Get the type of the Voxel class for use in the list
VoxelType = Voxel.class_type.instance_type

# Define the spec for the voxel system
voxel_system_spec = [
    ('voxels', types.ListType(VoxelType)),
    ('width', types.int64),
    ('height', types.int64),
    ('background_color', types.uint8[:]),
    ('plate_colors', types.DictType(types.int64, types.uint8[:])),  # Store plate colors
]

@jitclass(voxel_system_spec)
class VoxelSystem:
    def __init__(self, width, height, background_color=np.array([0, 0, 0, 255], dtype=np.uint8)):
        self.voxels = List.empty_list(VoxelType)
        self.width = width
        self.height = height
        self.background_color = background_color
        self.plate_colors = Dict.empty(key_type=types.int64, value_type=types.uint8[:])
    
    def add_voxel(self, position, color, size=1.0, plate_id=-1, elevation=0.0):
        """Add a voxel to the system"""
        voxel = Voxel(position.astype(np.float64), color.astype(np.uint8), float(size), plate_id, elevation)
        self.voxels.append(voxel)
    
    def create_sphere(self, center, radius, color, num_points=1000):
        """Create a sphere of voxels using Fibonacci sphere algorithm"""
        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
        
        for i in range(num_points):
            y = 1.0 - (i / float(num_points - 1)) * 2.0  # y goes from 1 to -1
            radius_at_y = math.sqrt(1.0 - y * y)  # radius at y
            
            theta = 2.0 * math.pi * i / golden_ratio  # golden angle increment
            
            x = math.cos(theta) * radius_at_y
            z = math.sin(theta) * radius_at_y
            
            # Scale by radius and translate by center
            point = np.array([
                center[0] + x * radius,
                center[1] + y * radius,
                center[2] + z * radius
            ], dtype=np.float64)
            
            self.add_voxel(point, color, 1.0)

@njit
def distance(a, b):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

@njit
def assign_plates(system, num_plates=10):
    """Assign tectonic plates to voxels"""
    # Create plate origins (random points on the sphere)
    plate_origins = []
    for i in range(num_plates):
        # Generate random point on sphere
        theta = random.random() * 2 * math.pi
        phi = math.acos(2 * random.random() - 1)
        
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        
        plate_origins.append(np.array([x, y, z], dtype=np.float64))
        
        # Generate a random color for this plate
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        system.plate_colors[i] = np.array([r, g, b, 255], dtype=np.uint8)
    
    # Assign each voxel to the nearest plate origin
    for i in prange(len(system.voxels)):
        voxel = system.voxels[i]
        min_dist = float('inf')
        closest_plate = -1
        
        # Normalize voxel position to get direction vector
        pos = voxel.position
        norm = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        if norm > 0:
            dir_vec = np.array([pos[0]/norm, pos[1]/norm, pos[2]/norm], dtype=np.float64)
            
            # Find closest plate origin
            for j in range(len(plate_origins)):
                dist = distance(dir_vec, plate_origins[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_plate = j
        
        # Update voxel plate ID
        voxel.plate_id = closest_plate

@njit
def calculate_elevation(system, center, radius):
    """Calculate elevation based on distance from plate boundaries"""
    for i in prange(len(system.voxels)):
        voxel = system.voxels[i]
        
        # Normalize position to get direction vector
        pos = voxel.position
        norm = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        if norm > 0:
            dir_vec = np.array([pos[0]/norm, pos[1]/norm, pos[2]/norm], dtype=np.float64)
            
            # Find distance to nearest plate boundary
            min_boundary_dist = float('inf')
            for j in range(len(system.voxels)):
                if i == j or system.voxels[j].plate_id == voxel.plate_id:
                    continue
                    
                other_pos = system.voxels[j].position
                other_norm = math.sqrt(other_pos[0]**2 + other_pos[1]**2 + other_pos[2]**2)
                if other_norm > 0:
                    other_dir = np.array([other_pos[0]/other_norm, other_pos[1]/other_norm, other_pos[2]/other_norm], dtype=np.float64)
                    dist = distance(dir_vec, other_dir)
                    if dist < min_boundary_dist:
                        min_boundary_dist = dist
            
            # Set elevation based on distance to boundary
            # Closer to boundary = higher elevation (mountains)
            elevation = min(1.0, max(0.0, 1.0 - min_boundary_dist * 5.0))
            voxel.elevation = elevation

def create_planet_with_tectonics(system, center, radius, num_points=50000, num_plates=10):
    """Create a planet with tectonic plates"""
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    
    # First create the sphere points
    for i in range(num_points):
        y = 1.0 - (i / float(num_points - 1)) * 2.0
        radius_at_y = math.sqrt(1.0 - y * y)
        
        theta = 2.0 * math.pi * i / golden_ratio
        
        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        
        point = np.array([
            center[0] + x * radius,
            center[1] + y * radius,
            center[2] + z * radius
        ], dtype=np.float64)
        
        # Add voxel with default color (will be updated later)
        system.add_voxel(point, np.array([0, 0, 0, 255], dtype=np.uint8), 1.5)
    
    # Assign tectonic plates
    print("Assigning tectonic plates...")
    start_time = time.time()
    assign_plates(system, num_plates)
    print(f"Assigned plates in {time.time() - start_time:.2f} seconds")
    
    # Calculate elevation based on plate boundaries
    print("Calculating elevation...")
    start_time = time.time()
    #calculate_elevation(system, center, radius)
    print(f"Calculated elevation in {time.time() - start_time:.2f} seconds")
    
    # Apply colors based on plate and elevation
    print("Applying colors...")
    start_time = time.time()
    apply_plate_colors(system)
    print(f"Applied colors in {time.time() - start_time:.2f} seconds")

@njit
def apply_plate_colors(system):
    """Apply colors based on plate membership and elevation"""
    for i in prange(len(system.voxels)):
        voxel = system.voxels[i]
        
        if voxel.plate_id in system.plate_colors:
            base_color = system.plate_colors[voxel.plate_id].copy()
            
            # Modify color based on elevation
            # Higher elevation = lighter color
            elevation_factor = 0.5 + voxel.elevation * 0.5
            
            base_color[0] = min(255, int(base_color[0] * elevation_factor))
            base_color[1] = min(255, int(base_color[1] * elevation_factor))
            base_color[2] = min(255, int(base_color[2] * elevation_factor))
            
            voxel.color = base_color

@njit(parallel=True)
def render_orthographic(system):
    """Render the voxels using orthographic projection"""
    image = np.zeros((system.height, system.width, 4), dtype=np.uint8)
    
    # Fill with background color
    image[:, :, 0] = system.background_color[0]  # R
    image[:, :, 1] = system.background_color[1]  # G
    image[:, :, 2] = system.background_color[2]  # B
    image[:, :, 3] = system.background_color[3]  # A
    
    # Simple orthographic projection: ignore z-coordinate for now
    for i in prange(len(system.voxels)):
        voxel = system.voxels[i]
        x = int(voxel.position[0] + system.width / 2)
        y = int(voxel.position[1] + system.height / 2)
        
        # Check if within bounds
        if 0 <= x < system.width and 0 <= y < system.height:
            # Simple alpha blending
            alpha = voxel.color[3] / 255.0
            image[y, x, 0] = int(voxel.color[0] * alpha + image[y, x, 0] * (1 - alpha))
            image[y, x, 1] = int(voxel.color[1] * alpha + image[y, x, 1] * (1 - alpha))
            image[y, x, 2] = int(voxel.color[2] * alpha + image[y, x, 2] * (1 - alpha))
            image[y, x, 3] = min(255, image[y, x, 3] + voxel.color[3])
    
    return image

def generate_planet():
    """Generate the planet and return the texture data"""
    # Create the voxel system
    width, height = 800, 800
    system = VoxelSystem(width, height)
    
    # Create a planet with tectonic plates
    center = np.array([0.0, 0.0, 0.0])
    radius = 300.0
    
    print("Creating planet with tectonic plates...")
    start_time = time.time()
    create_planet_with_tectonics(system, center, radius, 50000, 12)
    print(f"Created {len(system.voxels)} voxels in {time.time() - start_time:.2f} seconds")
    
    # Render the scene
    print("Rendering...")
    start_time = time.time()
    image_data = render_orthographic(system)
    print(f"Rendered in {time.time() - start_time:.2f} seconds")
    
    # Convert to a format suitable for DPG
    # DPG expects a flat array of floats in the range [0, 1]
    texture_data = np.zeros((height, width, 4), dtype=np.float32)
    texture_data[:, :, 0] = image_data[:, :, 0].astype(np.float32) / 255.0  # R
    texture_data[:, :, 1] = image_data[:, :, 1].astype(np.float32) / 255.0  # G
    texture_data[:, :, 2] = image_data[:, :, 2].astype(np.float32) / 255.0  # B
    texture_data[:, :, 3] = image_data[:, :, 3].astype(np.float32) / 255.0  # A
    
    return texture_data.flatten()

def dpgmain():
    # Initialize Dear PyGui
    dpg.create_context()
    dpg.create_viewport(title='Voxel Planet', width=800, height=800)
    
    # Generate the planet texture
    texture_data = generate_planet()
    
    # Create texture registry and add texture
    with dpg.texture_registry():
        dpg.add_raw_texture(width=800, height=800, default_value=texture_data, 
                           format=dpg.mvFormat_Float_rgba, tag="planet_texture")
    
    # Create main window with the texture
    with dpg.window(label="Voxel Planet", tag="primary_window", width=800, height=800):
        dpg.add_image("planet_texture", width=800, height=800)
    
    # Set primary window and show viewport
    dpg.set_primary_window("primary_window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Start the rendering loop
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    
    # Cleanup
    dpg.destroy_context()

if __name__ == "__main__":
    dpgmain()