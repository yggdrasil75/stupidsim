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
    ('density', types.float64),  # Add density for solid interior
]

@jitclass(voxel_spec)
class Voxel:
    def __init__(self, position, color, size=1.0, plate_id=-1, elevation=0.0, density=1.0):
        self.position = position
        self.color = color
        self.size = size
        self.plate_id = plate_id
        self.elevation = elevation
        self.density = density

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
    
    def add_voxel(self, position, color, size=1.0, plate_id=-1, elevation=0.0, density=1.0):
        """Add a voxel to the system"""
        voxel = Voxel(position.astype(np.float64), color.astype(np.uint8), float(size), plate_id, elevation, density)
        self.voxels.append(voxel)

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
    # Generate points throughout the entire volume of the sphere
    for i in range(num_points):
        # Generate a random point in a cube
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform(-1, 1)
        
        # Calculate distance from center
        dist = math.sqrt(x*x + y*y + z*z)
        
        # If point is outside the unit sphere, normalize it to the surface
        if dist > 1.0:
            x /= dist
            y /= dist
            z /= dist
            dist = 1.0
        
        # Scale by a random radius to fill the sphere
        r = random.uniform(0, 1)  # Uniform distribution in volume
        r = r ** (1/3)  # Correct for volume distribution
        
        # Apply the radius scaling
        x *= r
        y *= r
        z *= r
        dist *= r
        
        # Scale by planet radius and translate to center
        point = np.array([
            center[0] + x * radius,
            center[1] + y * radius,
            center[2] + z * radius
        ], dtype=np.float64)
        
        # Calculate density based on distance from center (higher density near core)
        density = 1.0 - dist  # Linear density gradient
        
        system.add_voxel(point, np.array([0, 0, 0, 255], dtype=np.uint8), 1.0, -1, 0.0, density)
    
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
    
    # Apply colors based on plate and elevation/density
    print("Applying colors...")
    start_time = time.time()
    apply_plate_colors(system)
    print(f"Applied colors in {time.time() - start_time:.2f} seconds")

@njit
def apply_plate_colors(system):
    """Apply colors based on plate membership, elevation, and density"""
    for i in prange(len(system.voxels)):
        voxel = system.voxels[i]
        
        if voxel.plate_id in system.plate_colors:
            base_color = system.plate_colors[voxel.plate_id].copy()
            
            # Modify color based on elevation and density
            # Higher elevation = lighter color
            # Higher density (deeper) = darker color
            elevation_factor = 0.5 + voxel.elevation * 0.5
            density_factor = 0.3 + 0.7 * (1.0 - voxel.density)  # Invert density for color (darker when denser)
            
            combined_factor = elevation_factor * density_factor
            
            base_color[0] = min(255, int(base_color[0] * combined_factor))
            base_color[1] = min(255, int(base_color[1] * combined_factor))
            base_color[2] = min(255, int(base_color[2] * combined_factor))
            
            voxel.color = base_color
        else:
            # Default color for interior (based on density)
            gray_value = int(100 + 100 * (1.0 - voxel.density))
            voxel.color = np.array([gray_value, gray_value, gray_value, 255], dtype=np.uint8)

@njit
def rotate_point(point, angle_x, angle_y, angle_z):
    """Rotate a point around the origin using Euler angles"""
    # Rotation around X axis
    y = point[1] * math.cos(angle_x) - point[2] * math.sin(angle_x)
    z = point[1] * math.sin(angle_x) + point[2] * math.cos(angle_x)
    
    # Rotation around Y axis
    x = point[0] * math.cos(angle_y) + z * math.sin(angle_y)
    z = -point[0] * math.sin(angle_y) + z * math.cos(angle_y)
    
    # Rotation around Z axis
    x_new = x * math.cos(angle_z) - y * math.sin(angle_z)
    y_new = x * math.sin(angle_z) + y * math.cos(angle_z)
    
    return np.array([x_new, y_new, z], dtype=np.float64)

@njit(parallel=True)
def render_orthographic(system, angle_x=0.0, angle_y=0.0, angle_z=0.0, distance=1.0):
    """Render the voxels using orthographic projection with rotation and zoom"""
    image = np.zeros((system.height, system.width, 4), dtype=np.uint8)
    
    # Fill with background color
    image[:, :, 0] = system.background_color[0]  # R
    image[:, :, 1] = system.background_color[1]  # G
    image[:, :, 2] = system.background_color[2]  # B
    image[:, :, 3] = system.background_color[3]  # A
    
    # Calculate center point
    center_x = system.width / 2
    center_y = system.height / 2
    
    # Create depth buffer for proper occlusion
    depth_buffer = np.full((system.height, system.width), float('inf'))
    
    # Sort voxels by distance from camera for proper rendering order
    sorted_indices = np.argsort(np.array([-np.linalg.norm(voxel.position) for voxel in system.voxels], dtype=np.float32), kind='quicksort')
    
    # Render each voxel with rotation
    for idx in prange(len(sorted_indices)):
        i = sorted_indices[idx]
        voxel = system.voxels[i]
        
        # Apply rotation to the voxel position
        rotated_pos = rotate_point(voxel.position, angle_x, angle_y, angle_z)
        
        # Apply distance scaling (zoom)
        scaled_pos = rotated_pos * distance
        
        # Calculate screen position
        x_center = int(scaled_pos[0] + center_x)
        y_center = int(scaled_pos[1] + center_y)
        
        # Skip if outside view
        if x_center < 0 or x_center >= system.width or y_center < 0 or y_center >= system.height:
            continue
        
        # Calculate depth (distance from camera)
        depth = np.linalg.norm(rotated_pos)
        
        # Use a single pixel per voxel instead of squares to avoid Moiré patterns
        # Or use anti-aliased circles for better quality
        
        # Method 1: Single pixel (fastest, no Moiré)
        if depth < depth_buffer[y_center, x_center]:
            image[y_center, x_center, :3] = voxel.color[:3]
            image[y_center, x_center, 3] = 255
            depth_buffer[y_center, x_center] = depth
        
        # Method 2: Small anti-aliased circle (better quality)
        # draw_anti_aliased_circle(image, depth_buffer, x_center, y_center, 
        #                         voxel.color, depth, radius=1)
    
    return image

@njit
def draw_anti_aliased_circle(image, depth_buffer, center_x, center_y, color, depth, radius=1):
    """Draw an anti-aliased circle to reduce Moiré patterns"""
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = center_x + dx
            y = center_y + dy
            
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                # Calculate distance from center
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist <= radius:
                    # Calculate alpha based on distance from edge
                    alpha = max(0.0, min(1.0, 1.0 - (dist / radius)))
                    
                    if depth < depth_buffer[y, x]:
                        # Blend with existing color
                        current_color = image[y, x]
                        new_r = int(color[0] * alpha + current_color[0] * (1 - alpha))
                        new_g = int(color[1] * alpha + current_color[1] * (1 - alpha))
                        new_b = int(color[2] * alpha + current_color[2] * (1 - alpha))
                        
                        image[y, x, 0] = new_r
                        image[y, x, 1] = new_g
                        image[y, x, 2] = new_b
                        image[y, x, 3] = 255
                        depth_buffer[y, x] = depth

class Camera:
    def __init__(self):
        self.yaw = 0.0      # Rotation around Y axis
        self.pitch = 0.0    # Rotation around X axis
        self.roll = 0.0     # Rotation around Z axis
        self.azimuth = 0.0  # Azimuth angle for orbital camera
        self.distance = 1.0  # Distance from center (zoom)
        self.auto_rotate = False
        self.rotation_speed = math.radians(1)  # 1 degree per second

    def update_rotation(self, dt):
        """Update rotation if auto-rotate is enabled"""
        if self.auto_rotate:
            self.yaw += self.rotation_speed * dt

def generate_planet(system, camera):
    """Generate the planet and return the texture data"""
    # Render the scene with current camera rotation
    image_data = render_orthographic(system, camera.pitch, camera.yaw, camera.roll, camera.distance)
    
    # Convert to a format suitable for DPG
    # DPG expects a flat array of floats in the range [0, 1]
    texture_data = np.zeros((system.height, system.width, 4), dtype=np.float32)
    texture_data[:, :, 0] = image_data[:, :, 0].astype(np.float32) / 255.0  # R
    texture_data[:, :, 1] = image_data[:, :, 1].astype(np.float32) / 255.0  # G
    texture_data[:, :, 2] = image_data[:, :, 2].astype(np.float32) / 255.0  # B
    texture_data[:, :, 3] = image_data[:, :, 3].astype(np.float32) / 255.0  # A
    
    return texture_data.flatten()

def update_texture(system, camera):
    """Update the texture with the current camera rotation"""
    texture_data = generate_planet(system, camera)
    dpg.set_value("planet_texture", texture_data)

def dpgmain():
    # Initialize Dear PyGui
    dpg.create_context()
    dpg.create_viewport(title='Voxel Planet', width=800, height=800)
    
    # Create the voxel system
    width, height = 800, 800
    system = VoxelSystem(width, height)
    
    # Create a planet with tectonic plates
    center = np.array([0.0, 0.0, 0.0])
    radius = 100.0
    
    print("Creating planet with tectonic plates...")
    start_time = time.time()
    create_planet_with_tectonics(system, center, radius, 500000, 12)
    print(f"Created {len(system.voxels)} voxels in {time.time() - start_time:.2f} seconds")
    
    # Create camera
    camera = Camera()
    
    # Generate the planet texture
    texture_data = generate_planet(system, camera)
    
    # Create texture registry and add texture
    with dpg.texture_registry():
        dpg.add_raw_texture(width=width, height=height, default_value=texture_data, 
                           format=dpg.mvFormat_Float_rgba, tag="planet_texture")
    
    # Create main window with the texture
    with dpg.window(label="Voxel Planet", tag="primary_window", width=800, height=800):
        with dpg.group(horizontal=True):
            # Image display
            dpg.add_image("planet_texture", width=800, height=800)
            
            # Control panel
            with dpg.group(width=200, height=200):
                # Rotation sliders
                dpg.add_text("Rotation Controls")
                dpg.add_slider_float(label="Yaw", min_value=-3.14, max_value=3.14, default_value=0.0, 
                                    callback=lambda s, a: update_camera_rotation(sender=s, app_data=a, system=system, camera=camera, rotation_type='yaw'))
                dpg.add_slider_float(label="Pitch", min_value=-3.14, max_value=3.14, default_value=0.0,
                                    callback=lambda s, a: update_camera_rotation(sender=s, app_data=a, system=system, camera=camera, rotation_type='pitch'))
                dpg.add_slider_float(label="Roll", min_value=-3.14, max_value=3.14, default_value=0.0,
                                    callback=lambda s, a: update_camera_rotation(sender=s, app_data=a, system=system, camera=camera, rotation_type='roll'))
                
                # Azimuth and distance sliders
                dpg.add_text("View Controls")
                dpg.add_slider_float(label="Azimuth", min_value=0.0, max_value=6.28, default_value=0.0,
                                    callback=lambda s, a: update_camera_rotation(sender=s, app_data=a, system=system, camera=camera, rotation_type='azimuth'))
                dpg.add_slider_float(label="Distance", min_value=0.1, max_value=10.0, default_value=1.0,
                                    callback=lambda s, a: update_camera_rotation(sender=s, app_data=a, system=system, camera=camera, rotation_type='distance'))
                
                # Auto-rotation button
                dpg.add_text("Auto Rotation")
                dpg.add_button(label="Toggle Auto Rotation", 
                              callback=lambda: toggle_auto_rotation(system, camera))
                
                # Reset button
                dpg.add_button(label="Reset View", 
                              callback=lambda: reset_view(system, camera))
    
    # Set primary window and show viewport
    dpg.set_primary_window("primary_window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Start the rendering loop with time tracking for auto-rotation
    last_time = time.time()
    while dpg.is_dearpygui_running():
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Update auto-rotation if enabled
        if camera.auto_rotate:
            camera.update_rotation(dt)
            update_texture(system, camera)
        
        dpg.render_dearpygui_frame()
    
    # Cleanup
    dpg.destroy_context()

def update_camera_rotation(sender, app_data, system, camera, rotation_type):
    """Update camera rotation based on slider changes"""
    if rotation_type == 'yaw':
        camera.yaw = app_data
    elif rotation_type == 'pitch':
        camera.pitch = app_data
    elif rotation_type == 'roll':
        camera.roll = app_data
    elif rotation_type == 'azimuth':
        camera.azimuth = app_data
    elif rotation_type == 'distance':
        camera.distance = app_data
    
    update_texture(system, camera)

def toggle_auto_rotation(system, camera):
    """Toggle auto-rotation on/off"""
    camera.auto_rotate = not camera.auto_rotate
    update_texture(system, camera)

def reset_view(system, camera):
    """Reset camera to default view"""
    camera.yaw = 0.0
    camera.pitch = 0.0
    camera.roll = 0.0
    camera.azimuth = 0.0
    camera.distance = 1.0
    camera.auto_rotate = False
    
    # Update sliders to match
    dpg.set_value("Yaw", 0.0)
    dpg.set_value("Pitch", 0.0)
    dpg.set_value("Roll", 0.0)
    dpg.set_value("Azimuth", 0.0)
    dpg.set_value("Distance", 1.0)
    
    update_texture(system, camera)

if __name__ == "__main__":
    dpgmain()