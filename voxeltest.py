import dearpygui.dearpygui as dpg
import numpy as np
from holder.voxelmesh import VoxelGrid, project_voxels_2d, rasterize_voxels
import time
import math

def create_sphere_voxels(resolution, radius=0.5):
    voxel_data = np.zeros(resolution, dtype=np.uint8)
    color_data = np.zeros((*resolution, 3), dtype=np.uint8)  # RGB colors
    
    center = np.array(resolution) / 2
    
    for x in range(resolution[0]):
        for y in range(resolution[1]):
            for z in range(resolution[2]):
                nx = (x - center[0]) / resolution[0]
                ny = (y - center[1]) / resolution[1]
                nz = (z - center[2]) / resolution[2]
                
                dist = np.sqrt(nx**2 + ny**2 + nz**2)
                if dist <= radius:
                    voxel_data[x, y, z] = 1
                    # Gradient color based on position
                    color_data[x, y, z] = [
                        int(255 * (0.5 + nx/2)),
                        int(255 * (0.5 + ny/2)),
                        int(255 * (0.5 + nz/2))
                    ]
    
    return voxel_data, color_data

class CameraController:
    def __init__(self):
        self.radius = 3.0  # Distance from center
        self.angle = 0.0  # Current angle in radians
        self.spinning = False
        self.last_time = time.time()
        
    def get_camera_position(self):
        # Calculate camera position in circular orbit
        x = self.radius * math.sin(self.angle)
        z = -self.radius * math.cos(self.angle)
        return x, 0.0, z
    
    def update(self):
        if self.spinning:
            current_time = time.time()
            delta_time = current_time - self.last_time
            self.last_time = current_time
            
            # Update angle at 1 degree per second
            self.angle += math.radians(1) * delta_time
            
            # Get new camera position
            x, y, z = self.get_camera_position()
            
            # Update GUI sliders
            dpg.set_value("cam_x", x)
            dpg.set_value("cam_y", y)
            dpg.set_value("cam_z", z)
            return True  # Indicate that an update is needed
        return False

def update_view():
    # Get current camera position
    cam_x = dpg.get_value("cam_x")
    cam_y = dpg.get_value("cam_y")
    cam_z = dpg.get_value("cam_z")
    
    eye = np.array([cam_x, cam_y, cam_z])
    lookat = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    # Render the voxel grid
    screen_positions, colors = project_voxels_2d([grid], eye, lookat, up)
    image = rasterize_voxels(screen_positions, colors, 800, 600)
    
    # Normalize image to 0-1 range for DPG
    normalized_image = image.astype(np.float32) / 255.0
    
    # Update the texture
    dpg.set_value("voxel_texture", normalized_image)

def toggle_spin():
    camera_controller.spinning = not camera_controller.spinning
    if camera_controller.spinning:
        dpg.configure_item("spin_button", label="Stop Spinning")
        # Initialize camera position
        x, y, z = camera_controller.get_camera_position()
        dpg.set_value("cam_x", x)
        dpg.set_value("cam_y", y)
        dpg.set_value("cam_z", z)
        camera_controller.last_time = time.time()
    else:
        dpg.configure_item("spin_button", label="Start Spinning")

if __name__ == "__main__":
    # Calculate resolution for approximately 1000 voxels
    # We'll use a cubic resolution (10x10x10 = 1000)
    resolution = (100, 100, 100)
    
    # Create sphere voxel data
    voxel_data, color_data = create_sphere_voxels(resolution)
    
    # Calculate scale so the total size is about 2 units
    total_size = 2.0
    scale = total_size / max(resolution)

    grid = VoxelGrid(
        id=1,
        resolution=resolution,
        origin=np.array([-1, -1, -1], dtype=np.float32),
        scale=scale,
        data=voxel_data,
        color=color_data  # Pass the full color array
    )

    # Generate LODs
    grid.generate_lods(levels=2)

    # Initialize Dear PyGui
    dpg.create_context()
    dpg.create_viewport(title='Voxel Sphere Viewer', width=800, height=600)

    # Create camera controller
    camera_controller = CameraController()

    with dpg.window(label="Voxel Viewer", tag='primary', width=800, height=600):
        # Create a texture to display our rendered image
        with dpg.texture_registry():
            dpg.add_raw_texture(width=800, height=600, default_value=np.zeros((600, 800, 3), dtype=np.float32), 
                                format=dpg.mvFormat_Float_rgb, tag="voxel_texture")

        dpg.add_image("voxel_texture", width=800, height=600)

        # Add controls for camera position
        with dpg.group(horizontal=True):
            dpg.add_slider_float(label="Camera X", tag="cam_x", default_value=0, min_value=-5, max_value=5)
            dpg.add_slider_float(label="Camera Y", tag="cam_y", default_value=0, min_value=-5, max_value=5)
            dpg.add_slider_float(label="Camera Z", tag="cam_z", default_value=-3, min_value=-10, max_value=-1)
        
        dpg.add_button(label="Update View", tag="update_button", callback=update_view)
        
        # Add spin button
        dpg.add_button(label="Start Spinning", tag="spin_button", callback=toggle_spin)

    # Initial render
    update_view()

    # Start the Dear PyGui application
    dpg.setup_dearpygui()
    dpg.set_primary_window('primary', True)
    dpg.show_viewport()

    # Main loop
    while dpg.is_dearpygui_running():
        # Update camera if spinning
        needs_update = camera_controller.update()
        
        # If camera was updated or there are other events, update the view
        #if needs_update or dpg.is_mouse_button_dragging() or dpg.is_key_down(dpg.mvKey_Control):
        update_view()
        
        # Render frame
        dpg.render_dearpygui_frame()

    dpg.destroy_context()