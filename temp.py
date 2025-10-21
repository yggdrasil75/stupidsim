import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.widgets import Button

# Set up the figure and 3D axis
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Create a sphere
def create_sphere(radius=1, resolution=50):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# Generate sphere
sphere_radius = 1
x_sphere, y_sphere, z_sphere = create_sphere(sphere_radius)

# Plot the sphere with transparency
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.2)

# Define tectonic plates as regions on the sphere
def create_plate_boundaries():
    plates = []
    
    # Plate 1: Pacific-like plate covering a large area
    plate1_u = np.linspace(0.2*np.pi, 1.8*np.pi, 10)
    plate1_v = np.linspace(0.1*np.pi, 0.9*np.pi, 10)
    plates.append((plate1_u, plate1_v, 'Pacific'))
    
    # Plate 2: Eurasian-like plate
    plate2_u = np.linspace(1.6*np.pi, 2.4*np.pi, 8)
    plate2_v = np.linspace(0.4*np.pi, 0.8*np.pi, 8)
    plates.append((plate2_u, plate2_v, 'Eurasian'))
    
    # Plate 3: African-like plate
    plate3_u = np.linspace(0.8*np.pi, 1.4*np.pi, 8)
    plate3_v = np.linspace(0.3*np.pi, 0.7*np.pi, 8)
    plates.append((plate3_u, plate3_v, 'African'))
    
    # Plate 4: South American-like plate
    plate4_u = np.linspace(1.2*np.pi, 1.8*np.pi, 6)
    plate4_v = np.linspace(0.1*np.pi, 0.5*np.pi, 6)
    plates.append((plate4_u, plate4_v, 'South American'))
    
    return plates

# Create plate boundaries
plates = create_plate_boundaries()

# Define colors for plates
plate_colors = ['red', 'green', 'blue', 'orange']

# Store plate data for animation
plate_data = []
plate_centers = []
angular_velocities = []
linear_velocities = []

# Initialize selected plate
selected_plate_idx = 0

# Plot plates and calculate velocities
for i, (u_range, v_range, name) in enumerate(plates):
    # Create grid for this plate
    u, v = np.meshgrid(u_range, v_range)
    
    # Convert to Cartesian coordinates
    x = sphere_radius * np.cos(u) * np.sin(v)
    y = sphere_radius * np.sin(u) * np.sin(v)
    z = sphere_radius * np.cos(v)
    
    # Store plate data
    plate_data.append((x, y, z, name))
    
    # Calculate plate center
    center_u = np.mean(u_range)
    center_v = np.mean(v_range)
    center_x = sphere_radius * np.cos(center_u) * np.sin(center_v)
    center_y = sphere_radius * np.sin(center_u) * np.sin(center_v)
    center_z = sphere_radius * np.cos(center_v)
    plate_centers.append((center_x, center_y, center_z))
    
    # Define angular velocity (radians per time unit)
    # Each plate has different rotation axis and speed
    if i == 0:  # Pacific plate - faster rotation
        angular_velocity = np.array([0.02, 0.03, 0.01])
    elif i == 1:  # Eurasian plate - slower rotation
        angular_velocity = np.array([0.005, 0.01, 0.008])
    elif i == 2:  # African plate
        angular_velocity = np.array([-0.01, 0.015, -0.005])
    else:  # South American plate
        angular_velocity = np.array([0.008, -0.006, 0.012])
    
    angular_velocities.append(angular_velocity)
    
    # Calculate linear velocity at the center: v = ω × r
    r = np.array([center_x, center_y, center_z])
    linear_velocity = np.cross(angular_velocity, r)
    linear_velocities.append(linear_velocity)

# Function to plot plates with velocity vectors
def plot_plates(highlight_idx=None):
    ax.cla()
    
    # Replot the sphere
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.2)
    
    # Plot each plate
    for i, (x, y, z, name) in enumerate(plate_data):
        alpha = 0.7 if highlight_idx is None or i == highlight_idx else 0.2
        ax.plot_surface(x, y, z, color=plate_colors[i], alpha=alpha, label=name)
        
        # Add plate name
        center_x, center_y, center_z = plate_centers[i]
        ax.text(center_x*1.2, center_y*1.2, center_z*1.2, name, fontsize=9)
    
    # If a plate is selected, highlight it and show its velocity vectors
    if highlight_idx is not None:
        i = highlight_idx
        center_x, center_y, center_z = plate_centers[i]
        
        # Plot angular velocity vector (rotation axis)
        av_magnitude = np.linalg.norm(angular_velocities[i])
        av_direction = angular_velocities[i] / av_magnitude if av_magnitude > 0 else angular_velocities[i]
        av_scale = 0.3  # Scale for visualization
        ax.quiver(center_x, center_y, center_z, 
                  av_direction[0], av_direction[1], av_direction[2], 
                  length=av_scale*av_magnitude, color='black', linewidth=3,
                  label='Angular Velocity', arrow_length_ratio=0.2)
        
        # Plot linear velocity vector
        lv_magnitude = np.linalg.norm(linear_velocities[i])
        lv_direction = linear_velocities[i] / lv_magnitude if lv_magnitude > 0 else linear_velocities[i]
        lv_scale = 0.5  # Scale for visualization
        ax.quiver(center_x, center_y, center_z, 
                  lv_direction[0], lv_direction[1], lv_direction[2], 
                  length=lv_scale*lv_magnitude, color='purple', linewidth=3,
                  label='Linear Velocity', arrow_length_ratio=0.2)
        
        # Add velocity information as text
        vel_text = f'{plates[i][2]} Plate:\nAngular Vel: {angular_velocities[i]}\nLinear Vel: {linear_velocities[i]}'
        ax.text2D(0.05, 0.95, vel_text, transform=ax.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Set axis limits and labels
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    title = 'Tectonic Plates with Velocity Vectors'
    if highlight_idx is not None:
        title += f'\nSelected: {plates[highlight_idx][2]} Plate'
    ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend for velocity vectors if a plate is selected
    if highlight_idx is not None:
        ax.legend(['Angular Velocity', 'Linear Velocity'], loc='upper right')

# Initial plot
plot_plates()

# Create button area
plt.subplots_adjust(bottom=0.2)
button_ax = []
buttons = []

# Create buttons for each plate
for i, (_, _, name) in enumerate(plates):
    ax_button = plt.axes([0.1 + i*0.2, 0.05, 0.15, 0.075])
    button = Button(ax_button, f'Select {name}', color=plate_colors[i], hovercolor='lightgray')
    button_ax.append(ax_button)
    buttons.append(button)

# Create "Show All" button
ax_all = plt.axes([0.1, 0.15, 0.8, 0.075])
button_all = Button(ax_all, 'Show All Plates', color='lightgray', hovercolor='white')

# Button callbacks
def create_callback(idx):
    def callback(event):
        global selected_plate_idx
        selected_plate_idx = idx
        plot_plates(highlight_idx=idx)
        plt.draw()
    return callback

def show_all_callback(event):
    plot_plates()
    plt.draw()

# Attach callbacks to buttons
for i, button in enumerate(buttons):
    button.on_clicked(create_callback(i))

button_all.on_clicked(show_all_callback)

# Animation function to show plate motion over time
def update(frame):
    ax.cla()
    
    # Replot the sphere
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.2)
    
    # Update and replot each plate
    for i, ((u_range, v_range, name), angular_velocity) in enumerate(zip(plates, angular_velocities)):
        # Rotate the plate coordinates
        rotation_angle = np.linalg.norm(angular_velocity) * frame / 10
        if rotation_angle != 0:
            rotation_axis = angular_velocity / np.linalg.norm(angular_velocity)
            
            # Create rotation matrix
            u = rotation_axis
            c = np.cos(rotation_angle)
            s = np.sin(rotation_angle)
            R = np.array([
                [c + u[0]**2*(1-c), u[0]*u[1]*(1-c) - u[2]*s, u[0]*u[2]*(1-c) + u[1]*s],
                [u[1]*u[0]*(1-c) + u[2]*s, c + u[1]**2*(1-c), u[1]*u[2]*(1-c) - u[0]*s],
                [u[2]*u[0]*(1-c) - u[1]*s, u[2]*u[1]*(1-c) + u[0]*s, c + u[2]**2*(1-c)]
            ])
            
            # Apply rotation to the original plate coordinates
            u, v = np.meshgrid(u_range, v_range)
            x = sphere_radius * np.cos(u) * np.sin(v)
            y = sphere_radius * np.sin(u) * np.sin(v)
            z = sphere_radius * np.cos(v)
            
            # Reshape for matrix multiplication
            original_coords = np.vstack([x.flatten(), y.flatten(), z.flatten()])
            rotated_coords = R @ original_coords
            
            # Reshape back to grid
            x_rotated = rotated_coords[0, :].reshape(x.shape)
            y_rotated = rotated_coords[1, :].reshape(y.shape)
            z_rotated = rotated_coords[2, :].reshape(z.shape)
            
            # Plot the rotated plate
            alpha = 0.7 if i == selected_plate_idx else 0.2
            ax.plot_surface(x_rotated, y_rotated, z_rotated, color=plate_colors[i], alpha=alpha)
            
            # Calculate new center
            center_u = np.mean(u_range) + angular_velocity[0] * frame / 100
            center_v = np.mean(v_range) + angular_velocity[1] * frame / 100
            center_x = sphere_radius * np.cos(center_u) * np.sin(center_v)
            center_y = sphere_radius * np.sin(center_u) * np.sin(center_v)
            center_z = sphere_radius * np.cos(center_v)
            
            # Add plate name
            ax.text(center_x*1.2, center_y*1.2, center_z*1.2, name, fontsize=9)
            
            # If this is the selected plate, show velocity vectors
            if i == selected_plate_idx:
                # Plot angular velocity vector
                av_magnitude = np.linalg.norm(angular_velocity)
                av_direction = angular_velocity / av_magnitude if av_magnitude > 0 else angular_velocity
                ax.quiver(center_x, center_y, center_z, 
                          av_direction[0], av_direction[1], av_direction[2], 
                          length=0.3*av_magnitude, color='black', linewidth=3, arrow_length_ratio=0.2)
                
                # Plot linear velocity vector
                r = np.array([center_x, center_y, center_z])
                current_linear_velocity = np.cross(angular_velocity, r)
                lv_magnitude = np.linalg.norm(current_linear_velocity)
                lv_direction = current_linear_velocity / lv_magnitude if lv_magnitude > 0 else current_linear_velocity
                ax.quiver(center_x, center_y, center_z, 
                          lv_direction[0], lv_direction[1], lv_direction[2], 
                          length=0.5*lv_magnitude, color='purple', linewidth=3, arrow_length_ratio=0.2)
                
                # Add velocity information as text
                vel_text = f'{name} Plate:\nAngular Vel: {angular_velocity}\nLinear Vel: {current_linear_velocity}'
                ax.text2D(0.05, 0.95, vel_text, transform=ax.transAxes, fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Set axis limits and labels
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Tectonic Plate Motion - {plates[selected_plate_idx][2]} Plate\nFrame {frame}')
    
    # Add legend if a plate is selected
    ax.legend(['Angular Velocity', 'Linear Velocity'], loc='upper right')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

# Create animation
ani = FuncAnimation(fig, update, frames=50, interval=200, blit=False, repeat=True)

plt.tight_layout()
plt.show()

# Print velocity information
print("Plate Velocity Information:")
print("--------------------------")
for i, (_, _, name) in enumerate(plates):
    print(f"{name} Plate:")
    print(f"  Angular Velocity: {angular_velocities[i]} radians/time unit")
    print(f"  Linear Velocity at center: {linear_velocities[i]} distance units/time unit")
    print(f"  Angular Speed: {np.linalg.norm(angular_velocities[i]):.4f} rad/time unit")
    print(f"  Linear Speed: {np.linalg.norm(linear_velocities[i]):.4f} distance units/time unit")
    print()