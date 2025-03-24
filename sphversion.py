import random
from matplotlib import pyplot as plt
import numpy as np

PLANET_RADIUS_KM = 1000

class Particle:
    def __init__(self, position, plate_id, mass=1.0):
        self.position = np.array(position, dtype=float) # 3D vector (x, y, z)
        self.plate_id = plate_id
        self.mass = mass
        self.density = 0.0
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float) # 3D velocity
        self.elevation = 0.0 # Will be derived from position
        
def lat_lon_to_cartesian(lat, lon, radius):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

def generate_particle_sphere(num_particles, planet_radius=PLANET_RADIUS_KM):
    """Generates particles distributed on a sphere using Fibonacci sphere algorithm."""
    particles = []
    indices = np.arange(0, num_particles, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_particles)
    theta = np.pi * (1 + 5**0.5) * indices

    for i in range(num_particles):
        x = planet_radius * np.cos(theta[i]) * np.sin(phi[i])
        y = planet_radius * np.sin(theta[i]) * np.sin(phi[i])
        z = planet_radius * np.cos(phi[i])
        particles.append(Particle(position=[x, y, z], plate_id=0)) # Initial plate_id = 0
    return particles


class Plate:
    def __init__(self, plate_id, center_x, center_y, movement_x, movement_y):
        self.plate_id = plate_id
        self.center_x = center_x
        self.center_y = center_y
        self.movement_x = movement_x
        self.movement_y = movement_y

    def move(self):
        self.center_x += self.movement_x
        self.center_y += self.movement_y
        #lat_resolution = world.shape[0]
        #lon_resolution = world.shape[1]
        #self.center_x = self.center_x % lat_resolution
        #self.center_y = self.center_y % lon_resolution

def assign_particles_to_plates(particles, plates):
    """Assigns particles to plates based on proximity to plate centers (Cartesian distance)."""
    for particle in particles:
        closest_plate_id = 0
        min_distance_sq = float('inf')
        for plate in plates:
            plate_center_cartesian = lat_lon_to_cartesian(plate.center_x, plate.center_y, PLANET_RADIUS_KM) # Convert plate center to cartesian
            distance_sq = np.sum((particle.position - plate_center_cartesian)**2) # Cartesian distance
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest_plate_id = plate.plate_id
        particle.plate_id = closest_plate_id
    return particles

def simulate_plate_motion_sph(particles, plates, time_step): # time_step added
    """Moves particles based on their assigned plate's movement vector."""
    for plate in plates:
        # Convert plate 2D movement to 3D velocity (simplified - needs refinement later)
        plate_velocity = np.array([plate.movement_x, plate.movement_y, 0.0]) * 0.01 #Scale down velocity and make it 3D. Z component is 0 for now.

        for particle in particles:
            if particle.plate_id == plate.plate_id:
                particle.velocity += plate_velocity # Apply plate velocity to particle
                particle.position += particle.velocity * time_step # Update position based on velocity and time step

                # Basic constraint to keep particles roughly on the sphere surface (very simplified)
                particle.position /= np.linalg.norm(particle.position) # Normalize to unit vector
                particle.position *= PLANET_RADIUS_KM # Scale back to planet radius

    return particles

def visualize_particle_world_3d(particles, ax):
    ax.clear()
    positions = np.array([p.position for p in particles])
    elevations = np.array([p.elevation for p in particles]) # Or use plate_ids for color

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=plt.cm.terrain(elevations / (2*np.max(np.abs(elevations))) + 0.5), # Color by elevation
               marker='o', s=20, alpha=0.8) # Adjust marker size and alpha

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect('equal')
    ax.view_init(elev=30, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
	
if __name__ == "__main__":
    num_particles = 5000  # Start with a smaller number for testing, increase later
    num_plates = 5
    num_steps = 100
    time_step = 0.1 # Adjust time step as needed

    particles = generate_particle_sphere(num_particles)
    plates = []
    world_size = 100 #dummy world size for plate generation, not really used in SPH.
    for i in range(num_plates):
        center_x = random.randint(0, world_size - 1)
        center_y = random.randint(0, world_size - 1)
        movement_x = random.uniform(-0.01, 0.01) # Reduced movement speed for SPH initial tests
        movement_y = random.uniform(-0.01, 0.01)
        plates.append(Plate(i + 1, center_x, center_y, movement_x, movement_y))

    particles = assign_particles_to_plates(particles, plates)

    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    world_history_particles = [] # Store particle history

    for step in range(num_steps):
        print(f"Simulating step {step + 1}/{num_steps}")
        particles = simulate_plate_motion_sph(particles, plates, time_step)

        # Derive elevation from radial position (example - refine this later for more meaningful elevation)
        for p in particles:
            p.elevation = np.linalg.norm(p.position) - PLANET_RADIUS_KM

        world_history_particles.append([p.__dict__.copy() for p in particles]) # Store particle state (copy dicts)

        visualize_particle_world_3d(particles, ax_3d)
        ax_3d.set_title(f'Particle World - Step {step + 1}')
        plt.pause(0.01) # Or plt.show() for static image

    plt.show()