import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider
import pickle
import sys
import math

PLANET_RADIUS_KM = 6371.0
ORBITAL_DISTANCE_AU = 1.0
AXIAL_TILT_DEGREES = 23.5

def lat_lon_to_cartesian(lat, lon, radius):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

def grid_index_to_lat_lon(lat_index, lon_index, lat_resolution, lon_resolution):
    latitude = (lat_index / lat_resolution) * 180.0 - 90.0
    longitude = (lon_index / lon_resolution) * 360.0 - 180.0
    return latitude, longitude

def lat_lon_to_grid_index(lat, lon, lat_resolution, lon_resolution):
    lat_index = int(((lat + 90.0) / 180.0) * lat_resolution)
    lon_index = int(((lon + 180.0) / 360.0) * lon_resolution)
    return lat_index, lon_index

def haversine_distance(lat1, lon1, lat2, lon2, radius=PLANET_RADIUS_KM):
    """
    Calculate the great-circle distance between two points on a sphere using the Haversine formula.
    Lat and Lon are in degrees. Radius is in km by default.
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = radius * c
    return distance

class Plate:
    def __init__(self, plate_id, center_x, center_y, movement_x, movement_y):
        self.plate_id = plate_id
        self.center_x = center_x
        self.center_y = center_y
        self.movement_x = movement_x
        self.movement_y = movement_y
        self.cells = set()

    def move(self):
        self.center_x += self.movement_x
        self.center_y += self.movement_y
        lat_resolution = world.shape[0]
        lon_resolution = world.shape[1]
        self.center_x = self.center_x % lat_resolution
        self.center_y = self.center_y % lon_resolution

def generate_initial_world_3d(lat_resolution=50, lon_resolution=100, radius_elevation=10, num_plates=5):
    world = np.zeros((lat_resolution, lon_resolution), dtype=float)
    for x in range(lat_resolution):
        for y in range(lon_resolution):
            world[x, y] = random.uniform(-radius_elevation, radius_elevation)

    plates = []
    for i in range(num_plates):
        center_x = random.randint(0, lat_resolution - 1)
        center_y = random.randint(0, lon_resolution - 1)
        movement_x = random.uniform(-0.1, 0.1)
        movement_y = random.uniform(-0.1, 0.1)
        plates.append(Plate(i + 1, center_x, center_y, movement_x, movement_y))

    plate_assignment = np.zeros_like(world, dtype=int)
    for x in range(lat_resolution):
        for y in range(lon_resolution):
            closest_plate_id = 0
            min_distance_sq = float('inf')
            for plate in plates:
                distance_sq = (x - plate.center_x)**2 + (y - plate.center_y)**2
                if distance_sq < min_distance_sq:
                    min_distance_sq = distance_sq
                    closest_plate_id = plate.plate_id
            plate_assignment[x, y] = closest_plate_id
            if closest_plate_id != 0:
                plates[closest_plate_id-1].cells.add((x,y))

    return world, plates, plate_assignment

def precalculate_distances(lat_resolution, lon_resolution):
    """Pre-calculates Haversine distances between all grid points."""
    distance_matrix = np.zeros((lat_resolution, lon_resolution, lat_resolution, lon_resolution), dtype=float)
    for lat_index1 in range(lat_resolution):
        for lon_index1 in range(lon_resolution):
            lat1, lon1 = grid_index_to_lat_lon(lat_index1, lon_index1, lat_resolution, lon_resolution)
            for lat_index2 in range(lat_resolution):
                for lon_index2 in range(lon_resolution):
                    if (lat_index1 == lat_index2 and lon_index1 == lon_index2): #Distance to self is 0
                        continue
                    lat2, lon2 = grid_index_to_lat_lon(lat_index2, lon_index2, lat_resolution, lon_resolution)
                    distance_degrees = haversine_distance(lat1, lon1, lat2, lon2, radius=1) # Angular distance in radians
                    distance_degrees = np.degrees(distance_degrees) # Convert to degrees
                    distance_matrix[lat_index1, lon_index1, lat_index2, lon_index2] = distance_degrees
    return distance_matrix

def get_spherical_neighbors(lat_index, lon_index, lat_resolution, lon_resolution, max_distance_degrees, distance_matrix):
    """Finds neighbors within a maximum spherical distance using pre-calculated matrix."""
    neighbors = []
    for ni in range(lat_resolution):
        for nj in range(lon_resolution):
            if ni == lat_index and nj == lon_index:
                continue # Skip self

            distance_degrees = distance_matrix[lat_index, lon_index, ni, nj]
            if distance_degrees <= max_distance_degrees:
                neighbors.append((ni, nj))
    return neighbors

def simulate_plate_tectonics_3d(world, plates, plate_assignment, step_size=1.0, lat_resolution=30, lon_resolution=60, neighbor_distance_degrees=10.0, distance_matrix=None):
    for plate in plates:
        plate.move()

    boundary_effect_grid = np.zeros_like(world, dtype=float) # Re-initialize for each step
    for x in range(lat_resolution):
        for y in range(lon_resolution):
            current_plate_id = plate_assignment[x, y]
            if current_plate_id == 0:
                continue

            neighbors = get_spherical_neighbors(x, y,  lat_resolution, lon_resolution, neighbor_distance_degrees, distance_matrix)

            for nx, ny in neighbors:
                neighbor_plate_id = plate_assignment[nx, ny]
                if neighbor_plate_id != current_plate_id and neighbor_plate_id != 0:
                    # Simplified Boundary Interaction: Elevation Adjustment based on plate difference
                    if plates[current_plate_id-1].plate_id < plates[neighbor_plate_id-1].plate_id: #Example interaction
                        boundary_effect_grid[x, y] += 0.001 * step_size # One plate pushes up
                        boundary_effect_grid[nx, ny] -= 0.001 * step_size # Other plate pushes down (or vice versa)


    world += boundary_effect_grid
    world = gaussian_filter(world, sigma=0.5) # Keep smoothing, but adjust if needed

    # Re-assign plate_assignment based on moved plates (VERY IMPORTANT for boundary updates)
    new_plate_assignment = np.zeros_like(world, dtype=int)
    for x in range(lat_resolution):
        for y in range(lon_resolution):
            closest_plate_id = 0
            min_distance_sq = float('inf')
            for plate in plates:
                distance_sq = (x - plate.center_x)**2 + (y - plate.center_y)**2
                if distance_sq < min_distance_sq:
                    min_distance_sq = distance_sq
                    closest_plate_id = plate.plate_id
            new_plate_assignment[x, y] = closest_plate_id
    plate_assignment = new_plate_assignment

    return world, plates, plate_assignment

def calculate_solar_radiation(lat, day_of_year):
    solar_constant = 1361
    axial_tilt_rad = np.radians(AXIAL_TILT_DEGREES)
    declination = axial_tilt_rad * np.sin(2 * np.pi * (day_of_year - 81) / 365)
    lat_rad = np.radians(lat)
    solar_zenith_angle = np.arccos(np.sin(lat_rad) * np.sin(declination) + np.cos(lat_rad) * np.cos(declination))
    radiation = solar_constant * np.cos(solar_zenith_angle)
    radiation = max(radiation, 0)
    return radiation

def calculate_temperature(lat, day_of_year, elevation=0):
    radiation = calculate_solar_radiation(lat, day_of_year)
    sea_level_temp = 15
    temp_drop_per_km = 6.5
    temperature = sea_level_temp + (radiation / 1361 * 30) - (elevation / 10) * temp_drop_per_km
    return temperature

def visualize_world_3d(world_data, ax, lat_resolution, lon_resolution):
    ax.clear()
    latitudes = np.linspace(-90, 90, lat_resolution)
    longitudes = np.linspace(-180, 180, lon_resolution)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    radius_grid = PLANET_RADIUS_KM + world_data
    x, y, z = lat_lon_to_cartesian(lat_grid, lon_grid, radius_grid)

    ax.plot_surface(x, y, z, facecolors=plt.cm.terrain(world_data / (2*world_data.max()) + 0.5),
                    rstride=1, cstride=1, linewidth=0, antialiased=False)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect('equal')
    ax.view_init(elev=30, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def save_simulation(filename, world, plates, plate_assignment, sim_params):
    """Saves the simulation state to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump({
            'world': world,
            'plates': plates,
            'plate_assignment': plate_assignment,
            'sim_params': sim_params
        }, f)
    print(f"Simulation saved to '{filename}'")

def load_simulation(filename):
    """Loads the simulation state from a file using pickle."""
    try:
        with open(filename, 'rb') as f:
            saved_state = pickle.load(f)
            return (
                saved_state['world'],
                saved_state['plates'],
                saved_state['plate_assignment'],
                saved_state['sim_params']
            )
    except FileNotFoundError:
        print(f"Save file '{filename}' not found. Starting a new simulation.")
        return None

if __name__ == "__main__":
    lat_resolution = 50 # Reduced for performance testing, increase later
    lon_resolution = 50 # Reduced for performance testing, increase later
    radius_elevation = 10
    num_plates = 5
    num_steps = 50
    step_size = 0.5
    neighbor_distance_degrees = 5.0 # Reduced neighbor distance to 5 degrees

    sim_params = {
        'lat_resolution': lat_resolution,
        'lon_resolution': lon_resolution,
        'radius_elevation': radius_elevation,
        'num_plates': num_plates,
        'num_steps': num_steps,
        'step_size': step_size,
        'neighbor_distance_degrees': neighbor_distance_degrees
    }

    save_file = "simulation_save.pkl"

    distance_matrix = precalculate_distances(lat_resolution, lon_resolution) # Pre-calculate distances

    if len(sys.argv) > 1:
        load_file = sys.argv[1]
        loaded_data = load_simulation(load_file)
        if loaded_data:
            world, plates, plate_assignment, loaded_params = loaded_data
            lat_resolution = loaded_params['lat_resolution']
            lon_resolution = loaded_params['lon_resolution']
            radius_elevation = loaded_params['radius_elevation']
            num_plates = loaded_params['num_plates']
            step_size = loaded_params['step_size']
            neighbor_distance_degrees = loaded_params['neighbor_distance_degrees']
            sim_params['neighbor_distance_degrees'] = neighbor_distance_degrees
            distance_matrix = precalculate_distances(lat_resolution, lon_resolution) # Re-calculate distance matrix on load (resolution might have changed)
            print(f"Simulation loaded from '{load_file}'.")
        else:
            world, plates, plate_assignment = generate_initial_world_3d(lat_resolution, lon_resolution, radius_elevation, num_plates)
            print("Starting a new simulation.")
    else:
        world, plates, plate_assignment = generate_initial_world_3d(lat_resolution, lon_resolution, num_plates=num_plates)
        print("Starting a new simulation.")


    world_history = []
    for step in range(num_steps):
        print(f"Simulating step {step + 1}/{num_steps}")
        world, plates, plate_assignment = simulate_plate_tectonics_3d(
            world, plates, plate_assignment, step_size=step_size,
            lat_resolution=lat_resolution, lon_resolution=lon_resolution,
            neighbor_distance_degrees=neighbor_distance_degrees,
            distance_matrix=distance_matrix # Pass distance_matrix
        )
        world_history.append(world.copy())

    print("Simulation complete. Preparing interactive visualization...")

    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    current_step = 0
    visualize_world_3d(world_history[current_step], ax_3d, lat_resolution, lon_resolution)
    ax_3d.set_title(f'World Evolution - Step {current_step + 1}')

    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    step_slider = Slider(
        ax=ax_slider,
        label='Step',
        valmin=0,
        valmax=num_steps - 1,
        valinit=current_step,
        valstep=1
    )

    def update(val):
        step = int(step_slider.val)
        visualize_world_3d(world_history[step], ax_3d, lat_resolution, lon_resolution)
        ax_3d.set_title(f'World Evolution - Step {step + 1}')
        fig.canvas.draw_idle()

    step_slider.on_changed(update)

    plt.show()

    save_simulation(save_file, world, plates, plate_assignment, sim_params)
    print(f"Simulation automatically saved to '{save_file}'.")