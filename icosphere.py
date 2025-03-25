from typing import Any
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import RadioButtons, Slider
import pickle, sys, math
from scipy.spatial import Delaunay

PLANET_RADIUS_KM = 6371.0
ORBITAL_DISTANCE_AU = 1.0
AXIAL_TILT_DEGREES = 23.5
MIN_PLATE_SPEED_CM_YR = 1.0  # ~1 cm/year (slow moving plates)
MAX_PLATE_SPEED_CM_YR = 10.0  # ~10 cm/year (fast moving plates)

def generate_icosphere(subdivisions=3, radius=1.0):
    """Generate an icosphere mesh with given number of subdivisions."""
    # Golden ratio
    t = (1.0 + math.sqrt(5.0)) / 2.0

    # Create initial icosahedron vertices
    vertices = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)
    ]

    # Normalize vertices to unit sphere
    vertices = [np.array(v)/np.linalg.norm(v) for v in vertices]

    # Create initial icosahedron faces
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]

    # Subdivide the mesh
    for _ in range(subdivisions):
        new_faces = []
        edge_vertices = {}

        for face in faces:
            # Get edge vertices
            edge_midpoints = []
            for i in range(3):
                v1, v2 = face[i], face[(i+1)%3]
                key = tuple(sorted((v1, v2)))
                if key not in edge_vertices:
                    # Create new vertex at midpoint
                    mid = (vertices[v1] + vertices[v2]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_vertices[key] = len(vertices)
                    vertices.append(mid)
                edge_midpoints.append(edge_vertices[key])

            # Create 4 new faces
            a, b, c = face
            d, e, f = edge_midpoints
            new_faces.extend([
                (a, d, f),
                (d, b, e),
                (f, e, c),
                (d, e, f)
            ])

        faces = new_faces

    # Convert to numpy arrays
    vertices = np.array(vertices) * radius
    faces = np.array(faces)

    return vertices, faces

def cartesian_to_lat_lon(x, y, z):
    """Convert cartesian coordinates to latitude/longitude."""
    lat = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

def lat_lon_to_cartesian(lat, lon, radius):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

def haversine_distance(lat1, lon1, lat2, lon2, radius=PLANET_RADIUS_KM):
    """Calculate great-circle distance between two points on a sphere."""
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
    def __init__(self, plate_id, center_vertex, movement_direction):
        self.plate_id = plate_id
        self.center_vertex = center_vertex
        # Initialize with a direction but no magnitude
        self.movement_direction = movement_direction / np.linalg.norm(movement_direction)
        self.speed_cm_yr = random.uniform(MIN_PLATE_SPEED_CM_YR, MAX_PLATE_SPEED_CM_YR)
        self.vertices = set()
        self.center_point = None

    def move(self, days_elapsed):
        """Move the plate based on real-world time scaling."""
        # Convert speed from cm/year to km/day
        speed_km_day = (self.speed_cm_yr / 100000) / 365.25
        movement_distance = speed_km_day * days_elapsed
        
        # Apply movement
        movement_vector = self.movement_direction * movement_distance
        self.center_point += movement_vector
        
        # Project back to sphere surface
        self.center_point = self.center_point / np.linalg.norm(self.center_point) * PLANET_RADIUS_KM

def generate_initial_world_spherical(subdivisions=3, radius=PLANET_RADIUS_KM, num_plates=5):
    """Generate initial world with spherical mesh and plates."""
    vertices, faces = generate_icosphere(subdivisions, radius)

    # Add random elevation to vertices using spherical harmonics for more natural distribution
    elevations = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        # Convert to spherical coordinates
        lat, lon = cartesian_to_lat_lon(*vertex)
        
        # Create more interesting initial elevations using noise
        noise = (np.sin(lon * 2) * np.cos(lat * 3) + 
                 np.sin(lon * 5) * np.cos(lat * 2)) * 5
        elevations[i] = noise + np.random.uniform(-2, 2)

    # Scale vertices with elevations
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis] * (radius + elevations[:, np.newaxis])

    # Create plates
    plates = []
    plate_assignment = np.zeros(len(vertices), dtype=int)

    # Generate random plate center locations on the sphere using Fibonacci sphere algorithm
    plate_centers = []
    for i in range(num_plates):
        y = 1 - (i / float(num_plates - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = math.pi * (3 - math.sqrt(5)) * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        plate_centers.append(np.array([x, y, z]) * PLANET_RADIUS_KM)

    for i in range(num_plates):
        center_point = plate_centers[i]
        
        # Create random movement direction (tangent to sphere)
        tangent = np.cross(center_point, np.random.randn(3))
        tangent = tangent / np.linalg.norm(tangent)
        
        plate = Plate(i + 1, -1, tangent)  # Note: speed is set in Plate.__init__
        plate.center_point = center_point.copy()
        plates.append(plate)

    # Assign vertices to nearest plate center
    for i, vertex in enumerate(vertices):
        min_dist = float('inf')
        closest_plate = None

        for plate in plates:
            dist = np.linalg.norm(vertex - plate.center_point)
            if dist < min_dist:
                min_dist = dist
                closest_plate = plate

        if closest_plate:
            plate_assignment[i] = closest_plate.plate_id
            closest_plate.vertices.add(i)

    return vertices, faces, plates, plate_assignment, elevations

def find_spherical_neighbors(vertices, faces, vertex_idx, max_distance_km):
    """Find neighbors within a certain distance on the sphere."""
    neighbors = set()
    center = vertices[vertex_idx]

    # First find direct face-connected neighbors
    for face in faces:
        if vertex_idx in face:
            for v in face:
                if v != vertex_idx:
                    neighbors.add(v)

    # Then check distance for all vertices
    final_neighbors = []
    for v in neighbors:
        # Calculate spherical distance
        lat1, lon1 = cartesian_to_lat_lon(*center)
        lat2, lon2 = cartesian_to_lat_lon(*vertices[v])
        dist = haversine_distance(lat1, lon1, lat2, lon2)

        if dist <= max_distance_km:
            final_neighbors.append(v)

    return final_neighbors

def simulate_plate_tectonics_spherical(vertices, faces, plates, plate_assignment, elevations, days_elapsed=30, max_neighbor_distance_km=1000):
    """Simulate plate tectonics on spherical mesh."""
    # Move plates
    for plate in plates:
        plate.move(days_elapsed)

    # Calculate boundary effects
    boundary_effects = np.zeros_like(elevations)

    for i, vertex in enumerate(vertices):
        current_plate_id = plate_assignment[i]
        if current_plate_id == 0:
            continue

        neighbors = find_spherical_neighbors(vertices, faces, i, max_neighbor_distance_km)

        for neighbor_idx in neighbors:
            neighbor_plate_id = plate_assignment[neighbor_idx]

            if neighbor_plate_id != current_plate_id and neighbor_plate_id != 0:
                # Simplified boundary interaction
                current_plate = plates[current_plate_id-1]
                neighbor_plate = plates[neighbor_plate_id-1]

                # Elevation adjustment based on plate movement
                boundary_effects[i] += 0.001 * step_size
                boundary_effects[neighbor_idx] -= 0.001 * step_size

    # Apply boundary effects
    elevations += boundary_effects

    # Smooth elevations
    smoothed_elevations = np.zeros_like(elevations)
    for i in range(len(vertices)):
        neighbor_indices = find_spherical_neighbors(vertices, faces, i, max_neighbor_distance_km)
        neighbor_elevations = [elevations[j] for j in neighbor_indices]
        smoothed_elevations[i] = np.mean([elevations[i]] + neighbor_elevations)

    elevations = smoothed_elevations

    # Update vertex positions with new elevations
    norm_vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    vertices = norm_vertices * (PLANET_RADIUS_KM + elevations[:, np.newaxis])

    # Reassign plate assignments based on new positions
    new_plate_assignment = np.zeros_like(plate_assignment)
    for plate in plates:
        plate.vertices.clear()

    for i, vertex in enumerate(vertices):
        min_dist = float('inf')
        closest_plate = None

        for plate in plates:
            dist = np.linalg.norm(vertex - plate.center_point)
            if dist < min_dist:
                min_dist = dist
                closest_plate = plate

        if closest_plate:
            new_plate_assignment[i] = closest_plate.plate_id
            closest_plate.vertices.add(i)

    return vertices, plates, new_plate_assignment, elevations

def calculate_solar_radiation(lat, lon, day_of_year, hour_of_day, elevation):
    """Calculate solar radiation at a given point considering axial tilt and time."""
    solar_constant = 1361  # W/m^2
    
    # Calculate declination (seasonal variation due to axial tilt)
    axial_tilt_rad = np.radians(AXIAL_TILT_DEGREES)
    declination = axial_tilt_rad * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    
    # Calculate hour angle (time of day)
    hour_angle = np.radians((hour_of_day - 12) * 15)  # 15 degrees per hour
    
    # Convert latitude to radians
    lat_rad = np.radians(lat)
    
    # Calculate solar zenith angle
    cos_zenith = (np.sin(lat_rad) * np.sin(declination) + (np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)))
    cos_zenith = np.clip(cos_zenith, 0, 1)  # Only consider daytime
    
    # Atmospheric absorption (simplified)
    air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - np.degrees(np.arccos(cos_zenith))) ** -1.6364)
    atmospheric_transmittance = 0.7 ** air_mass  # 70% transmittance per air mass
    
    # Elevation effect (thinner atmosphere at higher elevation)
    elevation_factor = 1 + (elevation / 10000)  # 10% increase per km
    
    # Total solar radiation
    radiation = solar_constant * cos_zenith * atmospheric_transmittance * elevation_factor
    return max(0, radiation)

def calculate_temperature(lat, lon, day_of_year, hour_of_day, elevation, water_fraction=0):
    """Calculate temperature considering solar radiation, elevation, and water bodies."""
    # Calculate solar radiation
    radiation = calculate_solar_radiation(lat, lon, day_of_year, hour_of_day, elevation)
    
    # Base temperature from radiation
    base_temp = (radiation / 200) - 10  # Scale radiation to reasonable temps
    
    # Elevation effect (6.5°C per km)
    elevation_effect = -6.5 * (elevation / 1000)
    
    # Water moderating effect (water has higher heat capacity)
    water_moderation = water_fraction * 5  # Water makes temps more moderate
    
    # Diurnal variation
    diurnal_variation = 10 * np.sin(np.radians(hour_of_day * 15))  # 15° per hour
    
    # Seasonal variation
    seasonal_variation = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    
    # Final temperature
    temperature = base_temp + elevation_effect + water_moderation + diurnal_variation + seasonal_variation
    
    # Never below absolute zero
    return max(-273, temperature)

def calculate_wind_patterns(lat, elevation, temperature_gradient):
    """Calculate prevailing wind direction based on latitude and elevation."""
    # Modified by elevation (mountains disrupt wind patterns)
    if elevation > 2000:
        return random.choice([0,1,2,3])  # Unpredictable in high mountains
    # Simplified wind patterns:
    if abs(lat) < 30:  # Trade winds
        if lat > 0: return 1  # NE trade winds
        else: return 3  # SE trade winds
    elif abs(lat) < 60:  # Westerlies
        return 2  # From west
    else:  # Polar easterlies
        return 0  # From east
    
cbar_obj = None  # Global variable to store the colorbar object

def visualize_world_spherical(vertices, faces, data, ax, data_type='elevation'):
    global cbar_obj
    ax.clear()

    if data_type == 'elevation':
        # Normalize elevations for coloring (using percentile to handle outliers)
        vmin = np.percentile(data, 5)
        vmax = np.percentile(data, 95)
        norm_data = (data - vmin) / (vmax - vmin)
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors based on vertex elevations
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.terrain(face_data)
        cmap = 'terrain'
        title = 'Elevation'

    elif data_type == 'temperature':
        # Normalize temperature data (-20 to 40°C)
        norm_data = (data + 20) / 60  # Scale -20°C to 40°C to 0-1
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.coolwarm(face_data)
        cmap = 'coolwarm'
        title = 'Temperature (°C)'

    elif data_type == 'rainfall':
        # Normalize rainfall data (0 to 200mm)
        norm_data = data / 200
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.Blues(face_data)
        cmap = 'Blues'
        title = 'Rainfall (mm)'

    # Plot the mesh with face colors
    mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                          triangles=faces, color='white',
                          edgecolor='none', alpha=1.0)
    mesh.set_array(face_data)
    mesh.set_cmap(cmap)

    # Add or update colorbar
    if cbar_obj is None:
        cbar_obj = plt.colorbar(mesh, ax=ax, shrink=0.5)
    else:
        cbar_obj.mappable.set_clim(vmin=np.min(face_data), vmax=np.max(face_data)) # Optional: update color limits if needed
        cbar_obj.mappable.set_array(face_data) # Update data for the colorbar

    if data_type == 'elevation':
        cbar_obj.set_label('Elevation (m)')
    elif data_type == 'temperature':
        cbar_obj.set_label('Temperature (°C)')
    elif data_type == 'rainfall':
        cbar_obj.set_label('Rainfall (mm)')
    cbar_obj.cmap = cmap

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect('equal')
    ax.view_init(elev=30, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title)

def save_simulation(filename, vertices, faces, plates, plate_assignment, elevations, sim_params):
    """Save the simulation state."""
    with open(filename, 'wb') as f:
        pickle.dump({
            'vertices': vertices,
            'faces': faces,
            'plates': plates,
            'plate_assignment': plate_assignment,
            'elevations': elevations,
            'sim_params': sim_params
        }, f)
    print(f"Simulation saved to '{filename}'")

def load_simulation(filename):
    """Load a saved simulation."""
    try:
        with open(filename, 'rb') as f:
            saved_state = pickle.load(f)
            return (
                saved_state['vertices'],
                saved_state['faces'],
                saved_state['plates'],
                saved_state['plate_assignment'],
                saved_state['elevations'],
                saved_state['sim_params']
            )
    except FileNotFoundError:
        print(f"Save file '{filename}' not found. Starting a new simulation.")
        return None
    
def calculate_rainfall(lat, lon, elevation, temperature, vertices, vertex_idx, max_distance_km=1000):
    """Calculate rainfall based on wind patterns, elevation, and nearby water."""
    # Determine wind direction
    wind_dir = calculate_wind_patterns(lat, elevation, 0)
    
    # Find upwind and downwind areas
    neighbors = find_spherical_neighbors(vertices, faces, vertex_idx, max_distance_km)
    
    # Calculate water availability (from nearby water bodies)
    water_availability = 0
    for n in neighbors:
        n_elevation = elevations[n]
        if n_elevation < 0:  # Water body
            dist = haversine_distance(lat, lon, *cartesian_to_lat_lon(*vertices[n]))
            water_availability += max(0, 1 - dist/500)  # Water influence up to 500km
    
    # Base precipitation based on latitude
    if abs(lat) < 10:  # ITCZ - lots of rain
        base_rain = 150
    elif abs(lat) < 30:  # Subtropics - dry
        base_rain = 20
    elif abs(lat) < 60:  # Temperate - moderate rain
        base_rain = 80
    else:  # Polar - dry
        base_rain = 10
    
    # Orographic precipitation
    orographic_effect = 0
    if elevation > 0:  # Only over land
        # Find highest point in wind direction
        max_upwind_elev = 0
        for n in neighbors:
            n_lat, n_lon = cartesian_to_lat_lon(*vertices[n])
            if is_upwind(lat, lon, n_lat, n_lon, wind_dir):
                max_upwind_elev = max(max_upwind_elev, elevations[n])
        
        if elevation > max_upwind_elev:  # We're on windward side
            orographic_effect = min(100, elevation / 20)  # 5mm per 100m
        else:  # Leeward side - rain shadow
            orographic_effect = -50
    
    # Temperature effect (more evaporation when warmer)
    temp_effect = max(0, temperature - 10)  # More rain above 10°C
    
    # Combine all effects
    rainfall = base_rain + orographic_effect + temp_effect + water_availability * 50
    return max(0, min(300, rainfall))  # Cap between 0-300mm

def is_upwind(lat1, lon1, lat2, lon2, wind_dir):
    """Check if point 2 is upwind of point 1 based on wind direction."""
    # Simplified check based on wind direction categories
    if wind_dir == 0:  # East wind
        return lon2 < lon1
    elif wind_dir == 1:  # North wind
        return lat2 > lat1
    elif wind_dir == 2:  # West wind
        return lon2 > lon1
    else:  # South wind
        return lat2 < lat1

if __name__ == "__main__":
    # Simulation parameters
    subdivisions = 5  # Controls mesh resolution (higher = more detailed)
    radius = PLANET_RADIUS_KM
    num_plates = 15 # Increased number of plates for more fragmentation
    num_steps = 12
    step_size = 1
    max_neighbor_distance_km = 1000  # Distance for plate boundary interactions
    days_per_step = 30  # Each step represents a month
    current_day = 0
    current_hour = 12  # Noon

    sim_params = {
        'subdivisions': subdivisions,
        'radius': radius,
        'num_plates': num_plates,
        'num_steps': num_steps,
        'step_size': step_size,
        'max_neighbor_distance_km': max_neighbor_distance_km
    }

    save_file = "spherical_simulation_save.pkl"

    if len(sys.argv) > 1:
        load_file = sys.argv[1]
        loaded_data = load_simulation(load_file)
        if loaded_data:
            vertices, faces, plates, plate_assignment, elevations, loaded_params = loaded_data
            subdivisions = loaded_params['subdivisions']
            radius = loaded_params['radius']
            num_plates = loaded_params['num_plates']
            step_size = loaded_params['step_size']
            max_neighbor_distance_km = loaded_params['max_neighbor_distance_km']
            print(f"Simulation loaded from '{load_file}'.")
        else:
            vertices, faces, plates, plate_assignment, elevations = generate_initial_world_spherical(
                subdivisions, radius, num_plates
            )
            print("Starting a new simulation.")
    else:
        vertices, faces, plates, plate_assignment, elevations = generate_initial_world_spherical(
            subdivisions, radius, num_plates
        )
        print("Starting a new simulation.")

    # Run simulation
    world_history = []
    for step in range(num_steps):
        print(f"Simulating step {step + 1}/{num_steps}")
        
        # Update time
        current_day += days_per_step
        current_day %= 365  # Wrap around year
        
        # Plate tectonics simulation
        vertices, plates, plate_assignment, elevations = simulate_plate_tectonics_spherical(
            vertices, faces, plates, plate_assignment, elevations,
            days_per_step, max_neighbor_distance_km
        )

        # Calculate water fraction for each vertex (simplified)
        water_fraction = np.where(elevations < 0, 1.0, 0.0)
        
        # Calculate climate variables
        temperatures = np.zeros(len(vertices))
        rainfall = np.zeros(len(vertices))
        
        for i, vertex in enumerate(vertices):
            lat, lon = cartesian_to_lat_lon(*vertex)
            temperatures[i] = calculate_temperature(lat, lon, current_day, current_hour, 
                                                elevations[i], water_fraction[i])
            rainfall[i] = calculate_rainfall(lat, lon, elevations[i], temperatures[i], 
                                        vertices, i)

        world_history.append((vertices.copy(), elevations.copy(), 
                            temperatures.copy(), rainfall.copy()))
    
    print("Simulation complete. Preparing interactive visualization...")

    # Set up visualization
    fig = plt.figure(figsize=(12, 10))
    ax_3d = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25, left=0.3)

    current_step = 0
    vertices, elevations, temperatures, rainfall = world_history[current_step]
    visualize_world_spherical(vertices, faces, elevations, ax_3d, 'elevation') # Initial call to create colorbar

    # Add radio buttons for view selection
    rax = plt.axes([0.05, 0.4, 0.15, 0.15])
    radio = RadioButtons(rax, ('Elevation', 'Temperature', 'Rainfall'))

    def update_view(label):
        vertices, elevations, temperatures, rainfall = world_history[current_step]
        if label == 'Elevation':
            visualize_world_spherical(vertices, faces, elevations, ax_3d, 'elevation')
        elif label == 'Temperature':
            visualize_world_spherical(vertices, faces, temperatures, ax_3d, 'temperature')
        elif label == 'Rainfall':
            visualize_world_spherical(vertices, faces, rainfall, ax_3d, 'rainfall')
        fig.canvas.draw_idle()

    radio.on_clicked(update_view)

    # Add step slider
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    step_slider = Slider(
        ax=ax_slider,
        label='Step',
        valmin=0,
        valmax=num_steps - 1,
        valinit=current_step,
        valstep=1
    )

    def update_step(val):
        global current_step
        current_step = int(step_slider.val)
        label = radio.value_selected
        update_view(label)

    step_slider.on_changed(update_step)

    plt.show()

    # Save the final state
    save_simulation(save_file, vertices, faces, plates, plate_assignment, elevations, sim_params)
    print(f"Simulation automatically saved to '{save_file}'.")