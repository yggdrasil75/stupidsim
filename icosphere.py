import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider
import pickle
import sys
import math
from scipy.spatial import Delaunay

PLANET_RADIUS_KM = 6371.0
ORBITAL_DISTANCE_AU = 1.0
AXIAL_TILT_DEGREES = 23.5

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
    def __init__(self, plate_id, center_vertex, movement_vector):
        self.plate_id = plate_id
        self.center_vertex = center_vertex  # Index of center vertex (not used anymore)
        self.movement_vector = movement_vector  # 3D movement vector
        self.vertices = set()  # Indices of vertices in this plate
        self.center_point = None  # Will be set during initialization

    def move(self):
        """Move the plate's center point according to its movement vector."""
        # The movement is applied to the center point in 3D space
        self.center_point += self.movement_vector
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
        # Random movement vector (tangent to sphere)
        center_point = plate_centers[i]
        tangent = np.cross(center_point, np.random.randn(3))
        tangent = tangent / np.linalg.norm(tangent)
        movement_vector = tangent * random.uniform(0.01, 0.1)

        plate = Plate(i + 1, -1, movement_vector)
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

def simulate_plate_tectonics_spherical(vertices, faces, plates, plate_assignment, elevations,
                                      step_size=1.0, max_neighbor_distance_km=1000):
    """Simulate plate tectonics on spherical mesh."""
    # Move plates
    for plate in plates:
        plate.move()

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

def calculate_solar_radiation(lat, day_of_year):
    solar_constant = 1361
    axial_tilt_rad = np.radians(AXIAL_TILT_DEGREES)
    declination = axial_tilt_rad * axial_tilt_rad * np.sin(2 * np.pi * (day_of_year - 81) / 365) # Corrected declination calculation
    declination = axial_tilt_rad * np.sin(2 * np.pi * (day_of_year - 81) / 365)
    lat_rad = np.radians(lat)
    hour_angle = np.radians(0) # Noon
    cos_zenith = (np.sin(lat_rad) * np.sin(declination)) + (np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
    solar_zenith_angle = np.arccos(np.clip(cos_zenith, -1.0, 1.0)) # Clip to handle potential floating point errors
    radiation = solar_constant * np.cos(solar_zenith_angle)
    radiation = max(radiation, 0)
    return radiation

def calculate_temperature(lat, day_of_year, elevation=0):
    radiation = calculate_solar_radiation(lat, day_of_year)
    sea_level_temp = 15
    temp_drop_per_km = 6.5
    temperature = sea_level_temp + (radiation / 1361 * 30) - (elevation / 1000) * temp_drop_per_km
    return temperature

def visualize_world_spherical(vertices, faces, elevations, ax):
    ax.clear()

    # Normalize elevations for coloring (using percentile to handle outliers)
    vmin = np.percentile(elevations, 5)
    vmax = np.percentile(elevations, 95)
    norm_elevations = (elevations - vmin) / (vmax - vmin)
    norm_elevations = np.clip(norm_elevations, 0, 1)
    
    # Create face colors based on vertex elevations
    face_elevations = np.mean(norm_elevations[faces], axis=1)
    colors = plt.cm.terrain(face_elevations)

    # Plot the mesh with face colors
    mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                          triangles=faces, color='white',
                          edgecolor='none', alpha=1.0)
    mesh.set_array(face_elevations)
    mesh.set_cmap('terrain')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect('equal')
    ax.view_init(elev=30, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

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

if __name__ == "__main__":
    # Simulation parameters
    subdivisions = 3  # Controls mesh resolution (higher = more detailed)
    radius = PLANET_RADIUS_KM
    num_plates = 15 # Increased number of plates for more fragmentation
    num_steps = 50
    step_size = 1
    max_neighbor_distance_km = 1000  # Distance for plate boundary interactions

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
        vertices, plates, plate_assignment, elevations = simulate_plate_tectonics_spherical(
            vertices, faces, plates, plate_assignment, elevations,
            step_size, max_neighbor_distance_km
        )
        world_history.append((vertices.copy(), elevations.copy()))

    print("Simulation complete. Preparing interactive visualization...")

    # Set up visualization
    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    current_step = 0
    vertices, elevations = world_history[current_step]
    visualize_world_spherical(vertices, faces, elevations, ax_3d)
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
        vertices, elevations = world_history[step]
        visualize_world_spherical(vertices, faces, elevations, ax_3d)
        ax_3d.set_title(f'World Evolution - Step {step + 1}')
        fig.canvas.draw_idle()

    step_slider.on_changed(update)

    plt.show()

    save_simulation(save_file, vertices, faces, plates, plate_assignment, elevations, sim_params)
    print(f"Simulation automatically saved to '{save_file}'.")