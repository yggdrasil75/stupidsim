import math

from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons, Slider
import numpy as np

from clouds import CloudSystem
from globals import ALBEDO_VALUES, PLANET_RADIUS_KM, SEA_LEVEL_PRESSURE_HPA, loadConfig, saveConfig
from humidity import calculate_humidity, calculate_rainfall
from plate import Plate, simulate_plate_tectonics_spherical
from pressure import calculate_pressure_with_circulation, update_pressure_systems
from storms import generate_storm_systems, update_storm_systems
from temperature import calculate_solar_radiation_for_vertex, calculate_sun_direction, calculate_temperature_from_radiation, calculate_temperature_with_greenhouse
from utils import cartesian_to_lat_lon, determine_surface_type
from viewer import visualize_world_spherical


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

def generate_initial_world_spherical(subdivisions=3, radius=PLANET_RADIUS_KM, num_plates=5, surface_pressures=None):
    """Generate initial world with spherical mesh and plates."""
    vertices, faces = generate_icosphere(subdivisions, radius)

    # Add random elevation to vertices using spherical harmonics for more natural distribution
    elevations = np.zeros(len(vertices))
    humidity_values = np.zeros(len(vertices))
    rainfall = np.zeros(len(vertices))
    if surface_pressures is None:
        surface_pressures = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        # Convert to spherical coordinates
        lat, lon = cartesian_to_lat_lon(*vertex)

        # Create more interesting initial elevations using noise
        noise = (np.sin(lon * 2) * np.cos(lat * 3) +
                 np.sin(lon * 5) * np.cos(lat * 2)) * 5
        elevations[i] = noise + np.random.uniform(-2, 2)
        surface_pressures[i] = SEA_LEVEL_PRESSURE_HPA # Initialize pressure to sea level
        humidity_values[i] = noise + np.random.uniform(-2, 2)
        rainfall[i] = noise + np.random.uniform(-2, 2)

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

    return vertices, faces, plates, plate_assignment, elevations, surface_pressures, humidity_values, rainfall

if __name__ == "__main__":
    config = loadConfig()
    # Simulation parameters
    subdivisions = config['simulation']['subdivisions']  # Controls mesh resolution (higher = more detailed)
    radius = PLANET_RADIUS_KM
    num_plates = config['plate_tectonics']['initial_num_plates'] #15 # Increased number of plates for more fragmentation
    num_steps = config['simulation']['num_steps']
    step_size = config['simulation']['step_size']
    max_neighbor_distance_km = config['simulation']['max_neighbor_distance_km']  # Distance for plate boundary interactions
    days_per_step = config['simulation']['days_per_step']  # Each step represents a month
    current_day = 0
    current_hour = 12  # Noon
    pressures = 0
    surface_pressures = None # Initialize surface_pressures to None

    sim_params = {
        'subdivisions': subdivisions,
        'radius': radius,
        'num_plates': num_plates,
        'num_steps': num_steps,
        'step_size': step_size,
        'max_neighbor_distance_km': max_neighbor_distance_km
    }


    vertices, faces, plates, plate_assignment, elevations, surface_pressures, humidity_values, rainfall = generate_initial_world_spherical(
        subdivisions, radius, num_plates, surface_pressures
    )
    print("Starting a new simulation.")
    cloud_system = CloudSystem(vertices, elevations)

    # Run simulation
    world_history = []
    active_storms = []
    for step in range(num_steps):
        print(f"Simulating step {step + 1}/{num_steps}")

        # Update time
        current_day += days_per_step
        current_day %= 365  # Wrap around year
        current_hour = (current_hour + 6) % 24  # Advance 6 hours each step

        # Update storm systems
        active_storms = update_storm_systems(active_storms, vertices, elevations, days_per_step)
        
        # Calculate water fraction for each vertex (simplified)
        water_fraction = np.where(elevations < 0, 1.0, 0.0)
        
        # Generate new storms periodically
        if step % 3 == 0:  # Every 3 steps
            new_storms = generate_storm_systems(vertices, elevations, current_day, num_storms=2)
            active_storms.extend(new_storms)
        


        # Calculate climate variables
        sun_direction = calculate_sun_direction(current_day, current_hour)
        temperatures = np.array([
            calculate_temperature_from_radiation(
                calculate_solar_radiation_for_vertex(vertex, sun_direction, elevations[i], SEA_LEVEL_PRESSURE_HPA,
                cloud_system.cloud_coverage[i], cloud_system.cloud_albedo[i]),
                elevations[i],
                water_fraction[i],
                SEA_LEVEL_PRESSURE_HPA
            )
            for i, vertex in enumerate(vertices)
        ])
        cloud_system.update_clouds(temperatures, humidity_values, surface_pressures, rainfall, current_day)
        # Plate tectonics simulation
        vertices, plates, plate_assignment, elevations = simulate_plate_tectonics_spherical(
            vertices, faces, plates, plate_assignment, elevations,
            days_per_step, max_neighbor_distance_km, step_size, 
            active_storms, water_fraction, temperatures, rainfall, humidity_values
        )
        # Update pressure systems with storms
        surface_pressures = update_pressure_systems(vertices, faces, elevations, temperatures, 
                                                current_day, current_hour, active_storms)

        for i, vertex in enumerate(vertices):
            lat, lon = cartesian_to_lat_lon(*vertex)
            
			
			# Determine surface type and albedo
            surface_type = determine_surface_type(elevations[i], temperatures[i], water_fraction[i])
            albedo = ALBEDO_VALUES[surface_type]

            # Update radiation calculation with albedo
            radiation = calculate_solar_radiation_for_vertex(
                vertex, sun_direction, elevations[i], surface_pressures[i],
                cloud_system.cloud_coverage[i], cloud_system.cloud_albedo[i]
            ) * (1 - albedo)

            # Get plate properties
            plate_id = plate_assignment[i]
            if plate_id > 0:
                plate = plates[plate_id-1]
                surface_pressures[i] = plate.pressure
                plate_temp = plate.temperature
            else:
                surface_pressures[i] = SEA_LEVEL_PRESSURE_HPA
                plate_temp = 15

            # Calculate surface pressure considering atmospheric layers
            surface_pressures[i] = calculate_pressure_with_circulation(elevations[i], temperatures[i], humidity_values[i])

            # Calculate temperature at surface considering atmospheric layers
            temperatures[i] = calculate_temperature_with_greenhouse(calculate_solar_radiation_for_vertex(vertex, sun_direction, 
                                                elevations[i], surface_pressures[i],cloud_system.cloud_coverage[i], cloud_system.cloud_albedo[i]),
                elevations[i], 
                water_fraction[i], 
                surface_pressures[i],
                humidity_values[i]
            )

            # Calculate humidity using surface conditions
            humidity_values[i] = calculate_humidity(surface_pressures[i], temperatures[i], water_fraction[i])

            # Rainfall calculation remains similar but uses the layered pressure
            rainfall[i] = calculate_rainfall(lat, lon, elevations[i], temperatures[i],
                                        surface_pressures[i], humidity_values[i],
                                        vertices, i, faces, elevations)

        world_history.append((vertices.copy(), elevations.copy(),
                            temperatures.copy(), rainfall.copy(), surface_pressures.copy())) # Store surface_pressures as well

    print("Simulation complete. Preparing interactive visualization...")

    # Set up visualization
    fig = plt.figure(figsize=(12, 10))
    ax_3d = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25, left=0.3)

    current_step = 0
    vertices, elevations, temperatures, rainfall, surface_pressures = world_history[current_step] # Unpack surface_pressures
    visualize_world_spherical(vertices, faces, elevations, ax_3d, 'elevation', elevations=elevations) # Initial call to create colorbar, pass elevations

    # Add radio buttons for view selection
    rax = plt.axes([0.05, 0.4, 0.15, 0.15])
    radio = RadioButtons(rax, ('Elevation', 'Temperature', 'Rainfall', 'Pressure'))

    ax_check = plt.axes([0.05, 0.25, 0.15, 0.1])
    check = CheckButtons(ax_check, ['Show Clouds', 'Show Storms'], [False, False])

    def update_overlays(label):
        # Get current visualization type
        current_view = radio.value_selected if hasattr(radio, 'value_selected') else 'elevation'
        update_view(current_view)

    check.on_clicked(update_overlays)

    def update_view(label):
        vertices, elevations, temperatures, rainfall, surface_pressures = world_history[current_step]
        show_clouds = check.get_status()[0]
        show_storms = check.get_status()[1]
        
        # Determine which data to visualize based on label
        if label == 'Elevation':
            vis_data = elevations
        elif label == 'Temperature':
            vis_data = temperatures
        elif label == 'Rainfall':
            vis_data = rainfall
        elif label == 'Pressure':
            vis_data = surface_pressures
        else:
            vis_data = elevations  # Default
        
        visualize_world_spherical(
            vertices, faces, 
            vis_data,
            ax_3d, 
            label.lower(),
            elevations=elevations,
            show_clouds=show_clouds,
            cloud_coverage=cloud_system.cloud_coverage,
            show_storms=show_storms,
            active_storms=active_storms
        )
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
    saveConfig(config)
    plt.title('StupidSim')

    plt.show()
