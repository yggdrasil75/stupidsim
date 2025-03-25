# your_main_script.py (replace with your original script name)
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from constants import *  # Import all constants
from plate import Plate
from utils import *
from world import generate_initial_world_spherical, simulate_plate_tectonics_spherical
from visualization import visualize_world_spherical


if __name__ == "__main__":
    # Simulation parameters
    subdivisions = 3  # Controls mesh resolution (higher = more detailed)
    radius = PLANET_RADIUS_KM
    num_plates = 15 # Increased number of plates for more fragmentation
    num_steps = 12
    step_size = 0.03
    max_neighbor_distance_km = 1000  # Distance for plate boundary interactions
    days_per_step = 30  # Each step represents a month
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


    vertices, faces, plates, plate_assignment, elevations, surface_pressures = generate_initial_world_spherical(
        subdivisions, radius, num_plates, surface_pressures
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
            days_per_step, max_neighbor_distance_km, step_size
        )

        # Calculate water fraction for each vertex (simplified)
        water_fraction = np.where(elevations < 0, 1.0, 0.0)

        # Calculate climate variables
        sun_direction = calculate_sun_direction(current_day, current_hour)
        temperatures = np.zeros(len(vertices))
        rainfall = np.zeros(len(vertices))
        humidity_values = np.zeros(len(vertices))
        surface_pressures = np.zeros(len(vertices)) # Initialize surface_pressures here for each step

        # In the main simulation loop:
        for i, vertex in enumerate(vertices):
            lat, lon = cartesian_to_lat_lon(*vertex)

            # Get plate properties
            plate_id = plate_assignment[i]
            if plate_id > 0:
                plate = plates[plate_id-1]
                plate_temp = plate.temperature
            else:
                plate_temp = 15

            # Calculate surface pressure considering atmospheric layers
            surface_pressures[i] = calculate_pressure_with_layers(elevations[i], plate_temp)

            # Calculate temperature at surface considering atmospheric layers
            temperatures[i] = calculate_temperature_with_altitude(
                calculate_temperature_from_radiation(
                    calculate_solar_radiation_for_vertex(vertex, sun_direction,
                                                    elevations[i], surface_pressures[i]),
                    elevations[i], water_fraction[i], surface_pressures[i]),
                elevations[i]
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

    def update_view(label):
        vertices, elevations, temperatures, rainfall, surface_pressures = world_history[current_step] # Unpack surface_pressures
        if label == 'Elevation':
            visualize_world_spherical(vertices, faces, elevations, ax_3d, 'elevation', elevations=elevations)
        elif label == 'Temperature':
            visualize_world_spherical(vertices, faces, temperatures, ax_3d, 'temperature', elevations=elevations)
        elif label == 'Rainfall':
            visualize_world_spherical(vertices, faces, rainfall, ax_3d, 'rainfall', elevations=elevations)
        elif label == 'Pressure':
            visualize_world_spherical(vertices, faces, temperatures, ax_3d, 'pressure', elevations=elevations) # Pass temperatures as data and elevations
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