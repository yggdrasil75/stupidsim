# world.py
import numpy as np
import math
import random
from utils import generate_icosphere, cartesian_to_lat_lon, haversine_distance, find_spherical_neighbors
from plate import Plate
from constants import PLANET_RADIUS_KM

def generate_initial_world_spherical(subdivisions=3, radius=PLANET_RADIUS_KM, num_plates=5, surface_pressures=None):
    """Generate initial world with spherical mesh and plates."""
    vertices, faces = generate_icosphere(subdivisions, radius)

    # Add random elevation to vertices using spherical harmonics for more natural distribution
    elevations = np.zeros(len(vertices))
    if surface_pressures is None:
        surface_pressures = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        # Convert to spherical coordinates
        lat, lon = cartesian_to_lat_lon(*vertex)

        # Create more interesting initial elevations using noise
        noise = (np.sin(lon * 2) * np.cos(lat * 3) +
                 np.sin(lon * 5) * np.cos(lat * 2)) * 5
        elevations[i] = noise + np.random.uniform(-2, 2)
        surface_pressures[i] = 1013.25 # Initialize pressure to sea level

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

    return vertices, faces, plates, plate_assignment, elevations, surface_pressures


def simulate_plate_tectonics_spherical(vertices, faces, plates, plate_assignment, elevations, days_elapsed=30, max_neighbor_distance_km=1000, step_size=0.03):
    """Simulate plate tectonics on spherical mesh with pressure and temperature effects."""
    # Move plates
    for plate in plates:
        plate.move(days_elapsed)

    # Calculate boundary effects
    boundary_effects = np.zeros_like(elevations)
    pressure_changes = np.zeros(len(vertices))
    temperature_changes = np.zeros(len(vertices))

    for i, vertex in enumerate(vertices):
        current_plate_id = plate_assignment[i]
        if current_plate_id == 0:
            continue

        neighbors = find_spherical_neighbors(vertices, faces, i, max_neighbor_distance_km)
        current_plate = plates[current_plate_id-1]

        for neighbor_idx in neighbors:
            neighbor_plate_id = plate_assignment[neighbor_idx]

            if neighbor_plate_id != current_plate_id and neighbor_plate_id != 0:
                # Simplified boundary interaction
                current_plate = plates[current_plate_id-1]
                neighbor_plate = plates[neighbor_plate_id-1]

                # Elevation adjustment based on plate movement
                boundary_effects[i] += 0.001 * step_size
                boundary_effects[neighbor_idx] -= 0.001 * step_size

                # Pressure changes at plate boundaries
                pressure_diff = current_plate.pressure - neighbor_plate.pressure
                pressure_changes[i] += pressure_diff * 0.01
                pressure_changes[neighbor_idx] -= pressure_diff * 0.01

                # Temperature changes at plate boundaries
                temp_diff = current_plate.temperature - neighbor_plate.temperature
                temperature_changes[i] += temp_diff * 0.01
                temperature_changes[neighbor_idx] -= temp_diff * 0.01

    # Apply boundary effects
    elevations += boundary_effects

    # Apply pressure and temperature changes
    for plate in plates:
        plate_vertices = list(plate.vertices)
        if plate_vertices:
            avg_pressure_change = np.mean(pressure_changes[plate_vertices])
            avg_temp_change = np.mean(temperature_changes[plate_vertices])

            # Update plate properties (dampened changes)
            plate.pressure += avg_pressure_change * 0.1
            plate.temperature += avg_temp_change * 0.1

            # Plate speed affected by temperature (warmer plates move faster)
            plate.speed_cm_yr *= (1 + 0.01 * (plate.temperature - 15))

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