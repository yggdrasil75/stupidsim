import h3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import random
from scipy.ndimage import gaussian_filter

PLANET_RADIUS_KM = 6371.0



def haversine_distance_h3(h3_index1, h3_index2, radius=PLANET_RADIUS_KM):
    """Calculate great-circle distance between H3 cell centers."""
    geo1 = h3.h3shape_to_geo(h3_index1)
    geo2 = h3.h3shape_to_geo(h3_index2)
    return haversine_distance(geo1[0], geo1[1], geo2[0], geo2[1], radius)

def haversine_distance(lat1, lon1, lat2, lon2, radius=PLANET_RADIUS_KM):
    """Haversine formula for distance between lat/lon points."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c

class Plate:
    def __init__(self, plate_id, center_h3_index, movement_x, movement_y):
        self.plate_id = plate_id
        self.center_h3_index = center_h3_index
        self.movement_x = movement_x
        self.movement_y = movement_y
        self.cells = set()

    def move(self, resolution):
        """Simplified plate movement by shifting the center H3 index."""
        center_geo = h3.h3shape_to_geo(self.center_h3_index)
        new_lat = center_geo[0] + self.movement_x * 0.5
        new_lon = center_geo[1] + self.movement_y * 0.5
        new_lon = (new_lon + 180) % 360 - 180
        new_lat = max(-90, min(90, new_lat))
        self.center_h3_index = h3.h3shape_to_geo(new_lat, new_lon, resolution)

def generate_h3_world(resolution=5, num_plates=5):
    """Generates H3 world grid with initial plate assignment."""
    center_h3 = h3.h3shape_to_geo(0, 0, resolution)
    all_h3_indices = list(h3.hex_range(center_h3, 200))

    world_data = {}
    for h3_index in all_h3_indices:
        geo_coords = h3.h3shape_to_geo(h3_index)
        world_data[h3_index] = {
            'elevation': 0.0,
            'plate_id': 0,
            'latitude': geo_coords[0],
            'longitude': geo_coords[1]
        }

    plates = []
    plate_center_indices = random.sample(all_h3_indices, num_plates)
    for i in range(num_plates):
        center_h3_index = plate_center_indices[i]
        movement_x = random.uniform(-0.02, 0.02)
        movement_y = random.uniform(-0.02, 0.02)
        plates.append(Plate(i + 1, center_h3_index, movement_x, movement_y))

    return assign_plates_to_world(world_data, plates, all_h3_indices), plates, resolution

def assign_plates_to_world(world_data, plates, all_h3_indices):
    """Assigns plate IDs to world cells based on distance to plate centers."""
    for h3_index in all_h3_indices:
        closest_plate_id = 0
        min_distance = float('inf')
        for plate in plates:
            distance = haversine_distance_h3(h3_index, plate.center_h3_index)
            if distance < min_distance:
                min_distance = distance
                closest_plate_id = plate.plate_id
        world_data[h3_index]['plate_id'] = closest_plate_id
        for plate in plates:
            plate.cells = set()
        for plate in plates:
            for h3_index_assign in all_h3_indices:
                if world_data[h3_index_assign]['plate_id'] == plate.plate_id:
                    plate.cells.add(h3_index_assign)
    return world_data

def simulate_plate_tectonics_h3(world_data, plates, resolution, all_h3_indices, step_size=0.5):
    """Simulates plate tectonics with movement and boundary interactions."""
    for plate in plates:
        plate.move(resolution)
    world_data = assign_plates_to_world(world_data, plates, all_h3_indices)

    boundary_effect_grid = {} # Dictionary to store elevation changes

    for h3_index in all_h3_indices:
        current_plate_id = world_data[h3_index]['plate_id']
        if current_plate_id == 0:
            continue

        neighbors_h3 = h3.h3_to_k_ring(h3_index, k=1) # Get neighbors (including self)
        for neighbor_h3_index in neighbors_h3:
            if neighbor_h3_index not in world_data: #Neighbor might be outside of our generated world range
                continue
            neighbor_plate_id = world_data[neighbor_h3_index]['plate_id']

            if current_plate_id != neighbor_plate_id and neighbor_plate_id != 0: # Boundary!
                plate1 = plates[current_plate_id - 1]
                plate2 = plates[neighbor_plate_id - 1]

                # Simplified Relative Movement (just difference in vectors)
                relative_movement_x = plate1.movement_x - plate2.movement_x
                relative_movement_y = plate1.movement_y - plate2.movement_y

                boundary_effect = 0
                # Approximate boundary type and elevation effect
                if abs(relative_movement_x) < 0.01 and abs(relative_movement_y) < 0.01: # Transform
                    boundary_effect += random.uniform(-0.05, 0.05) * step_size
                elif relative_movement_x > 0.01 or relative_movement_y > 0.01: # Divergent
                    boundary_effect += random.uniform(0.1, 0.2) * step_size
                    boundary_effect += random.uniform(0, 0.1) * step_size #Volcanism
                elif relative_movement_x < -0.01 or relative_movement_y < -0.01: # Convergent
                    boundary_effect += random.uniform(0.2, 0.3) * step_size
                    boundary_effect += random.uniform(0, 0.05) * step_size #Volcanism

                # Area Weighting (Latitude)
                current_lat = world_data[h3_index]['latitude']
                neighbor_lat = world_data[neighbor_h3_index]['latitude']
                area_weight = np.cos(np.radians((current_lat + neighbor_lat) / 2.0))
                boundary_effect *= area_weight
                boundary_effect *= 0.1 #Reduce overall boundary effect

                if h3_index not in boundary_effect_grid:
                    boundary_effect_grid[h3_index] = 0 # Initialize if not present
                boundary_effect_grid[h3_index] += boundary_effect

    # Apply boundary effects and erosion
    for h3_index in all_h3_indices:
        if h3_index in boundary_effect_grid:
            world_data[h3_index]['elevation'] += boundary_effect_grid[h3_index]

    # Basic Erosion (Smoothing) - Gaussian blur on elevations
    elevations = np.array([world_data[h3_index]['elevation'] for h3_index in all_h3_indices]).reshape(1,-1) #Reshape for gaussian filter
    smoothed_elevations = gaussian_filter(elevations, sigma=0.3).flatten() # Apply smoothing
    for i, h3_index in enumerate(all_h3_indices):
         world_data[h3_index]['elevation'] = smoothed_elevations[i]


    return world_data, plates


def visualize_h3_world(world_data, resolution, step=0, show_plate_ids=False):
    """Visualizes the H3 world on a 2D map using Cartopy, optionally showing plate IDs."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.stock_img()

    boundaries = []
    facecolors = []

    if show_plate_ids:
        cmap = plt.cm.get_cmap('tab20', 20)
        plate_ids = [data['plate_id'] for data in world_data.values()]
        min_plate_id = min(plate_ids)
        max_plate_id = max(plate_ids)

        for h3_index, data in world_data.items():
            boundary = h3.h3_to_geo_boundary(h3_index, geo_json=False)
            boundaries.append(boundary)
            plate_id = data['plate_id']
            if plate_id == 0:
                facecolors.append('lightgray')
            else:
                color_index = (plate_id - min_plate_id) % 20 if (max_plate_id - min_plate_id) > 0 else plate_id % 20
                facecolors.append(cmap(color_index))

    else: # Visualize elevation
        elevations = [data['elevation'] for data in world_data.values()]
        elevations_np = np.array(elevations)
        norm_elevations = (elevations_np - elevations_np.min()) / (elevations_np.max() - elevations_np.min() + 1e-9)
        cmap = plt.cm.terrain
        for h3_index, data in world_data.items():
            boundary = h3.h3_to_geo_boundary(h3_index, geo_json=False)
            boundaries.append(boundary)
            facecolors.append(cmap(norm_elevations[list(world_data.keys()).index(h3_index)]))

    collection = ax.add_collection(
        plt.PolygonCollection(boundaries, closed=True, facecolors=facecolors, linewidth=0.5, edgecolors='black'),
        transform=ccrs.PlateCarree()
    )

    title_suffix = " - Plate IDs" if show_plate_ids else " - Elevation"
    ax.set_title(f'H3 World - Resolution {resolution} - Step {step}{title_suffix}')
    plt.savefig(f'h3_world_step_{step:03d}.png')
    plt.show()


if __name__ == "__main__":
    resolution = 4 # Keep resolution relatively low for now
    num_plates = 7
    num_steps = 200

    world_data, plates, resolution = generate_h3_world(resolution, num_plates=num_plates)
    center_h3 = h3.h3shape_to_geo(0, 0, resolution)
    all_h3_indices = list(h3.hex_range(center_h3, 200))

    for step in range(num_steps):
        print(f"Simulating step {step + 1}/{num_steps}")
        world_data, plates = simulate_plate_tectonics_h3(world_data, plates, resolution, all_h3_indices, step_size=0.2) # Reduced step_size
        visualize_h3_world(world_data, resolution, step=step, show_plate_ids=False) # Visualize elevation now

    print("Simulation complete. Images saved as h3_world_step_*.png")