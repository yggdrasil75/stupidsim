#new plate assignment attempt
import numpy as np
from util import spherical_distance
import numpy as np


def assign_origins(self, vertices, num_plates, radius):
    random_indices = np.random.choice(len(vertices), num_plates, replace=False)
    plate_origins_indices = random_indices.copy()  # This will store the indices we return
    plate_origins = vertices[random_indices]  # This is just for distance checking
    
    # Check distances between all pairs of plate origins
    need_reassignment = True
    while need_reassignment:
        need_reassignment = False
        for i in range(len(plate_origins_indices)):
            for j in range(i+1, len(plate_origins_indices)):
                # Calculate spherical distance between two plate origins
                dist = spherical_distance(vertices[plate_origins_indices[i]], 
                                         vertices[plate_origins_indices[j]], 
                                         radius)
                
                # If too close, replace one of them with a new random vertex
                if dist < (radius / 10):
                    # Get all vertex indices not currently used as origins
                    all_indices = set(range(len(vertices)))
                    used_indices = set(plate_origins_indices)
                    available_vertices = list(all_indices - used_indices)
                    
                    if available_vertices:  # Ensure there are vertices left to choose from
                        # Select a new random index from available vertices
                        new_index = np.random.choice(available_vertices)
                        # Update both our tracking arrays
                        plate_origins_indices[j] = new_index
                        plate_origins[j] = vertices[new_index]
                        need_reassignment = True  # Need to check all pairs again
                    else:
                        raise ValueError("Not enough vertices to maintain minimum distance")
    
    return plate_origins_indices