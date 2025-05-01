
from collections import deque
import random

import numpy as np


def cart_to_sphere(p):
    x, y, z = p
    lon = np.atan2(y, x)
    lat = np.asin(z)
    return lat, lon

def plate_grow_worker_process(plate_id, work_queue_indices, plate_id_array, unassigned_indices_queue, assignment_lock, vertices_neighbors_getter):
    assigned_in_this_run = [] 
    find_neighbors = vertices_neighbors_getter
    local_queue = deque(work_queue_indices)

    while True:
        try:
            current_assigned_idx = local_queue.popleft()
        except IndexError:
            break 

        neighbors = find_neighbors(current_assigned_idx)
        potential_unassigned_neighbors = []

        plate_ids_vals = plate_id_array.get_obj() 
        for n_idx in neighbors:
             if plate_ids_vals[n_idx] == -1:
                 potential_unassigned_neighbors.append(n_idx)

        if not potential_unassigned_neighbors:
            continue

        if potential_unassigned_neighbors:
            selected_neighbor_idx = random.choice(potential_unassigned_neighbors)
        else:
            continue

        if selected_neighbor_idx != -1:
            with assignment_lock:
                current_plate_id_val = plate_id_array[selected_neighbor_idx]
                if current_plate_id_val == -1:
                    plate_id_array[selected_neighbor_idx] = plate_id
                    assigned_in_this_run.append(selected_neighbor_idx)

    if assigned_in_this_run:
        unassigned_indices_queue.put((plate_id, assigned_in_this_run))

@staticmethod
def batch_calculate_areas(faces, vertex_list, device='cuda'):
    facearea = []
    for face in faces:
        facearea.append(face.calculate_area(vertex_list))
        
