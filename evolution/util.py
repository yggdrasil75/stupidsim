from collections import deque, defaultdict
import random
import numpy as np
import time
from functools import wraps


_timing_stats = defaultdict(lambda: {'total_time': 0, 'call_count': 0})

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
        

def time_function(func):
    """
    Decorator to time function execution and store statistics.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        _timing_stats[func.__name__]['total_time'] += elapsed
        _timing_stats[func.__name__]['call_count'] += 1
        
        return result
    return wrapper

def get_timing_stats():
    """
    Returns timing statistics for all timed functions.
    
    Returns:
        dict: A dictionary with function names as keys and dictionaries 
              containing 'total_time' and 'avg_time' as values.
    """
    stats = {}
    for func_name, data in _timing_stats.items():
        if data['call_count'] > 0:
            stats[func_name] = {
                'total_time': data['total_time'],
                'avg_time': data['total_time'] / data['call_count'],
                'call_count': data['call_count']
            }
    return stats

def print_timing_stats():
    """
    Prints formatted timing statistics for all timed functions.
    """
    stats = get_timing_stats()
    if not stats:
        print("No timing statistics available.")
        return
    
    print("\nFunction Timing Statistics:")
    print("-" * 50)
    print(f"{'Function':<30} {'Calls':<10} {'Total Time (s)':<15} {'Avg Time (s)':<15}")
    print("-" * 50)
    
    for func_name, data in stats.items():
        print(f"{func_name:<30} {data['call_count']:<10} {data['total_time']:<15.6f} {data['avg_time']:<15.6f}")
    print("-" * 50)