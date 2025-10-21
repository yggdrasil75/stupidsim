import bisect
from collections import deque, defaultdict
import random
import numpy as np
import time
from functools import wraps
from numba import njit, float32, types, int64, int32, cuda, jit, prange
#from numba.cuda import jit
from numba.extending import overload
from numpy._typing._array_like import NDArray
import numpy.typing as npt
from typing import Literal

_timing_stats = defaultdict(lambda: {
    'timings': np.zeros(0, dtype=np.float32),  # Using numpy array for timings
    'total_time': 0.0,
    'call_count': 0,
})

@njit((float32[:](float32[:], float32[:])), cache=True)
def cross(a, b) -> NDArray:
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]], dtype=np.float32)

@njit((float32[:,:](float32[:,:], float32[:,:])), cache=True)
def cross_2d(a, b):
    result = np.empty_like(a)
    for i in range(a.shape[0]):
        result[i, 0] = a[i, 1]*b[i, 2] - a[i, 2]*b[i, 1]
        result[i, 1] = a[i, 2]*b[i, 0] - a[i, 0]*b[i, 2]
        result[i, 2] = a[i, 0]*b[i, 1] - a[i, 1]*b[i, 0]
    return result

@njit(float32(float32[:]), cache=True)
def norm(vec):
    sum_sq = np.float32(0.0)
    for x in vec:
        sum_sq += x * x
    return np.sqrt(sum_sq)

@njit(float32[:](float32[:]), cache=True)
def normalize(vec: npt.NDArray) -> npt.NDArray:
    n = norm(vec)
    if n < 1e-10:
        return vec
    return vec / n

@njit(cache=True)
def vnorm(vectors):
    """Compute norm for array of vectors [..., 3]"""
    result = np.empty(vectors.shape[:-1], dtype=vectors.dtype)
    for i in prange(vectors.shape[0]):
        for j in range(vectors.shape[1]):
            x = vectors[i, j, 0]
            y = vectors[i, j, 1]
            z = vectors[i, j, 2]
            result[i, j] = np.sqrt(x*x + y*y + z*z)
    return result

@njit
def dot(v1, v2):
    """Faster dot product for 3D vectors"""
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

@njit
def triangle_error_quadric(v0: NDArray[np.float32], v1: NDArray[np.float32], v2: NDArray[np.float32]) -> NDArray[np.float32]:
    normal = cross(v1 - v0, v2 - v0)
    normal = normalize(normal)
    d = -np.dot(normal, v0)
    plane = np.append(normal, d)
    
    K = np.outer(plane, plane)
    area = 0.5 * norm(cross(v1 - v0, v2 - v0))
    K *= area
    return K

@njit
def clip_value(value, min_val, max_val):
    """Clip a value between min and max."""
    return min(max(value, min_val), max_val)

def make_2D_array(lis):
    """Function to get 2D array from a list of lists
    """
    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = np.max(lengths)
    arr = np.zeros((n, max_len), np.float32)

    for i in range(n):
        arr[i, :lengths[i]] = lis[i]
    return arr, lengths

def time_function(func):
    #Decorator to time function execution and store statistics
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        stats = _timing_stats[func.__name__]
        stats['total_time'] += elapsed
        stats['call_count'] += 1
        
        # Efficiently grow numpy array for timings
        if stats['timings'].size == 0:
            stats['timings'] = np.array([elapsed], dtype=np.float32)
        else:
            stats['timings'] = np.append(stats['timings'], elapsed)
            
        return result
    return wrapper

def _calculate_percentiles(timings):
    """Helper function to calculate percentiles using numpy for efficiency"""
    if timings.size == 0:
        return {}
    
    return {
        'p99.9': np.percentile(timings, 99.9),
        'p99': np.percentile(timings, 99),
        'p95': np.percentile(timings, 95),
        'p90': np.percentile(timings, 90),
        'max': np.max(timings),
        'min': np.min(timings),
        'median': np.median(timings)
    }

def get_timing_stats(mode: Literal['basic', 'enhanced'] = 'basic'):
    """
    Returns timing statistics for all timed functions.
    
    Args:
        mode: 'basic' for minimal stats (calls, total, average), 
              'enhanced' for full stats with percentiles
    
    Returns:
        dict: A dictionary with function names as keys and statistics dictionaries
    """
    stats = {}
    for func_name, data in _timing_stats.items():
        if data['call_count'] > 0:
            stats[func_name] = {
                'call_count': data['call_count'],
                'total_time': data['total_time'],
                'avg_time': data['total_time'] / data['call_count'],
            }
            
            if mode == 'enhanced' and data['timings'].size > 0:
                
                stats[func_name]['percentiles'] = _calculate_percentiles(data['timings'])
    
    return stats


def print_timing_stats(mode: Literal['basic', 'enhanced'] = 'enhanced'):
    """
    Prints formatted timing statistics with control over verbosity.
    
    Args:
        mode: 'basic' for minimal output, 'enhanced' for full output with percentiles
    """
    stats = get_timing_stats(mode)
    if not stats:
        print("No timing statistics available.")
        return

    # Determine dynamic width for function name
    func_col_width = max(len(name) for name in stats.keys())
    func_col_width = max(func_col_width, len("Function"))  # ensure at least header width

    # Numeric formatting settings
    num_width = 12  # ensures consistent alignment (8 digits total + padding)
    float_fmt = f"{{:<{num_width}.6f}}"
    int_fmt = f"{{:<{num_width}d}}"

    if mode == 'basic':
        print("\nBasic Function Timing Statistics:")
        print("-" * (func_col_width + 3 * num_width + 8))
        header = (f"{'Function':<{func_col_width}} "
                  f"{'Calls':<{num_width}} {'Total (s)':<{num_width}} {'Avg (s)':<{num_width}}")
        print(header)
        print("-" * (func_col_width + 3 * num_width + 8))
        
        for func_name, data in stats.items():
            print(f"{func_name:<{func_col_width}} "
                  f"{int_fmt.format(data['call_count'])} "
                  f"{float_fmt.format(data['total_time'])} "
                  f"{float_fmt.format(data['avg_time'])}")
        
        print("-" * (func_col_width + 3 * num_width + 8))

    else:  # enhanced mode
        print("\nEnhanced Function Timing Statistics:")
        col_labels = ['Function', 'Calls', 'Total (s)', 'Avg (s)', 'Min (s)', 
                      'Median (s)', 'P99 (s)', 'P99.9 (s)', 'Max (s)']
        print("-" * (func_col_width + (len(col_labels) - 1) * num_width + 8))
        header = (f"{'Function':<{func_col_width}} "
                  f"{'Calls':<{num_width}} {'Total (s)':<{num_width}} {'Avg (s)':<{num_width}} "
                  f"{'Min (s)':<{num_width}} {'Median (s)':<{num_width}} "
                  f"{'P99 (s)':<{num_width}} {'P99.9 (s)':<{num_width}} {'Max (s)':<{num_width}}")
        print(header)
        print("-" * (func_col_width + (len(col_labels) - 1) * num_width + 8))
        
        for func_name, data in stats.items():
            p = data.get('percentiles', {})
            print(f"{func_name:<{func_col_width}} "
                  f"{int_fmt.format(data['call_count'])} "
                  f"{float_fmt.format(data['total_time'])} "
                  f"{float_fmt.format(data['avg_time'])} "
                  f"{float_fmt.format(p.get('min', 0))} "
                  f"{float_fmt.format(p.get('median', 0))} "
                  f"{float_fmt.format(p.get('p99', 0))} "
                  f"{float_fmt.format(p.get('p99.9', 0))} "
                  f"{float_fmt.format(p.get('max', 0))}")
        
        print("-" * (func_col_width + (len(col_labels) - 1) * num_width + 8))

@njit
def spherical_distance(point1: np.ndarray, point2: np.ndarray, radius: float) -> float:
    unit_vector1 = normalize(point1)
    unit_vector2 = normalize(point2)
    
    # Calculate the dot product between the two vectors
    dot_product = np.dot(unit_vector1, unit_vector2)
        
    # Calculate the central angle between the two points
    central_angle = np.arccos(dot_product)
    
    # Calculate the great-circle distance
    distance = radius * central_angle
    
    return distance