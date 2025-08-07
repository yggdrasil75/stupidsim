import bisect
from collections import deque, defaultdict
import random
import numpy as np
import time
from functools import wraps
from numba import njit, float32, types, int64
from numba.extending import overload
from numpy._typing._array_like import NDArray
import numpy.typing
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
def normalize(vec):
    n = norm(vec)
    if n < 1e-10:
        return vec
    return vec / n

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
    """
    Decorator to time function execution and store statistics, using numpy arrays for efficiency.
    """
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
    
    if mode == 'basic':
        print("\nBasic Function Timing Statistics:")
        print("-" * 60)
        print(f"{'Function':<25} {'Calls':<8} {'Total (s)':<12} {'Avg (s)':<12}")
        print("-" * 60)
        
        for func_name, data in stats.items():
            print(f"{func_name:<25} {data['call_count']:<8} {data['total_time']:<12.6f} {data['avg_time']:<12.6f}")
        
        print("-" * 60)
    
    else:  # enhanced mode
        print("\nEnhanced Function Timing Statistics:")
        print("-" * 110)
        header = (f"{'Function':<25} {'Calls':<8} {'Total (s)':<10} {'Avg (s)':<10} "
                  f"{'Min (s)':<10} {'Median (s)':<10} {'P99 (s)':<10} {'P99.9 (s)':<10} {'Max (s)':<10}")
        print(header)
        print("-" * 110)
        
        for func_name, data in stats.items():
            p = data.get('percentiles', {})
            print(f"{func_name:<25} {data['call_count']:<8} {data['total_time']:<10.6f} {data['avg_time']:<10.6f} "
                  f"{p.get('min', 0):<10.6f} {p.get('median', 0):<10.6f} "
                  f"{p.get('p99', 0):<10.6f} {p.get('p99.9', 0):<10.6f} {p.get('max', 0):<10.6f}")
        
        print("-" * 110)
