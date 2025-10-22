import numpy as np
from numba import int32, float64, boolean, types, deferred_type
from numba.experimental import jitclass
from numba.typed import List
import math

# Define the spec for the Octree node
node_spec = [
    ('center', float64[:]),
    ('half_size', float64),
    ('points', float64[:,:]),
    ('children', types.ListType(types.int32)),
    ('is_leaf', boolean),
    ('point_count', int32),
    ('max_points', int32),
    ('max_depth', int32)
]

@jitclass(spec=node_spec)
class OctreeNode:
    def __init__(self, center, half_size, max_points, max_depth):
        self.center = center
        self.half_size = half_size
        self.points = np.zeros((max_points, 3), dtype=np.float64)
        self.children = List.empty_list(types.int32)
        self.is_leaf = True
        self.point_count = 0
        self.max_points = max_points
        self.max_depth = max_depth
    
    def insert_point(self, point, nodes, depth=0):
        """Insert a point into the octree"""
        if not self.is_leaf:
            # Find which child should contain this point
            child_index = self._get_child_index(point)
            if child_index < len(self.children):
                nodes[self.children[child_index]].insert_point(point, nodes, depth + 1)
            return
        
        # Add point to this leaf node
        if self.point_count < self.max_points or depth >= self.max_depth:
            if self.point_count < len(self.points):
                self.points[self.point_count] = point
                self.point_count += 1
            return
        
        # Split the node
        self._split(nodes)
        
        # Redistribute existing points
        for i in range(self.point_count):
            child_index = self._get_child_index(self.points[i])
            if child_index < len(self.children):
                nodes[self.children[child_index]].insert_point(self.points[i], nodes, depth + 1)
        
        # Clear this node's points
        self.point_count = 0
        self.is_leaf = False
        
        # Insert the new point
        child_index = self._get_child_index(point)
        if child_index < len(self.children):
            nodes[self.children[child_index]].insert_point(point, nodes, depth + 1)
    
    def _split(self, nodes):
        """Split the current node into 8 children"""
        child_half_size = self.half_size / 2.0
        
        # Create 8 child nodes for each octant
        for i in range(8):
            child_center = self._get_child_center(i, child_half_size)
            child_node = OctreeNode(child_center, child_half_size, self.max_points, self.max_depth)
            # Use direct list append instead of storing indices
            nodes.append(child_node)
            self.children.append(len(nodes) - 1)
    
    def _get_child_center(self, index, child_half_size):
        """Calculate the center of a child node based on its index"""
        x = self.center[0] + child_half_size * (1 if (index & 1) else -1)
        y = self.center[1] + child_half_size * (1 if (index & 2) else -1)
        z = self.center[2] + child_half_size * (1 if (index & 4) else -1)
        return np.array([x, y, z], dtype=np.float64)
    
    def _get_child_index(self, point):
        """Determine which child octant contains the point"""
        index = 0
        if point[0] >= self.center[0]:
            index |= 1
        if point[1] >= self.center[1]:
            index |= 2
        if point[2] >= self.center[2]:
            index |= 4
        return index
    
    def query_range(self, center, radius, nodes, result):
        """Find all points within a sphere defined by center and radius"""
        if not self._intersects_sphere(center, radius):
            return
        
        if self.is_leaf:
            for i in range(self.point_count):
                point = self.points[i]
                if self._distance_squared(point, center) <= radius * radius:
                    result.append(point.copy())
        else:
            for child_index in self.children:
                nodes[child_index].query_range(center, radius, nodes, result)
    
    def _intersects_sphere(self, center, radius):
        """Check if this node's bounding box intersects with a sphere"""
        # Calculate the squared distance from sphere center to node's AABB
        d_min = 0.0
        for i in range(3):
            diff = center[i] - self.center[i]
            d_min += max(0, abs(diff) - self.half_size) ** 2
        
        return d_min <= radius * radius
    
    def _distance_squared(self, p1, p2):
        """Calculate squared distance between two points"""
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2

# Define the spec for the main Octree class
octree_spec = [
    ('root_index', int32),
    ('nodes', types.ListType(deferred_type())),  # Use deferred_type instead of specific class
    ('bounds_min', float64[:]),
    ('bounds_max', float64[:]),
    ('max_points_per_node', int32),
    ('max_depth', int32)
]

# Create a deferred type for the node
node_type = OctreeNode.class_type.instance_type

@jitclass(spec=octree_spec)
class Octree:
    def __init__(self, bounds_min, bounds_max, max_points_per_node=8, max_depth=10):
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.max_points_per_node = max_points_per_node
        self.max_depth = max_depth
        
        # Calculate center and half-size
        center = (bounds_min + bounds_max) / 2.0
        half_size = np.max(bounds_max - bounds_min) / 2.0
        
        # Create nodes list with deferred type
        self.nodes = List.empty_list(node_type)
        root_node = OctreeNode(center, half_size, max_points_per_node, max_depth)
        self.nodes.append(root_node)
        self.root_index = 0
    
    def insert(self, point):
        """Insert a point into the octree"""
        # Check if point is within bounds
        if not (np.all(point >= self.bounds_min) and np.all(point <= self.bounds_max)):
            return False
        
        self.nodes[self.root_index].insert_point(point, self.nodes)
        return True
    
    def insert_batch(self, points):
        """Insert multiple points into the octree"""
        for i in range(len(points)):
            self.insert(points[i])
    
    def query_sphere(self, center, radius):
        """Query all points within a sphere"""
        result = List.empty_list(float64[:])
        self.nodes[self.root_index].query_range(center, radius, self.nodes, result)
        return result
    
    def get_all_points(self):
        """Get all points in the octree"""
        result = List.empty_list(float64[:])
        self._collect_points(self.root_index, result)
        return result
    
    def _collect_points(self, node_index, result):
        """Recursively collect all points from the octree"""
        node = self.nodes[node_index]
        if node.is_leaf:
            for i in range(node.point_count):
                result.append(node.points[i].copy())
        else:
            for child_index in node.children:
                self._collect_points(child_index, result)

# Helper function to create an octree from point cloud data
def create_octree_from_points(points, max_points_per_node=8, max_depth=10):
    """Create an octree from a numpy array of points"""
    if len(points) == 0:
        raise ValueError("Points array is empty")
    
    bounds_min = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])])
    bounds_max = np.array([np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])])
    
    octree = Octree(bounds_min, bounds_max, max_points_per_node, max_depth)
    octree.insert_batch(points)
    
    return octree

# Example usage and testing
def example_usage():
    # Generate some random 3D points
    np.random.seed(42)
    points = np.random.rand(100, 3) * 100.0  # Reduced from 1000 to 100 for faster testing
    
    # Create octree
    octree = create_octree_from_points(points, max_points_per_node=16, max_depth=8)
    
    # Query points within a sphere
    query_center = np.array([50.0, 50.0, 50.0])
    query_radius = 20.0
    
    results = octree.query_sphere(query_center, query_radius)
    print(f"Found {len(results)} points within radius {query_radius}")
    
    # Get all points back
    all_points = octree.get_all_points()
    print(f"Total points in octree: {len(all_points)}")
    
    # Verify all original points are in the octree
    original_set = set(tuple(p) for p in points)
    octree_set = set(tuple(p) for p in all_points)
    
    print(f"All original points found in octree: {len(original_set - octree_set) == 0}")

if __name__ == "__main__":
    example_usage()