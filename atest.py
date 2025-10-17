import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import math
from util import normalize, time_function, print_timing_stats, norm, dot, cross, cross_2d
from numba import njit, prange, jit, float32, int32, typeof as numbatypeof, uint8, bool, types

epsilon = 0.00000001

#@time_function
@njit((float32(float32)), cache=True)
def fade(t: np.float32) -> np.float32:
    return 6*t**5 - 15*t**4 + 10*t**3

#@time_function
@njit((float32(float32,float32,float32)), cache=True)
def lerp(t: np.float32, a: np.float32, b: np.float32) -> np.float32:
    return a + t * (b - a)

#@time_function
@njit((float32(int32,float32,float32, float32)), cache=True)
def grad(hash: np.int32, x: np.float32, y: np.float32, z: np.float32) -> np.float32:
    h = hash & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


#@time_function
@njit((float32(int32[:],float32,float32,float32)), cache=True)
def pnoise3d(p: np.ndarray, x: np.float32, y: np.float32, z: np.float32) -> np.float32:
    iX = int(np.floor(x)) & 255
    iY = int(np.floor(y)) & 255
    iZ = int(np.floor(z)) & 255
    
    x -= np.floor(x)
    y -= np.floor(y)
    z -= np.floor(z)
    
    u = fade(x)
    v = fade(y)
    w = fade(z)
    
    A = p[iX] + iY
    AA = p[A] + iZ
    AB = p[A + 1] + iZ
    B = p[iX + 1] + iY
    BA = p[B] + iZ
    BB = p[B + 1] + iZ
    
    result = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
            grad(p[BA], x-1, y, z)), lerp(u, grad(p[AB], x, y-1, z),
            grad(p[BB], x-1, y-1, z))), lerp(v, lerp(u, grad(p[AA+1], x, y, z-1),
            grad(p[BA+1], x-1, y, z-1)), lerp(u, grad(p[AB+1], x, y-1, z-1),
            grad(p[BB+1], x-1, y-1, z-1))))
    return result

class PerlinNoise3D:
    def __init__(self):
        self.permutation: np.ndarray = np.arange(256, dtype=np.int32)
        np.random.shuffle(self.permutation)
        self.p: np.ndarray = np.tile(self.permutation, 2)
    
    def noise(self, x: np.float32, y: np.float32, z: np.float32) -> np.float32:
        return pnoise3d(self.p, x, y, z)

@time_function
@njit((float32[:,:](int32,float32,int32[:])),cache=True, parallel=True)
def noisebatch(num_points: np.int32, scale: np.float32, sp: np.ndarray) -> np.ndarray:
    # Preallocate array for maximum possible points
    points = np.zeros((num_points, 3), dtype=np.float32)
    
    for i in prange(num_points):
        x = np.float32(np.random.uniform(-scale, scale))
        y = np.float32(np.random.uniform(-scale, scale))
        z = np.float32(np.random.uniform(-scale, scale))
        
        # Use noise to create density variations
        noise_val = pnoise3d(sp, x * np.float32(0.5), y * np.float32(0.5), z * np.float32(0.5))
        
        if noise_val > 0.1:
            points[i] = [x, y, z]
    
    return points

@time_function
def generate_point_cloud(num_pointsa: int, scalea: float, seeda: int) -> np.ndarray:
    num_points = np.int32(num_pointsa)
    scale = np.float32(scalea)
    seed = np.int32(seeda)

    np.random.seed(seed)
    perlin = PerlinNoise3D()
    
    points = noisebatch(num_points, scale, perlin.p)
    return points

class VoxelGrid:
    def __init__(self, points: np.ndarray, voxel_size: np.float32):
        self.points = points
        self.voxel_size = voxel_size
        
        # Find bounds
        self.min_bounds = np.min(points, axis=0)
        self.max_bounds = np.max(points, axis=0)
        
        # Calculate grid dimensions
        self.dims = np.int32(np.ceil((self.max_bounds - self.min_bounds) / voxel_size).astype(int) + 1)
        
        # Create voxel grid
        #self.grid: np.ndarray = np.zeros(self.dims, dtype=np.bool_)
        self.grid = {}
        for point in points:
            voxel_idx = tuple(((point - self.min_bounds) // voxel_size).astype(int))
            if voxel_idx not in self.grid:
                self.grid[voxel_idx] = []
            self.grid[voxel_idx].append(point)
        self.grid_array = np.zeros(self.dims, dtype=np.bool_)
        for key in self.grid:
            self.grid_array[key[0], key[1], key[2]] = True

class AmanatidesWooRayTracer:
    def __init__(self, voxel_grid, image_width=800, image_height=600):
        self.voxel_grid = voxel_grid
        self.grid_array = voxel_grid.grid_array
        self.width = image_width
        self.height = image_height
        
        # Camera parameters
        self.camera_pos = np.array([0, 0, 15])  # Camera position
        self.look_at = np.array([0, 0, 0])      # Look at origin
        self.up = np.array([0, 1, 0])           # Up vector
        
        # Calculate camera basis
        self.forward = self.look_at - self.camera_pos
        self.forward = self.forward / np.linalg.norm(self.forward)
        self.right = np.cross(self.forward, self.up)
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        
        # Field of view
        self.fov = 60  # degrees
        self.aspect_ratio = self.width / self.height
        
        # Calculate screen dimensions
        self.screen_height = 2 * np.tan(np.radians(self.fov) / 2)
        self.screen_width = self.screen_height * self.aspect_ratio
        
    #@time_function
    def render(self):
        return numbarender(self.height, self.width, self.screen_width, self.screen_height,
                            self.forward, self.right, self.up,self.camera_pos, self.voxel_grid, self.grid_array)

@time_function
def numbarender(height, width, screen_width, screen_height, forward, right, up, camera_pos, voxel_grid, voxel_grid_np):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    max_distance = np.float32(25.0)

    vbound = voxel_grid.min_bounds
    vsize = voxel_grid.voxel_size
    voxel_grid_dims = voxel_grid.dims
    image = _render_parallel(image,  camera_pos, voxel_grid_np,
                            vsize, vbound, voxel_grid_dims, max_distance,
                            screen_height, screen_width, forward, right, up)
    
    return image

#@time_function
@njit(cache=True)
def voxel_traverse(ray_origin, ray_dir, voxel_grid, voxel_grid_voxel_size, voxel_grid_min_bounds, voxel_grid_dims):
    current_voxel = np.empty(3, dtype=np.int32)
    inv_dir = np.empty(3, dtype=np.float32)
    for i in range(3):
        current_voxel[i] = np.int32((ray_origin[i] - voxel_grid_min_bounds[i]) // voxel_grid_voxel_size)
        inv_dir[i] = ray_dir[i] if abs(ray_dir[i]) > epsilon else math.copysign(epsilon, ray_dir[i])
    max_t = 50.0
    step = np.sign(ray_dir)
    step_mask = (step > 0)
    
    next_voxel_boundary = ((current_voxel + step_mask) * voxel_grid_voxel_size + voxel_grid_min_bounds)

    t_max = (next_voxel_boundary - ray_origin) * inv_dir
    t_delta = voxel_grid_voxel_size / np.abs(inv_dir)

    t = 0.0
    max_steps = int(max_t / np.min(t_delta)) + 10

    for _ in range(max_steps):
        if not t < max_t:
            break
        voxel_index = (int(current_voxel[0]), int(current_voxel[1]), int(current_voxel[2]))
        
        if (0 <= current_voxel[0] < voxel_grid_dims[0] and 
            0 <= current_voxel[1] < voxel_grid_dims[1] and 
            0 <= current_voxel[2] < voxel_grid_dims[2]):
            
            if voxel_grid[voxel_index[0], voxel_index[1], voxel_index[2]]:
            #if voxel_grid[current_voxel[0], current_voxel[1], current_voxel[2]]:
                return True, t
        
        # Find next voxel
        min_axis = 0
        if t_max[1] < t_max[0]:
            min_axis = 1
        if t_max[2] < t_max[1]:
            min_axis = 2
        current_voxel[min_axis] += step[min_axis]
        t = t_max[min_axis]
        t_max[min_axis] += t_delta[min_axis]
    
    return False, 0.0

@njit(parallel=True, cache=True)
def _render_parallel(image, camera_pos, voxel_grid, vsize, vbound, dims, max_distance,
                     screen_height, screen_width, forward, right, up):
    height, width = image.shape[0], image.shape[1]
    
    inv_width = 1.0 / width
    inv_height = 1.0 / height
    screen_width_half = screen_width * 0.5
    screen_height_half = screen_height * 0.5

    for y in prange(height):
        sy = (1.0 - 2.0 * y * inv_height) * screen_height_half
        for x in range(width):
            sx = (2.0 * x * inv_width - 1.0) * screen_width_half
            ray_dir = forward + sx * right + sy * up
            ray_dir_norm = math.sqrt(ray_dir[0]*ray_dir[0] + ray_dir[1]*ray_dir[1] + ray_dir[2]*ray_dir[2])
            ray_dir = ray_dir / ray_dir_norm
            hit, distance = voxel_traverse(camera_pos, ray_dir, voxel_grid, vsize, vbound, dims)

            if hit:
                t = min(distance / max_distance, 1.0)
                r = int(t * 255)
                b = int((1 - t) * 255)
                image[y, x, 0] = r
                image[y, x, 1] = 0
                image[y, x, 2] = b
    
    return image



# Generate point cloud
print("Generating point cloud...")
point_cloud = generate_point_cloud(num_pointsa=15000, scalea=5.0, seeda=43)

# Create voxel grid
print("Creating voxel grid...")
voxel_size: np.float32 = np.float32(0.3)
voxel_grid: VoxelGrid = VoxelGrid(point_cloud, voxel_size=voxel_size)

# Render using Amanatides and Woo algorithm
print("Rendering with Amanatides-Woo ray tracing...")
tracer = AmanatidesWooRayTracer(voxel_grid, image_width=800, image_height=600)
rendered_image = tracer.render()

# Save as PNG
print("Saving image...")
img = Image.fromarray(rendered_image)
img.save('point_cloud_rendered.png')
print("Image saved as 'point_cloud_rendered.png'")

print_timing_stats()