import numpy as np
import numpy.typing as npt
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
def pnoise3d(p: npt.NDArray[np.int32], x: np.float32, y: np.float32, z: np.float32) -> np.float32:
    iX = np.int32(np.floor(x)) & 255
    iY = np.int32(np.floor(y)) & 255
    iZ = np.int32(np.floor(z)) & 255
    
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
    
    result = lerp(t=w, a=lerp(t=v, a=lerp(t=u, a=grad(hash=p[AA], x=x, y=y, z=z),
            b=grad(hash=p[BA], x=x-1, y=y, z=z)), b=lerp(t=u, a=grad(hash=p[AB], x=x, y=y-1, z=z),
            b=grad(hash=p[BB], x=x-1, y=y-1, z=z))), b=lerp(t=v, a=lerp(t=u, a=grad(hash=p[AA+1], x=x, y=y, z=z-1),
            b=grad(hash=p[BA+1], x=x-1, y=y, z=z-1)), b=lerp(t=u, a=grad(hash=p[AB+1], x=x, y=y-1, z=z-1),
            b=grad(hash=p[BB+1], x=x-1, y=y-1, z=z-1))))
    return result

class PerlinNoise3D:
    def __init__(self):
        self.permutation: npt.NDArray[np.int32] = np.arange(256, dtype=np.int32)
        np.random.shuffle(self.permutation)
        self.p: npt.NDArray[np.int32] = np.tile(self.permutation, 2)
    
    def noise(self, x: np.float32, y: np.float32, z: np.float32) -> np.float32:
        return pnoise3d(self.p, x, y, z)

@time_function
@njit((types.UniTuple(float32[:,:], 2)(int32,float32,int32[:])),cache=True, parallel=True)
def noisebatch(num_points: np.int32, scale: np.float32, sp: np.ndarray) -> tuple:
    # Preallocate arrays for points and colors
    points: np.ndarray = np.zeros((num_points, 3), dtype=np.float32)
    colors: np.ndarray = np.zeros((num_points, 4), dtype=np.float32)
    point_count = 0
    
    for i in prange(num_points):
        x: np.float32 = np.float32(np.random.uniform(-scale, scale))
        y: np.float32 = np.float32(np.random.uniform(-scale, scale))
        z: np.float32 = np.float32(np.random.uniform(-scale, scale))
        
        # Use noise to create density variations and color variations
        noise_val1: np.float32 = pnoise3d(sp, x * np.float32(0.5), y * np.float32(0.5), z * np.float32(0.5))
        noise_val2: np.float32 = pnoise3d(sp, x * np.float32(0.3), y * np.float32(0.3), z * np.float32(0.3))
        noise_val3: np.float32 = pnoise3d(sp, x * np.float32(0.7), y * np.float32(0.7), z * np.float32(0.7))
        noise_val4: np.float32 = pnoise3d(sp, x * np.float32(0.7), y * np.float32(0.7), z * np.float32(0.7))
        
        if noise_val1 > 0.1:
            # Assign colors based on different noise patterns
            r = np.float32((noise_val1 + 1.0) * 0.5)  # Red from first noise
            g = np.float32((noise_val2 + 1.0) * 0.5)  # Green from second noise
            b = np.float32((noise_val3 + 1.0) * 0.5)  # Blue from third noise
            a = np.float32((noise_val4 + 1.0) * 0.5)  # Alpha from fourth noise
            
            # Normalize colors to be more vibrant
            max_val = max(r, g, b)
            if max_val > 0:
                r = r / max_val
                g = g / max_val
                b = b / max_val
                a = a / max_val
            
            points[point_count] = [x, y, z]
            colors[point_count] = [r, g, b, a]
            point_count += 1
    
    # Trim arrays to actual size
    return points[:point_count], colors[:point_count]

@time_function
def generate_point_cloud(num_pointsa: int, scalea: float, seeda: int) -> tuple:
    num_points: np.int32 = np.int32(num_pointsa)
    scale: np.float32 = np.float32(scalea)
    seed: np.int32 = np.int32(seeda)

    np.random.seed(seed)
    perlin: PerlinNoise3D = PerlinNoise3D()
    
    points, colors = noisebatch(num_points, scale, perlin.p)
    return points, colors

class VoxelGrid:
    def __init__(self, points: np.ndarray, colors: np.ndarray, voxel_size: np.float32):
        self.points: np.ndarray = points
        self.colors: np.ndarray = colors
        self.voxel_size: np.float32 = voxel_size
        
        # Find bounds
        self.min_bounds: np.float32 = np.min(points, axis=0)
        self.max_bounds: np.float32 = np.max(points, axis=0)
        
        # Calculate grid dimensions
        self.dims: np.int32 = np.int32(np.ceil((self.max_bounds - self.min_bounds) / voxel_size).astype(int) + 1)
        
        # Create voxel grid with colors
        self.grid: dict = {}
        self.color_grid: dict = {}
        
        for i, point in enumerate(points):
            voxel_idx: tuple = tuple(((point - self.min_bounds) // voxel_size).astype(np.int32))
            if voxel_idx not in self.grid:
                self.grid[voxel_idx] = []
                self.color_grid[voxel_idx] = []
            self.grid[voxel_idx].append(point)
            self.color_grid[voxel_idx].append(colors[i])
        
        # Create grid arrays for fast access
        self.grid_array: npt.NDArray[np.bool_] = np.zeros(self.dims, dtype=np.bool_)
        self.color_array: npt.NDArray[np.float32] = np.zeros((*self.dims, 4), dtype=np.float32)  # Now includes alpha
        self.count_array: npt.NDArray[np.int32] = np.zeros(self.dims, dtype=np.int32)
        
        for key in self.grid:
            self.grid_array[key[0], key[1], key[2]] = True
            # Average colors for this voxel
            avg_color = np.mean(self.color_grid[key], axis=0)
            self.color_array[key[0], key[1], key[2]] = avg_color
            self.count_array[key[0], key[1], key[2]] = len(self.color_grid[key])

class AmanatidesWooRayTracer:
    def __init__(self, voxel_grid: VoxelGrid, image_width: np.int32 = np.int32(800), image_height: np.int32 = np.int32(600)):
        self.voxel_grid: VoxelGrid = voxel_grid
        self.grid_array: npt.NDArray[np.bool_] = voxel_grid.grid_array
        self.color_array: npt.NDArray[np.float32] = voxel_grid.color_array
        self.width: np.int32 = image_width
        self.height: np.int32 = image_height
        
        # Camera parameters
        self.camera_pos: npt.NDArray[np.float32] = np.array([0, 0, 15], np.float32)
        self.look_at: npt.NDArray[np.float32] = np.array([0, 0, 0], np.float32)
        self.up: npt.NDArray[np.float32] = np.array([0, 1, 0], np.float32)
        
        # Calculate camera basis
        self.forward = (self.look_at - self.camera_pos).astype(np.float32)
        self.forward = normalize(self.forward)
        self.right = cross(self.forward, self.up)
        self.right = normalize(self.right)
        self.up = cross(self.right, self.forward)
        
        # Field of view
        self.fov: np.float32 = np.float32(60)  # degrees
        self.aspect_ratio: np.float32  = np.float32(self.width / self.height)
        
        # Calculate screen dimensions
        self.screen_height: np.float32  = 2 * np.tan(np.radians(self.fov) / 2)
        self.screen_width: np.float32  = self.screen_height * self.aspect_ratio
        
    def render(self):
        return numbarender(self.height, self.width, self.screen_width, self.screen_height,
                            self.forward, self.right, self.up,self.camera_pos, self.voxel_grid, 
                            self.grid_array, self.color_array)

@time_function
def numbarender(height, width, screen_width, screen_height, forward, right, up, camera_pos, 
                voxel_grid, voxel_grid_np, color_array):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    max_distance = np.float32(25.0)

    vbound = voxel_grid.min_bounds
    vsize = voxel_grid.voxel_size
    voxel_grid_dims = voxel_grid.dims
    image = _render_parallel(image, camera_pos, voxel_grid_np, color_array,
                            vsize, vbound, voxel_grid_dims, max_distance,
                            screen_height, screen_width, forward, right, up)
    
    return image

@time_function
@njit(parallel=True, cache=True)
def _render_parallel(image, ray_origin, voxel_grid, color_array, vsize, vbound, dims, max_distance,
                     screen_height, screen_width, forward, right, up):
    height: np.int32 = image.shape[0]
    width: np.int32 = image.shape[1]
    epsilon = np.float32(0.00000000001)
    max_t: np.float32 = np.float32(50.0)
    max_steps: np.int32 = np.int32(123)

    max_distance = np.float32(max_distance)
    screen_height = np.float32(screen_height)  
    screen_width = np.float32(screen_width)
    
    inv_width: np.float32 = np.float32(1.0) / width
    inv_height: np.float32 = np.float32(1.0) / height
    screen_width_half: np.float32 = screen_width * np.float32(0.5)
    screen_height_half: np.float32 = screen_height * np.float32(0.5)
    #print(height)
    #print(width)
    for y in prange(height):
        sy: np.float32 = np.float32((np.float32(1.0) - np.float32(2.0) * y * inv_height) * screen_height_half)
        for x in prange(width):
            sx: np.float32 = np.float32((np.float32(2.0) * x * inv_width - np.float32(1.0)) * screen_width_half)
            print(f"working in:  {x},{y}")
            ray_dir = forward + sx * right + sy * up
            ray_dir = normalize(ray_dir)
            print(f"current ray dir: {ray_dir}")

            current_voxel = ((ray_origin - vbound) // vsize).astype(np.int32)
            print(f"cell at: {current_voxel}")
            inv_dir = np.where(np.abs(ray_dir) > epsilon, ray_dir, np.copysign(epsilon, ray_dir))
            print(f"inverse of the current ray: {inv_dir}")
            step = np.sign(ray_dir)
            print(f"current ray signs: {step}")
            step_mask = np.greater(step, 0)
            next_voxel_bound = ((current_voxel + step_mask) * vsize + vbound)
            print(f"next cell at: {next_voxel_bound}")
            t_max = (next_voxel_bound - ray_origin) * inv_dir
            print(f"t_max at: {t_max}")
            t_delta = vsize / np.abs(inv_dir)
            print(f"t_delta: {t_delta}")
            t = np.float32(0.0)
            
            # Alpha compositing variables
            accumulated_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            accumulated_alpha = np.float32(0.0)
            for _ in range(max_steps):
                if not t < max_t:
                    break
                if accumulated_alpha >= 1.0:  # Fully opaque, stop tracing
                    break
                
                #print(f"checking cell at {current_voxel}")
                if np.all((0 <= current_voxel) & (current_voxel < dims)):
                    if voxel_grid[current_voxel[0], current_voxel[1], current_voxel[2]]:
                        # Get the color and alpha from the color array
                        voxel_color = color_array[current_voxel[0], current_voxel[1], current_voxel[2]]
                        rgba = voxel_color
                        color_rgb = rgba[:3]
                        alpha = rgba[3]
                        
                        # Apply alpha compositing: front-to-back
                        if alpha > 0:
                            # Weight by current transparency
                            weight = alpha * (1.0 - accumulated_alpha)
                            accumulated_color += color_rgb * weight
                            accumulated_alpha += weight
                
                min_axis = 0
                if t_max[1] < t_max[0]:
                    min_axis = 1
                if t_max[2] < t_max[min_axis]:
                    min_axis = 2
                current_voxel[min_axis] += step[min_axis]
                t = t_max[min_axis]
                t_max[min_axis] += t_delta[min_axis]

            # Set final pixel color
            if accumulated_alpha > 0:
                # Blend with background (white)
                final_color = accumulated_color + (1.0 - accumulated_alpha) * np.array([1.0, 1.0, 1.0])
                image[y, x, 0] = np.uint8(final_color[0] * 255)
                image[y, x, 1] = np.uint8(final_color[1] * 255)
                image[y, x, 2] = np.uint8(final_color[2] * 255)
    
    return image


# Generate point cloud with colors
print("Generating point cloud with colors...")
point_cloud, point_colors = generate_point_cloud(num_pointsa=150000, scalea=5.0, seeda=43)

# Create voxel grid with colors
print("Creating voxel grid with colors...")
voxel_size: np.float32 = np.float32(0.3)
voxel_grid: VoxelGrid = VoxelGrid(point_cloud, point_colors, voxel_size=voxel_size)

# Render using Amanatides and Woo algorithm
print("Rendering with Amanatides-Woo ray tracing...")
tracer = AmanatidesWooRayTracer(voxel_grid, image_width=50, image_height=50)
rendered_image = tracer.render()

# Save as PNG
print("Saving image...")
img = Image.fromarray(rendered_image)
img.save('point_cloud_rendered.png')
print("Image saved as 'point_cloud_rendered.png'")

print_timing_stats()