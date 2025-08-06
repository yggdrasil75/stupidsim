import numpy as np

from globals import DEVICE
from holder.mesh import mesh
from holder.DeformableMesh import DeformableMesh

def create_sphere_mesh(radius=1.0, segments=16, rings=16, deformable=True):
    """
    Generate vertices and polygons for a topologically closed sphere using NumPy.
    This version avoids duplicate vertices at the seam to prevent issues with neighbor finding.
    """
    # Create theta and phi angles
    # theta goes from pole to pole (e.g., latitude)
    theta = np.linspace(0, np.pi, rings + 1, dtype=np.float32)
    
    # phi goes around the equator (e.g., longitude)
    # We generate `segments` points, not `segments + 1`, to avoid a duplicate seam.
    # The `[:-1]` removes the `2 * pi` endpoint which is identical to the `0` start point.
    phi = np.linspace(0, 2 * np.pi, segments + 1, dtype=np.float32)[:-1]
    
    # Create grid of angles
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Calculate spherical coordinates
    sin_theta = np.sin(theta_grid)
    cos_theta = np.cos(theta_grid)
    sin_phi = np.sin(phi_grid)
    cos_phi = np.cos(phi_grid)
    
    # Convert to Cartesian coordinates
    x = cos_phi * sin_theta
    y = cos_theta
    z = sin_phi * sin_theta
    
    # Stack coordinates and reshape to (N, 3)
    # The number of vertices is (rings + 1) * segments
    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1, dtype=np.float32) * radius
    
    # Generate polygons (quads converted to triangles)
    polygons = []
    for ring in range(rings):
        for segment in range(segments):
            # Vertex indices are based on a grid of size `segments` per ring.
            # Current vertex index in the current ring
            v0 = ring * segments + segment
            # Next vertex in the same ring (wraps around using modulo)
            v1 = ring * segments + (segment + 1) % segments
            # Next vertex in the next ring (wraps around using modulo)
            v2 = (ring + 1) * segments + (segment + 1) % segments
            # Current vertex in the next ring
            v3 = (ring + 1) * segments + segment
            
            # The top and bottom rings of triangles will be degenerate (zero area),
            # which is a standard and acceptable way to form the poles.
            # The important part is that the connectivity is correct.

            # Split quad into two triangles
            polygons.append([v0, v1, v2])
            polygons.append([v0, v2, v3])
    
    polys = np.array(polygons, dtype=np.int32)

    # Create a sphere mesh
    if deformable:
        sphere_mesh = DeformableMesh(
            id=0,
            _vertices=vertices,
            _polys=polys,
            _color=np.tile(np.array([[0.8, 0.2, 0.2]], dtype=np.float32), (len(vertices), 1))
        )
    else:
        sphere_mesh = mesh(
            id=0,
            _vertices=vertices,
            _polys=polys,
            _color=np.tile(np.array([[0.8, 0.2, 0.2]], dtype=np.float32), (len(vertices), 1))
        )
    return sphere_mesh