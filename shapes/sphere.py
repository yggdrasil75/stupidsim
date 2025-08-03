import torch

from globals import DEVICE
from holder.mesh import mesh
from holder.DeformableMesh import DeformableMesh

def create_sphere_mesh(radius=1.0, segments=16, rings=16, device=DEVICE, deformable: bool = True):
    """Generate vertices and polygons for a sphere using PyTorch"""
    # Create theta and phi angles
    theta = torch.linspace(0, torch.pi, rings + 1, device=device)
    phi = torch.linspace(0, 2 * torch.pi, segments + 1, device=device)
    
    # Create grid of angles
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
    
    # Calculate spherical coordinates
    sin_theta = torch.sin(theta_grid)
    cos_theta = torch.cos(theta_grid)
    sin_phi = torch.sin(phi_grid)
    cos_phi = torch.cos(phi_grid)
    
    # Convert to Cartesian coordinates
    x = cos_phi * sin_theta
    y = cos_theta
    z = sin_phi * sin_theta
    
    # Stack coordinates and reshape to (N, 3)
    vertices = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1) * radius
    
    # Generate polygons (quads converted to triangles)
    polygons = []
    for ring in range(rings):
        for segment in range(segments):
            v0 = ring * (segments + 1) + segment
            v1 = v0 + 1
            v2 = (ring + 1) * (segments + 1) + segment + 1
            v3 = (ring + 1) * (segments + 1) + segment
            
            # Split quad into two triangles
            polygons.append([v0, v1, v2])
            polygons.append([v0, v2, v3])
    
    polys= torch.tensor(polygons, dtype=torch.long, device=device)


    # Create a sphere mesh
    if deformable:
        sphere_mesh = DeformableMesh(
        id=0,
        _vertices=vertices,
        _polys=polys,
        _color=torch.tensor([[0.8, 0.2, 0.2]], dtype=torch.float32, device=DEVICE).expand(len(vertices), -1)
    )
    else:
        sphere_mesh = mesh(
            id=0,
            _vertices=vertices,
            _polys=polys,
            _color=torch.tensor([[0.8, 0.2, 0.2]], dtype=torch.float32, device=DEVICE).expand(len(vertices), -1)
        )
    return sphere_mesh