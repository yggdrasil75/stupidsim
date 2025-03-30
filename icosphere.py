import numpy as np
from typing import Tuple

def generate_icosphere(subdivisions: int = 3, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an icosphere mesh.
    
    Args:
        subdivisions: Number of subdivisions (default: 3)
        radius: Radius of the sphere (default: 1.0)
        
    Returns:
        Tuple of (vertices, faces) where:
        - vertices is a Nx3 numpy array of vertex positions
        - faces is a Mx3 numpy array of triangle indices
    """
    import icosphere as _icosphere
    vertices, faces = _icosphere.generate_icosphere(subdivisions, radius)
    
    # Convert to numpy arrays
    vert_array = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
    face_array = np.array([[f.a, f.b, f.c] for f in faces])
    
    #print(type(vert_array))
    #print(vert_array)

    return vert_array, face_array