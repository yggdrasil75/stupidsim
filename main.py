import matplotlib.pyplot as plt
from functools import lru_cache

from holder.globals import ELEVATION_MAX, ELEVATION_MIN, PLATES, SHAPE, SPHERE_SIZE, SUBDIVISIONS
from shapes.cube import Cube
from shapes.icosahedron import Icosahedron
from shapes.truncatedicosahedron import TruncatedIcosahedron
from shapes.TruncatedIcosidodecahedron import TruncatedIcosidodecahedron
from shapes.truncatedtetrahedron import truncatedTetrahedron
from system import System
from systemviwer import SystemViewer
from util import print_timing_stats

# --- Main Execution ---
if __name__ == "__main__":
    shape_type = SHAPE
    num_subdivisions = SUBDIVISIONS
    sphere_radius = SPHERE_SIZE
    plates = PLATES
    elevationmin = ELEVATION_MIN
    elevationmax = ELEVATION_MAX

    if shape_type == "icosahedron":
        shape = Icosahedron()
        shape.create_world(radius=sphere_radius)
        print(f"\nInitial Icosahedron created.")
    elif shape_type == "truncated":
        shape = TruncatedIcosahedron()
        shape.create_world(radius=sphere_radius)
        print(f"\nInitial Truncated Icosahedron created.")
    elif shape_type == "cube":
        shape = Cube()
        shape.create_world(radius=sphere_radius)
        print(f"\nInitial cube created.")
    elif shape_type == "tetrahedron":
        shape = truncatedTetrahedron()
        shape.create_world(radius=sphere_radius)
        print(f"\nInitial Truncated Icosahedron created.")
    elif shape_type == "TruncatedIcosidodecahedron":
        shape = TruncatedIcosidodecahedron()
        shape.create_world(radius=sphere_radius)
        print(f"\nInitial Truncated Icosadodecahedron created.")
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    shape.subdivide(sphere_radius, num_subdivisions)
    system = System(2, 1000)
    system.add_world(shape, 1, 365.25)
    system.update_environment()

    shape.genElevations(plates, elevationmin, elevationmax)
    #shape.simulate_water()

    # print(f'plates have the following vertex count: ')
    # for plate in shape.plates.values():
    #     print(f'{plate.plate_id} has {len(plate.vertices)}')
    
    #fig, ax, radio = shape.plot()
    # system.plot_system()
    # system.plot_world(0)
    # system.plot_world_environment(0)
    viewer = SystemViewer(system)
    viewer.launch()

    #ax.set_title(f'Sphere Approx. ({shape_type.capitalize()} Subdivided {num_subdivisions} Times)')
    print_timing_stats()
    plt.show(block=True)
    
