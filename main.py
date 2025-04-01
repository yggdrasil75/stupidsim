import icosphere
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Generate the icosphere
subdivisions = 3
radius = 6371  # Earth's radius in km
elevation_range = 10000
world = icosphere.initializeWorld(subdivisions, radius, elevation_range)


def plot_icosphere(world):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract vertices and faces
    vertices = np.array([[v.x, v.y, v.z] for v in world.vertices])
    faces = np.array([[world.vertex_indices[f.a], world.vertex_indices[f.b], world.vertex_indices[f.c]] for f in world.faces])

    # Create the 3D plot
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, cmap='terrain', linewidth=0.2, antialiased=True)


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Icosphere")
    ax.set_aspect("equal")

    plt.show()

plot_icosphere(world)




# Example of how to use other functions:
# Get latitude and longitude of a vertex
lat, lon = icosphere.cartesianLatLon(world.vertices[0])
print(f"Latitude: {lat}, Longitude: {lon}")


# Calculate distance between two vertices
distance = icosphere.sphericalDistanceCartesian(world.vertices[0], world.vertices[10], radius)
print(f"Distance: {distance} km")

# Find neighbors of a vertex
neighbors = icosphere.findSphericalNeighbors(world.vertices, world.faces, 0, 1000, radius)
print(f"Neighbors of vertex 0: {[world.vertex_indices[v]  for v in neighbors]}")  # Print indices for readability

# Calculate slope at a vertex
slope = icosphere.calculateSlope(world.vertices, world.faces, 0, radius)
print(f"Slope at vertex 0: {slope} radians")


# Run simulation (currently just returns the same world state multiple times)
face_history, vertex_indices = icosphere.run_simulation(subdivisions=3, radius=radius, elevationRange=elevation_range, steps=5)
# face_history will contain a list of face lists, one for each step of the simulation