import numpy as np
from typing import Tuple
import icosphere
from globals import PLANET_RADIUS_KM

def _generate_icosphere(subdivisions: int = 3, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    vertices, faces, vertexIndices = icosphere.generate_icosphere(subdivisions, radius)
    
    # Convert to numpy arrays
    vert_array = np.array([vertexToNparray(v) for v in vertices], dtype=np.float64)
    face_array = np.array([faceToNpArray(f) for f in faces], dtype=np.uint32)
    vertexIndicesPy = {}
    for i, v in enumerate(vertexIndices):
        vertexIndicesPy[i] = vertexToNparray(v)
    return vert_array, face_array, vertexIndicesPy

def cartesianLatLon(vertex: np.array):
    lat, lon = icosphere.cartesianLatLon(npArrayToVertex(vertex))
    return lat, lon

def latLonCartesian(lat: float, lon: float, radius: float):
    vertex = icosphere.latLonCartesian(lat, lon, radius)
    return vertexToNparray(vertex)

def haversineDistanceVertex(v1: np.ndarray, v2: np.ndarray, radius = PLANET_RADIUS_KM) -> float:
    return icosphere.haversineDistanceVertex(npArrayToVertex(v1), npArrayToVertex(v2), radius)

def HaversineDistance(lat1, lon1, lat2, lon2, radius = PLANET_RADIUS_KM) -> float:
    return icosphere.HaversineDistance(lat1, lon1, lat2, lon2, radius)

def findSphericalNeighbors(vertices, faces, vertexID, maxDistanceKM, radius = PLANET_RADIUS_KM):
    vertices = [npArrayToVertex(v) for v in vertices]
    facemap = [NPArrayToFace(face) for face in faces]
    neighbors = icosphere.findSphericalNeighbors(vertices, facemap, vertexID, maxDistanceKM, radius)
    npNeighbors = np.array(vertexToNparray(v) for v in neighbors)
    return npNeighbors

def vertexToNparray(vertex: icosphere.Vertex):
    x: float = vertex.x
    y: float = vertex.y
    z: float = vertex.z

    return np.array([x, y, z])

def npArrayToVertex(vert: np.ndarray) -> icosphere.Vertex:
    x: float = vert[0]
    y: float = vert[1]
    z: float = vert[2]
    #print(f"types are: {type(x)}")
    return icosphere.Vertex(x, y, z)

def faceToNpArray(face: icosphere.Face):
    return np.array([vertexToNparray(face.a), vertexToNparray(face.b), vertexToNparray(face.c)])

def NPArrayToFace(face: np.array):
    return icosphere.Face(npArrayToVertex(face[0]), npArrayToVertex(face[1]), npArrayToVertex(face[2]))