import random
from matplotlib.colorbar import Colorbar
from matplotlib.widgets import RadioButtons
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
from collections import deque
from itertools import permutations

# --- Data Holder Classes (Vertex, Face - unchanged) ---

class Vertex:
    """Represents a 3D vertex."""
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=float)
        self.elevation = 0.0
        self.plate_id = -1

    def normalize(self, radius=1.0):
        """Normalizes the vertex position to be on a sphere of given radius."""
        norm = np.linalg.norm(self.pos)
        if norm > 1e-9: # Avoid division by zero
            self.pos = (self.pos / norm) * radius

    def __repr__(self):
        return f"Vertex({self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f})"

class Face:
    """Represents a face defined by vertex indices."""
    def __init__(self, v_indices):
        if len(v_indices) < 3:
            raise ValueError("Face must have at least 3 vertices")
        self.v_indices = list(v_indices) # Store indices referencing the World's vertex list

    def get_vertices_pos(self, vertex_list):
        """Returns the 3D coordinates of the vertices forming this face."""
        return [vertex_list[i].pos for i in self.v_indices]

    def get_vertices_elevation(self, vertex_list):
        """Returns the elevation values of the vertices forming this face."""
        return [vertex_list[i].elevation for i in self.v_indices]

    def __repr__(self):
        return f"Face({self.v_indices})"

class Plate:
    """Represents a tectonic plate with velocity and associated vertices."""
    def __init__(self, plate_id):
        self.plate_id = plate_id
        self.velocity = np.random.uniform(-1, 1, 3)  # Angular velocity vector
        self.vertices = set()  # Indices of vertices belonging to this plate
        
    def add_vertex(self, vertex_idx):
        """Add a vertex to this plate."""
        self.vertices.add(vertex_idx)
        
    def get_boundary_vertices(self, world):
        """Return vertices on the boundary of this plate."""
        boundary = []
        for v_idx in self.vertices:
            neighbors = world._find_vertex_neighbors(v_idx)
            neighbor_plates = {world.vertices[n].plate_id for n in neighbors}
            if len(neighbor_plates) > 1:  # Boundary vertex
                boundary.append(v_idx)
        return boundary
    
# --- World Hierarchy ---

class World:
    """Base class for holding 3D shape data."""
    def __init__(self):
        self.vertices = [] # List of Vertex objects
        self.faces = []    # List of Face objects
        self.plates = {}

    def add_vertex(self, vertex):
        """Adds a vertex and returns its index."""
        self.vertices.append(vertex)
        return len(self.vertices) - 1

    def add_face(self, face):
        """Adds a face."""
        self.faces.append(face)

    def create_world(self, radius=1.0):
        """Abstract method to populate vertices and faces."""
        raise NotImplementedError("Subclasses must implement create_world()")

    def subdivide(self, radius=1.0):
        """Abstract method to subdivide the faces."""
        raise NotImplementedError("Subclasses must implement subdivide()")

    def genElevations(self, num_plates=7, min_height=0.0, max_height=1.0):
        """Generate elevation values based on tectonic plate simulation."""
        if not self.vertices:
            return
            
        # Reset all elevations to zero first
        for vertex in self.vertices:
            vertex.elevation = 0.0
            vertex.plate_id = -1  # Add plate_id attribute to vertices
            
        self._create_plates(num_plates)
        self._calculate_boundary_elevations(min_height, max_height)
        self._smooth_elevations(iterations=2)

    def _create_plates(self, num_plates):
        """Create plates and assign vertices to them."""
        self.plates = {}
        
        # Create plate objects
        for plate_id in range(num_plates):
            self.plates[plate_id] = Plate(plate_id)
            
        # Assign vertices to plates
        self._assign_tectonic_plates(num_plates)

    def _find_vertex_neighbors(self, vertex_idx):
        """Find all neighboring vertices (shared faces) for a given vertex."""
        neighbors = set()
        for face in self.faces:
            if vertex_idx in face.v_indices:
                for v_idx in face.v_indices:
                    if v_idx != vertex_idx:
                        neighbors.add(v_idx)
        return list(neighbors)

    def _assign_tectonic_plates(self, num_plates):
        """Assign vertices to tectonic plates using weighted region growing."""
        num_vertices = len(self.vertices)
        if num_plates <= 0 or num_plates > num_vertices:
            raise ValueError("num_plates must be positive and less than or equal to the number of vertices.")

        # Initialize all vertices as unassigned (-1)
        for v in self.vertices:
            v.plate_id = -1
        unassigned = set(range(num_vertices))

        # Choose random starting points for each plate
        available_starts = list(unassigned)
        if len(available_starts) < num_plates:
             raise ValueError(f"Cannot select {num_plates} unique start points from {len(available_starts)} available vertices.")
        plate_starts = np.random.choice(available_starts, size=num_plates, replace=False)

        # Create plate assignment queues
        plate_queues = {plate_id: deque() for plate_id in range(num_plates)}
        plate_active = {plate_id: True for plate_id in range(num_plates)}

        # Assign starting points and initialize queues
        for plate_id, start_idx in enumerate(plate_starts):
            if start_idx in unassigned:
                self.vertices[start_idx].plate_id = plate_id
                self.plates[plate_id].add_vertex(start_idx)
                unassigned.remove(start_idx)
                plate_queues[plate_id].append(start_idx)

        # Grow plates using a round-robin approach
        active_plates = num_plates
        while unassigned and active_plates > 0:
            progress_made_in_round = False
            plate_order = list(range(num_plates))
            random.shuffle(plate_order)

            for plate_id in plate_order:
                if not plate_active[plate_id] or not plate_queues[plate_id]:
                    if plate_active[plate_id]:
                       plate_active[plate_id] = False
                       active_plates -= 1
                    continue

                current_idx = plate_queues[plate_id].popleft()
                neighbors = self._find_vertex_neighbors(current_idx)
                unassigned_neighbors = [n for n in neighbors if n in unassigned]

                if not unassigned_neighbors:
                    if not plate_queues[plate_id]:
                        plate_active[plate_id] = False
                        active_plates -= 1
                    continue

                # Weighted random selection
                weights = []
                valid_neighbors = []
                for n_idx in unassigned_neighbors:
                    n_neighbors = self._find_vertex_neighbors(n_idx)
                    count = sum(1 for nn in n_neighbors if nn != current_idx and self.vertices[nn].plate_id == plate_id)
                    weights.append(count + 1)
                    valid_neighbors.append(n_idx)

                if not valid_neighbors:
                     if not plate_queues[plate_id]:
                         plate_active[plate_id] = False
                         active_plates -= 1
                     continue

                weights = np.array(weights, dtype=float)
                total_weight = weights.sum()

                if total_weight == 0:
                     selected_neighbor_idx = np.random.choice(valid_neighbors)
                else:
                    weights /= total_weight
                    try:
                        selected_neighbor_idx = np.random.choice(valid_neighbors, p=weights)
                    except ValueError:
                         selected_neighbor_idx = np.random.choice(valid_neighbors)

                # Assign the selected neighbor
                self.vertices[selected_neighbor_idx].plate_id = plate_id
                self.plates[plate_id].add_vertex(selected_neighbor_idx)
                if selected_neighbor_idx in unassigned:
                    unassigned.remove(selected_neighbor_idx)

                plate_queues[plate_id].append(selected_neighbor_idx)
                progress_made_in_round = True

            # Fallback assignment if no progress
            if not progress_made_in_round and unassigned:
                orphans = list(unassigned)
                assigned_in_fallback = False
                for v_idx in orphans:
                    if v_idx not in unassigned: continue

                    neighbors = self._find_vertex_neighbors(v_idx)
                    assigned_neighbors = [(n, self.vertices[n].plate_id) for n in neighbors if self.vertices[n].plate_id != -1]

                    if assigned_neighbors:
                        random_neighbor_idx, assigned_plate_id = random.choice(assigned_neighbors)
                        self.vertices[v_idx].plate_id = assigned_plate_id
                        self.plates[assigned_plate_id].add_vertex(v_idx)
                        unassigned.remove(v_idx)
                        assigned_in_fallback = True

                if not assigned_in_fallback and unassigned:
                     remaining_unassigned = list(unassigned)
                     for v_idx in remaining_unassigned:
                          if num_plates > 0:
                             plate_id = np.random.randint(num_plates)
                             self.vertices[v_idx].plate_id = plate_id
                             self.plates[plate_id].add_vertex(v_idx)
                             unassigned.remove(v_idx)
                          else: break

        if unassigned:
            self._assign_remaining_vertices(unassigned)

    def _assign_remaining_vertices(self, unassigned):
        """Assign any remaining unassigned vertices to the nearest plate."""
        for v_idx in unassigned:
            queue = deque()
            visited = set()
            
            neighbors = self._find_vertex_neighbors(v_idx)
            queue.extend(neighbors)
            visited.update(neighbors)
            
            found = False
            while queue and not found:
                current = queue.popleft()
                if self.vertices[current].plate_id != -1:
                    plate_id = self.vertices[current].plate_id
                    self.vertices[v_idx].plate_id = plate_id
                    self.plates[plate_id].add_vertex(v_idx)
                    found = True
                    break
                    
                new_neighbors = [n for n in self._find_vertex_neighbors(current) 
                            if n not in visited]
                queue.extend(new_neighbors)
                visited.update(new_neighbors)
                
            if not found:
                if self.plates:
                    plate_id = random.choice(list(self.plates.keys()))
                    self.vertices[v_idx].plate_id = plate_id
                    self.plates[plate_id].add_vertex(v_idx)
                    
    def _find_vertex_neighbors(self, vertex_idx):
        """Find all neighboring vertices (shared faces) for a given vertex."""
        neighbors = set()
        for face in self.faces:
            if vertex_idx in face.v_indices:
                for v_idx in face.v_indices:
                    if v_idx != vertex_idx:
                        neighbors.add(v_idx)
        return list(neighbors)
        
    def _calculate_boundary_elevations(self, min_height, max_height):
        """Calculate elevations based on plate interactions at boundaries."""
        # First pass: identify boundary vertices
        boundary_vertices = []
        for plate in self.plates.values():
            boundary_vertices.extend(plate.get_boundary_vertices(self))
            
        # Calculate relative movements at boundaries
        for v_idx in boundary_vertices:
            vertex = self.vertices[v_idx]
            neighbors = self._find_vertex_neighbors(v_idx)
            
            # Find all adjacent plates
            adjacent_plates = set()
            for n in neighbors:
                adjacent_plates.add(self.vertices[n].plate_id)
            adjacent_plates.add(vertex.plate_id)
            
            # Calculate relative movement vectors
            movement_vectors = []
            current_plate = self.plates[vertex.plate_id]
            
            for plate_id in adjacent_plates:
                if plate_id == vertex.plate_id:
                    continue
                    
                other_plate = self.plates[plate_id]
                rel_velocity = current_plate.velocity - other_plate.velocity
                
                # Project onto vertex normal
                normal = vertex.pos / np.linalg.norm(vertex.pos)
                movement = np.dot(rel_velocity, normal)
                movement_vectors.append(movement)
                
            # Elevation based on average relative movement
            if movement_vectors:
                avg_movement = np.mean(movement_vectors)
                vertex.elevation = min_height + (max_height - min_height) * (avg_movement + 1) / 2
                
    def _smooth_elevations(self, iterations=2):
        """Smooth elevation values across the sphere."""
        for _ in range(iterations):
            new_elevations = []
            for i, vertex in enumerate(self.vertices):
                neighbors = self._find_vertex_neighbors(i)
                if not neighbors:
                    new_elevations.append(vertex.elevation)
                    continue
                    
                neighbor_elevations = [self.vertices[n].elevation for n in neighbors]
                avg = np.mean([vertex.elevation] + neighbor_elevations)
                new_elevations.append(avg)
                
            for i, elevation in enumerate(new_elevations):
                self.vertices[i].elevation = elevation

    def _get_midpoint_vertex(self, v1_idx, v2_idx, midpoint_cache, next_level_vertices, radius):
        """
        Helper: Calculates or retrieves the midpoint vertex between two vertices.
        Adds the new vertex if it doesn't exist and normalizes it.
        Returns the index of the vertex in next_level_vertices.
        """
        key = tuple(sorted((v1_idx, v2_idx)))
        if key in midpoint_cache:
            # Map cached index (relative to additions) back to next_level_vertices index
            return midpoint_cache[key]

        v1 = self.vertices[v1_idx]
        v2 = self.vertices[v2_idx]
        mid_pos = (v1.pos + v2.pos) / 2.0
        new_v = Vertex(mid_pos[0], mid_pos[1], mid_pos[2])
        new_v.normalize(radius)

        # Add to the *accumulating* vertex list for the next level
        new_idx = len(next_level_vertices)
        next_level_vertices.append(new_v)
        midpoint_cache[key] = new_idx # Cache the index in the accumulating list
        return new_idx
        
    def _get_face_center_vertex(self, face, center_cache, next_level_vertices, radius):
        """
        Helper: Calculates or retrieves the center vertex of a face.
        Adds the new vertex if it doesn't exist and normalizes it.
        Returns the index of the vertex in next_level_vertices.
        """
        key = tuple(sorted(face.v_indices)) # Use sorted indices as cache key
        if key in center_cache:
             return center_cache[key]

        # Calculate geometric center
        center_pos = np.mean([self.vertices[i].pos for i in face.v_indices], axis=0)
        
        new_v = Vertex(center_pos[0], center_pos[1], center_pos[2])
        new_v.normalize(radius)

        # Add to the *accumulating* vertex list for the next level
        new_idx = len(next_level_vertices)
        next_level_vertices.append(new_v)
        center_cache[key] = new_idx # Cache the index in the accumulating list
        return new_idx

    def _compute_face_centroid(self, v_indices, vertices, radius):
        centroid = Vertex(0, 0, 0)
        for idx in v_indices:
            v = vertices[idx]
            centroid.pos[0] += v.pos[0]
            centroid.pos[1] += v.pos[1]
            centroid.pos[2] += v.pos[2]
        n = len(v_indices)
        centroid.pos[0] /= n
        centroid.pos[1] /= n
        centroid.pos[2] /= n
        centroid.normalize(radius)
        return centroid
    
    def plot(self, fig=None, ax=None, cmap='terrain', edge_color=None, alpha=0.9):
        """Plots the shape with interactive radio toggle for elevation/plate visualization."""
        if fig is None or ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Create radio button axes
        rax = plt.axes([0.05, 0.7, 0.15, 0.15])
        radio = RadioButtons(rax, ('Elevation', 'Plates'))
        
        # Store data needed for both visualization modes
        plot_data = {
            'fig': fig,
            'ax': ax,
            'cmap': cmap,
            'edge_color': edge_color,
            'alpha': alpha,
            'polygons': [],
            'face_elevations': [],
            'face_plates': [],
            'cbar_ax': None  # Store the colorbar axis separately
        }
        
        # Precompute face data
        for face in self.faces:
            face_verts = face.get_vertices_pos(self.vertices)
            if len(face_verts) >= 3:
                plot_data['polygons'].append(face_verts)
                # Elevation data
                elevs = face.get_vertices_elevation(self.vertices)
                plot_data['face_elevations'].append(np.mean(elevs))
                # Plate data (use most common plate in face)
                plates = [self.vertices[i].plate_id for i in face.v_indices]
                plot_data['face_plates'].append(max(set(plates), key=plates.count))
        
        # Initial plot (elevation)
        collection = self._create_collection(plot_data, mode='elevation')
        plot_data['collection'] = collection
        ax.add_collection3d(collection)
        
        # Set axes limits
        all_positions = np.array([v.pos for v in self.vertices])
        max_val = np.max(np.abs(all_positions)) * 1.1 if len(all_positions) > 0 else 1.1
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_aspect('equal')
        
        # Add colorbar
        plot_data['cbar'] = self._add_colorbar(plot_data, ax, mode='elevation')
        
        # Radio button callback
        def on_radio_change(label):
            # Remove old elements
            plot_data['collection'].remove()
            if plot_data['cbar'] is not None:
                plot_data['cbar'].remove()
            if plot_data['cbar_ax'] is not None:
                plot_data['cbar_ax'].remove()
            
            new_collection = self._create_collection(plot_data, mode=label.lower())
            plot_data['collection'] = new_collection
            plot_data['ax'].add_collection3d(new_collection)
            
            # Add new colorbar
            plot_data['cbar'] = self._add_colorbar(plot_data, ax, mode=label.lower())
            
            fig.canvas.draw_idle()
        
        radio.on_clicked(on_radio_change)
        
        return fig, ax, radio

    def _create_collection(self, plot_data, mode='elevation'):
        """Create the appropriate Poly3DCollection based on visualization mode."""
        if mode == 'elevation':
            norm = plt.Normalize(
                vmin=min(plot_data['face_elevations']),
                vmax=max(plot_data['face_elevations'])
            )
            cmap = plt.get_cmap(plot_data['cmap'])
            face_colors = cmap(norm(plot_data['face_elevations']))
        else:  # plates
            unique_plates = list(set(plot_data['face_plates']))
            plate_cmap = plt.get_cmap('tab20')  # Good for categorical data
            norm = plt.Normalize(vmin=0, vmax=len(unique_plates))
            face_colors = plate_cmap(norm([unique_plates.index(p) for p in plot_data['face_plates']]))
        
        return Poly3DCollection(
            plot_data['polygons'],
            facecolors=face_colors,
            linewidths=0.3 if plot_data['edge_color'] else 0,
            edgecolors=plot_data['edge_color'],
            alpha=plot_data['alpha']
        )
    
    def _add_colorbar(self, plot_data, ax, mode='elevation'):
        """Add appropriate colorbar based on visualization mode."""
        fig = plot_data['fig']
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        if mode == 'elevation':
            norm = plt.Normalize(
                vmin=min(plot_data['face_elevations']),
                vmax=max(plot_data['face_elevations'])
            )
            cmap = plt.get_cmap(plot_data['cmap'])
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(plot_data['face_elevations'])
            cbar = fig.colorbar(mappable, cax=cbar_ax, label='Elevation')
        else:  # plates
            unique_plates = sorted(list(set(plot_data['face_plates'])))
            plate_cmap = plt.get_cmap('tab20')
            norm = plt.Normalize(vmin=0, vmax=len(unique_plates)-1)
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=plate_cmap)
            mappable.set_array([unique_plates.index(p) for p in plot_data['face_plates']])
            cbar = fig.colorbar(mappable, cax=cbar_ax, label='Plate ID')
            cbar.set_ticks(range(len(unique_plates)))
            cbar.set_ticklabels(unique_plates)
        
        return cbar

class Icosahedron(World):
    """Represents an Icosahedron."""
    def __init__(self):
        super().__init__()

    def create_world(self, radius=1.0):
        """Creates the vertices and faces for a unit icosahedron."""
        self.vertices = []
        self.faces = []
        phi = (1 + math.sqrt(5)) / 2
        
        # Note: Using slightly different vertex definition for easier face indexing
        v_data = [
            (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
            (0, -1,  phi), (0,  1,  phi), (0, -1, -phi), (0,  1, -phi),
            ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1)
        ]
        
        for p in v_data:
            v = Vertex(*p)
            v.normalize(radius)
            self.add_vertex(v)

        # Add faces (triangles)
        f_indices = [
            # 5 faces around point 0
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            # 5 adjacent faces
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            # 5 faces around point 3
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            # 5 adjacent faces
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
        ]
        for f in f_indices:
            self.add_face(Face(f))

    def subdivide(self, radius=1.0, level = 3):
        for li in range(level):
            print(f"Subdividing Icosahedron with {len(self.faces)} faces...")
            midpoint_cache = {}
            new_faces = []
            next_level_vertices = list(self.vertices) # Start with existing vertices

            for face in self.faces:
                if len(face.v_indices) != 3:
                    print(f"Warning: Skipping subdivision of non-triangular face: {face}")
                    new_faces.append(face)
                    continue

                v1_idx, v2_idx, v3_idx = face.v_indices

                # Get or create midpoint vertices for each edge, adding to next_level_vertices
                m12_idx = self._get_midpoint_vertex(v1_idx, v2_idx, midpoint_cache, next_level_vertices, radius)
                m23_idx = self._get_midpoint_vertex(v2_idx, v3_idx, midpoint_cache, next_level_vertices, radius)
                m31_idx = self._get_midpoint_vertex(v3_idx, v1_idx, midpoint_cache, next_level_vertices, radius)

                # Create the four new faces using original and midpoint vertex indices
                # Indices must reference the combined list (original + new midpoints)
                new_faces.append(Face((v1_idx, m12_idx, m31_idx)))
                new_faces.append(Face((v2_idx, m23_idx, m12_idx)))
                new_faces.append(Face((v3_idx, m31_idx, m23_idx)))
                new_faces.append(Face((m12_idx, m23_idx, m31_idx))) # Center face

            self.vertices = next_level_vertices
            self.faces = new_faces
            print(f"Icosahedron subdivision complete. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")

class TruncatedIcosahedron(World):
    """Represents a Truncated Icosahedron (soccer ball)."""
    def __init__(self):
        super().__init__()

    def create_world(self, radius=1.0):
        """Creates the vertices and faces for a unit truncated icosahedron."""

        C0 = (1 + np.sqrt(5)) / 4
        C1 =  (1 + np.sqrt(5)) / 2
        C2 = (5 + np.sqrt(5)) / 4
        C3 = (2 + np.sqrt(5)) / 2
        C4 =  3 * (1 + np.sqrt(5)) / 4
        
        truncated_v_data = [
            ( 0.5,  0.0,   C4),
            ( 0.5,  0.0,  -C4),
            (-0.5,  0.0,   C4),
            (-0.5,  0.0,  -C4),
            (  C4,  0.5,  0.0),
            (  C4, -0.5,  0.0),
            ( -C4,  0.5,  0.0),
            ( -C4, -0.5,  0.0),
            ( 0.0,   C4,  0.5),
            ( 0.0,   C4, -0.5),
            ( 0.0,  -C4,  0.5),
            ( 0.0,  -C4, -0.5),
            ( 1.0,   C0,   C3),
            ( 1.0,   C0,  -C3),
            ( 1.0,  -C0,   C3),
            ( 1.0,  -C0,  -C3),
            (-1.0,   C0,   C3),
            (-1.0,   C0,  -C3),
            (-1.0,  -C0,   C3),
            (-1.0,  -C0,  -C3),
            (  C3,  1.0,   C0),
            (  C3,  1.0,  -C0),
            (  C3, -1.0,   C0),
            (  C3, -1.0,  -C0),
            ( -C3,  1.0,   C0),
            ( -C3,  1.0,  -C0),
            ( -C3, -1.0,   C0),
            ( -C3, -1.0,  -C0),
            (  C0,   C3,  1.0),
            (  C0,   C3, -1.0),
            (  C0,  -C3,  1.0),
            (  C0,  -C3, -1.0),
            ( -C0,   C3,  1.0),
            ( -C0,   C3, -1.0),
            ( -C0,  -C3,  1.0),
            ( -C0,  -C3, -1.0),
            ( 0.5,   C1,   C2),
            ( 0.5,   C1,  -C2),
            ( 0.5,  -C1,   C2),
            ( 0.5,  -C1,  -C2),
            (-0.5,   C1,   C2),
            (-0.5,   C1,  -C2),
            (-0.5,  -C1,   C2),
            (-0.5,  -C1,  -C2),
            (  C2,  0.5,   C1),
            (  C2,  0.5,  -C1),
            (  C2, -0.5,   C1),
            (  C2, -0.5,  -C1),
            ( -C2,  0.5,   C1),
            ( -C2,  0.5,  -C1),
            ( -C2, -0.5,   C1),
            ( -C2, -0.5,  -C1),
            (  C1,   C2,  0.5),
            (  C1,   C2, -0.5),
            (  C1,  -C2,  0.5),
            (  C1,  -C2, -0.5),
            ( -C1,   C2,  0.5),
            ( -C1,   C2, -0.5),
            ( -C1,  -C2,  0.5),
            ( -C1,  -C2, -0.5)
        ]

        for p in truncated_v_data:
            v = Vertex(*p)
            v.normalize(radius)
            self.add_vertex(v)

        faces = [
            (  0,  2, 18, 42, 38, 14 ),
            (  1,  3, 17, 41, 37, 13 ),
            (  2,  0, 12, 36, 40, 16 ),
            (  3,  1, 15, 39, 43, 19 ),
            (  4,  5, 23, 47, 45, 21 ),
            (  5,  4, 20, 44, 46, 22 ),
            (  6,  7, 26, 50, 48, 24 ),
            (  7,  6, 25, 49, 51, 27 ),
            (  8,  9, 33, 57, 56, 32 ),
            (  9,  8, 28, 52, 53, 29 ),
            ( 10, 11, 31, 55, 54, 30 ),
            ( 11, 10, 34, 58, 59, 35 ),
            ( 12, 44, 20, 52, 28, 36 ),
            ( 13, 37, 29, 53, 21, 45 ),
            ( 14, 38, 30, 54, 22, 46 ),
            ( 15, 47, 23, 55, 31, 39 ),
            ( 16, 40, 32, 56, 24, 48 ),
            ( 17, 49, 25, 57, 33, 41 ),
            ( 18, 50, 26, 58, 34, 42 ),
            ( 19, 43, 35, 59, 27, 51 ),
            (  0, 14, 46, 44, 12 ),
            (  1, 13, 45, 47, 15 ),
            (  2, 16, 48, 50, 18 ),
            (  3, 19, 51, 49, 17 ),
            (  4, 21, 53, 52, 20 ),
            (  5, 22, 54, 55, 23 ),
            (  6, 24, 56, 57, 25 ),
            (  7, 27, 59, 58, 26 ),
            (  8, 32, 40, 36, 28 ),
            (  9, 29, 37, 41, 33 ),
            ( 10, 30, 38, 42, 34 ),
            ( 11, 35, 43, 39, 31 )

        ]

        for f_idx_tuple in faces:
            # Ensure indices are integers if they aren't already
            int_indices = tuple(int(i) for i in f_idx_tuple)
            self.add_face(Face(int_indices))

    def subdivide(self, radius=1.0, level=3):
        for li in range(level):
            midpoint_cache = {}
            new_faces = []
            next_vertices = list(self.vertices)
            
            for face in self.faces:
                v_indices = face.v_indices
                n = len(v_indices)
                
                # Skip subdivision for certain levels based on face type
                if (n == 3 and li % 2 != 0) or (n == 5 and li % 4 != 0) or (n == 6 and not li):
                    new_faces.append(face)
                    continue
                    
                # Get all midpoints for this face
                mid_indices = []
                for i in range(n):
                    v1 = v_indices[i]
                    v2 = v_indices[(i+1)%n]
                    mid_idx = self._get_midpoint_vertex(v1, v2, midpoint_cache, next_vertices, radius)
                    mid_indices.append(mid_idx)
                
                # Create central face
                central_face = Face(tuple(mid_indices))
                new_faces.append(central_face)
                
                # Create surrounding faces
                for i in range(n):
                    v_idx = v_indices[i]
                    prev_mid = mid_indices[i-1] if i > 0 else mid_indices[-1]
                    curr_mid = mid_indices[i]
                    
                    if n == 3:  # Triangle becomes 3 quads (but your original creates 4 tris)
                        new_faces.append(Face((v_idx, curr_mid, prev_mid)))
                    else:  # N-gon becomes N triangles + 1 central N-gon
                        new_faces.append(Face((v_idx, curr_mid, prev_mid)))
            
            self.vertices = next_vertices
            self.faces = new_faces
            print(f"Subdivision level {li+1} complete. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")

class truncatedTetrahedron(World):
    def __init__(self):
        super().__init__()

    def create_world(self, radius=1):
        c0 = np.sqrt(2)/4
        c1 = 3 * np.sqrt(2) / 4

        v_data = [
            (c0, -c0, c1),
            (c0, c0, -c1),
            (-c0, c0, c1),
            (-c0, -c0, -c1),
            (c1, -c0, c0),
            (c1, c0, -c0),
            (-c1, c0, c0),
            (-c1, -c0, c0),
            (c0, -c1, c0),
            (c0, c1, -c0),
            (-c0, c1, c0),
            (-c0, -c1, -c0)            
        ]

        for p in v_data:
            v = Vertex(*p)
            v.normalize(radius)
            self.add_vertex(v)

        faces = [
            (0, 4, 5, 9, 10, 2),
            (1, 5, 4, 8, 11, 3),
            (2, 6, 7, 11, 8, 0),
            (3, 7, 6, 10, 9, 1),
            (0, 8, 4),
            (1, 9, 5),
            (2, 10, 6),
            (3, 11, 7)
        ]

        for f_idx_tuple in faces:
            # Ensure indices are integers if they aren't already
            int_indices = tuple(int(i) for i in f_idx_tuple)
            self.add_face(Face(int_indices))


    def subdivide(self, radius=1.0, level=3):
        for li in range(level):
            midpoint_cache = {}
            new_faces = []
            next_vertices = list(self.vertices)
            
            for face in self.faces:
                v_indices = face.v_indices
                n = len(v_indices)
                
                # # Skip subdivision for squares every other level
                # if n == 4 and li % 2 != 0:
                #     new_faces.append(face)
                #     continue
                    
                # Get all edge midpoints for this face
                mid_indices = []
                for i in range(n):
                    v1 = v_indices[i]
                    v2 = v_indices[(i+1)%n]
                    mid_idx = self._get_midpoint_vertex(v1, v2, midpoint_cache, next_vertices, radius)
                    mid_indices.append(mid_idx)
                
                # Get face centroid (for Catmull-Clark-like subdivision)
                centroid = self._compute_face_centroid(v_indices, next_vertices, radius)
                centroid_idx = len(next_vertices)
                next_vertices.append(centroid)
                
                # Create new faces
                for i in range(n):
                    v_idx = v_indices[i]
                    curr_mid = mid_indices[i]
                    prev_mid = mid_indices[i-1] if i > 0 else mid_indices[-1]
                    
                    if n == 4:
                        # For squares: Create a new quad (maintains shape)
                        next_mid = mid_indices[(i+1)%n]
                        new_faces.append(Face((v_idx, curr_mid, centroid_idx, prev_mid)))
                    else:
                        # For decagons/hexagons: Create a new n-gon face
                        new_faces.append(Face((v_idx, curr_mid, centroid_idx, prev_mid)))
            
            self.vertices = next_vertices
            self.faces = new_faces
            print(f"Subdivision level {li+1}: {len(self.vertices)} vertices, {len(self.faces)} faces")

class TruncatedIcosidodecahedron(World):
    def __init__(self):
        super().__init__()

    def create_world(self, radius=1):
        C0 = (3 + np.sqrt(5)) / 4
        C1 = (1 + np.sqrt(5)) / 2
        C2 = (5 + np.sqrt(5)) / 4
        C3 = (2 + np.sqrt(5)) / 2
        C4 = 3 * (1 + np.sqrt(5)) / 4
        C5 = (3 + np.sqrt(5)) / 2
        C6 = (5 + 3 * np.sqrt(5)) / 4
        C7 = (4 + np.sqrt(5)) / 2
        C8 = (7 + 3 * np.sqrt(5)) / 4
        C9 = (3 + 2 * np.sqrt(5)) / 2

        v_data = [
            ( 0.5,  0.5,   C9),
            ( 0.5,  0.5,  -C9),
            ( 0.5, -0.5,   C9),
            ( 0.5, -0.5,  -C9),
            (-0.5,  0.5,   C9),
            (-0.5,  0.5,  -C9),
            (-0.5, -0.5,   C9),
            (-0.5, -0.5,  -C9),
            (  C9,  0.5,  0.5),
            (  C9,  0.5, -0.5),
            (  C9, -0.5,  0.5),
            (  C9, -0.5, -0.5),
            ( -C9,  0.5,  0.5),
            ( -C9,  0.5, -0.5),
            ( -C9, -0.5,  0.5),
            ( -C9, -0.5, -0.5),
            ( 0.5,   C9,  0.5),
            ( 0.5,   C9, -0.5),
            ( 0.5,  -C9,  0.5),
            ( 0.5,  -C9, -0.5),
            (-0.5,   C9,  0.5),
            (-0.5,   C9, -0.5),
            (-0.5,  -C9,  0.5),
            (-0.5,  -C9, -0.5),
            ( 1.0,   C0,   C8),
            ( 1.0,   C0,  -C8),
            ( 1.0,  -C0,   C8),
            ( 1.0,  -C0,  -C8),
            (-1.0,   C0,   C8),
            (-1.0,   C0,  -C8),
            (-1.0,  -C0,   C8),
            (-1.0,  -C0,  -C8),
            (  C8,  1.0,   C0),
            (  C8,  1.0,  -C0),
            (  C8, -1.0,   C0),
            (  C8, -1.0,  -C0),
            ( -C8,  1.0,   C0),
            ( -C8,  1.0,  -C0),
            ( -C8, -1.0,   C0),
            ( -C8, -1.0,  -C0),
            (  C0,   C8,  1.0),
            (  C0,   C8, -1.0),
            (  C0,  -C8,  1.0),
            (  C0,  -C8, -1.0),
            ( -C0,   C8,  1.0),
            ( -C0,   C8, -1.0),
            ( -C0,  -C8,  1.0),
            ( -C0,  -C8, -1.0),
            ( 0.5,   C3,   C7),
            ( 0.5,   C3,  -C7),
            ( 0.5,  -C3,   C7),
            ( 0.5,  -C3,  -C7),
            (-0.5,   C3,   C7),
            (-0.5,   C3,  -C7),
            (-0.5,  -C3,   C7),
            (-0.5,  -C3,  -C7),
            (  C7,  0.5,   C3),
            (  C7,  0.5,  -C3),
            (  C7, -0.5,   C3),
            (  C7, -0.5,  -C3),
            ( -C7,  0.5,   C3),
            ( -C7,  0.5,  -C3),
            ( -C7, -0.5,   C3),
            ( -C7, -0.5,  -C3),
            (  C3,   C7,  0.5),
            (  C3,   C7, -0.5),
            (  C3,  -C7,  0.5),
            (  C3,  -C7, -0.5),
            ( -C3,   C7,  0.5),
            ( -C3,   C7, -0.5),
            ( -C3,  -C7,  0.5),
            ( -C3,  -C7, -0.5),
            (  C2,   C1,   C6),
            (  C2,   C1,  -C6),
            (  C2,  -C1,   C6),
            (  C2,  -C1,  -C6),
            ( -C2,   C1,   C6),
            ( -C2,   C1,  -C6),
            ( -C2,  -C1,   C6),
            ( -C2,  -C1,  -C6),
            (  C6,   C2,   C1),
            (  C6,   C2,  -C1),
            (  C6,  -C2,   C1),
            (  C6,  -C2,  -C1),
            ( -C6,   C2,   C1),
            ( -C6,   C2,  -C1),
            ( -C6,  -C2,   C1),
            ( -C6,  -C2,  -C1),
            (  C1,   C6,   C2),
            (  C1,   C6,  -C2),
            (  C1,  -C6,   C2),
            (  C1,  -C6,  -C2),
            ( -C1,   C6,   C2),
            ( -C1,   C6,  -C2),
            ( -C1,  -C6,   C2),
            ( -C1,  -C6,  -C2),
            (  C0,   C4,   C5),
            (  C0,   C4,  -C5),
            (  C0,  -C4,   C5),
            (  C0,  -C4,  -C5),
            ( -C0,   C4,   C5),
            ( -C0,   C4,  -C5),
            ( -C0,  -C4,   C5),
            ( -C0,  -C4,  -C5),
            (  C5,   C0,   C4),
            (  C5,   C0,  -C4),
            (  C5,  -C0,   C4),
            (  C5,  -C0,  -C4),
            ( -C5,   C0,   C4),
            ( -C5,   C0,  -C4),
            ( -C5,  -C0,   C4),
            ( -C5,  -C0,  -C4),
            (  C4,   C5,   C0),
            (  C4,   C5,  -C0),
            (  C4,  -C5,   C0),
            (  C4,  -C5,  -C0),
            ( -C4,   C5,   C0),
            ( -C4,   C5,  -C0),
            ( -C4,  -C5,   C0),
            ( -C4,  -C5,  -C0)
        ]

        for p in v_data:
            v = Vertex(*p)
            v.normalize(radius)
            self.add_vertex(v)

        faces = [
            (   0,   2,  26,  74, 106,  58,  56, 104,  72,  24 ),
            (   1,  25,  73, 105,  57,  59, 107,  75,  27,   3 ),
            (   4,  28,  76, 108,  60,  62, 110,  78,  30,   6 ),
            (   5,   7,  31,  79, 111,  63,  61, 109,  77,  29 ),
            (   8,   9,  33,  81, 113,  65,  64, 112,  80,  32 ),
            (  10,  34,  82, 114,  66,  67, 115,  83,  35,  11 ),
            (  12,  36,  84, 116,  68,  69, 117,  85,  37,  13 ),
            (  14,  15,  39,  87, 119,  71,  70, 118,  86,  38 ),
            (  16,  20,  44,  92, 100,  52,  48,  96,  88,  40 ),
            (  17,  41,  89,  97,  49,  53, 101,  93,  45,  21 ),
            (  18,  42,  90,  98,  50,  54, 102,  94,  46,  22 ),
            (  19,  23,  47,  95, 103,  55,  51,  99,  91,  43 ),
            (   0,  24,  48,  52,  28,   4 ),
            (   1,   5,  29,  53,  49,  25 ),
            (   2,   6,  30,  54,  50,  26 ),
            (   3,  27,  51,  55,  31,   7 ),
            (   8,  32,  56,  58,  34,  10 ),
            (   9,  11,  35,  59,  57,  33 ),
            (  12,  14,  38,  62,  60,  36 ),
            (  13,  37,  61,  63,  39,  15 ),
            (  16,  40,  64,  65,  41,  17 ),
            (  18,  19,  43,  67,  66,  42 ),
            (  20,  21,  45,  69,  68,  44 ),
            (  22,  46,  70,  71,  47,  23 ),
            (  72, 104,  80, 112,  88,  96 ),
            (  73,  97,  89, 113,  81, 105 ),
            (  74,  98,  90, 114,  82, 106 ),
            (  75, 107,  83, 115,  91,  99 ),
            (  76, 100,  92, 116,  84, 108 ),
            (  77, 109,  85, 117,  93, 101 ),
            (  78, 110,  86, 118,  94, 102 ),
            (  79, 103,  95, 119,  87, 111 ),
            (   0,   4,   6,   2 ),
            (   1,   3,   7,   5 ),
            (   8,  10,  11,   9 ),
            (  12,  13,  15,  14 ),
            (  16,  17,  21,  20 ),
            (  18,  22,  23,  19 ),
            (  24,  72,  96,  48 ),
            (  25,  49,  97,  73 ),
            (  26,  50,  98,  74 ),
            (  27,  75,  99,  51 ),
            (  28,  52, 100,  76 ),
            (  29,  77, 101,  53 ),
            (  30,  78, 102,  54 ),
            (  31,  55, 103,  79 ),
            (  32,  80, 104,  56 ),
            (  33,  57, 105,  81 ),
            (  34,  58, 106,  82 ),
            (  35,  83, 107,  59 ),
            (  36,  60, 108,  84 ),
            (  37,  85, 109,  61 ),
            (  38,  86, 110,  62 ),
            (  39,  63, 111,  87 ),
            (  40,  88, 112,  64 ),
            (  41,  65, 113,  89 ),
            (  42,  66, 114,  90 ),
            (  43,  91, 115,  67 ),
            (  44,  68, 116,  92 ),
            (  45,  93, 117,  69 ),
            (  46,  94, 118,  70 ),
            (  47,  71, 119,  95 )
        ]

        for f_idx_tuple in faces:
            # Ensure indices are integers if they aren't already
            int_indices = tuple(int(i) for i in f_idx_tuple)
            self.add_face(Face(int_indices))
    
    def subdivide(self, radius=1.0, level=3):
        for li in range(level):
            midpoint_cache = {}
            new_faces = []
            next_vertices = list(self.vertices)
            
            for face in self.faces:
                v_indices = face.v_indices
                n = len(v_indices)
                
                # # Skip subdivision for squares every other level
                # if n == 4 and li % 2 != 0:
                #     new_faces.append(face)
                #     continue
                    
                # Get all edge midpoints for this face
                mid_indices = []
                for i in range(n):
                    v1 = v_indices[i]
                    v2 = v_indices[(i+1)%n]
                    mid_idx = self._get_midpoint_vertex(v1, v2, midpoint_cache, next_vertices, radius)
                    mid_indices.append(mid_idx)
                
                # Get face centroid (for Catmull-Clark-like subdivision)
                centroid = self._compute_face_centroid(v_indices, next_vertices, radius)
                centroid_idx = len(next_vertices)
                next_vertices.append(centroid)
                
                # Create new faces
                for i in range(n):
                    v_idx = v_indices[i]
                    curr_mid = mid_indices[i]
                    prev_mid = mid_indices[i-1] if i > 0 else mid_indices[-1]
                    
                    if n == 4:
                        # For squares: Create a new quad (maintains shape)
                        next_mid = mid_indices[(i+1)%n]
                        new_faces.append(Face((v_idx, curr_mid, centroid_idx, prev_mid)))
                    else:
                        # For decagons/hexagons: Create a new n-gon face
                        new_faces.append(Face((v_idx, curr_mid, centroid_idx, prev_mid)))
            
            self.vertices = next_vertices
            self.faces = new_faces
            print(f"Subdivision level {li+1}: {len(self.vertices)} vertices, {len(self.faces)} faces")

class Cube(World):
    """Represents a Cube."""
    def __init__(self):
        super().__init__()

    def create_world(self, radius=1.0):
        """Creates the vertices and faces for a unit cube."""
        self.vertices = []
        self.faces = []
        
        # Create the 8 vertices of a cube
        v_data = [
            (-1, -1, -1),  # 0
            ( 1, -1, -1),  # 1
            ( 1,  1, -1),  # 2
            (-1,  1, -1),  # 3
            (-1, -1,  1),  # 4
            ( 1, -1,  1),  # 5
            ( 1,  1,  1),  # 6
            (-1,  1,  1)   # 7
        ]
        
        for p in v_data:
            v = Vertex(*p)
            v.normalize(radius)  # Normalize to make it a spherical cube
            self.add_vertex(v)

        # Add the 6 faces (quadrilaterals)
        f_indices = [
            (0, 1, 2, 3),  # Bottom face
            (4, 5, 6, 7),  # Top face
            (0, 1, 5, 4),  # Front face
            (1, 2, 6, 5),  # Right face
            (2, 3, 7, 6),  # Back face
            (3, 0, 4, 7)   # Left face
        ]
        
        for f in f_indices:
            self.add_face(Face(f))

    def subdivide(self, radius=1.0, level=3):
        """Subdivides faces into quadrilaterals using a Catmull-Clark like approach."""
        
        # Initial normalization check (if create_world didn't normalize)
        # for v in self.vertices:
        #     v.normalize(radius)
            
        for li in range(level):
            print(f"Starting Subdivision level {li+1}...")
            midpoint_cache = {} # Cache edge midpoints for this level
            new_faces = []
            # Important: Make a *copy* of the vertex list at the start of the level
            # because we will be appending new vertices to it during processing.
            current_level_vertices = list(self.vertices)
            
            # We will build the next vertex list incrementally
            next_vertices = list(current_level_vertices) 

            processed_faces = 0
            for face in self.faces:
                v_indices = face.v_indices
                n = len(v_indices)

                # We only handle quads in this specific subdivision logic
                # If you start with triangles or other shapes, you'd need different rules
                if n != 4:
                    # Keep non-quad faces as they are (or implement different subdivision)
                    print(f"Warning: Skipping non-quad face: {face}")
                    new_faces.append(face)
                    continue
                    
                # --- 1. Calculate Face Point ---
                face_center_pos = np.zeros(3, dtype=float)
                for v_idx in v_indices:
                    face_center_pos += current_level_vertices[v_idx].pos
                face_center_pos /= n
                
                center_v = Vertex(*face_center_pos)
                center_v.normalize(radius)
                
                # Add face point vertex to the list for the *next* level
                center_idx = len(next_vertices)
                next_vertices.append(center_v)

                # --- 2. Calculate Edge Midpoints ---
                mid_indices = []
                for i in range(n):
                    v1_idx = v_indices[i]
                    v2_idx = v_indices[(i + 1) % n] # Handle wrap-around
                    # Use the _get_midpoint_vertex helper which uses the cache
                    # Pass next_vertices so new midpoints are added correctly
                    mid_idx = self._get_midpoint_vertex(v1_idx, v2_idx, midpoint_cache, next_vertices, radius)
                    mid_indices.append(mid_idx)
                
                # --- 3. Create New Quadrilateral Faces ---
                # Connect original vertex -> edge midpoint -> face center -> previous edge midpoint
                for i in range(n):
                    v_orig_idx = v_indices[i] # Original vertex index
                    mid_curr_idx = mid_indices[i] # Midpoint of edge starting at v_orig_idx
                    # Midpoint of edge ending at v_orig_idx (handle wrap-around)
                    mid_prev_idx = mid_indices[i - 1] # Python's negative indexing handles wrap-around nicely
                    
                    # Create the new quad face
                    # Order: Original Corner -> Edge Midpoint -> Face Center -> Previous Edge Midpoint
                    new_quad = Face((v_orig_idx, mid_curr_idx, center_idx, mid_prev_idx))
                    new_faces.append(new_quad)

                processed_faces += 1


            # Update the world state for the next iteration or final result
            self.vertices = next_vertices
            self.faces = new_faces
            print(f"Subdivision level {li+1} complete. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")

# --- Main Execution ---
if __name__ == "__main__":
    shape_type = "cube" # Choose "icosahedron" or "truncated"
    num_subdivisions = 3     # Adjust level of detail
    sphere_radius = 1.0
    plates = 15
    elevationmin = -15000
    elevationmax = 15000


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


    # Subdivide the shape
    shape.subdivide(sphere_radius, num_subdivisions) # Call the correct subdivide

    shape.genElevations(plates, elevationmin, elevationmax)

    # # --- Plotting ---
    # fig = plt.figure(figsize=(9, 9))
    # ax = fig.add_subplot(111, projection='3d')

    # Plot the final shape
    #shape.plot(ax, cmap='terrain', edge_color='darkgreen', alpha=0.9)
    fig, ax, radio = shape.plot()

    ax.set_title(f'Sphere Approx. ({shape_type.capitalize()} Subdivided {num_subdivisions} Times)')
    plt.show()
    