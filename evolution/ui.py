import dearpygui.dearpygui as dpg
import random
import math
from organism import Organism

class GridSimulation:
    def __init__(self):
        self.grid_size = 100
        self.cell_size = 8
        self.current_z = 50
        self.organisms: list[Organism] = []
        self.organisms_tags = []
        self.drawlist_tag = "drawlist"
        self.running = False
        self.terrain = []  # 2D array representing terrain height
        self.show_top_view = False
        self.selected_organism = None  # Track which organism is selected in stats view
        
        # Generate terrain (simple Perlin noise for example)
        self.generate_terrain()
        
        # Create initial organisms placed on terrain
        for i in range(50):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            # Place organism 1 unit above terrain
            z = self.terrain[x][y] + 1
            self.organisms.append(Organism(x=x, y=y, z=z, oid=i))
    
    def generate_terrain(self):
        """Generate a simple terrain using basic noise"""
        self.terrain = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Simple diamond-square algorithm or basic noise
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Simple noise - in a real implementation you'd use Perlin/Simplex noise
                height = int((math.sin(x/10) + math.cos(y/10)) * 5 + self.grid_size/2)
                height = max(0, min(self.grid_size-1, height))
                self.terrain[x][y] = height
    
    def create_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="Grid Simulation", width=1200, height=800)
        
        with dpg.window(label="Main Window", tag="main_window", width=1200, height=800):
            # Create tab bar
            with dpg.tab_bar():
                # First tab - Simulation View
                with dpg.tab(label="Simulation"):
                    with dpg.group(horizontal=True):
                        # Drawing area (will contain either grid or top view)
                        with dpg.drawlist(width=1000, height=800, tag=self.drawlist_tag):
                            pass
                        
                        # Controls
                        with dpg.child_window(width=200, height=800):
                            dpg.add_slider_int(
                                label="Z Level",
                                min_value=0,
                                max_value=self.grid_size - 1,
                                default_value=self.current_z,
                                callback=self.update_z_level,
                                tag="z_slider"
                            )
                            
                            dpg.add_button(
                                label="Start/Stop Simulation",
                                callback=self.toggle_simulation
                            )
                            
                            dpg.add_button(
                                label="Add Organism",
                                callback=self.add_random_organism
                            )
                            
                            dpg.add_checkbox(
                                label="Show Top View",
                                default_value=self.show_top_view,
                                callback=self.toggle_top_view
                            )
                
                # Second tab - Stats View
                with dpg.tab(label="Organism Stats"):
                    with dpg.group(horizontal=True):
                        # Organism list
                        with dpg.child_window(width=300, height=800, tag="organism_list"):
                            dpg.add_text("Organisms:")
                            dpg.add_separator()
                            # Will be populated in update_stats_view
                        
                        # Organism details
                        with dpg.child_window(width=900, height=800, tag="organism_details"):
                            dpg.add_text("Select an organism to view details", tag="org_detail_text")
        dpg.set_primary_window("main_window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.update_display()
        self.update_stats_view()
    
    def draw_grid(self):
        dpg.delete_item(self.drawlist_tag, children_only=True)
        self.organisms_tags.clear()
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Horizontal lines
            dpg.draw_line(
                (0, i * self.cell_size),
                (self.grid_size * self.cell_size, i * self.cell_size),
                color=(50, 50, 50, 255),
                parent=self.drawlist_tag
            )
            # Vertical lines
            dpg.draw_line(
                (i * self.cell_size, 0),
                (i * self.cell_size, self.grid_size * self.cell_size),
                color=(50, 50, 50, 255),
                parent=self.drawlist_tag
            )
        
        # Draw terrain at current z level (as semi-transparent squares)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.terrain[x][y] == self.current_z:
                    dpg.draw_rectangle(
                        (x * self.cell_size, y * self.cell_size),
                        ((x+1) * self.cell_size, (y+1) * self.cell_size),
                        color=(100, 100, 100, 150),
                        fill=(100, 100, 100, 150),
                        parent=self.drawlist_tag
                    )
        
        # Draw organisms at current z level
        for org in self.organisms:
            if int(org.z) == self.current_z:
                points = org.get_triangle_points(self.cell_size, 0, 0)
                tag = dpg.draw_triangle(
                    (points[0], points[1]),
                    (points[2], points[3]),
                    (points[4], points[5]),
                    color=org.color,
                    fill=org.color,
                    parent=self.drawlist_tag
                )
                self.organisms_tags.append(tag)

    def draw_top_view(self):
        dpg.delete_item(self.drawlist_tag, children_only=True)
        
        # Draw terrain height map
        max_height = max(max(row) for row in self.terrain)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                height = self.terrain[x][y]
                # Normalize height to color (0-255)
                color_value = int((height / max_height) * 255)
                dpg.draw_rectangle(
                    (x * self.cell_size, y * self.cell_size),
                    ((x+1) * self.cell_size, (y+1) * self.cell_size),
                    color=(color_value, color_value, color_value, 255),
                    fill=(color_value, color_value, color_value, 255),
                    parent=self.drawlist_tag
                )
        
        # Draw organisms' paths
        for org in self.organisms:
            if len(org.path) > 1:
                # Convert path points to screen coordinates
                points = []
                for x, y in org.path:
                    points.append(x * self.cell_size + self.cell_size/2)
                    points.append(y * self.cell_size + self.cell_size/2)
                
                # Draw path
                dpg.draw_polyline(
                    points,
                    color=org.color,
                    thickness=1,
                    parent=self.drawlist_tag
                )
            
            # Draw current position
            dpg.draw_circle(
                (org.x * self.cell_size + self.cell_size/2, org.y * self.cell_size + self.cell_size/2),
                radius=2,
                color=org.color,
                fill=org.color,
                parent=self.drawlist_tag
            )

    def update_z_level(self, sender):
        self.current_z = dpg.get_value(sender)
        if not self.show_top_view:
            self.update_display()
    
    def toggle_simulation(self):
        self.running = not self.running
    
    def toggle_top_view(self, sender):
        self.show_top_view = dpg.get_value(sender)
        self.update_display()

    def update_display(self):
        if self.show_top_view:
            self.draw_top_view()
        else:
            self.draw_grid()
    
    def add_random_organism(self):
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        # Place organism 1 unit above terrain
        z = self.terrain[x][y] + 1
        self.organisms.append(Organism(x=x, y=y, z=z, oid=len(self.organisms)))
        self.update_display()
        self.update_stats_view()
    
    def update_stats_view(self):
        # Clear existing organism list
        dpg.delete_item("organism_list", children_only=True)
        
        # Add organisms to list
        with dpg.child_window(width=300, height=800, parent="organism_list"):
            dpg.add_text("Organisms:")
            dpg.add_separator()
            
            for org in self.organisms:
                with dpg.group(horizontal=True):
                    # Create a small drawlist for the colored circle
                    with dpg.drawlist(width=30, height=30):
                        dpg.draw_circle(
                            center=(15, 15), 
                            radius=5, 
                            color=org.color, 
                            fill=org.color
                        )
                    # Add button to select this organism
                    dpg.add_button(
                        label=f"Organism {org.oid}",
                        width=-1,
                        callback=lambda s, a, org=org: self.show_organism_details(org),
                        user_data=org
                    )
        
        # Update details if an organism is selected
        if self.selected_organism:
            self.show_organism_details(self.selected_organism)
        else:
            dpg.set_value("org_detail_text", "Select an organism to view details")
    
    def show_organism_details(self, organism):
        self.selected_organism = organism
        
        # Clear existing details
        dpg.delete_item("organism_details", children_only=True)
        
        # Add new details
        with dpg.child_window(width=900, height=800, parent="organism_details"):
            # Header with colored circle
            with dpg.group(horizontal=True):
                # Create a drawlist for the circle
                with dpg.drawlist(width=30, height=30):
                    dpg.draw_circle(
                        center=(15, 15), 
                        radius=5, 
                        color=organism.color, 
                        fill=organism.color
                    )
                dpg.add_text(f"Organism {organism.oid}", color=(255, 255, 0))
            
            dpg.add_separator()
            
            # Basic stats
            dpg.add_text(f"Position: ({organism.x:.1f}, {organism.y:.1f}, {organism.z:.1f})")
            dpg.add_text(f"Speed: {organism.speed:.2f}")
            dpg.add_text(f"Direction: {organism.direction:.2f} radians")
            dpg.add_text(f"Energy: {organism.energy}")
            dpg.add_text(f"Size: {organism.size}")
            
            # Memory section
            dpg.add_separator()
            dpg.add_text("Memory (remembered positions):")
            
            # Create a table for memory
            with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit):
                dpg.add_table_column(label="Position")
                dpg.add_table_column(label="Weight")
                
                for pos, weight in sorted(organism.memory.items(), key=lambda x: -x[1]):
                    with dpg.table_row():
                        dpg.add_text(f"{pos}")
                        dpg.add_text(f"{weight}")
            
            # Path section
            dpg.add_separator()
            dpg.add_text(f"Path history ({len(organism.path)} points)")
            
            # Draw path visualization
            with dpg.drawlist(width=800, height=300):
                if len(organism.path) > 1:
                    # Convert path points to screen coordinates
                    points = []
                    for x, y in organism.path:
                        points.append(x * 5)  # Scale down for visualization
                        points.append(y * 5)
                    
                    # Draw path
                    dpg.draw_polyline(
                        points,
                        color=organism.color,
                        thickness=2
                    )
                
                # Draw current position
                dpg.draw_circle(
                    (organism.x * 5, organism.y * 5),
                    radius=3,
                    color=(255, 255, 255),
                    fill=(255, 255, 255)
                )
    
    def update(self):
        if self.running:
            for org in self.organisms:
                org.move(self.grid_size)
                # Ensure organism stays 1 unit above terrain
                terrain_height = self.terrain[int(org.x)][int(org.y)]
                org.z = terrain_height + 1
        
        self.update_display()
    
    def run(self):
        self.create_gui()
        
        while dpg.is_dearpygui_running():
            self.update()
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()

# Create and run the simulation
if __name__ == "__main__":
    sim = GridSimulation()
    sim.run()