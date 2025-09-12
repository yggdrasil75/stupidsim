import dearpygui.dearpygui as dpg
import numpy as np
import math
import random

WIDTH=800
HEIGHT=600

# Material properties
MATERIALS = {
    "glass": {"reflection": 0.1, "refraction": 1.0, "absorption": 0.1, "color": (1.0, 0.0, 1.0, 0.39)},
    "stone": {"reflection": 0.01, "refraction": 0.0, "absorption": 1.0, "color": (1.0, 0.0, 0.0, 1.0)},
    "metal": {"reflection": 1.0, "refraction": 0.5, "absorption": 0.5, "color": (0.0, 1.0, 0.0, 1.0)}
}

class Wall:
    def __init__(self, x1, y1, x2, y2, material="stone"):
        self.start = (x1, y1)
        self.end = (x2, y2)
        self.material = material
        self.properties = MATERIALS[material]
        self.draw_id = None
        self.thickness = np.random.randint(1, 7)
        self.normal = None
        
    def draw(self):
        dpg.draw_line(self.start, self.end, color=self.properties["color"], thickness=self.thickness, parent="canvas")
    
    def get_normal(self):
        if self.normal is None:
            # Calculate normal vector to the wall
            dx = self.end[0] - self.start[0]
            dy = self.end[1] - self.start[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length == 0:
                return (0, 0)
                
            # Normalize
            dx /= length
            dy /= length
            
            # Return perpendicular vector (normal)
            self.normal = (-dy, dx)
        return self.normal
    
class world:

    def __init__(self):
        self.walls = []
        self.walls.append(Wall(5, 5, WIDTH-5, 5, "stone"))
        self.walls.append(Wall(WIDTH-5, 5, WIDTH-5, HEIGHT-5, "stone"))
        self.walls.append(Wall(WIDTH-5, HEIGHT-5,5, HEIGHT-5, "stone"))
        self.walls.append(Wall(5, HEIGHT-5, 5, 5, "stone"))

    def update_simulation(self):
        dpg.delete_item("canvas", children_only=True)
        for wall in self.walls:
            dpg.draw_line(wall.start, wall.end, color=wall.properties["color"], thickness=wall.thickness, parent="canvas")


def main():
    dpg.create_context()
    dpg.create_viewport(title="Light simulation 2d", width=WIDTH, height=HEIGHT)
    with dpg.window(label="test environment", tag="primary", width=WIDTH, height=HEIGHT) as o:
        dpg.add_drawlist(width=-1, height=-1, tag="canvas")
    simworld = world()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)

    while dpg.is_dearpygui_running():
        simworld.update_simulation()
        dpg.draw_line((10, 10), (100, 100), color=(255, 0, 0, 255), thickness=1, parent="canvas")
        dpg.render_dearpygui_frame()

    
if __name__ == "__main__":
    main()