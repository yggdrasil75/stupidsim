from dataclasses import field
import dearpygui.dearpygui as dpg
from shapes.sphere import create_sphere_mesh
from world import World, render_world
import torch

# Default simulation settings
settings = {
    "segments": 64,
    "rings": 64,
    "plate_count": 15,
    "sea_level": 0.0,
    "min_height": -1.0,
    "max_height": 1.0,
    "rainfall_rate": 0.1,
    "evaporation_rate": 0.05,
    "water_flow_max": 1.0,
    "water_flow_min": 0.01
}

def start_simulation():
    """Callback for the start button - launches the simulation with current settings"""
    # Update the World class defaults with our settings
    World.sphere_mesh = field(default_factory=lambda: create_sphere_mesh(
        segments=settings["segments"], 
        rings=settings["rings"]
    ))
    World.sea_level = torch.tensor(settings["sea_level"], dtype=torch.float32)
    World.min_height = torch.tensor(settings["min_height"], dtype=torch.float32)
    World.max_height = torch.tensor(settings["max_height"], dtype=torch.float32)
    World.plate_count = torch.tensor(settings["plate_count"], dtype=torch.int32)
    World.rainfall_rate = torch.tensor(settings["rainfall_rate"], dtype=torch.float32)
    World.evaporation_rate = torch.tensor(settings["evaporation_rate"], dtype=torch.float32)
    World.water_flow_max = torch.tensor(settings["water_flow_max"], dtype=torch.float32)
    World.water_flow_min = torch.tensor(settings["water_flow_min"], dtype=torch.float32)
    
    # Close the menu window and start the simulation
    dpg.delete_item("main_window")
    render_world()

def show_main_menu():
    """Creates and shows the main menu interface"""
    dpg.create_context()
    dpg.create_viewport(title='Procedural World Generator', width=600, height=500)
    
    with dpg.window(label="Main Menu", tag="main_window", width=600, height=500):
        dpg.add_text("Procedural World Generator", color=(0, 200, 255))
        dpg.add_separator()
        
        with dpg.collapsing_header(label="World Settings", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_text("Mesh Resolution:")
                dpg.add_input_int(label="Segments", min_value=8, max_value=256, 
                                 default_value=settings["segments"], 
                                 callback=lambda s, a: settings.__setitem__("segments", a))
                dpg.add_input_int(label="Rings", min_value=8, max_value=256, 
                                default_value=settings["rings"], 
                                callback=lambda s, a: settings.__setitem__("rings", a))
            
            dpg.add_slider_int(label="Plate Count", min_value=1, max_value=50, 
                             default_value=settings["plate_count"], 
                             callback=lambda s, a: settings.__setitem__("plate_count", a))
            
            with dpg.group(horizontal=True):
                dpg.add_text("Height Range:")
                dpg.add_input_float(label="Min", default_value=settings["min_height"], 
                                   callback=lambda s, a: settings.__setitem__("min_height", a))
                dpg.add_input_float(label="Max", default_value=settings["max_height"], 
                                   callback=lambda s, a: settings.__setitem__("max_height", a))
                dpg.add_input_float(label="Sea Level", default_value=settings["sea_level"], 
                                   callback=lambda s, a: settings.__setitem__("sea_level", a))
        
        with dpg.collapsing_header(label="Water Simulation"):
            dpg.add_input_float(label="Rainfall Rate", default_value=settings["rainfall_rate"], 
                              callback=lambda s, a: settings.__setitem__("rainfall_rate", a))
            dpg.add_input_float(label="Evaporation Rate", default_value=settings["evaporation_rate"], 
                               callback=lambda s, a: settings.__setitem__("evaporation_rate", a))
            
            with dpg.group(horizontal=True):
                dpg.add_text("Water Flow:")
                dpg.add_input_float(label="Min", default_value=settings["water_flow_min"], 
                                   callback=lambda s, a: settings.__setitem__("water_flow_min", a))
                dpg.add_input_float(label="Max", default_value=settings["water_flow_max"], 
                                   callback=lambda s, a: settings.__setitem__("water_flow_max", a))
        
        dpg.add_separator()
        dpg.add_button(label="Start Simulation", width=200, height=40, callback=start_simulation)
        dpg.add_text("Note: Higher resolution settings will require more memory\nand may impact performance.", 
                    color=(200, 200, 0))

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    
    dpg.destroy_context()

if __name__ == "__main__":
    show_main_menu()