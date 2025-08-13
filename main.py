import dearpygui.dearpygui as dpg
from world import render_world

viewport = None
dpgcontext = None

def update_slider_limits():
    # Ensure mountains are always higher than depths
    depths = dpg.get_value("depths_value")
    dpg.configure_item("hills_value", min_value=depths)
    
    # Ensure depths are always lower than mountains
    hills = dpg.get_value("hills_value")
    dpg.configure_item("depths_value", max_value=hills)

def show_configure_window():
    # Remove the main menu window
    dpg.delete_item("main")
    
    with dpg.window(label="Configure Simulation", width=400, height=300, tag="config"):
        # World Resolution Slider
        dpg.add_slider_int(
            label="World Resolution",
            min_value=64,
            max_value=512,
            default_value=64,
            clamped=True,
            format="%d",
            callback=lambda s, a, u: None,
            width=200,
            tag="resolution_value"
        )
        dpg.add_text("Higher will be slower but better", color=[150, 150, 150])
        
        # Depths Slider
        dpg.add_slider_int(
            label="Depths",
            min_value=3000,  # Reasonable minimum
            max_value=7000,  # Will be updated when mountains changes
            default_value=6357,
            clamped=True,
            format="%d km",
            callback=update_slider_limits,
            width=200,
            tag="depths_value"
        )
        dpg.add_text("Depths are at 6357 km on Earth", color=[150, 150, 150])
        
        # Mountains Slider
        dpg.add_slider_int(
            label="Mountains",
            min_value=6357,  # Will be updated when depths changes
            max_value=9000,  # Reasonable maximum
            default_value=6378,
            clamped=True,
            format="%d km",
            callback=update_slider_limits,
            width=200,
            tag="hills_value"
        )
        dpg.add_text("The tallest mounts on Earth are at 6378 km", color=[150, 150, 150])
        
        dpg.add_slider_int(
            label="Plates",
            min_value=10,
            max_value=25,
            default_value=20,
            clamped=True,
            tag="plate_count_value"
        )
        dpg.add_text("Earth has 15 plates, please set this above your goal though as randomization may drop a few plates.")

        # Start Simulation Button
        dpg.add_button(
            label="Start Simulation",
            callback=start_simulation,
            width=100,
            height=30
        )

def start_simulation():
    global viewport, dpgcontext
    # Remove the config window
    resolution=dpg.get_value("resolution_value")
    min_height=dpg.get_value("depths_value")
    max_height=dpg.get_value("hills_value")
    plate_count=dpg.get_value("plate_count_value")
    dpg.delete_item("config")
    # Start the simulation with the configured parameters
    print(f"Creating world with the following: res of {resolution}, depths are at {min_height}, mountain peaks are at {max_height}, plates have a max of {plate_count}")
    render_world(
        resolution=resolution,
        min_height=min_height,
        max_height=max_height,
        plate_count=plate_count,
        viewport = viewport,
        context = dpgcontext
    )

def main():
    global viewport, dpgcontext
    dpgcontext = dpg.create_context()
    viewport = dpg.create_viewport(title='Simulation Menu', width=600, height=400)
    
    with dpg.window(label="Main Menu", tag="main", width=600, height=400):
        dpg.add_text("Simulation Main Menu", pos=[200, 50])
        
        # Play Button
        dpg.add_button(
            label="Play",
            callback=show_configure_window,
            pos=[250, 150],
            width=100,
            height=50
        )
        
        # Quit Button
        dpg.add_button(
            label="Quit",
            callback=lambda: dpg.stop_dearpygui(),
            pos=[250, 220],
            width=100,
            height=50
        )
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    #dpg.destroy_context()

if __name__ == "__main__":
    main()