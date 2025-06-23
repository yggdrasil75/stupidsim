import dearpygui.dearpygui as dpg

class SimulationUI:
    def __init__(self):
        dpg.create_context()
        self.game = GameOfLife()
        self.setup_ui()

    def setup_ui(self):
        with dpg.window(tag="Main Window"):
            with dpg.group(horizontal=True):
                # Control panel
                with dpg.child_window(width=300):
                    dpg.add_text("Evolution Simulation")
                    dpg.add_button(label="Start", callback=lambda: self.game.set_running(True))
                    dpg.add_button(label="Stop", callback=lambda: self.game.set_running(False))
                    dpg.add_button(label="Step", callback=self.game.step)
                    dpg.add_button(label="Clear", callback=self.game.clear_grid)
                    dpg.add_button(label="Randomize", callback=self.game.randomize_grid)
                    
                    dpg.add_spacer(height=10)
                    dpg.add_slider_int(label="Grid Size", min_value=10, max_value=100, 
                                     default_value=self.game.grid_size, callback=self.game.change_grid_size)
                    
                    # Environment settings
                    dpg.add_spacer(height=10)
                    dpg.add_text("Environment Settings:")
                    self.axial_tilt_slider = dpg.add_slider_float(
                        label="Axial Tilt (degrees)", 
                        min_value=0.0, max_value=45.0, 
                        default_value=23.44, 
                        callback=self.game.update_environment_settings
                    )
                    self.latitude_slider = dpg.add_slider_float(
                        label="Latitude (degrees)", 
                        min_value=0.0, max_value=90.0, 
                        default_value=45.0, 
                        callback=self.game.update_environment_settings
                    )

                    # Z-axis viewing controls
                    dpg.add_spacer(height=10)
                    dpg.add_text("Z-Axis View:")
                    with dpg.group(horizontal=True):
                        dpg.add_radio_button(
                            items=["Slider", "Top", "Bottom"],
                            default_value="Slider",
                            callback=self.game.change_z_view_mode
                        )
                    self.z_slider = dpg.add_slider_int(
                        label="Z Level",
                        min_value=0,
                        max_value=self.game.max_z_level,
                        default_value=0,
                        callback=self.game.change_z_level,
                        show=False  # Initially hidden unless in slider mode
                    )
                    
                    # Status displays
                    self.time_text = dpg.add_text("Time: Day 1, 00:00", tag="time_text")
                    self.daylight_text = dpg.add_text(f"Daylight: {self.game.current_daylight/2}h", tag="daylight_text")
                    self.light_level_text = dpg.add_text(f"Light: {self.game.light_level:.2f}", tag="light_level_text")
                    self.temp_text = dpg.add_text(f"Temp: {self.game.temperature:.2f}", tag="temp_text")
                    self.date_text = dpg.add_text("Date: Jan 1", tag="date_text")
                
                with dpg.child_window(tag="game_window"):
                    with dpg.drawlist(width=self.game.grid_size*self.game.cell_size, 
                                    height=self.game.grid_size*self.game.cell_size, 
                                    tag="drawlist"):
                        pass
        dpg.set_viewport_resize_callback(self.game.on_viewport_resize)
        dpg.set_primary_window("Main Window", True)
        dpg.create_viewport(title="Evolution", width=850,height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self):
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
            

if __name__ == "__main__":
    UI = SimulationUI()
    UI.run()