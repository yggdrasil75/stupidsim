from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from system import System


class SystemViewer:
    def __init__(self, system: System):
        self.system: System = system
        self.current_world_index = 0 if system.world else None
        
    def launch(self):
        plt.ion()  # Interactive mode on
        self.fig = plt.figure(figsize=(15, 8))
        
        # Create buttons
        self.ax_system = self.fig.add_subplot(121, projection='3d')
        self.ax_world = self.fig.add_subplot(122, projection='3d')
        
        # Add control buttons
        self.prev_btn_ax = self.fig.add_axes([0.4, 0.05, 0.1, 0.05])
        self.next_btn_ax = self.fig.add_axes([0.5, 0.05, 0.1, 0.05])
        self.prev_btn = Button(self.prev_btn_ax, 'Previous World')
        self.next_btn = Button(self.next_btn_ax, 'Next World')
        
        self.prev_btn.on_clicked(self.prev_world)
        self.next_btn.on_clicked(self.next_world)
        
        self.update_display()
        
    def update_display(self):
        self.ax_system.clear()
        self.ax_world.clear()
        
        # Plot system view
        self.system.plot_system(ax=self.ax_system)
        
        # Plot current world if available
        if self.current_world_index is not None:
            world = self.system.world[self.current_world_index]
            world.plot(ax=self.ax_world)
            self.ax_world.set_title(f'World {self.current_world_index}')
        
        self.fig.canvas.draw()
    
    def prev_world(self, event):
        if self.system.world and self.current_world_index > 0:
            self.current_world_index -= 1
            self.update_display()
    
    def next_world(self, event):
        if self.system.world and self.current_world_index < len(self.system.world) - 1:
            self.current_world_index += 1
            self.update_display()