import dearpygui.dearpygui as dpg
import numpy as np

dpg.create_context()

# Create a 255x255 RGBA texture using numpy array
texture_array = np.zeros((255, 255, 4), dtype=np.float32)

# Fill the texture with gradient data
for r in range(1, 256):
    for g in range(1, 256):
        texture_array[r-1, g-1] = [r/255, g/255, r/255, g/255]

# Flatten the array for Dear PyGui
texture_data = texture_array.ravel()

with dpg.texture_registry(show=True):
    dpg.add_raw_texture(width=255, height=255, default_value=texture_data, 
                        format=dpg.mvFormat_Float_rgba, tag="texture_tag")

def update_dynamic_texture(sender, app_data, user_data):
    new_color = np.array(dpg.get_value(sender)) / 255
    # Update the entire texture with the new color
    texture_array[:, :] = new_color
    # Update the texture data
    texture_data[:] = texture_array.ravel()

with dpg.window(label="Tutorial"):
    dpg.add_image("texture_tag")
    dpg.add_color_picker((255, 0, 255, 255), label="Texture",
                         no_side_preview=True, alpha_bar=True, width=200,
                         callback=update_dynamic_texture)

dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()