import dearpygui.dearpygui as dpg
import math

dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()

size = 5
verticies = [
        [-size, -size, -size],  # 0 near side
        [ size, -size, -size],  # 1
        [-size,  size, -size],  # 2
        [ size,  size, -size],  # 3
        [-size, -size,  size],  # 4 far side
        [ size, -size,  size],  # 5
        [-size,  size,  size],  # 6
        [ size,  size,  size],  # 7
        [-size, -size, -size],  # 8 left side
        [-size,  size, -size],  # 9
        [-size, -size,  size],  # 10
        [-size,  size,  size],  # 11
        [ size, -size, -size],  # 12 right side
        [ size,  size, -size],  # 13
        [ size, -size,  size],  # 14
        [ size,  size,  size],  # 15
        [-size, -size, -size],  # 16 bottom side
        [ size, -size, -size],  # 17
        [-size, -size,  size],  # 18
        [ size, -size,  size],  # 19
        [-size,  size, -size],  # 20 top side
        [ size,  size, -size],  # 21
        [-size,  size,  size],  # 22
        [ size,  size,  size],  # 23
    ]

colors = [
        [255,   0,   0, 150],
        [255, 255,   0, 150],
        [255, 255, 255, 150],
        [255,   0, 255, 150],
        [  0, 255,   0, 150],
        [  0, 255, 255, 150],
        [  0,   0, 255, 150],
        [  0, 125,   0, 150],
        [128,   0,   0, 150],
        [128,  70,   0, 150],
        [128, 255, 255, 150],
        [128,   0, 128, 150]
    ]

with dpg.window(label="tutorial", width=550, height=550):

    with dpg.drawlist(width=500, height=500):

        with dpg.draw_layer(tag="main pass", depth_clipping=True, perspective_divide=True, cull_mode=dpg.mvCullMode_Back):

            with dpg.draw_node(tag="cube"):

                dpg.draw_triangle(verticies[1],  verticies[2],  verticies[0], color=[0,0,0.0],  fill=colors[0])
                dpg.draw_triangle(verticies[1],  verticies[3],  verticies[2], color=[0,0,0.0],  fill=colors[1])
                dpg.draw_triangle(verticies[7],  verticies[5],  verticies[4], color=[0,0,0.0],  fill=colors[2])
                dpg.draw_triangle(verticies[6],  verticies[7],  verticies[4], color=[0,0,0.0],  fill=colors[3])
                dpg.draw_triangle(verticies[9],  verticies[10], verticies[8], color=[0,0,0.0],  fill=colors[4])
                dpg.draw_triangle(verticies[9],  verticies[11], verticies[10], color=[0,0,0.0], fill=colors[5])
                dpg.draw_triangle(verticies[15], verticies[13], verticies[12], color=[0,0,0.0], fill=colors[6])
                dpg.draw_triangle(verticies[14], verticies[15], verticies[12], color=[0,0,0.0], fill=colors[7])
                dpg.draw_triangle(verticies[18], verticies[17], verticies[16], color=[0,0,0.0], fill=colors[8])
                dpg.draw_triangle(verticies[19], verticies[17], verticies[18], color=[0,0,0.0], fill=colors[9])
                dpg.draw_triangle(verticies[21], verticies[23], verticies[20], color=[0,0,0.0], fill=colors[10])
                dpg.draw_triangle(verticies[23], verticies[22], verticies[20], color=[0,0,0.0], fill=colors[11])

x_rot = 10
y_rot = 45
z_rot = 0

view = dpg.create_fps_matrix([0, 0, 50], 0.0, 0.0)
proj = dpg.create_perspective_matrix(math.pi*45.0/180.0, 1.0, 0.1, 100)
model = dpg.create_rotation_matrix(math.pi*x_rot/180.0 , [1, 0, 0])*\
                        dpg.create_rotation_matrix(math.pi*y_rot/180.0 , [0, 1, 0])*\
                        dpg.create_rotation_matrix(math.pi*z_rot/180.0 , [0, 0, 1])

dpg.set_clip_space("main pass", 0, 0, 500, 500, -1.0, 1.0)
dpg.apply_transform("cube", proj*view*model)

dpg.show_viewport()
while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()

dpg.destroy_context()