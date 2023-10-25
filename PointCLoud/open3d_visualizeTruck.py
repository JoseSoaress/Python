import cv2, time
import numpy as np
import open3d as o3d

#vis= o3d.visualization.Visualizer()#creat an object
#vis.create_window(window_name='MyWindow', width=800, height=800, visible=True)#creat a window
#vis.add_geometry(pcd)#add the first PointCLoud
#vis.poll_events()#keeps the code running?
pcd = o3d.io.read_point_cloud('Truck.ply')

o3d.visualization.draw_geometries([pcd])
