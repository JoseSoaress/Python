import cv2, time
import open3d as o3d
import numpy as np

#o3d.visualization.Visualizer.remove_geometry()
#pcd = o3d.visualization.Visualizer.create_window(self, window_name='pum', width=1920, height=1080, left=50, top=50, visible=True)    
#vis.create_window()  
#vis.update_geometry()
#vis.run()    

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()

print("Reading PointCLoud")
pcd = o3d.io.read_point_cloud('Truck.ply')

vis= o3d.visualization.Visualizer()#creat an object
vis.create_window(window_name='MyWindow', width=800, height=800, visible=True)#creat a window

print("FIrst Render")
vis.add_geometry(pcd)#add the first PointCLoud
vis.poll_events()
vis.run() #after use this funtion need to presse "q" to keep the program going (
#vis.destroy_window()

time.sleep(1.5) #give time for the viewer see the diferences
print("Removing the first render")
#remove the geometry from the window
#vis.remove_geometry(pcd,reset_bounding_box=False)#just removes thoose points from the window******+needs to be False to work
vis.clear_geometries()#clear all the geometries in the window
vis.poll_events()
vis.update_renderer  #not neede in this example

time.sleep(1.5)#give time for the viewer see the diferences
print("Second Render")
#add the geometry again in the same window
vis.add_geometry(pcd,reset_bounding_box=False) #allow to save view_point
#vis.update_geometry(pcd)
vis.poll_events()
vis.update_renderer()

#vis.run() 
raw_input("Press Enter to continue...")

time.sleep(1.5)#give time for the viewer see the diferences
print("done")

vis.destroy_window()
#del vis #might be needed in some examples


#COde to save a view_point #(might be usufull when rendering all tehe point cluds.


# if __name__ == "__main__":
    # pcd = o3d.io.read_point_cloud("../../TestData/fragment.pcd")
    # save_view_point(pcd, "viewpoint.json")
    # load_view_point(pcd, "viewpoint.json")