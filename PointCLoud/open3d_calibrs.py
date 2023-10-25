import freenect
import cv2, time
import sys
import termios
import curses#termios and curses is to flush the stdin


import csv
import plyfile

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

#for writhe the FPS in the image
fps=1
font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText = (10,20)
fontScale              = 0.7
fontColor              = (0,230,0)
lineType               = 2

width = 640 #450
height = 340 #240
widthF = width
heightF = height 

#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    #array = cv2.cvtColor(array,cv2.COLOR_RGB2HSV)
    array = array[:,:,::-1]
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    #array = array.astype(dtype=np.uint8)
    array = array.astype(dtype=np.uint16)
    #array = cv2.cvtColor(array,cv2.COLOR_GRAY2RGB)
    #array = cv2.cvtColor(array,cv2.COLOR_RGB2HSV)
    return array
    
def depth2xyzuv(depth, u=None, v=None):
    if u is None or v is None:
        u,v = np.mgrid[:340,:640]  
        #u,v = np.mgrid[:640,:340]  
        #u,v = np.mgrid[:height,:width]  
        #u,v = np.mgrid[widthF,heightF]
        #u,v = np.mgrid[:480,:640]  
        #u,v = np.mgrid[:640,:480]  
    # Build a 3xN matrix of the d,u,v data
    C = np.vstack((u.flatten(), v.flatten(), depth.flatten(), 0*u.flatten()+1))
    # Project the duv matrix into xyz using xyz_matrix()
    X,Y,Z,W = np.dot(xyz_matrix(),C)
    X,Y,Z = X/W, Y/W, Z/W
    xyz = np.vstack((-X,-Y,Z)).transpose()
    xyz = xyz[Z<0,:]
    # Project the duv matrix into U,V rgb coordinates using rgb_matrix() and xyz_matrix()
    #U,V,_,W = np.dot(np.dot(uv_matrix(), xyz_matrix()),C)
    #U,V = U/W, V/W
    #uv = np.vstack((U,V)).transpose()
    #uv = uv[Z<0,:]
    return xyz
    
def uv_matrix():
  #"""Returns a matrix you can use to project XYZ coordinates (in meters) into  U,V coordinates in the kinect RGB image"""
  rot = np.array([[ 9.99846e-01,   -1.26353e-03,   1.74872e-02], 
                  [-1.4779096e-03, -9.999238e-01,  1.225138e-02],
                  [1.747042e-02,   -1.227534e-02,  -9.99772e-01]])
  trans = np.array([[1.9985e-02, -7.44237e-04,-1.0916736e-02]])
  m = np.hstack((rot, -trans.transpose()))
  m = np.vstack((m, np.array([[0,0,0,1]])))
  KK = np.array([[529.2, 0, 329, 0],
                 [0, 525.6, 267.5, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
  m = np.dot(KK, (m))
  return m

def xyz_matrix():
  fx = 594.21
  fy = 591.04
  a = -0.0030711
  b = 3.3309495
  cx = 339.5
  cy = 242.7
  mat = np.array([[1/fx, 0, 0, -cx/fx],
                  [0, -1/fy, 0, cy/fy],
                  [0,   0, 0,    -1],
                  [0,   0, a,     b]])
  return mat
  
    
if __name__ == "__main__":
 
    # fx = 594.21
    # fy = 591.04
    # cx = 339.5
    # cy = 242.7
    #o3d.set_intrinsics(self, widthF,heightF, fx, fy, cx, cy)
    
    
    depth = get_depth()  
    depth = cv2.resize(depth, (widthF,heightF)) 
    #depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2RGB)
    
    frame = get_video()
    frame = cv2.resize(frame, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)

    RGB = o3d.geometry.Image(frame)
    D = o3d.geometry.Image(depth)    
    
    depthpoints = depth2xyzuv(depth)
    #depthpoints = cv2.resize(depthpoints, (640,480)) #resize give extra frames (need to check the best loss-reward ratio)
    
    vis= o3d.visualization.Visualizer()#creat an object
    vis.create_window(window_name='MyWindow', width=800, height=800,left=490, top=10, visible=True)#creat a window

    #rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(frame, depth)
    #rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(frame, depth,depth_trunc=config["max_depth"],  convert_rgb_to_intensity=convert_rgb_to_intensity)
    #depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2RGB)
    # plt.subplot(1, 2, 1)
    # plt.title('Redwood grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Redwood depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()
    
    # plt.subplot(1, 2, 1)
    # plt.title('Redwood grayscale image')
    # plt.imshow(frame)
    # plt.subplot(1, 2, 2)
    # plt.title('Redwood depth image')
    # plt.imshow(depth)
    # plt.show()
    #o3d.visualization.draw_geometries([depthpoints])    
    
    #print(depth)
    #RGB = o3d.geometry.Image(frame)
    #D = o3d.geometry.Image(depth)
    #rgbd = o3d.geometry.Image(depthpoints)
   
    #rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(RGB, D)
    #rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(RGB, D#first 2 don't have good results
    #rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(RGB, D)
    #rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(RGB, D)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(RGB, D,depth_scale=2000.0, depth_trunc=3, convert_rgb_to_intensity=False)

    
    # plt.subplot(1, 2, 1)
    # plt.title('Redwood grayscale image')
    # plt.imshow(rgbd_image.color),fx=f
    # plt.subplot(1, 2, 2)
    # plt.title('Redwood depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()
    
    # fx = 594.21
    # fy = 591.04
    # cx = 339.5
    # cy = 242.7
    
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(640,340,594.21,591.04,339.5, 242.7))
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(width = 640,height=340,x = 594.21,    fy = 591.04,    cx = 339.5,    cy = 242.7))
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault))
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault))
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault))
    
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(depthpoints, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault))

   # Flip it, otherwise the pointcloud will be upside down 
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

   
    vis.add_geometry(pcd)#add the first PointCLoud
    #vis.remove_geometry(pcd)
    vis.clear_geometries()
    #vis.run()

    #print(o3d.camera.PinholeCameraIntrinsic)

    #plt.imshow(np.asarray(depth))
    #plt.show()
    
    
    threshold = 0.05
    icp_iteration = 100
    save_image = False   



      # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(depthpoints)
    # o3d.visualization.draw_geometries([pcd])
    i=0
    while 1:
        start = time.time()#start time to calcule FPS
        
        #get a frame from RGB camera
        frame = get_video()
        frame = cv2.resize(frame, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)

         #---------------------------------------------------------
        #get a frame from depth sensor
        depth = get_depth()  
        depth = cv2.resize(depth, (widthF,heightF)) 
        #depthpoints = depth2xyzuv(depth)    
        #depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2RGB);
        
        
        RGB = o3d.geometry.Image(frame)
        D = o3d.geometry.Image(depth)
        #rgbd = o3d.geometry.Image(depthpoints)
   
        #rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(RGB, D)
        #rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(RGB, D#first 2 don't have good results
        #rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(RGB, D)
        #rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(RGB, D)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(RGB, D,depth_scale=800.0, depth_trunc=1.5, convert_rgb_to_intensity=False)
    
        
        #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault))
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault))
        #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    
        #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault))
    
        
        # Flip it, otherwise the pointcloud will be upside down 
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        #raw_input("Press Enter to continue...")

        vis.add_geometry(pcd,reset_bounding_box=False)#add the first PointCLoud

        
        #vis.update_geometry(temp)
        vis.poll_events()
        vis.update_renderer()

        #vis.run()
        vis.remove_geometry(pcd,reset_bounding_box=False)#needs to be False to work
        #vis.poll_events()

        #v.reset()
        # print "Create csv header..."
        # f = open("/home/gerry/depthtest.csv",'a')
        # f.write("x,y,z\n")
        # f.close()
        # print "writing to text file...please wait...."
        # with open("/home/gerry/depthtest.csv", 'a') as f:
            # csvwriter = csv.writer(f)
            # csvwriter.writerows(depthpoints)   
        # print "finished writing to text file..."
        # print "done"
        
        #---------------------------------------------------------
         #display RGB image
        frame = cv2.resize(frame, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)
        frame = frame[:,:,::-1]
        cv2.putText(frame,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType) 
        cv2.imshow('RGB image',frame)
         #---------------------------------------------------------
        #display depth image
        depth = cv2.resize(depth, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)
        cv2.putText(depth,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.imshow('Depth image',depth)
        #---------------------------------------------------------            
        # Main - Code
        
        #Compute FPS
        time_taken = time.time() -start
        fps = 1./time_taken
         #---------------------------------------------------------
         
        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()