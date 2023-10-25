from __future__ import print_function

import os, sys, numpy as np
import argparse
#from scipy import misc
import caffe #lib for Mlearning
import tempfile
from math import ceil
import cv2, time #opencv
import flowiz as fz #lib to convert flow in .png files
import freenect #kinect lib
import frame_convert2#anotherkinectlib

import matplotlib.pyplot as plt
import open3d as o3d

#--------------------------------------arguments for call the python program-----------------------------------------------------------------------
parser = argparse.ArgumentParser()
#FlowNet2 //My GPU as no memory
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2/FlowNet2_weights.caffemodel.h5')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2/FlowNet2_deploy.prototxt.template')
#FlowNet2-c /8.5 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-c/FlowNet2-c_weights.caffemodel')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-c/FlowNet2-c_deploy.prototxt.template')
#FlowNet2-C /5 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-C/FlowNet2-C_weights.caffemodel')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-C/FlowNet2-C_deploy.prototxt.template')
#FlowNet2-cs /6.7 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-cs/FlowNet2-cs_weights.caffemodel')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-cs/FlowNet2-cs_deploy.prototxt.template')
#FlowNet2-CS /3.5 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-CS/FlowNet2-CS_weights.caffemodel')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-CS/FlowNet2-CS_deploy.prototxt.template')
#FlowNet2-css /5.5 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-css/FlowNet2-css_weights.caffemodel')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-css/FlowNet2-css_deploy.prototxt.template')
#FlowNet2-CSS /2.67 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-CSS/FlowNet2-CSS_weights.caffemodel.h5')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-CSS/FlowNet2-CSS_deploy.prototxt.template')
#FlowNet2-css-ft-sd /5.5 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-css-ft-sd/FlowNet2-css-ft-sd_weights.caffemodel.h5')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-css-ft-sd/FlowNet2-css-ft-sd_deploy.prototxt.template')
#FlowNet2-css-ft-sd /2.6 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-CSS-ft-sd/FlowNet2-CSS-ft-sd_weights.caffemodel.h5')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-CSS-ft-sd/FlowNet2-CSS-ft-sd_deploy.prototxt.template')
#Flowet2-s /10.4 fps
parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-s/FlowNet2-s_weights.caffemodel')
parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-s/FlowNet2-s_deploy.prototxt.template')
#FlowNet2-S /7.8 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-S/FlowNet2-S_weights.caffemodel.h5')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-S/FlowNet2-S_deploy.prototxt.template')
#FlowNet2-SD /5.7 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template')
#FlowNet2-ss / 8 fps
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-ss/FlowNet2-ss_weights.caffemodel')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-ss/FlowNet2-ss_deploy.prototxt.template')

parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')
args = parser.parse_args()

if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)#debug
if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)#debug
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ colors for write in comand promp-------------------------------------------------------------------
RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"
CSI="\x1B["
print(GREEN+ "Used Network:" + CSI + "0m", RED+'%s'% args.caffemodel+CSI + "0m") #Debug
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ vars to calcule FPS and writhe them -------------------------------------------------------------------
fps=1
font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText    = (10,20)
fontScale              = 0.7
fontColor              = (0,230,0)
lineType               = 2
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------Global_Variables----------------------------------------------------------------------------------------
frame = np.float32()
frame_1 = np.float32()

depth = np.float32()
depth_1 = np.float32()

Fflow=np.float32()
width = 640 #450
height = 340 #240
widthF = width
heightF = height 
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------Main vars for OpticalFlow(flownet2.0)-------------------------------------------------------------------
num_blobs = 2
vars = {}
vars['TARGET_WIDTH'] = width
vars['TARGET_HEIGHT'] = height
divisor = 64.
vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)
vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);
tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
proto = open(args.deployproto).readlines()
for line in proto:
    for key, value in vars.items():
        tag = "$%s$" % key
        line = line.replace(tag, str(value))
    tmp.write(line)
tmp.flush()
if not args.verbose:
    caffe.set_logging_disabled()
caffe.set_mode_gpu()
caffe.set_device(args.gpu)
net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)
#-------------------------------------------------------------------------------------------------------------------------------------
def get_video():#function to get RGB image from kinect
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    #array = array[:,:,::-1]
    return array
def get_depth(): #function to get depth image from kinect
    array,_ = freenect.sync_get_depth()
    array = array.astype(dtype=np.uint16)
    return array
def readFlow(name):#funtion to read flow form a file
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]
    f = open(name, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')
    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)
def writeFlow(name, flow):#funtion to writte flow in a folder
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close() 

#-------this line is here to get the first image for allow first iteration to happen------------------------------------
frame_1 = frame   #frame T-1 equal to frame T, after that we get the Frame T.
depth_1 = depth
    
frame = get_video()    #get a frame from RGB camera
frame = cv2.resize(frame, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)

depth = get_depth()    #get a frame from depth sensor
depth = cv2.resize(depth, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)
#--------------------------------------------------------------------------------------------------------------------------------------  

if __name__ == "__main__":
	
    #start = time.time()#start timer fos fps calcule   
    #------------------------------ FAST ---------------------------------------------------------------
    fast = cv2.FastFeatureDetector(threshold=25, nonmaxSuppression=True)    # Initiate FAST object with default values
    #--------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------    O3D --------------------------------------------------------------------------------
    vis= o3d.visualization.Visualizer()#creat an object
    vis.create_window(window_name='3DScene', width=800, height=800,left=490, top=10, visible=True)#creat a window
    #--------------------------------------------------------------------------------------------------------------------------------------------
    
    frame_1 = frame   #frame T-1 equal to frame T, after that we get the Frame T.
    depth_1 = depth
    
    frame = get_video()    #get a frame from RGB camera
    frame = cv2.resize(frame, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)

    depth = get_depth()    #get a frame from depth sensor
    depth = cv2.resize(depth, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)
    depth.flatten();
    
    RGB = o3d.geometry.Image(frame)
    D = o3d.geometry.Image(depth)    
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(RGB, D,depth_scale=800.0, depth_trunc=1.5, convert_rgb_to_intensity=False)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault))

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    vis.add_geometry(pcd)
    vis.clear_geometries
        
    while 1:
       
        start = time.time()#start timer fos fps calcule
        frame_1 = frame   #frame T-1 equal to frame T, after that we get the Frame T.
        depth_1 = depth
    
        frame = get_video()    #get a frame from RGB camera
        frame = cv2.resize(frame, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)
    
        depth = get_depth()    #get a frame from depth sensor
        depth = cv2.resize(depth, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)
        
        v2 = depth.astype(np.float) #conversion to float to allow the calcule depth = (depth * -0.0030711016)  +  3.3309495161
        # #beggining of the calcule from grayscale to centimeters
        v2 = 100/(v2 *  -0.0030711016 + 3.3309495161)
        #end of the calcule from grayscale to centimeters
        depth = v2.astype(np.uint16) #conversion to unsigned int
        #depth[depth>350] = 0  #clear the points above 350 cm
      
        

       
        RGB = o3d.geometry.Image(frame)
        D = o3d.geometry.Image(depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(RGB, D,depth_scale=800.0, depth_trunc=1.5, convert_rgb_to_intensity=False)
         
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        #pcd.voxel_down_sample(voxel_size = 0.05);#not sure if it works
        pcd.remove_none_finite_points(remove_nan = True, remove_infinite = True) # funtion to remove points that are 0 or infinite.
        
        #pcd.normalize_normals()
        #pcd.remove_radius_outlier(10,0.005) #remove points that are allow in a range
        
        # Flip it, otherwise the pointcloud will be upside down 
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        points = [  [0, 0, 0]  ]      
            
       
       #add a sphere to represent camera position in O3D
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                #mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([1, 0 ,0])     
        vis.add_geometry(mesh_sphere,reset_bounding_box=False)
         #---------------------------------------------------------
        #add a point to represent camera position
        CamPoint = o3d.geometry.PointCloud()
        CamPoint.points = o3d.utility.Vector3dVector(points)
        vis.add_geometry(CamPoint,reset_bounding_box=False)
        #---------------------------------------------------------
        
        #raw_input("Press Enter to continue...")     
        vis.add_geometry(pcd,reset_bounding_box=False)#add the first PointCLoud
        vis.poll_events()
        vis.update_renderer()
        #vis.run()
        vis.clear_geometries()
        #vis.remove_geometry(pcd,reset_bounding_box=False)#needs to be False to work
 #--------------------------------------------------------------------------------------------------------------------------------------------
 #------------------------------Main Code for OpticalFlow(flownet2.0)-------------------------------------------------------------------
        input_data = []    #Force to clean the var        
        
        input_data.append(frame_1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        input_data.append(frame[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
           
        input_dict = {}    #Force to clean the var  
        for blob_idx in range(num_blobs):
            input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
            
        net.forward(**input_dict)
        blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
            
        #Fflow = fz._flow2color(blob) #convert the flow file to a png file for better understanding and vizualization
     
        #kp = fast.detect(frame,None)
        #img = cv2.drawKeypoints(frame, kp, color=(255,0,0))
        
        
        #FPS calcule:
        time_taken = time.time() -start
        fps = 1./time_taken
            
        #4. write FPS counter in the images
        #cv2.putText(depth,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType) #writhe fps on the image
        cv2.putText(frame,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType) #writhe fps on the image
        #cv2.putText(Fflow,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType)#writhe fps on the image
            
        #4. show the RGB and the FLOW frame!
        cv2.imshow('Depth image',depth)
        cv2.imshow("RGB-Image", frame)
        #cv2.imshow("OpticalFlow", Fflow) 
    #------------------------------------------------------------------------------------------------------------------------------------------
        key=cv2.waitKey(1)
        if key == ord('x'):
            break
    cv2.destroyAllWindows