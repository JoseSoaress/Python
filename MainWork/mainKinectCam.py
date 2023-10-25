from __future__ import print_function

import os, sys, numpy as np
import argparse

import caffe #lib for Mlearning
import flowiz as fz #lib to convert flow in .png files

import tempfile
from math import ceil
import cv2, time #opencv



import freenect #kinect lib

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

#sintelExample
#parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models-sintel/FlowNet2-Sintel/FlowNet2-CSS-Sintel_weights.caffemodel.h5')
#parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models-sintel/FlowNet2-Sintel/FlowNet2-CSS-Sintel_deploy.prototxt.template')
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
    #cv2.putText(array,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType) 
    return array
def get_depth(): #function to get depth image from kinect
    #array = frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])
    array,_ = freenect.sync_get_depth()
    #array = 255 * np.logical_and(array > 350 , array <= 1000)
    array = array.astype(np.uint16)
    #array = cv2.cvtColor(array,cv2.COLOR_GRAY2RGB);
    #array = cv2.cvtColor(array,cv2.COLOR_RGB2HSV);
    #cv2.putText(array,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType)
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
frame = get_video()    #get a frame from RGB camera
frame = cv2.resize(frame, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)
#--------------------------------------------------------------------------------------------------------------------------------------  

while True: 
	
    start = time.time()#start timer fos fps calcule
    frame_1 = frame   #frame T-1 equal to frame T, after that we get the Frame T.
  
    frame = get_video()    #get a frame from RGB camera
    frame = cv2.resize(frame, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)

    depth = get_depth()    #get a frame from depth sensor
    depth = cv2.resize(depth, (640, 340)) #resize give extra frames (need to check the best loss-reward ratio)

 #--------------------------------------------------------------------------------------------------------------------------------------------
 #------------------------------Main Code for OpticalFlow(flownet2.0)-------------------------------------------------------------------
    input_data = []    #Force to clean the var        
    #input_data.append(frame_1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    #input_data.append(frame[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
      
    input_data.append(frame_1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    input_data.append(frame[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]) 
    
    input_dict = {}    #Force to clean the var  
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
        
    net.forward(**input_dict)
    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        
    Fflow = fz._flow2color(blob) #convert the flow file to a png file for better understanding and vizualization
    #Fflow = fz._flow2uv(blob)
    #FPS calcule:
    time_taken = time.time() -start
    fps = 1./time_taken
        
    #4. write FPS counter in the images
    #cv2.putText(depth,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType) #writhe fps on the image
    #cv2.putText(frame,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType) #writhe fps on the image
    cv2.putText(Fflow,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType)#writhe fps on the image
        
    #4. show the RGB and the FLOW frame!
    cv2.imshow('Depth image',depth)
    cv2.imshow("RGB-Image", frame)
    cv2.imshow("OpticalFlow", Fflow) 
#------------------------------------------------------------------------------------------------------------------------------------------
    key=cv2.waitKey(1)
    if key == ord('x'):
		break

cv2.destroyAllWindows

