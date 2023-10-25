#!/usr/bin/env python2.7
from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import cv2, time #opencv
import flowiz as fz
import freenect #kinect lib

parser = argparse.ArgumentParser()
#the first 2 arguments allow to choose another network from the comand line.
parser.add_argument('--caffemodel', help='path to model',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-c/FlowNet2-c_weights.caffemodel')
parser.add_argument('--deployproto', help='path to deploy prototxt template',default='/home/soares/workspace/flownet2/models/flownet2-models/FlowNet2-c/FlowNet2-c_deploy.prototxt.template')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')
args = parser.parse_args()

if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)


#--------------------------------------Global_Variables-----------------------------------------
flag=0
frame = np.float32()
frame_1 = np.float32()
Fflow=np.float32()
width = -1
height = -1

#show fps vars
fps=1
font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText = (10,20)
fontScale              = 0.7
fontColor              = (0,230,0)
lineType               = 2

#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    #cv2.putText(array,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType) 
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    array = cv2.cvtColor(array,cv2.COLOR_GRAY2RGB);
    array = cv2.cvtColor(array,cv2.COLOR_RGB2HSV);
    #array = cv2.cvtColor(array,cv2.COLOR_RGB2BRG);
    #cv2.putText(array,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType)
    return array
    
def readFlow(name):
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
def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close() 

#colors for write in comand promp
RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"
CSI="\x1B["
print(GREEN+ "Used Network:" + CSI + "0m", RED+'%s'% args.caffemodel+CSI + "0m") #Debug
#print(CSI+"31;40m" + "Used Network:" + CSI + "0m")
#--------------------------------------------------------------------------------------------


while True: 
	
    start = time.time()#start timer fos fps calcule	
    frame_1 = frame   #frame T-1 equal to frame T, after that we get the Frame T.
    #get a frame from RGB camera
    frame = get_video()
    frame = cv2.resize(frame, (640, 340)) #resize give extra frames (need to check the best loss-reward ratio)
    #get a frame from depth sensor
    depth = get_depth()
    depth = cv2.resize(depth, (640, 340)) #resize give extra frames (need to check the best loss-reward ratio)
    
 #---------------------------------------------------------------------------------------------------------------------------
    if flag > 0 : #Permite testar e vizualizar a frame anterior(depois da primeira iteracao e que temos as 2 imagens

        num_blobs = 2
        input_data = []
        img0 = frame_1#misc.imread(args.img0)
        if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        img1 = frame#misc.imread(args.img1)
        if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

        if width != input_data[0].shape[3] or height != input_data[0].shape[2]:
            width = input_data[0].shape[3]
            height = input_data[0].shape[2]

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
                #otimizacao realizada a versao original que melhorou a velocidade do programa.
        if not args.verbose:
            caffe.set_logging_disabled()
        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()
        net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)
            
        input_dict = {}
        for blob_idx in range(num_blobs):
            input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

        #
        # There is some non-deterministic nan-bug in caffe#don't understand this coment
        #
       
        i = 1
        while i<=1: #tenho que trabalahr esta parte do codigo este while em tempo real nao e necessario
            i+=1

            net.forward(**input_dict)

            containsNaN = False
            for name in net.blobs:
                blob = net.blobs[name]
                has_nan = np.isnan(blob.data[...]).any()

                if has_nan:
                    print('blob %s contains nan' % name)
                    containsNaN = True

            if not containsNaN:
             #   print('Succeeded.')
                break
            else:
                print('**************** FOUND NANs, RETRYING ****************')

        blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        
        Fflow = fz._flow2color(blob) #convert the flow file to a png file for better understanding and vizualization       
        #FPS calcule:
        time_taken = time.time() -start
        fps = 1./time_taken      
        #4. write FPS counter in the images
        cv2.putText(depth,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType) #writhe fps on the image
        cv2.putText(frame,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType) #writhe fps on the image
        cv2.putText(Fflow,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType)#writhe fps on the image
        
        #4. show the RGB and the FLOW frame!
        cv2.imshow('Depth image',depth)
        cv2.imshow("RGB-Image", frame)
        cv2.imshow("OpticalFlow", Fflow) 
#-------------------------------------------------------------------------------------------------------------------------
    flag=1;
    key=cv2.waitKey(1)
    if key == ord('x'):
		break


cv2.destroyAllWindows

#def calculateFlow(image_i,image1)
#rreturn flow.astype(np.float32)
