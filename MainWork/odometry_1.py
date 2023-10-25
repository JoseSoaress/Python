#!/usr/bin/env python
# Python libs
import sys, time

# numpy and scipy
import numpy as np

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy
import ros_numpy
import tf
import math

#import pcl

# Ros Messages
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 
# We do not use cv_bridge it does not support CompressedImage in python
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt

class vis_odometry():

    def __init__(self):
        self.br = CvBridge() #allow conversion from RosData to OPenCv data
        self.loop_rate = rospy.Rate(8)        # Node cycle rate (in Hz).
                
        self.feature_params = dict( maxCorners = 300,
                            qualityLevel = 0.3,
                            minDistance = 5,
                            blockSize = 5 )
                            
        #self.focal_length = 1.0#3.35 #kinect value(mean between x and y ) 3.3985
        #self.pp = (320,240)#(0,0) #kinect value  --
        self.prob = 0.999    
        
        #self.focal_length = 3.35#3.35 #kinect value(mean between x and y ) 3.3985
        self.focal_length = 515.6# calibracao ROS      
        self.pp = (330.27965,259.16179) #calibracao usada ROS
        #self.pp = (319.5,239.5) #MeioImagem
        
        self.lk_params=dict(winSize  = (21,21), 
                                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,  30,  0.01))

        # self.lk_params=dict(winSize  = (21,21), 
                                         # maxLevel = 5,
                                        # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,  30,  0.01))
        
        #self.lk_params=dict(winSize  = (21,21), 
        #                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,  30,  0.01))
      
        # self.lk_params = dict( winSize  = (15,15),
                  # maxLevel = 2,
                  # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                  
        # self.lk_params = dict(winSize  = (21,21), 
               # maxLevel = 2,                #allows to use piramids or not for big movements!
               # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
       
       #variables to calculate visual odometry
        self.R = np.zeros(shape=(3, 3))#rotation matrix
        self.t  = np.zeros(shape=(3, 1))#translation matrix            

        self.id = 0
        self.n_features = 0
        
        self.K = np.array( [[515.83235,   0.     , 330.27965],     [0.     , 515.40825, 259.16179],       [0.     ,   0.     ,   1.     ]], dtype=np.float64)
        self.dist_coefficients = np.array([0.186108, -0.396983, 0.002316, -0.002322, 0.000000],dtype=np.float64)
        
        self.detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        #self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=False)
        
        self.last_frame = None
        self.last_depth = None
        self.frame_aux = None
        self.depth_aux = None   
        
        self.old_frame = None
        self.new_frame = None
        self.old_depth = None
        self.new_depth = None
        
        self.p_frame_1 = None
        self.p_frame = None
        
        self.depth_p_1 = None
        self.depth_p = None
        
        self.transformT = tf.TransformBroadcaster()
        
        # Subscribers
        #self.subscriber = rospy.Subscriber("/camera/rgb/image_color", Image, self.rgb_callback,  queue_size = 1)
        #self.subscriber = rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.rgb_callback,  queue_size = 1)
        self.subscriber = rospy.Subscriber("/camera/rgb/image_rect_mono", Image, self.rgb_callback,  queue_size = 1)
        
        self.subscriber2 = rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect", Image, self.depth_callback,  queue_size = 1)
        #self.subscriber2 = rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect_raw", Image, self.depth_callback,  queue_size = 1)
        #self.subscriber2 = rospy.Subscriber("/camera/depth/image_rect", Image, self.depth_callback,  queue_size = 1)

        # Publishers
        self.pub = rospy.Publisher('FeaturesRGB', Image,queue_size=5)
        self.pub2 = rospy.Publisher('FeaturesDepth', Image,queue_size=5)

    def rgb_callback(self, data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        #rospy.loginfo('Image received...rgb')
        try:
            self.last_frame = self.br.imgmsg_to_cv2(data) #Converts from RosMessage to CV2
            #self.frame_1 = self.last_frame
        except CvBridgeError as e:
            print(e)       
  
    def depth_callback(self, data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        #rospy.loginfo('Image received...depth')
        try:
            self.last_depth = self.br.imgmsg_to_cv2(data) #Converts from RosMessage to CV2
        except CvBridgeError as e:
            print(e)      
           
    def start(self):
        rospy.loginfo("Starting Program")
        rospy.sleep(2)
        self.old_frame = self.last_frame
        self.old_depth = self.last_depth
        self.depth_p_1 = []# #clear array
        self.depth_p = []   #clear array
        while not rospy.is_shutdown():#main loop

            print("Starts Here")
            if self.n_features < 150:#
                Raw_p_frame_1 = self.detector.detect(self.old_frame)   #points detected from the previous frame      -> self.old_frame    || old_gray
                self.p_frame_1 = np.array([x.pt for x in Raw_p_frame_1], dtype=np.float32).reshape(-1, 1, 2)#converts the data to match the next funtion data type  (original form)                 

          ##----------------------Calculation of the good points and its matches -----------------------------
            self.new_frame = self.last_frame    #Get the last RGB-frame
            self.new_depth = self.last_depth    #Get the last depth-frame
            self.p_frame, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.new_frame, self.p_frame_1, None,**self.lk_params)#calculate the opticalFLow
            good_p_1 = self.p_frame_1[st == 1]  #Good points from image (T)
            good_p = self.p_frame[st == 1]      #Good points from image (T+1)                   
            self.n_features = good_p.shape[0]      
            #print("Number of Features",self.n_features)     
          ##------------------ Clear the points that are outside the image range x>=0, x<640 || y>=0, y<480 ------ This is here, because the function calcOpticalFlowPyrLK predict the points movements and they can be outside the range
            x = good_p[:,0] 
            y =  good_p[:,1]
            mask = np.logical_and(np.logical_and(x>=0, x<640), np.logical_and(y>=0, y<480)) #mask to remove outsiders
            good_p = good_p[mask] #aply mask to take the outsiders
            good_p_1 = good_p_1[mask] #aply the same mask on the others points, to have the same points
                #this may seem redundant but, we need to check in both directions
            x = good_p_1[:,0] 
            y =  good_p_1[:,1]
            mask = np.logical_and(np.logical_and(x>=0, x<640), np.logical_and(y>=0, y<480)) #mask to remove outsiders
            good_p_1 = good_p_1[mask] #aply mask to take the outsiders
            good_p = good_p[mask] #aply the same mask on the others points, to have the same points
           ## ----------------------------------------------------------------------------------------------------------------------------------------   
           ##------------------  Check the depth in the features-points and form a numpy array --------------- 
             ##I must corvert the array from Float to INt in order to make the loop and acess the depth values. This introduce an error but is needed
            good_p_1_aux = good_p_1.astype(int)    #some issues with this. But needed for the loop
            good_p_aux = good_p.astype(int)         
            for point in good_p_aux:#loop for remove the points    
               self.depth_p = np.append(self.depth_p,self.new_depth[point[1],point[0]])      
            for point in good_p_1_aux:#loop for remove the points    
                self.depth_p_1 = np.append(self.depth_p_1,self.old_depth[point[1],point[0]])
                # We will clear this arrays later in the code for the next iteration self.depth_p and self.depth_p_1
            ## ----------------------------------------------------------------------------------------------------------------------------------------   
            ##------------------ Clear the points that are outsiderss in the depth. Depth = 0 mean that we don't have a value in that point. So... We can remove it ------
            mask = np.logical_and(self.depth_p > 0, self.depth_p < 3.50)
            self.depth_p = self.depth_p[mask]
            self.depth_p_1 = self.depth_p_1[mask]
            good_p_1 = good_p_1[mask] #aply mask to take the outsiders
            good_p = good_p[mask] #aply the same mask on the others points, to have the same points         
                #i could clear good_p_1 and good_p but if i just have goodpoints far away(>3500), i would lost track of the points
            
            mask = np.logical_and(self.depth_p_1 > 0, self.depth_p_1 < 3.50)
            self.depth_p_1 = self.depth_p_1[mask]
            self.depth_p = self.depth_p[mask]
            good_p_1 = good_p_1[mask] #aply mask to take the outsiders
            good_p = good_p[mask] #aply the same mask on the others points, to have the same points
            
            #self.p_frame_1 = good_p.reshape(-1,1,2)#update the old points to the new ones
            keypoints = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in good_p] #convert to keypoints   IT is here because all the outsiders are removed at this point               
            keypoints2 = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in good_p_1]
            # ## ----------------------------------------------------------------------------------------------------------------------------------------   
             ##------Calcule of the scale------------------------------------------------
            print(self.depth_p)
            print(self.depth_p_1)
            scale_dif = np.subtract(self.depth_p_1 , self.depth_p)   
            #scale_dif = np.subtract(self.depth_p , self.depth_p_1)            
            #scale_dif = np.subtract(self.old_depth, self.new_depth)       
            #mask = np.logical_and(scale_dif<20, scale_dif>-20)#testing this value
            #scale_dif = scale_dif[mask]
            
            if (np.count_nonzero(scale_dif==0) >= scale_dif.shape[0]*0.6): # if more than 3/4 of the numbers are  0. There is no movement
                scale = 0
            else: #if there is movement, the scale is calculated
                scale = np.mean(scale_dif)     
                
            #scale = np.mean(scale_dif)     #FOr testing
            #scale = 0.1
            print("Scale")            
            print(scale_dif)
            
            depth_aux1=self.new_depth
            depth_aux1 =np.uint8(depth_aux1)
            depth_aux1 = cv2.cvtColor(depth_aux1, cv2.COLOR_GRAY2RGB)
            depth_aux2=self.old_depth
            depth_aux2 =np.uint8(depth_aux2)
            depth_aux2 = cv2.cvtColor(depth_aux2, cv2.COLOR_GRAY2RGB)  
            
            cv2.drawKeypoints(depth_aux1, keypoints,depth_aux1, color=(255,0,0))   
            cv2.drawKeypoints(depth_aux2, keypoints2,depth_aux2, color=(0,255,0))   
            
            self.pub2.publish(self.br.cv2_to_imgmsg(depth_aux1))
            self.pub.publish(self.br.cv2_to_imgmsg(depth_aux2))
            #self.pub2.publish(self.br.cv2_to_imgmsg(self.new_frame))
            #self.pub.publish(self.br.cv2_to_imgmsg(self.old_frame))   
            ## -----------------------------------------------------------------------------------------------
            ##----------------------Clear my variavels-----------------------------
            self.depth_p_1 =  []    #Clear the array to append more points in the next iteration
            self.depth_p =  []          ##Clear the array to append more points in the next iteration
            self.old_frame = self.new_frame# The old frame can now be updated
            self.old_depth = self.new_depth#   The old depth frame can be updated 
            ## -----------------------------------------------------------------------------------------------            
            ##----------------------Calculation of the R and t matrix -----------------------------
            self.p_frame_1 = good_p.reshape(-1,1,2)#update the old points to the new ones      
            
            if self.id < 2:#Inicialize for the first 2 frames
                E, _ = cv2.findEssentialMat(good_p, good_p_1, self.focal_length, self.pp, cv2.  LMEDS, self.prob, 1, None)
                _, self.R, self.t, _ = cv2.recoverPose(E, good_p_1, good_p, self.R, self.t, self.focal_length, self.pp, None)
                self.R = np.eye(3)#clear the rotation matrix 
                self.t  = np.zeros(shape=(3, 1))#translation matrix go to 0 for have a start points     
            else:#
                E, _ = cv2.findEssentialMat(good_p, good_p_1, self.focal_length, self.pp, cv2.LMEDS, self.prob, 1, None)
                _, R, t, _ = cv2.recoverPose(E, good_p_1, good_p, self.R.copy(), self.t.copy(), self.focal_length, self.pp, None)
                
            # if self.id < 2:#Inicialize for the first 2 frames
                # E, _ = cv2.findEssentialMat(good_p, good_p_1, self.focal_length, self.pp, cv2.RANSAC, self.prob, 0.2, None)
                # _, self.R, self.t, _ = cv2.recoverPose(E, good_p_1, good_p, self.R, self.t, self.focal_length, self.pp, None)
                # self.R = np.eye(3)#clear the rotation matrix (bugs,,,,)
                # self.t  = np.zeros(shape=(3, 1))#translation matrix go to 0 for have a start points     
            # else:
                # E, _ = cv2.findEssentialMat(good_p, good_p_1, self.focal_length, self.pp, cv2.RANSAC, self.prob, 0.2, None)
                # _, R, t,_ = cv2.recoverPose(E, good_p_1, good_p, self.R.copy(), self.t.copy(), self.focal_length, self.pp, None)
       
                t = t[::-1,:]   #the translation happens in Z but i want the movemente b in x. SO i shift every position once
                #scale = (scale) * 0.01    #conversion to meters for work with ROS
                print("Scale in Meters:", scale)

                if(abs(scale) > 0.005): #and abs(t[0][0]) > abs(t[1][0]) and abs(t[0][0]) > abs(t[2][0]):
                #if(scale > 0.05) and abs(t[1][0]) > abs(t[0][0]) and abs(t[1][0]) > abs(t[2][0]):                
                #if(scale > 0.05):# and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0]):
                    
                    self.p_frame_1 = good_p.reshape(-1,1,2)#update the old points to the new ones
                    auxt = self.t
                    self.t = self.t + (scale*self.R.dot(abs(t)))                    
                    auxR = self.R   #Auxiliar var to check if the rotation makes sense
                    self.R = (np.linalg.inv(R)).dot(self.R)     ##np.linalg.inv(self.R))     
                    aux_dif = self.R - auxR#Matrix with the diference of the new rotation against the old one
                    if scale > 0 :
                        print("A andar para a frente")
                    else:
                        print("A andar para tras")
                    print(scale)          
                    if (np.any(np.logical_or(aux_dif > 0.35, aux_dif < -0.35))):# if the diference is too big, keep the old value and ignore the new [0.35 rad = 20]
                        self.R = auxR
                        self.t =auxt 
                    
                    print(t)
             
                #self.t = self.t + scale*self.R.dot(t)
                #self.R = R.dot(self.R)                             
            self.id= self.id +1 #may overflow!?
            
             
              ##------------------------COnverstion to quarternion---------------------------------##-----          
            #get the real part of the quaternion first
            r = np.math.sqrt(float(1)+self.R[0,0]+self.R[1,1]+self.R[2,2])*0.5
            i = (self.R[2,1]-self.R[1,2])/(4*r)
            j = (self.R[0,2]-self.R[2,0])/(4*r)
            k = (self.R[1,0]-self.R[0,1])/(4*r)
            ##---------------Send the O dometry information to a node to see it in RVIZ------------#            
            #self.transformT.sendTransform((0,0,0.5),(k,i,j,r),rospy.Time.now(),"me","camera_link")            
            
            #self.transformT.sendTransform((self.t[0],0,0),(k,i,j,r),rospy.Time.now(),"me","camera_link")        
            #self.transformT.sendTransform((0,0,0),(k,i,j,r),rospy.Time.now(),"me","camera_link")             
            #self.transformT.sendTransform(self.t,(k,i,j,r),rospy.Time.now(),"me","camera_link")
           
            #self.transformT.sendTransform((0,0,0),(k,i,j,r),rospy.Time.now(), "camera_link","me")                       
            #self.transformT.sendTransform((self.t[0],0,0),(k,i,j,r),rospy.Time.now(), "camera_link","me")         
            self.transformT.sendTransform(self.t,(k,i,j,r),rospy.Time.now(), "camera_link","me")#this      
               
            #self.transformT.sendTransform(self.t,(k,i,j,r),rospy.Time.now(),"me","camera_link")
            #self.transformT.sendTransform(self.t,(0.0, 0.0, 0.0, 1.0),rospy.Time.now(),"me","camera_link")            
            #self.transformT.sendTransform((0,0,0.5),(k,i,j,r),rospy.Time.now(),"camera_link","me")
            #self.transformT.sendTransform(self.t,(0.0, 0.0, 1.0, 0.0),rospy.Time.now(),"camera_link","me")  

            #self.transformT.sendTransform(self.t,(k,i,j,r),rospy.Time.now(),"me","camera_link")

           ##-----------------------------------------------------------
            
            self.frame_aux= self.last_frame #just to show in RVIZ      
            #self.frame_aux = cv2.cvtColor(self.frame_aux, cv2.COLOR_GRAY2RGB)
            self.last_depth = self.last_depth.astype(float)
            self.depth_aux=self.last_depth
            self.depth_aux =np.uint8(self.depth_aux)
            self.depth_aux = cv2.cvtColor(self.depth_aux, cv2.COLOR_GRAY2RGB)
                               
                   

            #cv2.drawKeypoints(self.frame_aux, keypoints,self.frame_aux, color=(255,0,0))      
            #self.pub.publish(self.br.cv2_to_imgmsg(self.frame_aux))
            #cv2.drawKeypoints(self.depth_aux, keypoints,self.depth_aux, color=(0,0,255))            
         
                    
            self.loop_rate.sleep()  #defines the loop rate!
        rospy.loginfo("Closing Program")

def main():
    #'''Initializes and cleanup ros node'''
    #rospy.init_node('vis_odometry', anonymous=True)
    rospy.init_node('vis_odometry')
    my_node = vis_odometry()
    my_node.start()
  

if __name__ == '__main__':
    
    main()