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

# Ros Messages
from sensor_msgs.msg import Image
# We do not use cv_bridge it does not support CompressedImage in python
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt


class vis_odometry():

    def __init__(self):
        self.br = CvBridge() #allow conversion from RosData to OPenCv data
        self.loop_rate = rospy.Rate(15)        # Node cycle rate (in Hz).
        
        #variables to calculate visual odometry
        self.R = np.zeros(shape=(3, 3))#rotation matrix
        self.t  = np.zeros(shape=(3, 3))#translation matrix
        self.focal_length = 318.8560
        self.pp = (607.1928, 185.2157)
        self.lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.id = 0
        self.n_features = 0
        
        self.detector = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=False)
        
        self.frame =    None
        self.frame_1=   np.float32()
        self.frame_aux = None
        
        self.depth_aux = None
        self.depth =    None
        # Subscribers
        #self.subscriber = rospy.Subscriber("/camera/rgb/image_color", Image, self.rgb_callback,  queue_size = 1)
        self.subscriber = rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.rgb_callback,  queue_size = 1)
        
        #self.subscriber2 = rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect", Image, self.depth_callback,  queue_size = 1)
        self.subscriber2 = rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect_raw", Image, self.depth_callback,  queue_size = 1)
        
        # Publishers
        self.pub = rospy.Publisher('FeaturesRGB', Image,queue_size=5)
        self.pub2 = rospy.Publisher('FeaturesDepth', Image,queue_size=5)

    def rgb_callback(self, data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        #rospy.loginfo('Image received...rgb')
        try:
            self.frame = self.br.imgmsg_to_cv2(data) #Converts from RosMessage to CV2
        except CvBridgeError as e:
            print(e)       
    
    def depth_callback(self, data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        #rospy.loginfo('Image received...depth')
        try:
            self.depth = self.br.imgmsg_to_cv2(data) #Converts from RosMessage to CV2
        except CvBridgeError as e:
            print(e)      
        
        
    def start(self):
        rospy.loginfo("Starting Program")
       
        while not rospy.is_shutdown():

            if self.frame is not None:
                if self.depth is not None:
                   
                    #gray=cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    #p = self.detector.detect(gray)   
                    p = self.detector.detect(self.frame)   
                    self.frame_aux=self.frame 

                    self.depth = self.depth.astype(float)
                    self.depth_aux=self.depth
                    self.depth_aux =np.uint8(self.depth_aux)
                    self.depth_aux = cv2.cvtColor(self.depth_aux, cv2.COLOR_GRAY2RGB)
                    
                    cv2.drawKeypoints(self.frame_aux, p,self.frame_aux, color=(255,0,0))
                    self.pub.publish(self.br.cv2_to_imgmsg(self.frame_aux))
                    cv2.drawKeypoints(self.depth_aux, p,self.depth_aux, color=(255,0,0)) 
                    self.pub2.publish(self.br.cv2_to_imgmsg(self.depth_aux))

            self.loop_rate.sleep()
        rospy.loginfo("Closing Program")


def main():
    #'''Initializes and cleanup ros node'''
    rospy.init_node('vis_odometry', anonymous=True)
    my_node = vis_odometry()
    my_node.start()
  

if __name__ == '__main__':
    main()

