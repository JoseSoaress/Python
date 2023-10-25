#!/usr/bin/env python
import rospy

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from cv_bridge import CvBridge, CvBridgeError


#open the image
img = cv2.imread('ex.jpg')
img2 = cv2.imread('livros.jpg')

bridge = CvBridge()
cv_image = bridge.imgmsg_to_cv2(img, "bgr8")

#cv2.imshow("RawImage", img)
#cv2.imshow("RawImage2", img2)

#gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray_img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

rospy.init_node("HelloWorld")
rate = rospy.Rate(2)
while not rospy.is_shutdown():
    print 'Hello World'   
    rate.sleep()