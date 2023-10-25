#import the necessary modules
#PYTHONPATH=/usr/local/lib/python2.7/dist-packages
#sys.path.append "/usr/local/lib/python2.7/dist-packages"
import cv2, time
import numpy as np
import freenect

#for writhe the FPS in the image
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
    #array = cv2.cvtColor(array,cv2.COLOR_RGB2HSV)
    #array = array[:,:,::-1]
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(dtype=np.uint8)
    #array = cv2.cvtColor(array,cv2.COLOR_GRAY2RGB)
    #array = cv2.cvtColor(array,cv2.COLOR_RGB2HSV)
    return array
    
if __name__ == "__main__":
  
    while 1:
        start = time.time()#start time to calcule FPS
        
        #get a frame from RGB camera
        frame = get_video()
         #---------------------------------------------------------
        #get a frame from depth sensor
        depth = get_depth()
         #---------------------------------------------------------
        
        #display RGB image
        cv2.putText(frame,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType) 
        cv2.imshow('RGB image',frame)
         #---------------------------------------------------------
        #display depth image
        cv2.putText(depth,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.imshow('Depth image',depth)
        #---------------------------------------------------------    
     
        #Compute FPS
        time_taken = time.time() -start
        fps = 1./time_taken
         #---------------------------------------------------------
         
        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
