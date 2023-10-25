#import the necessary modules
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
    array = cv2.cvtColor(array,cv2.COLOR_GRAY2RGB)
    #array = cv2.cvtColor(array,cv2.COLOR_RGB2HSV)
    return array

#-------this line is here to get the first image for allow first iteration to happen------------------------------------
frame_1 = frame   #frame T-1 equal to frame T, after that we get the Frame T.
depth_1 = depth
    
frame = get_video()    #get a frame from RGB camera
frame = cv2.resize(frame, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)

depth = get_depth()    #get a frame from depth sensor
depth = cv2.resize(depth, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)
#--------------------------------------------------------------------------------------------------------------------------------------  



if __name__ == "__main__":
    
    focal_length = 318.8560
    pp = (607.1928, 185.2157)
    lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    R = np.zeros(shape=(3, 3))
    t = np.zeros(shape=(3, 3))

    #detector = cv2.FastFeatureDetector_create()    # Initiate FAST object with default values
    detector = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True)
    #detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True, type=2 )
    #detector = cv2.FastFeatureDetector_create(threshold=50)
    #detector = cv2.FastFeatureDetector_create(threshold=75, nonmaxSuppression=True)    # Initiate FAST object with default values
    
    #pythhe Frame T.
    depth_1 = depth
    frame_1 = frame   #frame T-1 equal to frame T, after that we get the Frame T.
    
    depth = get_depth()    #get a frame from depth sensor
    depth = cv2.resize(depth, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)
    #depth.flatten();    
    
    frame = get_video()    #get a frame from RGB camera
    frame = cv2.resize(frame, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #converts the image to gray.
    p = detector.detect(gray )           
    p_aux_1=p
    
    p_1 = np.array([x.pt for x in p_aux_1], dtype=np.float32).reshape(-1, 1, 2)#converts the data to match the next funtion data type
     #   Calculate optical flow between frames, st holds status
      #  of points from frame to frame
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame_1, frame, p_1, None,**lk_params)
    
    while 1:
        start = time.time()#start time to calcule FPS
  
        frame_1 = frame   #frame T-1 equal to frame T, after that we get the Frame T.
        depth_1 = depth
        #get a frame from RGB camera
        frame = get_video()
        frame = cv2.resize(frame, (widthF,heightF)) 
         #---------------------------------------------------------
        #get a frame from depth sensor
        depth = get_depth()
        depth = cv2.resize(depth, (widthF,heightF)) #resize give extra frames (need to check the best loss-reward ratio)
        y=22
        x=17
        depth = np.roll(depth, y, axis=0)
        depth = np.roll(depth,x,axis=1)
        depth[:y, :] = 2040  #y axix
        depth[:, :x] = 2040#x axix
         #---------------------------------------------------------
        
        # Disable nonmaxSuppression
        #detector.setBool('nonmaxSuppression',0)       
        p_1 = np.array([x.pt for x in p], dtype=np.float32).reshape(-1, 1, 2)#converts the data to match the next funtion data type
        
        gray=cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)   #converts the image to gray.
        p = detector.detect(gray)       
        #p = detector.detect(frame)#points of the actual frame
        
        # Calculate optical flow between frames, st holds status
        # of points from frame to frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame_1, frame, p_1, None,**lk_params)
        
        good_p_1 = p_1[st == 1]
        good_p = p1[st == 1]
        
        #for x in range(widthF):
            #for y in range(heightF):
                #r,g,b = im.getpixel((x,y))
                #if b < g and b < r or r==g==b:
                    #out.putpixel((x,y), 0)
        
        
        
        E, _ = cv2.findEssentialMat(good_p, good_p_1, focal_length, pp, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(E, good_p_1, good_p, R, t, focal_length, pp, None)

        n_features = good_p.shape[0]
              
        #print(t)
        print(good_p)
        print(n_features)
        print('\nGo!\n')
        
        img2 = frame
        img3 = depth
        cv2.drawKeypoints(frame, p,img2, color=(255,0,0))
        cv2.drawKeypoints(depth, p,img3, color=(255,0,0))
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
