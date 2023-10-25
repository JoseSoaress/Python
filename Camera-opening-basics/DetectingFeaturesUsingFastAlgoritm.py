import cv2, time
import matplotlib.pyplot as plt
import numpy as np

camera_port = 0
#create an object. zero for external camera
video=cv2.VideoCapture(camera_port)

detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)    # Initiate FAST object with default values
#detector = cv2.FastFeatureDetector()    # Initiate FAST object with default values


#for writhe in the image
fps=1
font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText    = (10,20)
fontScale              = 0.7
fontColor              = (0,230,0)
lineType               = 2

def detect(img):

    p = fast.detect(img,None)
    return np.array([x.pt for x in p], dtype=np.float32).reshape(-1, 1, 2)        
   # return p


while True: 
        
    #start  fps coutner
    start = time.time()
	#create a frame object
    check, frame = video.read()
    #frame = cv2.resize(frame, (650, 340))
    
    #write the number of frames in the image
    #cv2.putText(frame,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType)
    
    # print "Threshold: ", fast.getInt('threshold')
    # print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
    # print "neighborhood: ", fast.getInt('type')
    # print "Total Keypoints with nonmaxSuppression: ", len(kp)

	#converting to grayscale
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  	#show the frame!  
    cv2.imshow("Capturing", frame)

	#cv2.imshow("Capturing_Gray", gray) 
    #kp = detect(gray) 
    kp = detector.detect(gray,None)

    working_img = frame
    #img2 = cv2.drawKeypoints(frame, kp, color=(255,0,0))
    
    cv2.drawKeypoints(frame, kp,working_img, color=(255,0,0))
    cv2.putText(working_img,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType)
    
    cv2.imshow("FAST", working_img)

    #fps counter
    time_taken = time.time() - start
    fps = 1./time_taken
    #print(fps)

	#7. for playing
    key=cv2.waitKey(1) 
    if key == ord('x'):
		break
	

#2. Shutdown the camera
video.release()

cv2.destroyAllWindows