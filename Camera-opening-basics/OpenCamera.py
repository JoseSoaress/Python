import cv2, time


camera_port = 0
#creat an object. zero for external camera
video=cv2.VideoCapture(camera_port)

#for writh in the image
fps=1
font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText    = (10,20)
fontScale              = 0.7
fontColor              = (0,230,0)
lineType               = 2

while True: 
	
    #start  fps coutner
    start = time.time()
	#creat a frame object
    check, frame = video.read()
    frame = cv2.resize(frame, (650, 340) )
    #write the number of frames in the image
    cv2.putText(frame,"FPS = {:.2f}".format(fps),topLeftCornerOfText, font, fontScale, fontColor, lineType)


  	#show the frame
    cv2.imshow("Capturing", frame)

    #converting to grayscale
	#gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#cv2.imshow("Capturing_Gray", gray)  

	#for press any key to out (miliseconds)
	#cv2.waitKey(0)
    
    #fps counter
    time_taken = time.time() -start
    fps = 1./time_taken
    #print(fps)

	#for playing
    key=cv2.waitKey(1) 
    if key == ord('x'):
		break
	

#Shutdown the camera
video.release()

cv2.destroyAllWindows