import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

#%matplotlib notebook

frames=0

#open the image
img = cv2.imread('ex.jpg')
img2 = cv2.imread('livros.jpg')

cv2.imshow("RawImage", img)
cv2.imshow("RawImage2", img2)

#grayscale convertion
gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#cv2.imshow("GraysImage", BW_img)
#contours testing


#finding harris corners

#img- Input image, it should be grayscale and float32 type.
#blockSize- It is the size of neighbourhood considered for corner detection
#ksize- Aperture parameter of Sobel derivative used.
#k- Harris detector free parameter in the equation
#dst = cv2.cornerHarris(BW_img,blockSize,Ksize,k)

dst = cv2.cornerHarris(gray_img,2,3,0.08)#funcao de harris para detecao de cantos exemplo
dst = cv2.dilate(dst, None) #apenas para realcar os cantos

dst2 = cv2.cornerHarris(gray_img2,2,3,0.08)#funcao de harris para detecao de cantos exemplo
dst2 = cv2.dilate(dst2, None) #apenas para realcar os cantos

#dst2 = cv2.threshold(dst,0.01*dst.max(),255,0)

img[dst>0.03*dst.max()]=[0,0,255]
img2[dst2>0.03*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
cv2.imshow('dst2',img2)
#-________________________________________________-#

#-________________________________________________-#
#find centroids
#dst2 = np.uint8(dst2)

#ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst2)

# define the criteria to stop and refine the corners

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv2.cornerSubPix(gray_img2,np.float32(centroids),(5,5),(-1,-1),criteria)
#rever esta funcao 
#Desenha os novos cantos

# res = np.hstack((centroids,corners))
# res = np.int0(res)
# img[res[:,1],res[:,0]]=[0,0,255]
# img[res[:,3],res[:,2]] = [0,255,0]

#-________________________________________________-#

#-________GoodFeaturesToTrack_____Testing__________-#

corners = cv2.goodFeaturesToTrack(gray_img,25,0.01,10)
corners = np.int0(corners)
for i in corners:
 x,y = i.ravel()
 cv2.circle(img,(x,y),3,255,-1)
 

corners2 = cv2.goodFeaturesToTrack(gray_img2,25,0.01,10)
corners2 = np.int0(corners2)
for i in corners2:
 x2,y2 = i.ravel()
 cv2.circle(img2,(x2,y2),3,255,-1)
 
plt.imshow(img),plt.show()

plt.imshow(img2),plt.show()





while True: 
	frames = frames + 1 #number of frames

	
	#7. for playing
	key=cv2.waitKey(1)
	if key == ord('q'):
		break
			
print(frames/30) #show the video time

cv2.destroyAllWindows