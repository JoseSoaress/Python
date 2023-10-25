import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

frames=0

#open the image
room = cv2.imread('room.jpg')
books = cv2.imread('books.jpg')

cv2.imshow("RawImageRoom", room) #show raw image -room
cv2.imshow("RawImageBooks", books) #show raw image - books

#grayscale convertion
gray_img_room=cv2.cvtColor(room, cv2.COLOR_BGR2GRAY)
gray_img_books=cv2.cvtColor(books, cv2.COLOR_BGR2GRAY)

#Finding corners

#finding harris corners - Method 1
#dst = cv2.cornerHarris(BW_img,blockSize,Ksize,k)
#img- Input image, it should be grayscale and float32 type.
#blockSize- It is the size of neighbourhood considered for corner detection
#ksize- Aperture parameter of Sobel derivative used.
#k- Harris detector free parameter in the equation

dst_room = cv2.cornerHarris(gray_img_room,2,3,0.08) #Funtion to detect
dst_room = cv2.dilate(dst, None) #Highlight corners

dst_books = cv2.cornerHarris(gray_img_books,2,3,0.08) #Funtion to detect
dst_books = cv2.dilate(dst2, None) #Highlight corners

room[dst_room>0.03*dst.max()]=[0,0,255]
books[dst_books>0.03*dst.max()]=[0,0,255]

cv2.imshow('dst_room',room)
cv2.imshow('dst2_books',books)
#-_____Method 1 end_____________-#


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

corners = cv2.goodFeaturesToTrack(gray_img_room,25,0.01,10)
corners = np.int0(corners)
for i in corners:
 x,y = i.ravel()
 cv2.circle(room,(x,y),3,255,-1)
 

corners2 = cv2.goodFeaturesToTrack(gray_img_books,25,0.01,10)
corners2 = np.int0(corners2)
for i in corners2:
 x2,y2 = i.ravel()
 cv2.circle(books,(x2,y2),3,255,-1)
 
plt.imshow(room),plt.show()

plt.imshow(books),plt.show()

while True: 
	frames = frames + 1 #number of frames

	key=cv2.waitKey(1)
	if key == ord('q'):
		break
			
print(frames/30) #show the video time

cv2.destroyAllWindows