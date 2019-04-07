import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from math import sqrt 

img = cv.imread('image5.jpg')
edges = cv.Canny(img,100,200)
#print(edges.shape)
canny = cv.imwrite('canny.png',edges)
canny = cv.imread('canny.png')

gray = cv.cvtColor(canny,cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray,150,0.01,10)
corners = np.int0(corners)


print(corners.shape)
rcorners = np.reshape(corners,(corners.shape[0],2))
#print(rcorners)

count = 0
distances = (corners.shape[0],2)
distances = np.zeros(distances)	
for i in range(0,corners.shape[0]-2):							#change back to 98
	count = 0
	for j in range(1,corners.shape[0]-1):						#change back to 99
		dist = sqrt((rcorners[i][0]-rcorners[j][0])**2+(rcorners[i][1]-rcorners[j][1])**2)
		if dist < 20:								#the area you want to check for adjacent points
			count = count+1
			if count > 2:							#number of points per area
				distances[i] = rcorners[i]
				break 
#print(distances)	
#rx = rcorners[:,0]
#ry = rcorners[:,1]
#print(rx.shape)



#rsx = np.sort(rx)
#print(rsx)
#rsy = np.sort(ry)
#print(rsy)

#for i in range(0,98):
#	j = i+1 
#	k = i+2
#	diff = rsx[i]
#	if 








#X = [1,2,3,4,5]
kmeans = KMeans(n_clusters=4, random_state=0).fit_predict(distances)
#print(type(kmeans))
plt.scatter(distances[:, 0], distances[:, 1], c=kmeans)

#for i in corners:
#    x,y = i.ravel()
#    coord = coord.append((x,y))
 #   cv.circle(canny,(x,y),3,255,-1)

#cv.imshow("Hola",canny)
#plt.show()
#cv.waitKey(0)

#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges)
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])



plt.imshow(canny),plt.show()
