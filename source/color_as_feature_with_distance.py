import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from math import sqrt 
from sklearn.cluster import KMeans

img = cv.imread('image13.jpg')
img2 = cv.imread('image13.jpg')
#print(img.shape)
#print(type(img))


#for i in img2:
#	for j in i:
	#	print(i.shape)
	#	maxi = np.amax(j)
		#print(maxi)
#		j[0] = (j[0]/(np.amax(j)+1))*255
#		j[1] = (j[1]/(np.amax(j)+1))*255
#		j[2] = (j[2]/(np.amax(j)+1))*255
#	j[1] = (j[1]/(max([i[0],i[1],i[2]))
#	j[2] = (i[2]/(max([i[0],i[1],i[2]))
#		j[0] = (j[0]/(j[0]+j[1]+j[2]))
#		j[1] = (j[1]/(j[0]+j[1]+j[2]))
#		j[2] = (j[2]/(j[0]+j[1]+j[2]))
#print(img2)


size = (img.shape[0],img.shape[1])
norm = np.zeros(size)
xcoord = []
ycoord = []
 
x=-2
for indi,i in enumerate(img2):
	for indj,j in enumerate(i):
		x = int(int(j[0])+int(j[2])-2.5*int(j[1]))	#-2*int(j[1]))
		norm[indi][indj] = max(50,x)	
		#max(50,x)

ret, norm = cv.threshold(norm,80,255,cv.THRESH_BINARY)
			
for indi,i in enumerate(norm):					#this loop checks if there are any pixels which have x more than 50( i is row)
	for indj, j in enumerate(i): 				# j is each pixel in row i
		if j>50:					#j>50				# condition check
			#print(norm[indi][indj],indi,indj)	#print coordinates of all points which has value more than 50
			xcoord.append(indi)
			ycoord.append(indj)

xcoord = np.array(xcoord)
ycoord = np.array(ycoord)
coord = np.column_stack((xcoord,ycoord))
#print(coord.shape)
count = 0
distances = (coord.shape[0],2)
distances = np.zeros(distances)	
for i in range(0,coord.shape[0]-2):							#change back to 98
	count = 0
	for j in range(1,coord.shape[0]-1):						#change back to 99
		dist = sqrt((coord[i][0]-coord[j][0])**2+(coord[i][1]-coord[j][1])**2)
		if dist < 15:								#the area you want to check for adjacent points
			count = count+1
			if count > 15:							#number of points per area
				distances[i] = coord[i]

				break 

kmeans = KMeans(n_clusters=4, random_state=0).fit_predict(distances)
#print(type(kmeans))
#fig = plt.figure()
plt.scatter(distances[:, 1], distances[:, 0], c=kmeans)
plt.imshow(img),plt.show()
#a = fig.add_sub_plot
#plt.figure()
#plt.imshow(norm)
#plt.show()
#print(norm)
#print(img2)
#imgplot = plt.imshow(img)
#cv.imshow("plain",img)
#cv.imshow("normalised",img2)
#color = ('b','g','r')
#for i,col in enumerate(color):
#    histr = cv.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
#plt.imshow(img),plt.show()
cv.waitKey(0)
