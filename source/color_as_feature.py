import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import math
img = cv.imread('face1.jpg')
img2 = cv.imread('face1.jpg')
#print(img.shape)

xcoord = []
ycoord = []
zcoord = []
size = (img.shape[0],img.shape[1])
norm = np.zeros(size)

x=-2								#create an arbitrary integer to store computed value at each pixel
for indi,i in enumerate(img2):					#indi index of i & i iterates through each row of pixels 
	for indj,j in enumerate(i):				#indj index of j & j iterates through each column of row i (so j is a pixel)
		x = int(int(j[0])+int(j[2])-2*int(j[1]))	#-2*int(j[1]))    x = red + blue - 2*green value in pixel 
		norm[indi][indj] = max(50,x)			#max(50,x)	save coordinates of pixels with x more than 50		

for indi,i in enumerate(norm):					#this loop checks if there are any pixels which have x more than 50( i is row)
	for indj, j in enumerate(i): 				# j is each pixel in row i
		if j>50:					#j>50				# condition check
			#print(norm[indi][indj],indi,indj)	#print coordinates of all points which has value more than 50
			xcoord.append(indi)
			ycoord.append(indj)
			zcoord.append(j)
xcoord = np.array(xcoord)
ycoord = np.array(ycoord)
zcoord = np.array(zcoord)
coord = np.column_stack((xcoord,ycoord,zcoord))
#print(norm.shape)
#print(coord.shape)
#print(ycoord.shape)
#rcoord = np.reshape(coord,(coord.shape[1],2))
kmeans = KMeans(n_clusters=4, random_state=0).fit_predict(coord)
#centroid_labels = [centroids[i] for i in kmeans]
#print(centroid_labels)
#print(kmeans.shape)
cluster1 = []
cl1x = []
cl1y = []
cluster2 = []
cl2x = []
cl2y = []
cluster3 = []
cl3x = []
cl3y = []
cluster4 = []
cl4x = []
cl4y = []

#color =[]

for index,i in enumerate(kmeans):
	if i==0:
		cluster1.append(zcoord[index]**2)
		cl1x.append(xcoord[index])
		cl1y.append(ycoord[index])
#		color.append('r')
	if i==1:
		cluster2.append(zcoord[index]**2)
		cl2x.append(xcoord[index])
		cl2y.append(ycoord[index])
#		color.append('b')
	if i==2:
		cluster3.append(zcoord[index]**2)
		cl3x.append(xcoord[index])
		cl3y.append(ycoord[index])
#		color.append('g')
	if i==3:
		cluster4.append(zcoord[index]**2)
		cl4x.append(xcoord[index])
		cl4y.append(ycoord[index])
#		color.append('y')
		
cluster1 = np.array(cluster1)
cl1x = np.array(cl1x)
cl1y = np.array(cl1y)
cluster2 = np.array(cluster2)
cl2x = np.array(cl2x)
cl2y = np.array(cl2y)
cluster3 = np.array(cluster3)
cl3x = np.array(cl3x)
cl3y = np.array(cl3y)
cluster4 = np.array(cluster4)
cl4x = np.array(cl4x)
cl4y = np.array(cl4y)

value1 = np.sum(cluster1)/len(cluster1)
value2 = np.sum(cluster2)/len(cluster2)
value3 = np.sum(cluster3)/len(cluster3)
value4 = np.sum(cluster4)/len(cluster4)

values = [value1,value2,value3,value4] 
lip = values.index(max(values))
print(lip)

norm = np.zeros(size)

if lip == 0:
	implot = plt.imshow(img)
	plt.scatter(cl1y, cl1x, c='r')
if lip == 1:
	implot = plt.imshow(img)
	plt.scatter(cl2y, cl2x, c='r')
if lip == 2:
	implot = plt.imshow(img)
	plt.scatter(cl3y, cl3x, c='r')
if lip == 3:
	implot = plt.imshow(img)
	plt.scatter(cl4y, cl4x, c='r')
	

#print(value1,value2,value3,value4)
#print(zcoord.shape)
#print(type(kmeans))
#print(type(kmeans))
#plt.scatter(coord[:, 0], coord[:, 1], c=kmeans)

#print(rcoord.shape)
#plt.imshow(norm)						#display the coordinates as an image. if pixel < 50 then black is displayed 
plt.show()

cv.waitKey(0)
