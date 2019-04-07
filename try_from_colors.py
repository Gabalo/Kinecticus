import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('image5.jpg')
img2 = cv.imread('image5.jpg')
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

x=-2
for indi,i in enumerate(img2):
	for indj,j in enumerate(i):
		x = int(int(j[0])+int(j[2])-2*int(j[1]))	#-2*int(j[1]))
		norm[indi][indj] = max(50,x)			#max(50,x)			

#for indi,i in enumerate(norm):
#	for indj, j in enumerate(i):
#		if j>50:					#j>50			
#			print(norm[indi][indj],indi,indj)
			#ccord = np.append(
#print(norm.shape)
#plt.scatter(norm[:,0],norm[:,1])
plt.imshow(norm)
plt.show()
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
