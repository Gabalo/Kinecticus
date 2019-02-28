import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('face1.jpg')
img2 = cv.imread('face1.jpg')
print(img.shape)


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
			print(norm[indi][indj],indi,indj)	#print coordinates of all points which has value more than 50

print(norm.shape)
plt.imshow(norm)						#display the coordinates as an image. if pixel < 50 then black is displayed 
plt.show()

cv.waitKey(0)
