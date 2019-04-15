import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from math import sqrt 
from sklearn.cluster import KMeans
import freenect

#function to get RGB image from kinect

images = 10
if __name__ == "__main__":
	
	for(i=0,i<images,i++):
		name = 'images/image'+str(i)
		save_name = 'results/image'+str(i)
		img = cv.imread(name)
		#img = get_video()
		img2=img

		size = (img.shape[0],img.shape[1])
		norm = np.zeros(size)
		xcoord = []
		ycoord = []
		 
		x=-2
		for indi,i in enumerate(img2):
			for indj,j in enumerate(i):
				x = int(int(j[0])+int(j[2])-2.5*int(j[1]))	#-2.5*int(j[1]))
				norm[indi][indj] = max(50,x)	
				#max(50,x)

		ret, norm = cv.threshold(norm,100,255,cv.THRESH_BINARY)
					
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
		limit = min(100,coord.shape[0])	
		for i in range(0,limit-2):			#change back to 98
			count = 0
			for j in range(1,limit-1):		#change back to 99
				dist = sqrt((coord[i][0]-coord[j][0])**2+(coord[i][1]-coord[j][1])**2)
				if dist < 15:				#the area you want to check for adjacent points
					count = count+1
					if count > 15:			#number of points per area
						distances[i] = coord[i]

						break 

#		kmeans = KMeans(n_clusters=4, random_state=0).fit_predict(distances)
		kmeans = KMeans(n_clusters=2, random_state=0)
		if distances.any:		
			kmeans.fit(distances)
		
		centers = kmeans.cluster_centers_
		centers = centers.astype(int)
		if centers.any:		
			print(centers)
			centro = centers[np.nonzero(centers)]		
			
			print centro
	
		#avgy = int(np.sum(np.nonzero(centers[0,:]))/np.size(np.nonzero(centers[0,:])))
		#avgx = int(np.sum(np.nonzero(centers[1,:]))/np.size(np.nonzero(centers[1,:]))) 
			
		#print(avgx)
		#print(avgy)

		#plt.scatter(distances[:, 1], distances[:, 0])
		#plt.imshow(img),plt.show()
		
			centro=np.ravel(centro)
			centro = tuple(centro)
			centro = centro[::-1]		
			if centro:
				cv.circle(img,centro,40,(0,0,255),2,8)
		cv2.imwrite(save_name,img)
		
		k = cv.waitKey(5) & 0xFF
        	if k == 27:
            		break
cv2.destroyAllWindows()
