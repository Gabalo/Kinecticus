import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('images/keanu.png')
edges = cv.Canny(img,100,200)
canny = cv.imwrite('canny.png',edges)
canny = cv.imread('canny.png')

gray = cv.cvtColor(canny,cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray,100,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv.circle(canny,(x,y),3,255,-1)

cv.imshow("Hola",canny)
cv.waitKey(0)

#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges)
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#plt.show()


#plt.imshow(img),plt.show()