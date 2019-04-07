#import the necessary modules
import freenect
import cv2
import numpy as np
 
#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
 
#function to get depth image from kinect
def get_depth():
 	array,_ = freenect.sync_get_depth()
	array = array.astype(np.uint8)
	array[array >= 100] = 255
	array[array < 100] = 0
	return array

def show_depth():
    	global threshold
    	global current_depth

    	depth, timestamp = freenect.sync_get_depth()
    	depth = 255 * np.logical_and(depth >= current_depth - threshold,
                                 depth <= current_depth + threshold)
    	depth = depth.astype(np.uint8)
    	return depth


	
if __name__ == "__main__":
	
	threshold = 200
	current_depth = 900

	depth = show_depth()
	print depth.shape
	frame = get_video()
	print frame.shape

	while 1:
		#get a frame from RGB camera
		original = get_video()
		#get a frame from depth sensor
		depth = show_depth()
		depth_inv = cv2.bitwise_not(depth)						
		#display RGB image
		frame = cv2.bitwise_or(original,original,mask=depth_inv)
		depth = cv2.merge((depth,depth,depth))				
		frame = cv2.add(frame,depth)

		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		
		ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)

		corners = cv2.goodFeaturesToTrack(thresh,100,0.01,10)
		corners = np.int0(corners)

		for i in corners:
    			x,y = i.ravel()
    			cv2.circle(thresh,(x,y),3,255,-1)	

		cv2.imshow('RGB image',thresh)
		#display depth image
		cv2.imshow('Depth image',depth)
		cv2.imshow('Normal',original)
	 
		# quit program when 'esc' key is pressed
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
		    break
	cv2.destroyAllWindows()
