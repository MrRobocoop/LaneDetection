import numpy as np
import cv2

def line_fit(binary_warped, window_size, line_margin, vertical_margin):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]*2/4:,:], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)

	margin = window_size

	leftx_base = np.argmax(histogram[10:midpoint]) + margin/2
	rightx_base = np.argmax(histogram[midpoint:-margin/2]) + midpoint
	if rightx_base == midpoint:
		rightx_base = midpoint*2 - margin/2

	# Choose the number of sliding windows
	nwindows = 10
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	lefty_current = binary_warped.shape[0]
	righty_current = binary_warped.shape[0]
	# Set the width of the windows +/- margin

	# Set minimum number of pixels found to recenter window
	minpix = 5
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((abs(nonzeroy-lefty_current)<vertical_margin )&(nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((abs(nonzeroy-righty_current)<vertical_margin )&(nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			lefty_current = (win_y_high + win_y_low)/2
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
			righty_current = (win_y_high + win_y_low)/2

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)
	if rightx_base == midpoint*2 - margin/2:
		right_lane_inds = np.array([])

	if leftx_base == margin / 2:
		left_lane_inds = np.array([])

	leftx = np.array([0])
	lefty = np.array([0])
	rightx = np.array([0])
	righty = np.array([0])
	# Extract left and right line pixel positions
	try:
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
	except:
		pass
	try:
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
	except:
		pass

	left_fit = [];
	right_fit = [];
	print np.size(leftx)
	print np.size(rightx)
	if np.size(leftx)<line_margin:
		pass
	else:

	# Fit a second order polynomial to each
		if np.size(leftx)<line_margin*3:
			try:
				left_fit = np.polyfit(lefty, leftx, 1)
			except:
				pass
		else:
			try:
			#left_fit = np.polyfit(lefty, leftx, 2)
				left_fit = np.polyfit(lefty, leftx, 3)
			except:
				pass
	if np.size(rightx)<line_margin:
		pass
	else:
		if np.size(rightx)<line_margin*3:
			try:
				right_fit = np.polyfit(righty, rightx, 1)
			except:
				pass
		else:
			try:
			#left_fit = np.polyfit(lefty, leftx, 2)
				right_fit = np.polyfit(righty, rightx, 3)
			except:
				pass
	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds
	ret['leftx'] = leftx
	ret['lefty'] = lefty
	ret['rightx'] = rightx
	ret['righty'] = righty
	return ret