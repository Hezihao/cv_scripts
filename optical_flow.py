#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

# calculation of euclidian distance
def eucl_dist(m, n, o, p):
	return np.sqrt(m**2 + n**2 + o**2 + p**2)

# define parameters in corner detector
maxCorners = 10
qualityLvl = 0.01
minDist = 10

# predefined parameters for optical flow Lucas Kanade
former_frame = []
lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))		# 10 is the number of reference points

# open object file
file = cv2.VideoCapture('../../Videos/output.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('../../Videos/OF_trace_with_SIFT.avi', fourcc, 40.0, (450,800))

# SIFT feature detector
sift = cv2.xfeatures2d.SIFT_create()

while(file.isOpened()):
	ret, frame = file.read()
	if(ret):
		frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# do feature extraction
		# 1) with Harris corner detector
		points = cv2.goodFeaturesToTrack(frame_g, maxCorners, qualityLvl, minDist)
		# 2) with SIFT feature detector
		key_points, des = sift.detectAndCompute(frame_g, None)
		key_points_for_LK = np.array([])
		for kp in key_points:
			key_points_for_LK = np.append(key_points_for_LK, [kp.pt[0],kp.pt[1]])
		key_points_for_LK = np.float32(key_points_for_LK.reshape(int(len(key_points_for_LK)/2),1,2))
		if(len(former_frame)):
			# calculate Optical Flow with:
			# 1) Harris corners
			# p_op_flow, status, error = cv2.calcOpticalFlowPyrLK(former_frame, frame_g, points, None, **lk_params)
			# 2) SIFT feature
			p_op_flow, status, error = cv2.calcOpticalFlowPyrLK(former_frame, frame_g, key_points_for_LK, None, **lk_params)

			# obviously all the errors are not equal/close, cause there is spatial information missing
			# if we have another channel of 'depth', we can reconstruct the spatial relationship and estimate
			# the 3D motion of camera(if motion_obj known)/object(if motion_obj known)
			# print('Error between frames: \n'+str(error))
			relevant_pts = p_op_flow[status==1]
			#former_relevant_pts = points[status==1]
			former_relevant_pts = key_points_for_LK[status==1]
			it = 0
			for i, (rel_pt, form_rel_pt) in enumerate(zip(relevant_pts, former_relevant_pts)):
				x, y = rel_pt.ravel()
				form_x, form_y = form_rel_pt.ravel()
				# correction of optical flow analysis
				# 1) delete those who didn't change position in image, applied here
				# 2) trace back from real-time frame to the former frame, 
				#    if the point_i(calculated while trace-back)doesn't match former_point_i, discard it. More CP-consuming.
				if eucl_dist(x, y, form_x, form_y)>2:
					relevant_pts[it] = relevant_pts[i]
					former_relevant_pts[it] = former_relevant_pts[i]
					it += 1
			relevant_pts = relevant_pts[:it]
			former_relevant_pts = former_relevant_pts[:it]
		# visualization
		'''
		for point in points:
			cv2.circle(frame, (point[0][0], point[0][1]), 3, (0,0,255), 1)	# Harris features
		'''
		cv2.drawKeypoints(frame,key_points,frame)							# SIFT features
		if(len(former_frame)):												# optical flow trace
			for i, (pt, f_pt) in enumerate(zip(relevant_pts, former_relevant_pts)):
				cv2.line(frame, (pt[0],pt[1]), (f_pt[0],f_pt[1]), (0, 255, 0), 1)
		cv2.imshow('Motion', frame)
		#out.write(frame)
		former_frame = frame_g
	else:
		break
	if(cv2.waitKey(25) &0xFF == ord('q')):
		break
file.release()
#out.release()
cv2.destroyAllWindows()