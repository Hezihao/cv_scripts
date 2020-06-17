#!/usr/bin/env python3

import cv2
import numpy as np

# work-flow:
'''
1) undistort the raw image data from camera
2) transfer into grayscale image, and apply Canny filter on it
3) apply color filter on it(cause the lane line is of yellow) in order to extract line
4) combine both and binarize it
5) get to the bird-eye view image with warping
6) apply sliding window onto the bird-eye view and detect centers of lines in it
7) interpolation inbetween those nodes with 2nd(3rd) order curves
8) map it back to normal view and overlap it with original(undistorted image)
'''
# setting
CALC_DISTORSION = False
DISP_DISTORSION = False

cal_img_prefix = '../../Images/calibration'
source = '../../Videos/lane.mp4'

# pattern size in every picture
num = [[9,5],[9,6],[9,6],[7,4],[7,5],
	   [9,6],[9,6],[9,6],[9,6],[9,6],
	   [9,6],[9,6],[9,6],[9,6],[9,6],
	   [9,6],[9,6],[9,6],[9,6],[9,6]]

sp_points_list = []
pl_points_list = []

def calibrate_it():
	for i in range(1,21):
		cal_img_name = cal_img_prefix+str(i)+'.jpg'
		cal_img = cv2.imread(cal_img_name)
		cal_img_g = cv2.cvtColor(cal_img, cv2.COLOR_BGR2GRAY)
		cal_img_g = cv2.cvtColor(cal_img_g, cv2.COLOR_GRAY2BGR)
		ret, corners = cv2.findChessboardCorners(cal_img_g, (num[i-1][0], num[i-1][1]))		
		if(ret):
			sp_points = np.zeros((num[i-1][0]*num[i-1][1], 3), np.float32)
			sp_points[:, :2] = np.mgrid[0:num[i-1][0], 0:num[i-1][1]].T.reshape(-1,2)
			sp_points_list.append(sp_points)
			pl_points_list.append(corners)
			#cv2.drawChessboardCorners(cal_img_g, (num[i-1][0], num[i-1][1]), corners, ret)
			#cv2.imshow('Image Calibration'+str(i), cal_img_g)
		else:
			print(str(i-1)+' rounds of calibration finished.')
			return 0, None, None
	ret, mat, dist, rvecs, tvecs = cv2.calibrateCamera(sp_points_list, pl_points_list, (cal_img.shape[0],cal_img.shape[1]), None, None)
	return 1, mat, dist

def my_resize(img):
	return cv2.resize(img, (0,0), None, 0.25, 0.25)

# extract yellow & white area from image
def color_filter(img):
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	min_white = np.array([0, 0, 221])
	max_white = np.array([180, 30, 255])
	min_yellow = np.array([15, 43, 46])
	max_yellow = np.array([48, 255, 255])
	filtered_white = cv2.inRange(img_hsv, min_white, max_white)
	filtered_yellow = cv2.inRange(img_hsv, min_yellow, max_yellow)
	img_filtered = cv2.bitwise_or(filtered_white, filtered_yellow)

	return img_filtered

# extract edges in image
def edge_filter(img):
	img_blured = cv2.GaussianBlur(img, (5,5), 0)
	img_edge = cv2.Canny(img_blured, 80, 100)
	k = np.ones((5,5))
	#img_edge = cv2.dilate(img_edge, k)
	#img_edge = cv2.erode(img_edge, k)
	img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, k)
	return img_edge

if __name__ =='__main__':
	# get distorsion parameters and camera matrix
	if(CALC_DISTORSION):
		ret, mat, dist = calibrate_it()
		print(mat,dist)
		if(ret and DISP_DISTORSION):
			for i in range(1,21):
				cal_img_name = cal_img_prefix+str(i)+'.jpg'
				cal_img = cv2.imread(cal_img_name)
				cal_img_undist = cv2.undistort(cal_img, mat, dist)
				cal_img = my_resize(cal_img)
				cal_img_undist = my_resize(cal_img_undist)
				disp = np.hstack((cal_img, cal_img_undist))
				cv2.imshow('Distorted IMG'+str(i), disp)
	else:
		mat = np.array([[1.16043008e+03, 0.00000000e+00, 6.66452525e+02],
 						[0.00000000e+00, 1.15718357e+03, 3.91458780e+02],
 						[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
		dist = np.array([[-0.27118956, 0.13279919, -0.00090874, 0.00068669, -0.25289427]])
	# fetch video frames and play them
	vid = cv2.VideoCapture(source)
	while(vid.isOpened()):
		ret, frame = vid.read()
		if(ret):
			# transfer into grayscale image
			frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# undistorsion & visualization
			frame_undist = cv2.undistort(frame, mat, dist)
			frame_g_undist = cv2.undistort(frame_g, mat, dist)
			# extract area of relevant colors
			frame_color_filtered = color_filter(frame_undist)
			# extract edges
			frame_edge = edge_filter(frame_g_undist)
			# combination of information from color & edge
			#frame_preprocessed = cv2.bitwise_and(frame_color_filtered, frame_edge)
			frame_preprocessed = cv2.bitwise_or(frame_color_filtered, frame_edge)
			#frame_preprocessed_close = cv2.morphologyEx(frame_preprocessed, cv2.MORPH_CLOSE, np.ones((15,15)))
			
			# prepare data for visualization
			frame_origin = my_resize(frame)
			frame_g = my_resize(cv2.cvtColor(frame_g, cv2.COLOR_GRAY2BGR))
			frame_undist = my_resize(frame_undist)
			frame_color_filtered = my_resize(cv2.cvtColor(frame_color_filtered ,cv2.COLOR_GRAY2BGR))
			frame_edge = my_resize(cv2.cvtColor(frame_edge ,cv2.COLOR_GRAY2BGR))
			frame_preprocessed = my_resize(cv2.cvtColor(frame_preprocessed ,cv2.COLOR_GRAY2BGR))
			#frame_preprocessed_close = my_resize(cv2.cvtColor(frame_preprocessed_close ,cv2.COLOR_GRAY2BGR))
			img_for_disp = np.hstack((frame_origin, frame_undist, frame_color_filtered))
			img_for_disp = np.vstack((img_for_disp, np.hstack((frame_edge, frame_preprocessed, frame_undist))))
			cv2.imshow('Lane Detection', img_for_disp)
		if(cv2.waitKey(25) & 0xFF == ord('q')):
			break
