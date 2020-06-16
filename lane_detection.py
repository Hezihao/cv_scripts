#!/usr/bin/env python3

import cv2
import numpy as np

# setting
DISP_DISTORSION = False

cal_img_prefix = '../../Images/calibration'
source = '../../Videos/lane.mp4'
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

def yellow_filter(img):
	filtered = np.zeros((img.shape[0],img.shape[1],1), np.float32)
	filtered[:,:,0] = (img[:,:,1]+img[:,:,2])/255
	filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
	return filtered

if __name__ =='__main__':
	# get distorsion parameters and camera matrix
	ret, mat, dist = calibrate_it()
	if(ret and DISP_DISTORSION):
		for i in range(1,21):
			cal_img_name = cal_img_prefix+str(i)+'.jpg'
			cal_img = cv2.imread(cal_img_name)
			cal_img_undist = cv2.undistort(cal_img, mat, dist)
			cal_img = my_resize(cal_img)
			cal_img_undist = my_resize(cal_img_undist)
			disp = np.hstack((cal_img, cal_img_undist))
			cv2.imshow('Distorted IMG'+str(i), disp)
		
	cv2.waitKey(0)
	'''
	vid = cv2.VideoCapture(source)
	while(vid.isOpened()):
		ret, frame = vid.read()
		frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_g = cv2.cvtColor(frame_g, cv2.COLOR_GRAY2BGR)
		frame_y = yellow_filter(frame)
		# visualization
		frame = my_resize(frame)
		frame_g = my_resize(frame_g)
		frame_y = my_resize(frame_y)
		img_for_disp = np.hstack((frame, frame_g, frame_y))
		if(ret):
			cv2.imshow('Lane Detection', img_for_disp)
		if(cv2.waitKey(25) & 0xFF == ord('q')):
			break
	'''