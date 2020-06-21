#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# configuration of functionalities
CALC_DISTORSION = False
DISP_DISTORSION = False
cal_img_prefix = '../../Images/calibration'
source = '../../Videos/lane.mp4'

# calibration
sp_points_list = []
pl_points_list = []

# pattern size in every picture
num = [[9,5],[9,6],[9,6],[7,4],[7,5],
	   [9,6],[9,6],[9,6],[9,6],[9,6],
	   [9,6],[9,6],[9,6],[9,6],[9,6],
	   [9,6],[9,6],[9,6],[9,6],[9,6]]

# corresponding points for doing perspective transform
# this trapezoid should follow the shape of lane: trapezoid --(warp)--> rectangle
# instead of wrapping a lane in it
points = np.float32([[580, 460], [700, 460], [200, 720], [1096, 720]])
points_warped = np.float32([[300, 0], [950, 0], [300, 720], [950, 720]])
# test of camera type b
#points = np.float32([[362, 287], [437, 287], [125, 450], [685, 450]])
#points_warped = np.float32([[93, 0], [718, 0], [116, 450], [593, 450]])
warp_Mat = cv2.getPerspectiveTransform(points, points_warped)
warp_back_Mat = cv2.getPerspectiveTransform(points_warped, points)

# sliding window
# img_size = []
neighborhood_radius = 120//2
window_size = [160, 90] 		# width, height
#lane_color = (78, 152, 235) 	# orange
lane_color = (113, 204, 46) 	# green
trust = [[],[]]
nodes_reliable = [[],[]]

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

# apply sliding window to image
def sliding_window(img):
	global trust

	img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	hist_init = np.sum(img[int(-window_size[1])*4:, :], axis=0)
	# get index of starting points of both lines
	left_box = []
	right_box = []
	trust_buff = [[],[]]
	half_width = hist_init.shape[0]//2
	half_window_width = window_size[0]//2
	if(len(nodes_reliable[0])):
		nodes_reliable[0][0] = np.argmax(hist_init[:half_width])
		ll_bound = nodes_reliable[0][0]-neighborhood_radius
		lr_bound = nodes_reliable[0][0]+neighborhood_radius
		first_window = img[-window_size[1]:, ll_bound:lr_bound]
		if(np.sum(first_window)>50000):	nodes_reliable[0][0] = int(np.mean(np.where(first_window>0)[1]))+ll_bound
	else:
		nodes_reliable[0].append(np.argmax(hist_init[:half_width]))
	if(len(nodes_reliable[1])):
		rl_bound = nodes_reliable[1][0]-neighborhood_radius
		rr_bound = nodes_reliable[1][0]+neighborhood_radius
		first_window = img[-window_size[1]:, rl_bound:rr_bound]
		if(np.sum(first_window)>50000):	nodes_reliable[1][0] = int(np.mean(np.where(first_window>0)[1]))+rl_bound
	else:
		nodes_reliable[1].append(np.argmax(hist_init[-half_width:])+half_width)
	# initializing position of windows
	left_box.append(nodes_reliable[0][0])
	right_box.append(nodes_reliable[1][0])
	for window_t in range(img.shape[0]//window_size[1]):		# x//y can help us getting integer out of the devision
		upper_boundary = img.shape[0]-(window_t+1)*window_size[1]
		lower_boundary = img.shape[0]-window_t*window_size[1]
		
		left_ltop = (left_box[window_t]-half_window_width, upper_boundary)
		left_rbot = (left_box[window_t]+half_window_width, lower_boundary)
		right_ltop = (right_box[window_t]-half_window_width, upper_boundary)
		right_rbot = (right_box[window_t]+half_window_width, lower_boundary)
		cv2.rectangle(img_bgr, left_ltop, left_rbot, (0,255,0), 5)
		cv2.rectangle(img_bgr, right_ltop, right_rbot, (0,255,255), 5)
		# calculate mid point of next box
		left_window = img[upper_boundary:lower_boundary, left_ltop[0]:left_rbot[0]]
		right_window = img[upper_boundary:lower_boundary, right_ltop[0]:right_rbot[0]]
		right_window = cv2.GaussianBlur(right_window, (15,15), 0)
		# get mean value of horizontal coordinates of all the non-zero elements
		# np.where() is returning 2 lists, 1st one is indicating the 1st dimension of input list
		# here we need the column indices, which is the 2nd dimension of a piece of img element
		if(np.sum(left_window)>50000): 
			left_box.append(int(np.mean(np.where(left_window>0)[1]))+left_ltop[0])
			trust_buff[0].append(1)
		else: 
			left_box.append(left_box[-1])
			trust_buff[0].append(0)
		if(np.sum(right_window)>50000):
			right_box.append(int(np.mean(np.where(right_window>0)[1]))+right_ltop[0])
			trust_buff[1].append(1)
		else:
			right_box.append(right_box[-1])
			trust_buff[1].append(0)
	trust = trust_buff
	return [left_box[:-1], right_box[:-1]], img_bgr

# 2nd order curve
def f_2nd(x,a,b,c):
	return a*x**2 + b*x + c

# 3rd order curve
def f_3rd(x,a,b,c,d):
	return a*x**3 + b*x**2 + c*x + d

# curve-fitting
def fit_lane(nodes):
	nodes_v = []
	nodes_u = []
	for node, i in zip(nodes,range(len(nodes))):
		if(node):
			nodes_u.append(node)
			nodes_v.append(img_size[0]-i*window_size[1]-window_size[1]//2)
	#params, cov = curve_fit(f_3rd, nodes_v, nodes_u)
	params, cov = curve_fit(f_2nd, nodes_v, nodes_u)
	return params


if __name__ =='__main__':
	global img_size

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
			# get image size
			img_size = frame.shape
			# transfer into grayscale image
			frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# undistorsion & visualization
			frame_undist = cv2.undistort(frame, mat, dist)
			for point in points:
				cv2.circle(frame_undist, (point[0],point[1]), 10, (0,0,255), -1)
			#cv2.imshow('Undistorted Image', frame_undist)
			frame_g_undist = cv2.undistort(frame_g, mat, dist)
			# extract area of relevant colors
			frame_color_filtered = color_filter(frame_undist)
			# extract edges
			frame_edge = edge_filter(frame_g_undist)
			# combination of information from color & edge
			# frame_preprocessed = cv2.bitwise_and(frame_color_filtered, frame_edge)
			frame_preprocessed = cv2.bitwise_or(frame_color_filtered, frame_edge)
			#frame_preprocessed_close = cv2.morphologyEx(frame_preprocessed, cv2.MORPH_CLOSE, np.ones((15,15)))
			# warp(perspective transform) to bird-eye view image
			frame_warped = cv2.warpPerspective(frame_preprocessed, warp_Mat, (1280, 720))
			# get position of lines
			nodes, frame_boxed = sliding_window(frame_warped)
			for i in range(2):
				nodes_lis = nodes[i]
				trust_lis = trust[i]
				for num, (node, tr) in enumerate(zip(nodes_lis, trust_lis)):
					if(not len(nodes_reliable[i])==img_size[0]//window_size[1]):
						if(tr):	nodes_reliable[i].append(node)
						else: nodes_reliable[i].append(0)
					else:
						if(tr): 
							nodes_reliable[i][num] = node
			# curve-fitting with detected nodes
			frame_lane = frame_warped.copy()
			frame_lane[:,:] = 0
			frame_lane = cv2.cvtColor(frame_lane, cv2.COLOR_GRAY2BGR)
			#[params_left, params_right] = [fit_lane(nodes[0]), fit_lane(nodes[1])]
			[params_left, params_right] = [fit_lane(nodes_reliable[0]), fit_lane(nodes_reliable[1])]
			for i in range(img_size[0]):
				#cv2.line(frame_lane, (int(f_3rd(i, *params_left)), i), (int(f_3rd(i+1, *params_left)), i+1), lane_color, 50)
				#cv2.line(frame_lane, (int(f_3rd(i, *params_right)), i), (int(f_3rd(i+1, *params_right)), i+1), lane_color, 50)
				cv2.line(frame_lane, (int(f_2nd(i, *params_left)), i), (int(f_2nd(i+1, *params_left)), i+1), lane_color, 50)
				cv2.line(frame_lane, (int(f_2nd(i, *params_right)), i), (int(f_2nd(i+1, *params_right)), i+1), lane_color, 50)			
			# warp image back to FPV
			frame_lane_back = cv2.warpPerspective(frame_lane, warp_back_Mat, (1280,720))
			frame_lane_back = cv2.bitwise_or(frame_lane_back, frame_undist)
			for i, pt in enumerate(nodes_reliable[1]):
				cv2.circle(frame_lane, (pt, 720-i*90-45), 5, (0,0,255), -1)
			cv2.imshow('Overlapping Result', frame_lane_back)
			# prepare data for visualization
			frame_origin = my_resize(frame)
			frame_g = my_resize(cv2.cvtColor(frame_g, cv2.COLOR_GRAY2BGR))
			frame_undist = my_resize(frame_undist)
			frame_color_filtered = my_resize(cv2.cvtColor(frame_color_filtered ,cv2.COLOR_GRAY2BGR))
			frame_edge = my_resize(cv2.cvtColor(frame_edge ,cv2.COLOR_GRAY2BGR))
			frame_preprocessed = my_resize(cv2.cvtColor(frame_preprocessed ,cv2.COLOR_GRAY2BGR))
			frame_warped = my_resize(cv2.cvtColor(frame_warped, cv2.COLOR_GRAY2BGR))
			frame_boxed = my_resize(frame_boxed)
			frame_lane = my_resize(frame_lane)
			frame_lane_back = my_resize(frame_lane_back)

			#frame_preprocessed_close = my_resize(cv2.cvtColor(frame_preprocessed_close ,cv2.COLOR_GRAY2BGR))
			first_row = np.hstack((frame_origin, frame_undist, frame_color_filtered))
			second_row = np.hstack((frame_edge, frame_preprocessed, frame_warped))
			third_row = np.hstack((frame_boxed, frame_lane, frame_lane_back))
			img_for_disp = np.vstack((first_row, second_row, third_row))
			cv2.imshow('Lane Detection', img_for_disp)
		if(cv2.waitKey(25)& 0xFF == ord('q')):
			break
