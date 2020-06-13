#!/usr/bin/env python3

import cv2
import numpy as np

# define parameters in feature extraction
maxCorners = 10
qualityLvl = 0.1
minDist = 10

# open object file
file = cv2.VideoCapture('../../Videos/output.avi')

while(file.isOpened()):
	ret, frame = file.read()
	if(ret):
		frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# do feature extraction
		points = cv2.goodFeaturesToTrack(frame_g, maxCorners, qualityLvl, minDist)
		for point in points:
			cv2.circle(frame, (point[0][0], point[0][1]), 3, (0,0,255), 1)
			cv2.imshow('Motion', frame)
	else:
		break
	if(cv2.waitKey(25) &0xFF == ord('q')):
		break
file.release()
cv2.destroyAllWindows()