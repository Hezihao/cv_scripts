#!/usr/bin/env python3

import cv2
import numpy as np

cap = cv2.VideoCapture('../../Videos/IMG_8351.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../../Videos/output.avi', fourcc, 20.0, (225,400))

num_frames = 0
while(cap.isOpened()):
	num_frames += 1
	ret, frame = cap.read()
	if(ret):
		frame_resized = cv2.resize(frame,(225,400))
		cv2.imshow('Video',frame_resized)
		#out.write(cv2.flip(frame_resized, 0))
		out.write(frame_resized)
	else:
		print('THE END')
		break
	if(cv2.waitKey(25) &0xFF == ord('q')):
		break
print('Number of frames is: '+str(num_frames))
cap.release()
out.release()
cv2.destroyAllWindows()
