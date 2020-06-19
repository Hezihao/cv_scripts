# cv_scripts
Functionalities implemented with OpenCV.

## Lane Detector
### Intro:
The following steps are included in this simple demo:
1) undistort the raw image data from camera
2) transfer into grayscale image, and apply Canny filter on it
3) apply color filter on it(cause the lane line is of yellow) in order to extract line
4) combine both and binarize it
5) get to the bird-eye view image with warping
6) apply sliding window onto the bird-eye view and detect centers of lines in it
7) interpolation inbetween those nodes with 2nd(3rd) order curves
8) map it back to normal view and overlap it with original(undistorted image)

### Examples:
1) Output of detection on warped image:
<br /><p align="center"><img src="https://github.com/Hezihao/cv_scripts/blob/master/IMG/example_lane_detector.png" width="650" height="400"></p>
2) Overlap with original data:
<br /><p align="center">...in process...</p>

## Optical Flow
### Intro:
Following steps are envolved in calculation of optical flow:
1) Feature extraction: Harris corners & SIFT feature are covered by now.
<br />SIFT provides a more rich feature detection and hopefully a better optical flow.
2) Calculation of optical flow with:
      <br />
      <code>points, status, error = cv2.calcOpticalFlowPyrLK(frame_1, frame_2, feature_points, (...next_points...), \**lk_params)</code>
      <br />
      see documentation of function at https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
      this function is based on Lukas-Kanade method, which results as a sparse set(compared to Farneback method), and there are three assumptions:
      <br />
      - constant intensity
      - continuious/small motion
      - spatial continuity
      
3) Visualization of features/traces with:
      <br />
      <code>cv2.circle(Img, (point_x, point_y), radius, (color_b, color_g, color_r), width)</code>
      <br />
      <code>cv2.line(Img, (pa_x, pa_y), (pb_x, pb_y), (color_b, color_g, color_r), width)</code>
### Examples
1) Original data:
<br /><p align="center"><img src="https://github.com/Hezihao/cv_scripts/blob/master/IMG/original_img.png" width="325" height="400"></p>
2) Detected features:
<br />Harris
<br /><p align="center"><img src="https://github.com/Hezihao/cv_scripts/blob/master/IMG/Harris_features.png" width="325" height="400"></p>
<br />SIFT
<br /><p align="center"><img src="https://github.com/Hezihao/cv_scripts/blob/master/IMG/SIFT_features.png" width="325" height="400"></p>
3) Calculated optical flow:
<br />Harris
<br /><p align="center"><img src="https://github.com/Hezihao/cv_scripts/blob/master/IMG/of_frame_with_Harris.png" width="325" height="400"></p>
<br />SIFT
<br /><p align="center"><img src="https://github.com/Hezihao/cv_scripts/blob/master/IMG/of_frame_with_SIFT.png" width="325" height="400"></p>
4) Overall traces
<br />Harris
<br /><p align="center"><img src="https://github.com/Hezihao/cv_scripts/blob/master/IMG/of_trace_with_Harris.png" width="325" height="400"></p>
<br />SIFT
<br /><p align="center"><img src="https://github.com/Hezihao/cv_scripts/blob/master/IMG/of_trace_with_SIFT.png" width="325" height="400"></p>
