# cv_scripts
Functionalities implemented with OpenCV.

## Optical Flow
Following steps are envolved in calculation of optical flow:
1) Feature extraction: Harris corners & SIFT feature are covered by now.
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
![image](https://github.com/Hezihao/cv_scripts/blob/master/IMG/original_img.png)
2) Detected features:
![image](https://github.com/Hezihao/cv_scripts/blob/master/IMG/Harris_features.png)
3) Calculated optical flow:
![image](https://github.com/Hezihao/cv_scripts/blob/master/IMG/of_trace_with_Harris.png)
