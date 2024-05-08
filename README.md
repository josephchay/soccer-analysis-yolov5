# Football Analysis

---

## Table of Contents
1. [Project Description](#project-description)
2. [Detection from Pre-trained YOLOv8x Model](#detection-from-pre-trained-yolov8x-model)
3. [Detection from Custom-trained YOLOv5 Model](#detection-from-custom-trained-yolov5-model)
4. [Detection Tracking](#detection-tracking)
   1. [Analysis of Player Images for Color Clustering](#analysis-of-player-images-for-color-clustering)
   2. [Ball Detection Interpretation between Frames](#ball-detection-interpretation-between-frames)
   3. [Ball Handler Detection](#ball-handler-detection)
   4. [Team Ball Control](#team-ball-control)
   5. [Camera Movement Tracking](#camera-movement-tracking)
   6. [Perspective Transformation & Speed and Distance Estimation](#perspective-transformation--speed-and-distance-estimation)
5. [Limitations](#limitations)
    1. [Consistent Ball Tracking](#consistent-ball-tracking)
    2. [Current Ball Handler](#current-ball-handler)

---

## Project Description
This project explores the analysis of football matches using YOLOv5 and OpenCV.

---

## Detection from Pre-trained YOLOv8x Model
![image](https://github.com/josephchay/football-analysis-yolov8x/assets/136827046/1dc02fef-a1fd-4810-8d1c-c90d1791973b)

The image shows a sample snapshot of detections from a pre-trained YOLOv8x model of a test football match video.

---

## Detection from Custom-trained YOLOv5 Model
![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/b0b8f9b2-c69f-4704-a222-f77d83459ef5)

The image shows a sample snapshot of detections from a custom-trained YOLOv5 model of a test football match video.
The dataset used can be found [here](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1#).

---

## Detection Tracking
Each bounding box is tracked across frames using multiple factors such as object movement, appearance similarity.

---

### Analysis of Player Images for Color Clustering
![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/e4893a19-419a-4651-8bf7-2c6b29439ffb)
The `save_player_imgs` function in `main.py` extracts player images from the video frames and saves them in the `outputs/players/` directory by default.

The `color_assignment.ipynb` notebook analyzes player images to determine the dominant colors of their uniforms
Then color clustering is applied to extract main colors.

1. Displaying Images
The player images are converted from OpenCV's default BGR color format to RGB to ensure colors are represented correctly when displayed with `matplotlib`.

2. Clustering with KMeans
To differentiate the player from the background, the notebook utilizes the KMeans clustering algorithm from scikit-learn. 
The images are reshaped into a two-dimensional array of pixel values and fed into the KMeans algorithm configured to identify two clusters.  

3. Extracting Dominant Colors
After applying KMeans clustering, the resulting clusters are then used to identify and extract the dominant colors. 
It considers the corners of the images, which are most likely to represent the background. 
By determining the most frequent cluster in these corner pixels, the cluster that corresponds to the background is identified, by which the opposite is considered to be the player. 
With the player’s cluster identified, the centroid of this cluster—calculated by KMeans as the mean of the cluster’s color values—represents the dominant color of the player's uniform.

### Ball Detection Interpretation between Frames
![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/8fbbcba5-6664-4a51-8b97-c3784092b80d)

Occasionally, between frames, the ball is not being detected. Interpolation can be done between frames to estimate the ball's position by considering the two closest frames where the ball was detected.
Then the ball's detection can be placed roughly along the line of the two points. 

### Ball Handler Detection
![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/1fce4a93-73fe-42ca-a209-b7e5f75aa3c0)

The ball is determined to which player is the closest by comparing the distances between the ball's center and the bottom corners of each player's bounding box.
All the players are iterated over, comparing the computed distances with the threshold, and keeps track of the minimum distance found.
If a player is within the threshold and closer than any previously checked players, they are assigned as the closest to the ball.

### Team Ball Control
![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/e73abe50-76a9-4ba9-9e0c-ad3da75fcf4a)

The control for both sides of the team is calculated by capturing all the history frames up to the current frame, of which team controlled the ball at each frame.
It calculates the control percentage for each team by dividing the number of frames each team controlled the ball by the total number of frames in which either team had control.

### Camera Movement Tracking
![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/b728caa3-e9c4-44c2-8389-fcac2371cdfb)

The `CameraMovement` class method `get_camera_movement` handles the task of detecting camera movement over a sequence of video frames, starting with an initial frame and continuing through subsequent frames.
An array `camera_movement` is used to store the movement detected in each frame, starting with zero movement assumed for the initial frame. 
It processes the first frame to grayscale and uses `cv2.goodFeaturesToTrack` to detect points of interest based on the provided feature parameters. 
These points, which ideally represent stable features in the environment, are then tracked frame-to-frame using `cv2.calcOpticalFlowPyrLK`, which calculates the optical flow to estimate how each point has moved between the current and subsequent frames. 
For each new frame, it compares the new position of these points against their original positions to determine the camera's movement. 
The greatest distance a point has moved (if it exceeds a minimal distance threshold) is recorded as the camera movement for that frame. This is crucial for applications that require understanding of scene dynamics or camera stability, such as video stabilization and motion analysis.

### Perspective Transformation & Speed and Distance Estimation
![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/f1f814e7-7aa2-4462-8a18-d210bdbcaa94)

The `ViewTransformer` class transforms points from a video frame's perspective to a normalized top-view perspective of a football field, enhancing the analysis of player movements by standardizing the viewpoint. 
Upon initialization, it defines the pixel coordinates (`pixel_vertices`) of the corners of the football court as seen in the video and maps these to the actual dimensions of the court (`target_vertices`) in meters. 
The class uses OpenCV's `cv2.getPerspectiveTransform` to compute a transformation matrix that aligns these two sets of coordinates. 
The `transform_point` method then checks if a given point (representing a player's position) is within the defined court area and, if so, applies the transformation matrix to convert this point to the top-view perspective. 
This transformation is integrated into player tracking data through the `add_transformed_position_to_tracks` method, which iteratively applies the transformation to each tracked player's position across video frames, thereby allowing precise tracking of player movements relative to the real-world dimensions of the football field.

![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/2fccf006-8350-434f-85e0-3abe3903e3b8)

The `SpeedAndDistanceEstimator` class complements the `ViewTransformer` by calculating and annotating the speed and distance traveled by each player in a football video analysis. 
Upon initialization, it sets parameters for frame intervals and the video frame rate to facilitate these calculations. 
Using transformed player positions from the `ViewTransformer`, the class determines the distance between positions at specified frame intervals and computes the speed in m/s and km/h. 
These metrics are then dynamically annotated on the video frames for each player, excluding the ball and referees. 

---

## Limitations
### Consistent Ball Tracking
![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/4ad6b925-eec8-4503-adaa-e62e225aacd1)

Sometimes, it's possible that sometimes the detection position can be slightly off due to the interpolation method used.

![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/c9f87452-e8ab-4f88-b657-9a819c7bab77)

Occasionally, the ball's actual position is not detected.

### Current Ball Handler
![image](https://github.com/josephchay/football-analysis-yolov5/assets/136827046/b1e58523-7a2e-42fd-ae88-5c793c39c3c7)

The current ball handler sometimes is not well detected as it detects by only using the "2D" view of the video. 
The image shows even though the goalkeeper made a shot sending the ball flying pass the currently marked (red) player, 
however because based on the 2D view, the player is still the closest to the ball, the player is still marked as the current ball handler.
