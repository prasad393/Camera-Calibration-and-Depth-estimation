#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import cv2
import os
import math

def calculate_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point2 - point1)
    return distance

def calculate_midpoint(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    midpoint = (point1 + point2) / 2
    return midpoint

def image_shower(image):
    cv2.namedWindow('window',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('window',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def Convert_to_3D(pixel_coordinate,K, scalar):
    K_inverse = np.linalg.inv(K)
    pixel_homogeneous = np.append(pixel_coordinate, 1).reshape(3, 1)
    ray_direction = np.dot(K_inverse, pixel_homogeneous)
    Point__3D = scalar*ray_direction.flatten()
    return Point__3D



# In[80]:


# FOR Camera calibration 

checkerboard_size = (7, 7)

# Define the object points of the checkerboard
objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

objpoints = []  
imgpoints = [] 

for i in range(12):
    img = cv2.imread(f'C:/Users/VISHAL.000/Desktop/Computer vision/Task2/Camera Calibration and Depth Estimation-20230506/Goal_1_Dataset/frame_{i}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    
    if ret == True: 
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
    else :
        print("frame",i)
        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(f'Distortion coefficients: {dist}')


# In[8]:


#For undistrotion
image=cv2.imread(f'C:/Users/VISHAL.000/Desktop/Computer vision/Task2/Camera Calibration and Depth Estimation-20230506/Goal_1_Dataset/frame_500.jpg')
image_shape = image.shape[1::-1]
newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist,image_shape, 1)

# Undistort image
undistorted_img = cv2.undistort(image, mtx, dist, None, newcameramtx)

x1 = 100  
y1 = 40  
x2 = 1150  
y2 = 900 

# Crop the image
cropped_image = undistorted_img[y1:y2,x1:x2]


image_shower(image)
image_shower(undistorted_img)
image_shower(cropped_image)

cv2.imwrite("C:/Users7VISHAL.000/Desktop/Computer vision/Task2/Camera Calibration and Depth Estimation-20230506/undistroid_images/4/4.1.jpg",image)
cv2.imwrite("C:/Users/VISHAL.000/Desktop/Computer vision/Task2/Camera Calibration and Depth Estimation-20230506/undistroid_images/4/4.2.jpg",undistorted_img)
cv2.imwrite("C:/Users/VISHAL.000/Desktop/Computer vision/Task2/Camera Calibration and Depth Estimation-20230506/undistroid_images/4/4.3.jpg",cropped_image)


# In[19]:


# FOR Depth estimation

pattern_size = (7, 7)  
square_size = 1

object_points = []  
image_points = []  

object_points.clear()
image_points.clear()
for j in range(pattern_size[1]):
    for i in range(pattern_size[0]):
        object_points.append((i * square_size, j * square_size, 0))
        
# Choose image        
i=1897
image = cv2.imread(f'C:/Users/VISHAL.000/Desktop/Computer vision\Task2/Camera Calibration and Depth Estimation-20230506/Goal_2_dataset/frame_{i}.jpg', cv2.IMREAD_GRAYSCALE)
found, corners = cv2.findChessboardCorners(image, pattern_size, None)
if found:
        cv2.cornerSubPix(image, corners, (11, 11), (-1, -1),criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        image_points.append(corners)

image_shower(image)


# In[3]:


# Perform camera calibration
object_points_per_image = [np.array(object_points, dtype=np.float32)] * len(image_points)
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points_per_image, image_points,
                                                                    image.shape[::-1], None, None)

print("Camera matrix:")
print(camera_matrix)



# In[20]:


C=np.array([0,0,0])
corner_points = np.array([corners[0],      # Corners of outer Edge from 7x7 chekkerboard.
                          corners[6],
                          corners[42],
                          corners[48]])

b = np.mean(corner_points, axis=0)         # Finding the center(b) of checkerbord in image plan. 
Q = Convert_to_3D(b,new_camera_matrix,1)

a = calculate_midpoint(corners[0],corners[6])
P = Convert_to_3D(a,new_camera_matrix,1)

PQ=calculate_distance(Q,P)
QC=calculate_distance(Q,C)
AB= 0.18975                      # The value is in meter.
BC = QC*AB/PQ
print("Distance:",BC)


# In[73]:


Corner_0=(Convert_to_3D(corners[0],camera_matrix,Distance))*100
Corner_6=(Convert_to_3D(corners[6],camera_matrix,Distance))*100
Real_width =37.95
calculated_width=calculate_distance(Corner_0,Corner_6)
Error=calculated_width-Real_width
print("Real Width of 7X7 checkerbord(cm):",Real_width)
print("Calculated width of 7X7 checkerbord(cm):",calculated_width)
print("Error in cemtimeter:",Error)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#For get frames from video

import cv2

# Load the video
cap = cv2.VideoCapture('C:/Users/VISHAL.000/Downloads/Camera Calibration and Depth Estimation-20230506/videoHD1.avi')

# Set the frame rate of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set the frame number to start extracting frames from
frame_num = 0

while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()

    if not ret:
        break

    # Extract frames at a certain interval (e.g., every 2 seconds)
    if frame_num % (0.1 * fps) == 0:
        # Save the frame as an image
        cv2.imwrite(f'C:/Users/VISHAL.000/Desktop/Computer vision/Task2/Camera Calibration and Depth Estimation-20230506/Goal_2_all_image/frame_{frame_num}.jpg', frame)

    # Increment the frame number
    frame_num += 1

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




