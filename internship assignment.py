# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 01:19:55 2023

@author: HP
"""

import numpy as np
import cv2
  
video_path = "C:\\Users\\HP\\Downloads\\AI Assignment video.mp4"
# Capturing video through video
video = cv2.VideoCapture(video_path)
  
# Start a while loop
while(1):
      
    # Reading the video from the
    # video in image frames
    _, imageFrame = video.read()
  
    # Convert the imageFrame in 
    # BGR(RGB color space) to 
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
  
    # Set range for white color and 
    # define mask
    white_lower = np.array([0, 0, 0], np.uint8)
    white_upper = np.array([0, 0, 255], np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)
  
    # Set range for green color and 
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
  
    # Set range for orange color and
    # define mask
    orange_lower = np.array([5, 50, 50], np.uint8)
    orange_upper = np.array([15, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    # Set range for yellow color and 
    # define mask
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
      
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")
      
    # For white color
    white_mask = cv2.dilate(white_mask, kernal)
    res_white = cv2.bitwise_and(imageFrame, imageFrame, 
                              mask = white_mask)
      
    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask = green_mask)
      
    # For orange color
    orange_mask = cv2.dilate(orange_mask, kernal)
    res_orange = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = orange_mask)
    
    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = yellow_mask)
   
    # Creating contour to track white color
    contours, hierarchy = cv2.findContours(white_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h), 
                                       (255, 255, 255), 2)
              
            cv2.putText(imageFrame, "white Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255))    
  
    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h),
                                       (0, 255, 0), 2)
              
            cv2.putText(imageFrame, "Green Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0))
  
    # Creating contour to track orange color
    contours, hierarchy = cv2.findContours(orange_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 165, 0), 2)
              
            cv2.putText(imageFrame, "orange Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 165, 0))
            
    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h), 
                                       (255, 255, 0), 2)
              
            cv2.putText(imageFrame, "yellow Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 0))   
              
    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        
        cv2.destroyAllWindows()
        break     
