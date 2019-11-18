#!/usr/bin/env python
## Author: Geonhee
## Date: November, 11, 2019
# Purpose: Ros node to use Yolact  using Pytorch

import sys
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt

# For getting realsense image
import socket


# ROS
import rospy
import rospkg
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from yolact_ros_msgs.msg import Detections
from yolact_ros_msgs.msg import Detection
from yolact_ros_msgs.msg import Box
from yolact_ros_msgs.msg import Mask
from cv_bridge import CvBridge, CvBridgeError

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

class SaveImg:
    def __init__(self):
        print ("Initalization about SaveImg class")
        
        self.bridge = CvBridge()
        save_server = rospy.Service('/save_image', SetBool, self.server_cb)
        self.count = 0

    def server_cb(self, data):
        self.get_data()
        self.count = self.count +1

        return SetBoolResponse(True, "Save!!!")

    def get_data(self):
        # Data options (change me)
        im_height = 720
        im_width = 1280
        tcp_host_ip = '127.0.0.1'
        #tcp_host_ip = '192.168.0.5'
        tcp_port = 50000
        buffer_size = 4098 # 4 KiB

        color_img = np.empty((im_height,im_width, 3))
        depth_img = np.empty((im_height,im_width))

        # Connect to server
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((tcp_host_ip, tcp_port))

        # Connect to server
        intrinsics = None
        
        #Ping the server with anything
        tcp_socket.send(b'asdf')

        # Fetch TCP data:
        #     color camera intrinsics, 9 floats, number of bytes: 9 x 4
        #     depth scale for converting depth from uint16 to float, 1 float, number of bytes: 4
        #     depth image, im_width x im_height uint16, number of bytes: im_width x im_height x 2
        #     color image, im_width x im_height x 3 uint8, number of bytes: im_width x im_height x 3
        data = b''
        while len(data) < (10*4 + im_height*im_width*5):
            data += tcp_socket.recv(buffer_size)

        # Reorganize TCP data into color and depth frame
        intrinsics = np.fromstring(data[0:(9*4)], np.float32).reshape(3, 3)
        depth_scale = np.fromstring(data[(9*4):(10*4)], np.float32)[0]
        depth_img = np.fromstring(data[(10*4):((10*4)+im_width*im_height*2)], np.uint16).reshape(im_height, im_width)
        color_img = np.fromstring(data[((10*4)+im_width*im_height*2):], np.uint8).reshape(im_height, im_width, 3)
        depth_img = depth_img.astype(float) * depth_scale
        
        # Color ndarray to img
        tmp_color_data = np.asarray(color_img)
        tmp_color_data.shape = (im_height,im_width,3)
        tmp_color_image = cv2.cvtColor(tmp_color_data, cv2.COLOR_RGB2BGR)

        # Depth ndarray to img
        #tmp_depth_data = np.asarray(depth_img)
        #tmp_depth_data.shape = (im_height,im_width)
        #tmp_depth_data = tmp_depth_data.astype(float)/1000

        cv2.imwrite(os.path.join('.', 'saved_img_%d.png' %self.count), tmp_color_image)
        #cv2.imwrite(os.path.join('.', 'test-depth.png'), tmp_depth_data)

        return tmp_color_image

def main():
    rospy.init_node('save_img_node', anonymous=True)
    detect_ = SaveImg()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
