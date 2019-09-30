#!/home/juhk/anaconda3/envs/torch/bin/python
import rospy

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
 
class Gray():
    def __init__(self):
        self._sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback, queue_size=1)
 
        self.bridge = CvBridge()
 
    def callback(self, image_msg):
 
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
 
        #cv_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
 
        cv2.imshow('cv_image', cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def main(self):
        rospy.spin()
 
if __name__ == '__main__':
    rospy.init_node('gray')
    node = Gray()
    node.main()




