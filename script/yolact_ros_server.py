#!/usr/bin/env python
## Author: Geonhee
## Date: November, 11, 2019
# Purpose: Ros node to use Yolact  using Pytorch

import sys
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
 
# Yolact
#sys.path.append(os.path.join(os.path.dirname(__file__), "yolact"))
import yolact
from yolact.data.coco import COCODetection, get_label_map
from yolact.data.config import MEANS, COLORS
from yolact.yolact import Yolact
from yolact.utils.augmentations import BaseTransform, FastBaseTransform, Resize
from yolact.utils.functions import MovingAverage, ProgressBar
from yolact.layers.box_utils import jaccard, center_size
from yolact.utils import timer
from yolact.utils.functions import SavePath
from yolact.layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from yolact.data.config import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
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
import math
import matplotlib.pyplot as plt

# For getting realsense image
import socket


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})


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
from yolact_ros_msgs.msg import GraspPt
from cv_bridge import CvBridge, CvBridgeError

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='/home/geonhee-ml/catkin_build_ws/src/yolact_ros/src/yolact/weight/yolact_base_1234_100000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)


class DetectImg:
    def __init__(self, net:Yolact):
        print ("Initalization about DetectImg class")
        self.save_cnt = 0
        self.net = net
        
        self.bridge = CvBridge()
        self.detections_pub = rospy.Publisher("detections",numpy_msg(Detections),queue_size=10)
        self.com_pub = rospy.Publisher("com_info",numpy_msg(GraspPt),queue_size=10)
        self.image_pub = rospy.Publisher("cvimage_published",Image,queue_size=10)

        start_server = rospy.Service('/start_instance_seg', SetBool, self.server_cb)

    def server_cb(self, data):
        self.evaluate(self.net)

        return SetBoolResponse(True, "Success")

    def evaluate(self, net:Yolact, train_mode=False):
        net.detect.use_fast_nms = args.fast_nms
        cfg.mask_proto_debug = args.mask_proto_debug

        self.evalimage()
            
    def evalimage(self):
        cv_img = self.get_data()
        #frame = torch.from_numpy(cv2.imread(path)).cuda().float()
        frame = torch.from_numpy(cv_img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self.net(batch)

        self.prep_display(preds, frame, None, None, undo_transform=False)

    def get_orientation_from_mask(self, num_dets_to_consider, img_numpy, detections, mask):
        print ("-----------------------------")
        print ("The number of evaluation: ",self.save_cnt)

        mask_data = mask.cpu()
        mask_data = mask_data.numpy()
        mask_data = mask_data.astype(np.int64)
        mask_data = mask_data * 100
         
        # The number of detected objects
        num_object = num_dets_to_consider
        center_of_mass_instance = GraspPt()

        for num_ in range(0, num_object):
            print ("Class name:", detections.detections[num_object -1 - num_].class_name) # Start point is different
            print ("Score:", detections.detections[num_object -1 - num_].score)
            mask_h_sum = 0
            mask_w_sum = 0
            sum_count = 0
            
            for i in range(0, mask_data.shape[1]):
                for j in range(0,mask_data.shape[2]):
                    if mask_data[num_,i,j,0] != 0:
                        mask_h_sum += i
                        mask_w_sum += j
                        sum_count += 1

            mask_h = mask_h_sum / sum_count
            mask_w = mask_w_sum / sum_count

            mask_h_index = int(mask_h)
            mask_w_index = int(mask_w)

            print ("center of mass x :", mask_w_index)
            print ("center of mass y :", mask_h_index)

            mask_data[num_, mask_h_index, mask_w_index, 0] = 255

            img = mask_data[num_, :, :, 0]
        
            rows = img.shape[0]
            cols = img.shape[1]
            
            x = np.ones((rows, 1))
            y = np.ones((1, cols))

            for i in range(2, cols+1):
                m = np.ones((rows, 1))*i
                x = np.hstack((x,m))

            for i in range(2, rows+1):
                n = np.ones((1, cols))*i
                y = np.vstack((y,n))

            area = img.sum()
            f_img = img.astype(np.float)

            im_x = (f_img * x)
            im_y = (f_img * y)

            meanx = im_x.sum()/area
            meany = im_y.sum()/area

            x = x - meanx
            y = y - meany

            a_img = f_img * (x * x)
            b_img = f_img * (x * y)
            c_img = f_img * (y * y)
            
            a = a_img.sum()
            b = b_img.sum()*2
            c = c_img.sum() 

            denom = b*b + (a-c)*(a-c)

            if denom == 0:
                thetamin = 2 * 3.14 * random.random(0,1)
                thetamax = 2 * 3.14 * random.random(0,1)
                roundness = 1
            else:
                sin2thetamin = b/math.sqrt(denom)
                sin2thetamax = -sin2thetamin
                cos2thetamin = (a-c)/math.sqrt(denom)
                cos2thetamax = -cos2thetamin

                thetamin = math.atan2(sin2thetamin, cos2thetamin)/2
                thetamax = math.atan2(sin2thetamax, cos2thetamax)/2

                lmin = 0.5*(c+a) - 0.5*(a-c)*cos2thetamin - 0.5*b*sin2thetamin
                lmax = 0.5*(c+a) - 0.5*(a-c)*cos2thetamax - 0.5*b*sin2thetamax

                roundness = lmin/lmax
            
            if thetamin > 0:
                pass
            elif thetamin < 0:
                thetamin = np.pi + thetamin

            print ("Theta_min(deg)",thetamin * 57.325)

            point_x = 50*math.sin(thetamin)
            point_y = 50*math.cos(thetamin)

            rotation_point_x = mask_h_index - point_x
            rotation_point_y = mask_w_index - point_y

            rotation_point_x = int(rotation_point_x)
            rotation_point_y = int(rotation_point_y)
            
            mask_data[num_, rotation_point_x, rotation_point_y, 0] = 255
            
            center_of_mass_instance.class_name.append(detections.detections[num_object -1 - num_].class_name)
            center_of_mass_instance.score.append(detections.detections[num_object -1 - num_].score)
            center_of_mass_instance.com_x.append(mask_w_index)
            center_of_mass_instance.com_y.append(mask_h_index)

            center_of_mass_instance.angle.append(thetamin)
        
        self.com_pub.publish(center_of_mass_instance)
        self.save_cnt = self.save_cnt +1 
        cv2.imwrite("/home/geonhee-ml/Desktop/%d_img_raw_%d.jpg" %(self.save_cnt, thetamin * 57.325), img_numpy)
        for i in range(0, num_object):
            cv2.imwrite("/home/geonhee-ml/Desktop/%d_img_mask_%d.jpg" %(self.save_cnt, center_of_mass_instance.angle[i] * 57.325), mask_data[i,:,:,:])
        print ("-----------------------------")

    def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, image_header=Header()):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        with torch.no_grad():
            detections = Detections() 

            if undo_transform:
                img_numpy = undo_image_transformation(img, w, h)
                img_gpu = torch.Tensor(img_numpy).cuda()
            else:
                img_gpu = img / 255.0
                h, w, _ = img.shape
            
            with timer.env('Postprocess'):
                t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                                crop_masks        = args.crop,
                                                score_threshold   = args.score_threshold)
                torch.cuda.synchronize()

            with timer.env('Copy'):
                if cfg.eval_mask_branch:
                    # Masks are drawn on the GPU, so don't copy
                    masks = t[3][:args.top_k]
                classes, scores, boxes = [x[:args.top_k].cpu().numpy() for x in t[:3]]

            num_dets_to_consider = min(args.top_k, classes.shape[0])
            for j in range(num_dets_to_consider):
                if scores[j] < args.score_threshold:
                    num_dets_to_consider = j
                    break
            
            if num_dets_to_consider == 0:
                # No detections found so just output the original image
                return (img_gpu * 255).byte().cpu().numpy()

            # Quick and dirty lambda for selecting the color for a particular index
            # Also keeps track of a per-gpu color cache for maximum speed
            def get_color(j, on_gpu=None):
                global color_cache
                color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
                
                if on_gpu is not None and color_idx in color_cache[on_gpu]:
                    return color_cache[on_gpu][color_idx]
                else:
                    color = COLORS[color_idx]
                    if not undo_transform:
                        # The image might come in as RGB or BRG, depending
                        color = (color[2], color[1], color[0])
                    if on_gpu is not None:
                        color = torch.Tensor(color).to(on_gpu).float() / 255.
                        color_cache[on_gpu][color_idx] = color
                    return color

            # First, draw the masks on the GPU where we can do it really fast
            # Beware: very fast but possibly unintelligible mask-drawing code ahead
            # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
            if args.display_masks and cfg.eval_mask_branch:
                # After this, mask is of size [num_dets, h, w, 1]
                masks = masks[:num_dets_to_consider, :, :, None]
                
                # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
                colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
                masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

                # This is 1 everywhere except for 1-mask_alpha where the mask is
                inv_alph_masks = masks * (-mask_alpha) + 1
                
                # I did the math for this on pen and paper. This whole block should be equivalent to:
                #    for j in range(num_dets_to_consider):
                #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
                masks_color_summand = masks_color[0]
                if num_dets_to_consider > 1:
                    inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                    masks_color_cumul = masks_color[1:] * inv_alph_cumul
                    masks_color_summand += masks_color_cumul.sum(dim=0)

                img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
                
            # Then draw the stuff that needs to be done on the cpu
            # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
            img_numpy = (img_gpu * 255).byte().cpu().numpy()
            
            print("Num dets: ",  num_dets_to_consider)
            if args.display_text or args.display_bboxes:
                for j in reversed(range(num_dets_to_consider)):
                    x1, y1, x2, y2 = boxes[j, :]
                    color = get_color(j)
                    score = scores[j]

                    if args.display_bboxes:
                        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 2)

                    if args.display_text:
                        _class = cfg.dataset.class_names[classes[j]]
                        text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                        font_face = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = 0.6
                        font_thickness = 1

                        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                        text_pt = (x1, y1 - 10)
                        text_color = [255, 255, 255]

                        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
                    det = Detection()
                    det.box.x1 = x1
                    det.box.y1 = y1
                    det.box.x2 = x2
                    det.box.y2 = y2
                    det.class_name = _class
                    det.score = score
                    mask_shape = np.shape(masks[j])
                    #print("Shape: ", mask_shape)
                    #mask_bb = np.squeeze(masks[j].cpu().numpy(), axis=2)[y1:y2,x1:x2] # Crop
                    mask_bb = np.squeeze(masks[j].cpu().numpy(), axis=2)[:,:] # Every mask (1280 * 720)
                    #print("Box: x1:", x1,", x2: ",x2,", y1: ",y1,", y2: ",y2)
                    #print("Mask in box shape: ", np.shape(mask_bb))
                    mask_rs = np.reshape(mask_bb, -1)
                    #print("New shape: ", np.shape(mask_rs))
                    #print("Mask:\n",mask_bb)
                    det.mask.height = y2 - y1
                    det.mask.width = x2 - x1
                    det.mask.mask = np.array(mask_rs, dtype=bool)
                    detections.detections.append(det)
                detections.header.stamp = image_header.stamp
                detections.header.frame_id = image_header.frame_id
            
            self.detections_pub.publish(detections)                
            self.get_orientation_from_mask(num_dets_to_consider, img_numpy, detections, masks)
   
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(img_numpy, "bgr8"))
            except CvBridgeError as e:
                print(e)
            
            #return num_dets_to_consider, img_numpy, masks

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

        #cv2.imwrite(os.path.join('.', 'test.png'), tmp_color_image)
        #cv2.imwrite(os.path.join('.', 'test-depth.png'), tmp_depth_data)

        return tmp_color_image

def main():
    parse_args()

    rospy.init_node('yolact_ros', anonymous=True)
    if args.config is not None:
        set_cfg(args.config)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()


        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        net.detect.use_fast_nms = True
        cfg.mask_proto_debug = False

        detect_ = DetectImg(net)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
