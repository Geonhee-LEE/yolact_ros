#!/home/juhk/anaconda3/envs/torch/bin/python

import sys
sys.path.append('/home/juhk/catkin_ws/src/test_/test/src/yolact')

from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

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

import matplotlib.pyplot as plt
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
from sensor_msgs.msg import Image as Images
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

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
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
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

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})



from multiprocessing.pool import ThreadPool
from queue import Queue

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])




class detect:
    def __init__(self):
        rospy.init_node('yolact_node', anonymous=False)
        self.image_pub = rospy.Publisher("/usb_cam/image_raw",Images)
        self.bridge = CvBridge()

    def evalvideo(self,net:Yolact, path:str):
        # If the path is a digit, parse it as a webcam index
        is_webcam = path.isdigit()
        
        if is_webcam:
            vid = cv2.VideoCapture(int(path))
        else:
            vid = cv2.VideoCapture(path)
        
        if not vid.isOpened():
            print('Could not open video "%s"' % path)
            exit(-1)
        
        net = CustomDataParallel(net).cuda()
        transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
        frame_times = MovingAverage(100)
        fps = 0
        # The 0.8 is to account for the overhead of time.sleep
        frame_time_target = 1 / vid.get(cv2.CAP_PROP_FPS)
        running = True

        def cleanup_and_exit():
            print()
            pool.terminate()
            vid.release()
            cv2.destroyAllWindows()
            exit()

        def get_next_frame(vid):
            return [vid.read()[1] for _ in range(args.video_multiframe)]

        def transform_frame(frames):
            with torch.no_grad():
                frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
                return frames, transform(torch.stack(frames, 0))

        def eval_network(inp):
            with torch.no_grad():
                frames, imgs = inp
                return frames, net(imgs)

        def prep_frame(inp):
            with torch.no_grad():
                frame, preds = inp
                return self.prep_display(preds, frame, None, None, undo_transform=False, class_color=True)

        frame_buffer = Queue()
        video_fps = 0

        # All this timing code to make sure that 
        def play_video():
            nonlocal frame_buffer, running, video_fps, is_webcam

            video_frame_times = MovingAverage(100)
            frame_time_stabilizer = frame_time_target
            last_time = None
            stabilizer_step = 0.0005

            while running:
                frame_time_start = time.time()

                if not frame_buffer.empty():
                    next_time = time.time()
                    if last_time is not None:
                        video_frame_times.add(next_time - last_time)
                        video_fps = 1 / video_frame_times.get_avg()
                    cv2.imshow(path, frame_buffer.get())
                    last_time = next_time

                self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame_buffer.get(), "bgr8"))

                if cv2.waitKey(1) == 27: # Press Escape to close
                    running = False

                buffer_size = frame_buffer.qsize()
                if buffer_size < args.video_multiframe:
                    frame_time_stabilizer += stabilizer_step
                elif buffer_size > args.video_multiframe:
                    frame_time_stabilizer -= stabilizer_step
                    if frame_time_stabilizer < 0:
                        frame_time_stabilizer = 0

                new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)

                next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                target_time = frame_time_start + next_frame_target - 0.001 # Let's just subtract a millisecond to be safe

                # This gives more accurate timing than if sleeping the whole amount at once
                while time.time() < target_time:
                    time.sleep(0.001)
                


        extract_frame = lambda x, i: (x[0][i] if x[1][i] is None else x[0][i].to(x[1][i]['box'].device), [x[1][i]])

        # Prime the network on the first frame because I do some thread unsafe things otherwise
        print('Initializing model... ', end='')
        eval_network(transform_frame(get_next_frame(vid)))
        print('Done.')

        # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
        sequence = [prep_frame, eval_network, transform_frame]
        pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
        pool.apply_async(play_video)

        active_frames = []

        print()


        while vid.isOpened() and running:
            start_time = time.time()

            # Start loading the next frames from the disk
            next_frames = pool.apply_async(get_next_frame, args=(vid,))
            
            # For each frame in our active processing queue, dispatch a job
            # for that frame using the current function in the sequence
            for frame in active_frames:
                frame['value'] = pool.apply_async(sequence[frame['idx']], args=(frame['value'],))
            
            # For each frame whose job was the last in the sequence (i.e. for all final outputs)
            for frame in active_frames:
                if frame['idx'] == 0:
                    frame_buffer.put(frame['value'].get())

            # Remove the finished frames from the processing queue
            active_frames = [x for x in active_frames if x['idx'] > 0]

            # Finish evaluating every frame in the processing queue and advanced their position in the sequence
            for frame in list(reversed(active_frames)):
                frame['value'] = frame['value'].get()
                frame['idx'] -= 1

                if frame['idx'] == 0:
                    # Split this up into individual threads for prep_frame since it doesn't support batch size
                    active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, args.video_multiframe)]
                    frame['value'] = extract_frame(frame['value'], 0)

            
            # Finish loading in the next frames and add them to the processing queue
            active_frames.append({'value': next_frames.get(), 'idx': len(sequence)-1})
            
            # Compute FPS
            frame_times.add(time.time() - start_time)
            fps = args.video_multiframe / frame_times.get_avg()

            print('\rProcessing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d    ' % (fps, video_fps, frame_buffer.qsize()), end='')
        
        cleanup_and_exit()

    def prep_display(self,dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
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
        
        if args.display_text or args.display_bboxes:
            str_ = ""
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                if args.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if args.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    #pub = rospy.Publisher('chatter',String,queue_size=10)
                    #rate = rospy.Rate(50) #10hz
                    #str_ += text_str
            #rospy.loginfo(str_)
            #pub.publish(str_)
            #rate.sleep()
        
        return img_numpy

if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

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

        if args.image is None and args.video is None and args.images is None:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None        

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        net.detect.use_fast_nms = args.fast_nms
        cfg.mask_proto_debug = args.mask_proto_debug

        detect_ = detect()
        detect_.evalvideo(net, args.video)


