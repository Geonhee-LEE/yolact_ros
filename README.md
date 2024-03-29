# yolact_ros

The `yolact_ros` is based on [Yolact](https://github.com/dbolya/yolact) and is integrated with ROS(Robot Operating System)

[![Yolact(You Only Look At CoefficienTs) with ROS and Webcam](http://img.youtube.com/vi/Qn949mpmndI/0.jpg)](https://www.youtube.com/watch?v=Qn949mpmndI&feature=youtu.be)

# Requirement 

* ROS(Kinetic)
* GPU supporting CUDA

# Installation
 - Set up a Python3 environment.
 - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
   - conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch
 - Install some other packages:
   ```Shell
   # Cython needs to be installed before pycocotools
   pip install cython
   pip install opencv-python pillow pycocotools matplotlib 
   
   conda install opencv
   pip install pyyaml
   pip install rospkg

   
   ```
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/dbolya/yolact.git
   cd yolact
   ```
 - If you'd like to train YOLACT, download the COCO dataset and the 2014/2017 annotations. Note that this script will take a while and dump 21gb of files into `./data/coco`.
   ```Shell
   sh data/scripts/COCO.sh
   ```
 - If you'd like to evaluate YOLACT on `test-dev`, download `test-dev` with this script.
   ```Shell
   sh data/scripts/COCO_test.sh
   ```


# Weight 
| Image Size | Backbone      | FPS  | mAP  | Weights                                                                                                              |  |
|:----------:|:-------------:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|--------|
| 550        | Resnet50-FPN  | 42.5 | 28.2 | [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)  | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EUVpxoSXaqNIlssoLKOEoCcB1m0RpzGq_Khp5n1VX3zcUw) |
| 550        | Darknet53-FPN | 40.0 | 28.7 | [yolact_darknet53_54_800000.pth](https://drive.google.com/file/d/1dukLrTzZQEuhzitGkHaGjphlmRJOjVnP/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/ERrao26c8llJn25dIyZPhwMBxUp2GdZTKIMUQA3t0djHLw)
| 550        | Resnet101-FPN | 33.0 | 29.8 | [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing)      | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EYRWxBEoKU9DiblrWx2M89MBGFkVVB_drlRd_v5sdT3Hgg)
| 700        | Resnet101-FPN | 23.6 | 31.2 | [yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view?usp=sharing)     | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/Eagg5RSc5hFEhp7sPtvLNyoBjhlf2feog7t8OQzHKKphjw)

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands.


In my workspace, After downloaded the [weight](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing), copy and paste to the _trained_model path_.



# Yolact ros with Visual grasping


[![Yolact(You Only Look At CoefficienTs) grasp](http://img.youtube.com/vi/bBZfp4Ve7Uw/0.jpg)](https://www.youtube.com/watch?v=bBZfp4Ve7Uw&feature=youtu.be)

Conda env: 
- torch11py36(Custom computer)
- py36_ros(NSCL computer)

## Capture image with customized config (In my case, Nobrand)
```
rosrun yolact_ros yolact_capture_img.py  --trained_model=/home/geonhee-ml/rl_ws/src/yolact_ros/src/yolact/weight/yolact_base_1234_100000.pth  --score_threshold=0.3 --top_k=100 --image=/home/geonhee-ml/rl_ws/src/yolact_ros/src/yolact/image/116.jpg
```


## Launch

```
./realsense
```

```
roscore
```

(torch11py36) conda env
```
rosrun yolact_ros yolact_tcp_img.py  --trained_model=/home/geonhee-ml/rl_ws/src/yolact_ros/src/yolact/weight/yolact_base_1234_100000.pth  --score_threshold=0.3 --top_k=100 --image=/home/geonhee-ml/rl_ws/src/yolact_ros/src/yolact/image/116.jpg 

```

### Save image from realsense through ros serveice

```
rosrun yolact_ros yolact_save_img.py 
```

```
rosservice call /save_image "data: false" 
```

### Run ros server for getting call and sending instance information

```
rosrun yolact_ros yolact_ros_server.py  --trained_model=/home/geonhee-ml/rl_ws/src/yolact_ros/src/yolact/weight/yolact_base_1234_100000.pth  --score_threshold=0.3 --top_k=100 
```
