## 3D-ZeF: A 3D Zebrafish Tracking Benchmark Dataset

This repository contains the code and scripts for the paper *3D-ZeF: A 3D Zebrafish Tracking Benchmark Dataset*

The paper investigates tracking multiple zebrafish simuntaniously from two unique camea views, in order to generate 3D trajectories of each zebrafish.
In the paper we present our novel zebrafish 3D tracking dataset, recorded in a laboratory environment at Aalborg, Denmark. The data is publicly available at our [MOT Challenge page](https://motchallenge.net/data/3D-ZeF20).


As this is a collection of research code, one might find some occasional rough edges. We have tried to clean up the code to a decent level but if you encounter a bug or a regular mistake, please report it in our issue tracker. 


### Zebrafish Tracker Overview
#### Tracker Pipeline 
Use the "pipeline-env.yml" Anaconda environment and run the provided Pipeline bat scripts for running the entire tracking pipeline at a given directory path. The directory path should contain:
 
 * cam1.mp4 and cam2.mp4, of the top and front views respectively, or the image folders ImgF and ImgT.
 * cam1_references.json and cam2_references.json, containing the world <-> camera correspondance, and the cam1_intrinsic.json and cam2_intrinsic.json files containg the intrinsic camera parameters..
 * cam1.pkl and cam2.pkl containing the camera calibrations.
 * cam1.pkl and cam2.pkl are crated using the JsonToCamera.py script in the reconstruction folder. When using the references and intrinsic json files from the MOTChallenge data folder, you need to rename the camT_\*.json files to cam1_\*.json and camF_\*.json to cam2_\*.json.  
 * settings.ini file such as the provided dummy_settings.ini, with line 2, 13, 14, 41 and 42 adjusted to the specific videos.
 * Have an empty folder called "processed".


##### Script Descriptions

ExtractBackground.py
    
    Creates a median image of the video, used for background subtraction in BgDetector.py

BgDetector.py
    
    Runs thorugh the specified camera view and detects bounding boxes and keypoint for the zebrafish in all of the frames of the video. Can also provide previously detected bounding boxes, in which case they are processed for the next step.

TrackerVisual.py
    
    Runs through the specified camera view and construct initial 2D tracklets from the detected keypoints.
    
TrackletMatching.py
    
    Goes through the 2D tracklets from TrackerVisual.py and tries to make them into 3D tracklets.
    
FinalizeTracks.py
    
    Goes through the 3D tracklets from TrackeltMatching.py, and combiens them into full 3D tracks



#### Faster RCNN
To train and evaluate using the Faster RCNN method, go to the folder "modules/fasterrcnn".
Utilize the provided docker file. if there are any problems read the DOCKER_README.md file.
The Faster RCNN Code is based on the torchvision object detection fine tuning guide: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

##### Pretrained models

The utilized pretrained Faster RCNN models are available at this [Dropbox link](https://www.dropbox.com/s/fesalzi16usruso/3DZeF_pretrained_fasterrcnn.zip?dl=0)

##### Script Descriptions

train.py
    
    Trains a Faster RCNN with a ResNet50 FPN bakbone. Expects the data to be in a subfolder called "data" and with the child folders "train" and "valid". In each of these folders all the extracted images and a file called "annotations.csv" should be placed. The annotations should be in the AAU VAP Bounding Box annotation format.

evaluateVideo.py
    
    Runs the Faster RCNN on a provided video with the provided model weights. The output is placed in the provided output dir.

evaluateImages.py
    
    Runs the Faster RCNN on a provided images with the provided model weights. The output is placed in the provided output dir.


#### MOT METRICS
In order to obtain the MOT metrics of the complete tracking output, the MOTMetrics.bat script should be run. For this task the Anaconda environment described in "motmetrics-enc.yml" should be used.
This requires that at the provided directory path, there is a folder called "gt" with a "annotations_full.csv" file or the gt.txt file from the MOT Challenge page, and an empty folder called "metrics". When using gt.txt the --useMOTFormat should be supplied.
The metrics are calculated using the py-motmetrics package: https://github.com/cheind/py-motmetrics (Version used: 1.1.3.  Retrieved: 30th August 2019)



### License

All code is licensed under the MIT license, except for the pymotmetrics and torchvision code reference, which are subject to their respective licenses when applicable.



### Acknowledgements
Please cite the following paper if you use our code or dataset:

```TeX
@InProceedings{Pedersen_2020_CVPR,
author = {Pedersen, Malte and Haurum, Joakim Bruslund and Bengtson, Stefan Hein and Moeslund, Thomas B.},
title = {3D-ZeF: A 3D Zebrafish Tracking Benchmark Dataset},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```