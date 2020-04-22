import os 
import utils
import torch
import argparse       
import cv2
import datetime
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import pickle as pkl
from PIL import Image
import pandas as pd                  
from custom_dataset import CustomDataset                                                                                                                                                                      
from train import initializeModel

@torch.no_grad()
def test(args):

    path = args["path"]
    camId = args["camID"]
    weight_path = args["weights"]

    # train on the GPU or on the CPU, if a GPU is not available 
    if torch.cuda.is_available():
        device = torch.device('cuda') 
        print("Using GPU")
    else:
        print("WARNING: Using CPU")
        device = torch.device('cpu')

    # our dataset has two classes only - background and zebrafish
    num_classes = 2

    model = initializeModel(False, num_classes)
    model.load_state_dict(torch.load(weight_path)["model_state_dict"])

    cpu_device = torch.device("cpu")

    vid_path = os.path.join(path, "cam{}.mp4".format(camId))

    cap = cv2.VideoCapture(vid_path)

    if not cap.isOpened():
        print("VIDEO NOT FOUND: {}".format(vid_path))

    # move model to the device (GPU/CPU)
    model.to(device)
    model.eval()

    id_map = {0:"Background", 1:"Zebrafish"}
    output_df = pd.DataFrame(columns=["Filename", "Object ID", "Annotation tag", "Upper left corner X", "Upper left corner Y", "Lower right corner X", "Lower right corner Y", "Confidence"])

    frameCount = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        frameCount += 1

        if frameCount % 1000 == 0:
            print(frameCount)

        if ret:
            img = F.to_tensor(img)
            img = [img.to(device)]

            outputs = model(img)
            outputs = {k: v.to(cpu_device) for k, v in outputs[0].items()}

            bboxes = outputs["boxes"].cpu().detach().numpy()
            labels = outputs["labels"].cpu().detach().numpy()
            scores = outputs["scores"].cpu().detach().numpy()
            filename = str(frameCount).zfill(6)+".png"

            output_dict = {"Filename": [filename]*len(scores),
                            "Frame": [frameCount]*len(scores),
                            "Object ID": [-1]*len(scores),
                            "Annotation tag": [id_map[x] for x in labels],
                            "Upper left corner X": list(bboxes[:,0]),
                            "Upper left corner Y": list(bboxes[:,1]),
                            "Lower right corner X": list(bboxes[:,2]),
                            "Lower right corner Y": list(bboxes[:,3]),
                            "Confidence": list(scores)}
            output_df = pd.concat([output_df, pd.DataFrame.from_dict(output_dict)], ignore_index=True)
        else:
            break
    output_df.to_csv(path,"processed", "boundingboxes_2d_cam{}.csv".format(camId), index=False, sep=",")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--path", help="Path to folder")
    ap.add_argument("-c", "--camId", help="Camera ID. top = 1 and front = 2")
    ap.add_argument("-w", "--weights", help="Path to the trained model weights")
    args = vars(ap.parse_args())
    test(args)