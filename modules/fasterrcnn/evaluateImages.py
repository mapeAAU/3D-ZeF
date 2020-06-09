import os 
import utils
import torch
import argparse     
import datetime
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import pickle as pkl
import cv2
from PIL import Image
import pandas as pd                  
from custom_dataset import CustomDataset                                                                                                                                                                      

def initializeModel(pretrained, num_classes):
    """
    Loads the Faster RCNN ResNet50 model from torchvision, and sets whether it is COCO pretrained, and adjustes the heds box predictor to our number of classes.

    Input:
        pretrained: Whether to use a CoCo pretrained model
        num_classes: How many classes we have:

    Output:
        model: THe initialized PyTorch model
    """
    
    # Load model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    
@torch.no_grad()
def test(args):

    path = args["path"]
    camId = args["camId"]
    weight_path = args["weights"]
    startFrame = args["startFrame"]
    endFrame = args["endFrame"]
    outputPath = args["outputPath"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

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

    if camId == 1:
        imgs_path = os.path.join(path, "imgT")
    else:
        imgs_path = os.path.join(path, "imgF")

    if not os.path.isdir(imgs_path):
        print("Could not find image folder {0}".format(imgs_path))
        sys.exit()

    # move model to the device (GPU/CPU)
    model.to(device)
    model.eval()

    id_map = {0:"Background", 1:"Zebrafish"}
    output_df = pd.DataFrame(columns=["Filename", "Object ID", "Annotation tag", "Upper left corner X", "Upper left corner Y", "Lower right corner X", "Lower right corner Y", "Confidence"])

    frameCount = 0

    filenames = [f for f in sorted(os.listdir(imgs_path)) if os.path.splitext(f)[-1] in [".png", ".jpg"]]

    for filename in filenames:

        frameCount = int(filename[:-4])
        

        if frameCount % 1000 == 0:
            print(frameCount)

        if frameCount < startFrame:
            continue

        if frameCount > endFrame:
            frameCount -= 1
            break
            
        img = cv2.imread(os.path.join(imgs_path, filename))
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

    output_df.to_csv(os.path.join(outputPath, "boundingboxes_2d_cam{}.csv".format(camId)), index=False, sep=",")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--path", help="Path to folder")
    ap.add_argument("-c", "--camId", help="Camera ID. top = 1 and front = 2")
    ap.add_argument("-w", "--weights", help="Path to the trained model weights")
    ap.add_argument("-sf", "--startFrame", default=1, type=int)
    ap.add_argument("-ef", "--endFrame", default=10000, type=int)
    ap.add_argument("-o", "--outputPath", default="./processed")
    args = vars(ap.parse_args())
    test(args)