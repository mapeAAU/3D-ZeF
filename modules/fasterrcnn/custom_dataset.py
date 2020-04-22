import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path, data_type, img_ext='.png', transforms=None, min_area = 30):
        self.path = path
        self.transforms = transforms
        self.data_type = data_type
        self.img_path = os.path.join(self.path, self.data_type, "Zebrafish")
        self.min_area = min_area

        # Load annotations
        ann_path = os.path.join(self.path, self.data_type, "annotations.csv")
        self.df = pd.read_csv(ann_path, sep=";")
        self.tags = ['background'] + list(self.df["Annotation tag"].unique())

        # Get a sorted list of paths of annotated images - images without annotations are ignored
        self.imgs = sorted(self.df['Filename'].unique())
    
    def __getitem__(self, idx):
        # load images 
        img_path = os.path.join(self.img_path, self.imgs[idx])
        img = Image.open(img_path)#.convert("RGB")
        image_id = torch.tensor([idx])

        # Load bounding boxes and labels from the given frame
        bb = self.df[self.df['Filename'] == self.imgs[idx]]

        boxes = []
        labels = []
        # Check for bounding boxes in the image
        for i in range(len(bb)):
            # Read the bounding box positions
            xmin = bb.loc[bb.index[i],'Upper left corner X']
            ymin = bb.loc[bb.index[i],'Upper left corner Y']
            xmax = bb.loc[bb.index[i],'Lower right corner X']
            ymax = bb.loc[bb.index[i],'Lower right corner Y']

            if (xmax-xmin)*(ymax-ymin) < self.min_area:
                continue


            boxes.append([xmin, ymin, xmax, ymax])

            # Convert labels from annotation tag to index number (birth = 1, piglet = 2)
            labels.append(self.tags.index(bb.loc[bb.index[i],'Annotation tag']))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(bb),), dtype=torch.int64) 

        # Set target attributes
        target = {}
        target["image_id"] = image_id
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_events(self,delimiter):
        """
        An event is specified by the first part of the filename before a given delimiter

        Example
            filename: 2019_05_10_14_55_57-00000.png
            delimiter: -
            extension: .png
            image number: 00000
            event: 2019_05_10_14_55_57

        """
        events = []
        for f in self.imgs:
            tmp = f[:(f.find(delimiter))]
            # Append unique event-names to the list
            if tmp in events: continue 
            events.append(tmp)

        return events

    def __len__(self):
        return len(self.imgs)
