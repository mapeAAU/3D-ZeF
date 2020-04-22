import os
import pandas as pd
import numpy as np
import argparse

def combineAnnotations(args):
    """
    Combines the bbox and point annotations for a specified caemra view

    Input:
        directory: Path to the root folder. Within there should be a folder called "GT", containing the "bbox_annotations_'camView'.csv" and "point_annotations_'camView'_'fishID'.csv" files
        camView: String determining which camera view is combined

    Output:
        A file called "annotations_'camView'.csv" in the GT folder under 'directory'
    """

    directory = args["directory"]
    cam_view = args["camView"]

    gt_directory = os.path.join(directory, "GT")

    bbox_annotations_path = os.path.join(gt_directory, "bbox_annotations_"+cam_view.lower()+".csv")
    bbox_annotations = pd.read_csv(bbox_annotations_path, sep=";")
    fish_IDs = bbox_annotations["Object ID"].unique()

    new_df = pd.DataFrame()

    for fish_ID in sorted(fish_IDs):
        print(fish_ID)
        point_annotations_path = os.path.join(gt_directory, "point_annotations_"+cam_view.lower()+"_"+str(fish_ID)+".csv")
        point_annotations = pd.read_csv(point_annotations_path, sep=";")
        bbox_id = bbox_annotations[bbox_annotations["Object ID"] == fish_ID]

        point_frames = point_annotations["Filename"].unique()
        bbox_frames = bbox_id["Filename"].unique()
        
        print(len(set(bbox_frames)), len(set(point_frames)))
        print(set(bbox_frames).symmetric_difference(set(point_frames)))
        assert set(bbox_frames) == set(point_frames)

        for filename in sorted(point_frames):
            bbox = bbox_id[bbox_id["Filename"] == filename]
            point = point_annotations[point_annotations["Filename"] == filename]
            ann_dict = {"Filename":filename, "Object ID": [fish_ID], "Annotation tag": [bbox["Annotation tag"].values[0]], "Upper left corner X": [int(bbox["Upper left corner X"])], "Upper left corner Y": [int(bbox["Upper left corner Y"])] , "Lower right corner X": [int(bbox["Lower right corner X"])], "Lower right corner Y": [int(bbox["Lower right corner Y"])], "Occlusion": [int(bbox["Occlusion"])], "X": [int(point["X"])], "Y": [int(point["Y"])]}
            new_df = pd.concat([new_df, pd.DataFrame.from_dict(ann_dict)], ignore_index=True)

    new_df.to_csv(os.path.join(gt_directory, "annotations_"+cam_view.lower()+".csv"), sep=";", index=False)





if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Combine the bbox and point annotations for a camera view")
    ap.add_argument("-d", "--directory", help="Path to video directory", type=str)
    ap.add_argument("-c", "--camView", help="Camera view (top/front)", type=str)
    args = vars(ap.parse_args())

    combineAnnotations(args)