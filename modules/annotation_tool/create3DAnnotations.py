import os
import pandas as pd
import numpy as np
import argparse

import sys
sys.path.append('../../')
from common.utility import prepareCams
from modules.reconstruction.Triangulate import Triangulate



def create3DAnnotations(args):
    """
    Combines the 2D annotation files into a single 3D annotation file

    Input:
        directory: Path to the root folder. Within there should be a folder called "GT", containing the "annotations_top.csv" and "annotations_front.csv" files
        syncTop: The sync frame for the top view
        syncFront: The sync frame for the front view

    Output:
        A file called "annotations_full.csv" in the GT folder under 'directory'. Also aoutput a txt file per view with frames and ids where the reprojection error is above 20.0
    """

    directory = args["directory"]
    synFrameTop = args["syncTop"]
    synFrameFront = args["syncFront"]

    camera1_offset = max(0,synFrameFront-synFrameTop)
    camera2_offset = max(0,synFrameTop-synFrameFront)  

    print("Front",synFrameFront, camera2_offset)
    print("Top", synFrameTop, camera1_offset)

    gt_directory = os.path.join(directory, "GT")

    cams = prepareCams(directory)

    tr = Triangulate()

    top = pd.read_csv(os.path.join(gt_directory, "annotations_top.csv"), sep=";")
    front = pd.read_csv(os.path.join(gt_directory, "annotations_front.csv"), sep=";")

    top["Org Frame"] = [int(x[:-4]) for x in top["Filename"].values.tolist()]
    front["Org Frame"] = [int(x[:-4]) for x in front["Filename"].values.tolist()]

    top["Frame"] = top["Org Frame"] + camera1_offset
    front["Frame"] = front["Org Frame"] + camera2_offset

    
    top_frames = top["Frame"].values.tolist()
    front_frames = front["Frame"].values.tolist()
    frames = list(set(top_frames + front_frames))

    top_ids = top["Object ID"].values.tolist()
    front_ids = front["Object ID"].values.tolist()
    ids = list(set(top_ids + front_ids))

    df = pd.DataFrame()

    cam1_err = []
    cam2_err = []
    
    for frame in sorted(frames):
        for fish_ID in sorted(ids):
            data_dict = None
            
            top_subdf = top[(top["Object ID"] == fish_ID) & (top["Frame"] == frame)]
            front_subdf = front[(front["Object ID"] == fish_ID) & (front["Frame"] == frame)]

            if len(top_subdf) == 0:
                data_dict = {"frame": [frame],
                            "id": [fish_ID],
                            "3d_x": [None],
                            "3d_y": [None],
                            "3d_z": [None],
                            "cam1_reproj_err": [None],
                            "cam2_reproj_err": [None],
                            "cam1_frame": [None],
                            "cam1_x": [None],
                            "cam1_y": [None],
                            "cam1_tl_x": [None],
                            "cam1_tl_y": [None],
                            "cam1_br_x": [None],
                            "cam1_br_y": [None],
                            "cam1_occlusion": [None],
                            "cam2_frame": [front_subdf["Org Frame"].values[0]],
                            "cam2_x": [front_subdf["X"].values[0]],
                            "cam2_y": [front_subdf["Y"].values[0]],
                            "cam2_tl_x": [front_subdf["Upper left corner X"].values[0]],
                            "cam2_tl_y": [front_subdf["Upper left corner Y"].values[0]],
                            "cam2_br_x": [front_subdf["Lower right corner X"].values[0]],
                            "cam2_br_y": [front_subdf["Lower right corner Y"].values[0]],
                            "cam2_occlusion": [front_subdf["Occlusion"].values[0]]
                            }

            elif len(front_subdf) == 0:
                data_dict = {"frame": [frame],
                            "id": [fish_ID],
                            "3d_x": [None],
                            "3d_y": [None],
                            "3d_z": [None],
                            "cam1_reproj_err": [None],
                            "cam2_reproj_err": [None],
                            "cam1_frame": [top_subdf["Org Frame"].values[0]],
                            "cam1_x": [top_subdf["X"].values[0]],
                            "cam1_y": [top_subdf["Y"].values[0]],
                            "cam1_tl_x": [top_subdf["Upper left corner X"].values[0]],
                            "cam1_tl_y": [top_subdf["Upper left corner Y"].values[0]],
                            "cam1_br_x": [top_subdf["Lower right corner X"].values[0]],
                            "cam1_br_y": [top_subdf["Lower right corner Y"].values[0]],
                            "cam1_occlusion": [top_subdf["Occlusion"].values[0]],
                            "cam2_frame": [None],
                            "cam2_x": [None],
                            "cam2_y": [None],
                            "cam2_tl_x": [None],
                            "cam2_tl_y": [None],
                            "cam2_br_x": [None],
                            "cam2_br_y": [None],
                            "cam2_occlusion": [None]
                            }
            else:
                t1Pt = [float(top_subdf["X"].values[0]), float(top_subdf["Y"].values[0])]
                t2Pt = [float(front_subdf["X"].values[0]), float(front_subdf["Y"].values[0])]

                # 1) Triangulate 3D point
                p,d = tr.triangulatePoint(t1Pt, t2Pt, cams[1], cams[2], correctRefraction=True)

                p1 = cams[1].forwardprojectPoint(*p)
                p2 = cams[2].forwardprojectPoint(*p)

                # 2) Calc re-projection errors
                pos1 = np.array(t1Pt)
                err1 = np.linalg.norm(pos1-p1)
                pos2 = np.array(t2Pt)
                err2 = np.linalg.norm(pos2-p2)
                err = err1 + err2     

                if err1 > 20.0:
                    print("Frame {} - cam frame {} - cam 1 - fish {} - reporj error {}".format(frame, top_subdf["Org Frame"].values[0], fish_ID, err1))
                    cam1_err.append("Frame {} - cam frame {} - cam 1 - fish {} - reporj error {}".format(frame, top_subdf["Org Frame"].values[0], fish_ID, err1))

                if err2 > 20.0:
                    print("Frame {} - cam frame {} - cam 2 - fish {} - reporj error {}".format(frame, front_subdf["Org Frame"].values[0], fish_ID, err2))
                    cam2_err.append("Frame {} - cam frame {} - cam 2 - fish {} - reporj error {}".format(frame, front_subdf["Org Frame"].values[0], fish_ID, err2))
                
                data_dict = {"frame":frame,
                            "id":[fish_ID],
                            "3d_x":[p[0]],
                            "3d_y":[p[1]],
                            "3d_z":[p[2]],
                            "cam1_reproj_err": [err1],
                            "cam2_reproj_err": [err2],
                            "cam1_frame":[top_subdf["Org Frame"].values[0]],
                            "cam1_x": [top_subdf["X"].values[0]],
                            "cam1_y":[top_subdf["Y"].values[0]],
                            "cam1_tl_x": [top_subdf["Upper left corner X"].values[0]],
                            "cam1_tl_y": [top_subdf["Upper left corner Y"].values[0]],
                            "cam1_br_x": [top_subdf["Lower right corner X"].values[0]],
                            "cam1_br_y": [top_subdf["Lower right corner Y"].values[0]],
                            "cam1_occlusion": [top_subdf["Occlusion"].values[0]],
                            "cam2_frame": [front_subdf["Org Frame"].values[0]],
                            "cam2_x": [front_subdf["X"].values[0]],
                            "cam2_y": [front_subdf["Y"].values[0]],
                            "cam2_tl_x": [front_subdf["Upper left corner X"].values[0]],
                            "cam2_tl_y": [front_subdf["Upper left corner Y"].values[0]],
                            "cam2_br_x": [front_subdf["Lower right corner X"].values[0]],
                            "cam2_br_y": [front_subdf["Lower right corner Y"].values[0]],
                            "cam2_occlusion": [front_subdf["Occlusion"].values[0]]
                            }

            df = pd.concat([df, pd.DataFrame.from_dict(data_dict)], ignore_index=True)

    df = df.sort_values(by=["id", "frame"])
    df.to_csv(os.path.join(gt_directory, "annotations_full.csv"), sep=";", index=False)

    with open(os.path.join(gt_directory, "cam1_reproj_erros.txt"), "w") as f:
        f.write("\n".join(cam1_err))
        
    with open(os.path.join(gt_directory, "cam2_reproj_erros.txt"), "w") as f:
        f.write("\n".join(cam2_err))
        


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create 3D annotations")
    ap.add_argument("-d", "--directory", help="Path to video directori", type=str)
    ap.add_argument("-s1", "--syncTop", help="Sync frame for top camera", type=int)
    ap.add_argument("-s2", "--syncFront", help="Sync frame for front camera", type=int)
    args = vars(ap.parse_args())

    create3DAnnotations(args)