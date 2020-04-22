import cv2
import numpy as np
import pandas as pd
import argparse
import os

import sys
sys.path.append('../../')
from common.utility import prepareCams
from modules.reconstruction.Triangulate import Triangulate


clickEvent_top = False
mouseX_top, mouseY_top = None, None
clickEvent_front = False
mouseX_front, mouseY_front = None, None

def click_front(event, x, y, flags, param):
	# grab references to the global variables
    global mouseX_front, mouseY_front, clickEvent_front
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX_front,mouseY_front = x,y
        clickEvent_front = True


def click_top(event, x, y, flags, param):
	# grab references to the global variables
    global mouseX_top, mouseY_top, clickEvent_top
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX_top,mouseY_top = x,y
        clickEvent_top = True


def saveAnnotationFile(df, filename):
    """
    Saves the provided df at the provided path

    Input:
        df: Pandas dataframe
        filename: Path to the output file
    """
    df = df.sort_values(by=["Filename"])
    df.to_csv(filename, sep=";", index=False)
    print("Saved: {}".format(filename))


def setupFiles(view, directory, fishID):
    """
    Retreives all the annoation and image files for the provided fishID and camera view

    Input:
        directory: Path to the root folder
        fishID: ID of the fish which is being annotated
    """

    point_filename = os.path.join(directory, "GT", "point_annotations_"+view+"_" + str(fishID) + ".csv")
    images = os.path.join(directory, view)
    bbFile = os.path.join(directory, "GT", "bbox_annotations_"+view+".csv")

    bb_df = pd.read_csv(bbFile, sep=";")
    
    bb_df["Width"] = bb_df["Lower right corner X"] - bb_df["Upper left corner X"]
    bb_df["Height"] = bb_df["Lower right corner Y"] - bb_df["Upper left corner Y"]

    bb_max_w = int(bb_df["Width"].max())
    bb_max_h = int(bb_df["Height"].max())
    bb_df_id = bb_df[bb_df["Object ID"] == fishID]


    if os.path.isfile(point_filename):
        point_df = pd.read_csv(point_filename, sep=";")
    else:
        point_df = pd.DataFrame(columns=["Filename", "Object ID", "X", "Y"])

    return point_filename, images, bb_df_id, point_df, (bb_max_w, bb_max_h)


def pointCorrection(args):
    """
    Function used to correct point annotations wher ethe reproejction error is too large. Shows both caemra views, and reports the reproejction error when provided new points.

    Input:
        directory: Root path containing the GT and top/front image folders
        synTop: sync frame for the top view
        syncFront: sync frame for the front view
        startFrame: Frame which the annotation corrector starts at
        fishID: The id of the fish which annotations should be corrected
        bboxPadding: How much the bounding box annotations should be corrected
    """

    global clickEvent_top, clickEvent_front

    directory = args["directory"]
    sync_top = args["syncTop"]#"top"
    sync_front = args["syncFront"]#"front"
    print(args)

    cam1_offset = max(0,sync_front-sync_top)
    cam2_offset = max(0,sync_top-sync_front)     
    print("Front", sync_front, cam2_offset)
    print("Top", sync_top, cam1_offset)

    frame_numb = args["startFrame"]
    fishID = args["fishID"]
    bb_padding = args["bboxPadding"]
    frames = 7190

    point_filename_front, images_front, bb_df_id_front, point_df_front, front_bb_max = setupFiles("front", directory, fishID)
    point_filename_top, images_top, bb_df_id_top, point_df_top, top_bb_max = setupFiles("top", directory, fishID)

    cams = prepareCams(directory)
    tr = Triangulate()

    active = True

    cv2.namedWindow('top')
    cv2.setMouseCallback("top", click_top)
    cv2.namedWindow('front')
    cv2.setMouseCallback("front", click_front)

    while(active):
        points = {"top": [], "front": []}
        print()
        for view_tuple in [(bb_df_id_top, point_df_top, cam1_offset, "top", images_top, top_bb_max), (bb_df_id_front, point_df_front, cam2_offset, "front", images_front, front_bb_max)]:
            
            bb_df_id, point_df, offset, view, images, bb_max = view_tuple
            bb_max_w, bb_max_h = bb_max
            frame_id = str(frame_numb-offset).zfill(6) + ".png"
            print(offset)
			
            if frame_numb > frames:
                break
            print(view, frame_id)
			
            bb_series = bb_df_id[bb_df_id["Filename"] == frame_id]
            print(bb_series)
            

            frame = cv2.imread(os.path.join(images, frame_id))

            ## Draw bounding box
            bb_tlx = int(bb_series["Upper left corner X"])
            bb_tly = int(bb_series["Upper left corner Y"])
            bb_w = int(bb_series["Lower right corner X"]) - bb_tlx
            bb_h = int(bb_series["Lower right corner Y"]) - bb_tly

            width_diff = bb_max_w - bb_w
            height_diff = bb_max_h - bb_h

            bb_tlx_min = max(0, bb_tlx - width_diff//2 - bb_padding)
            bb_tly_min = max(0, bb_tly - height_diff//2 - bb_padding)
            bb_tlx_max = min(2704, bb_tlx + bb_w + width_diff//2 + bb_padding)
            bb_tly_max = min(1520, bb_tly + bb_h + height_diff//2 + bb_padding)

            cv2.rectangle(frame, (bb_series["Upper left corner X"],bb_series["Upper left corner Y"]),(bb_series["Lower right corner X"],bb_series["Lower right corner Y"]),(255,0,0), 1)

            ## Draw current poitn annotation, if it exists
            point_tmp = point_df[point_df["Filename"] == frame_id]
            point_series = point_tmp[point_tmp["Object ID"] == fishID]

            if len(point_series) > 0:
                x = point_series["X"]
                y = point_series["Y"]
                points[view] = [x,y]

                cv2.circle(frame,(x,y),10,(255,0,0),1)
                cv2.circle(frame,(x,y),1,(255,0,0),-1)

            # Crop out the ROI and show image
            points[view+"_frame"] = [frame, [bb_tly_min, bb_tly_max, bb_tlx_min, bb_tlx_max]]
        
        # calc reproj
        point_front = points["front"]
        point_top = points["top"]
        front_frame, front_bbs = points["front_frame"]
        top_frame, top_bbs = points["top_frame"]
		
        t1Pt = [float(point_top[0]), float(point_top[1])]
        t2Pt = [float(point_front[0]), float(point_front[1])]

        # 1) Triangulate 3D point
        p,d = tr.triangulatePoint(t1Pt, t2Pt, cams[1], cams[2], correctRefraction=True)

        p1 = cams[1].forwardprojectPoint(*p)
        p2 = cams[2].forwardprojectPoint(*p)

        # 2) Calc re-projection errors
        pos1 = np.array(t1Pt)
        err1 = np.linalg.norm(pos1-p1)
        pos2 = np.array(t2Pt)
        err2 = np.linalg.norm(pos2-p2)
        
        
        cv2.circle(top_frame,(int(p1[0]), int(p1[1])),10,(0,255,0),1)
        cv2.circle(front_frame,(int(p2[0]), int(p2[1])),10,(0,255,0),1)

        front_frame = front_frame[front_bbs[0]:front_bbs[1], front_bbs[2]:front_bbs[3]]
        top_frame = top_frame[top_bbs[0]:top_bbs[1], top_bbs[2]:top_bbs[3]]
        cv2.putText(top_frame,"{:.4f}".format(err1), 
                                ((50,50)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1,
                                (0,0,0),
                                2)
        
        cv2.putText(front_frame,"{:.4f}".format(err2), 
                                ((50,50)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1,
                                (0,0,0),
                                2)
        cv2.imshow("front", front_frame)
        cv2.imshow("top", top_frame)



        # Wait for click event. At the same time wait for key events
        while(not (clickEvent_top or clickEvent_front)):
            k = cv2.waitKey(1) & 0xFF

            if k == ord("e") or k == ord("E"):
                active = False
                break

            elif k == ord("a") or k == ord("A"):
                frame_numb -= 1
                break

            elif k == ord("d") or k == ord("D"):
                frame_numb += 1
                break

            elif k == ord("s") or k == ord("S"):
                saveAnnotationFile(point_df_top, point_filename_top)
                saveAnnotationFile(point_df_front, point_filename_front)

        
        # On click event add new point to point df. Removes previous annotation for the same id and frame filename if ti exists
        if clickEvent_top:

            frame_id = str(frame_numb-cam1_offset).zfill(6) + ".png"
            point_df_top = point_df_top[point_df_top["Filename"] != frame_id]

            ann_dict = {"Filename":frame_id, "Object ID": [fishID], "X": [mouseX_top+top_bbs[2]], "Y":[mouseY_top+top_bbs[0]]}
            point_df_top = pd.concat([point_df_top, pd.DataFrame.from_dict(ann_dict)], ignore_index=True)

            clickEvent_top = False    

			
        if clickEvent_front:

            frame_id = str(frame_numb-cam2_offset).zfill(6) + ".png"
            point_df_front = point_df_front[point_df_front["Filename"] != frame_id]
			
            ann_dict = {"Filename":frame_id, "Object ID": [fishID], "X": [mouseX_front+front_bbs[2]], "Y":[mouseY_front+front_bbs[0]]}
            point_df_front = pd.concat([point_df_front, pd.DataFrame.from_dict(ann_dict)], ignore_index=True)

            clickEvent_front = False    


    saveAnnotationFile(point_df_top, point_filename_top)
    saveAnnotationFile(point_df_front, point_filename_front)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make corrections to the point annotation for specific fish in video")
    ap.add_argument("-d", "--directory", help="Path to video directory", type=str)
    ap.add_argument("-s1", "--syncTop", help="Sync frame for the to pview", type=int)
    ap.add_argument("-s2", "--syncFront", help="Sync frame for the front view", type=int)
    ap.add_argument("-sf", "--startFrame", help="Start frame", type=int, default = 1)
    ap.add_argument("-bb", "--bboxPadding", help="Padding added to the ROI", type=int, default=50)
    ap.add_argument("-fid", "--fishID", help="Fish ID", type=int, default=1)
    args = vars(ap.parse_args())

    pointCorrection(args)