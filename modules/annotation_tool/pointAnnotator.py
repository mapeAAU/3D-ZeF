import cv2
import numpy as np
import pandas as pd
import argparse
import os

clickEvent = False
mouseX, mouseY = None, None
frame_crop = None

def click(event, x, y, flags, param):
	# grab references to the global variables
    global mouseX, mouseY, clickEvent, frame_crop
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX,mouseY = x,y
        clickEvent = True
        cv2.circle(frame_crop,(x,y),100,(255,0,0),-1)

def saveAnnotationFile(df, filename):
    """
    Saves the provided df at the provided path

    Input:
        df: Pandas dataframe
        filename: Path to the output file
    """

    df = df.sort_values(by=["Filename"])
    df.to_csv(filename, sep=";", index=False)


def pointAnnotator(args):
    """
    Point annotatio ntool. based on the bounding boxes from the AAU Bounding Box annotation tool. 

    Input:
        directory: Root folder containing the GT fodler and the "Top" or "Front" folders containing the extracted frames
        camView: String designated which view is being annotated
        startFrame: The frame which the tool should start at
        fishID: The fish ID which is being annotated
        bboxPadding: How much extra than the bbox should be shown
    """

    global clickEvent, frame_crop

    directory = args["directory"]
    cam_view = args["camView"]

    frame_numb = args["startFrame"]
    fishID = args["fishID"]
    bb_padding = args["bboxPadding"]
    frames = 7190
    
    active = True

    # Prepare  the different paths to point and bbox annotation files, and the extracted camerea frames
    point_filename = os.path.join(directory, "GT", "point_annotations_"+cam_view.lower() + "_" + str(fishID) + ".csv")
    images = os.path.join(directory, cam_view)
    bbFile = os.path.join(directory, "GT", "bbox_annotations_"+cam_view.lower()+".csv")
    print(point_filename)


    # Read bboxes
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

    cv2.namedWindow('frame')
    cv2.setMouseCallback("frame", click)

    while(active):
        frame_id = str(frame_numb).zfill(6) + ".png"

        bb_series = bb_df_id[bb_df_id["Filename"] == frame_id]
		
        if frame_numb > frames:
            break
        print(frame_numb)

        if len(bb_series) == 0:
            frame_numb += 1
            continue

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

            cv2.circle(frame,(x,y),10,(255,0,0),1)
            cv2.circle(frame,(x,y),1,(255,0,0),-1)

        # Crop out the ROI and show image
        frame_crop = frame[bb_tly_min:bb_tly_max, bb_tlx_min:bb_tlx_max]
        cv2.imshow("frame", frame_crop)


        # Wait for click event. At the same time wait for key events
        while(not clickEvent):
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
                saveAnnotationFile(point_df, point_filename)

        
        # On click event add new point to point df. Removes previous annotation for the same id and frame filename if ti exists
        if clickEvent:

            point_df = point_df[point_df["Filename"] != frame_id]

            ann_dict = {"Filename":frame_id, "Object ID": [fishID], "X": [mouseX+bb_tlx_min], "Y":[mouseY+bb_tly_min]}
            point_df = pd.concat([point_df, pd.DataFrame.from_dict(ann_dict)], ignore_index=True)

            frame_numb += 1
            clickEvent = False    
            print(mouseX, mouseY)
            print(ann_dict)
            print(bb_series)

    saveAnnotationFile(point_df, point_filename)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make point annotation for specific fish in video from specified camera view")
    ap.add_argument("-d", "--directory", help="Path to video directori", type=str)
    ap.add_argument("-c", "--camView", help="Camera view", type=str)
    ap.add_argument("-sf", "--startFrame", help="Start frame", type=int, default = 1)
    ap.add_argument("-bb", "--bboxPadding", help="Padding added to the ROI", type=int, default=50)
    ap.add_argument("-fid", "--fishID", help="Fish ID", type=int, default=1)
    args = vars(ap.parse_args())

    pointAnnotator(args)