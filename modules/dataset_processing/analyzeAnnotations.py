import os
import numpy as np
import pandas as pd
import argparse

def getOcclusions(df):
    """
    Finds all occlusions in the annotated dataset, provided in a pandas dataframe

    Input:
        df: Pandas DataFrame contatining the dataset
    
    Output: 
        occlusion_id_cam: Dictinary containing a dict per fish ID. The dict per fish ID contains two keys (cam1/cam2), each holding a list of tuples, where each tuple is the start and endframe of an occlusion

    """
    
    unique_ids = df["id"].unique()
    id_map = {id_n:idx for idx, id_n in enumerate(unique_ids)}
    
    occlusion_id_cam = {id_n: {"cam1":[], "cam2": []} for id_n in id_map.keys()}

    for id_n in unique_ids:
        id_df = df[df["id"] == id_n]

        for cam in ["cam1", "cam2"]:

            id_df_cam = id_df.dropna(subset=["{}_occlusion".format(cam)])
            
            occlusions = id_df_cam["{}_occlusion".format(cam)].values
            frames = id_df_cam["{}_frame".format(cam)].unique()
            
            occ_tuples = []

            startFrame = -1
            for idx, frame in enumerate(frames[:-1]):
                
                if occlusions[idx] == 0:
                    continue
                
                if startFrame == -1:
                    startFrame = frame
                
                if occlusions[idx+1] == 0:
                    occ_tuples.append((startFrame, frame))
                    startFrame = -1

            occlusion_id_cam[id_n][cam] = occ_tuples

    return occlusion_id_cam


def getBBox(df, cam_key, frame, fid):
    """
    Returns the bounding box of a given fish in a given frame/cam view.

    Input:
        df: Dataset Pandas dataframe
        cam_key: String designating the camera view (cam1 or cam2)
        frame: int indicating the frame which should be looked up
        fid: int indicating the fish ID

    Output:
        Tuple containing the top left x/y coordiante and the bottom right x/y coordinates
    """
    df_f = df[(df["id"] == fid) & (df["{}_frame".format(cam_key)] == frame)]
    return (df_f["{}_tl_x".format(cam_key)].values, df_f["{}_tl_y".format(cam_key)].values, df_f["{}_br_x".format(cam_key)].values, df_f["{}_br_y".format(cam_key)].values)


def checkIntersection(bbox1, bbox2):
    """
    Checks whether two bounding boxes are intersecting

    Input:
        bbox1: List of bounding box coordiantes (top left and bottom right x/y)
        bbox2: List of bounding box coordiantes (top left and bottom right x/y)
    
    Output:
        Boolean
    """
    return  (bbox1[2] >= bbox2[0]) and (bbox1[0] <= bbox2[2]) and (bbox1[3] >= bbox2[1]) and (bbox1[1] <= bbox2[3])


def analyzeOcclusions(df):
    """
    Function that given occlusion data calculates a variety of metrics.

    Input:
        df: Pandas Dataframe containing the annotated dataset

    Output:
        stats_dict: A dict containig statistics on different aspect of occlusions. Contains both dicts (OL, OL-raw, FBO, FBO-raw) and single values (cam1-occ, cam2-occ, cam1-occRatio, cam2-occRatio)
        occlusions_tuple_dict: A dict of dicts containing the occlusion ata per fish per camera view (output from the getOcclusions function)
    """
    
    occlusions_tuple_dict = getOcclusions(df)

    stats_dict = {"OL": {}, "OL-raw": {}, "FBO": {}, "FBO-raw": {}}

    max_frame = {"cam1": np.nanmax(df["cam1_frame"].values),
                 "cam2": np.nanmax(df["cam2_frame"].values)}
                 
    min_frame = {"cam1": np.nanmin(df["cam1_frame"].values),
                 "cam2": np.nanmin(df["cam2_frame"].values)}


    stats_dict["cam1-occ"] = df["cam1_occlusion"].sum()
    stats_dict["cam2-occ"] = df["cam2_occlusion"].sum()
    stats_dict["cam1-occRatio"] = df["cam1_occlusion"].sum() / len(df["cam1_frame"][np.isfinite(df["cam1_frame"])])
    stats_dict["cam2-occRatio"] = df["cam2_occlusion"].sum() / len(df["cam2_frame"][np.isfinite(df["cam2_frame"])])


    for id_key in occlusions_tuple_dict.keys():
        for cam_key in occlusions_tuple_dict[id_key].keys():
            occlusions_tuple_list = occlusions_tuple_dict[id_key][cam_key]
            
            # Get time where the fish is part of an occlusion i.e. amount of consecutive frames with occlusion
            occlusions_length = np.asarray([x[1]-x[0]+1 for x in occlusions_tuple_list]) # end-start + 1, (+1 in order to count the last frame also)


            # Get time between occlusions, i.e. amount of consecutive frames without occlusions
            between_occlusions = [(occlusions_tuple_list[idx+1][0]-1) - (occlusions_tuple_list[idx][1] + 1) +1 for idx in range(len(occlusions_tuple_list[:-1]))]

            if occlusions_tuple_list[-1][1] < max_frame[cam_key]:
                between_occlusions.append(max_frame[cam_key] - (occlusions_tuple_list[-1][1] + 1) +1)
                
            if occlusions_tuple_list[0][0] > min_frame[cam_key]:
                between_occlusions.append((occlusions_tuple_list[0][0]-1) - min_frame[cam_key] +1)
            
            between_occlusions = np.asarray(between_occlusions)
            
            stats_dict["OL"]["{}-{}".format(id_key, cam_key)] = np.mean(occlusions_length)
            stats_dict["OL-raw"]["{}-{}".format(id_key, cam_key)] =  list(occlusions_length)
            stats_dict["FBO"]["{}-{}".format(id_key, cam_key)] =  np.mean(between_occlusions)
            stats_dict["FBO-raw"]["{}-{}".format(id_key, cam_key)] =  list(between_occlusions)


    # Find concurrent occlusions
    for cam_key in ["cam1", "cam2"]:
        occ_list = []
        for id_key in occlusions_tuple_dict.keys():
            occ_list.extend([(x[0], x[1], id_key) for x in occlusions_tuple_dict[id_key][cam_key]])

        occ_list = sorted(occ_list, key=lambda x: x[0])

        n_occ_list = len(occ_list)
        search_horizon = len(df["id"].unique())-1
        event_counter = 0
        events = []

        for idx in range(n_occ_list):
            occ_s = occ_list[idx]

            short_list = [occ_s]

            for idx_future in range(1, search_horizon+1):
                idx_next = idx + idx_future
                if idx_next >= n_occ_list:
                    continue

                occ_n = occ_list[idx_next]

                # Check if occlusion event is already accounted for in an event
                if len(events) > 0 and set(occ_n).intersection(set(map(tuple, events))):
                    continue
                
                # Check if the occlusion sequences start at the same time and the bounding boxes overlap (counts as 1 event! )
                bbox_s = getBBox(df, cam_key, occ_s[0], occ_s[-1])
                bbox_n = getBBox(df, cam_key, occ_s[0], occ_n[-1])

                if occ_s[0] == occ_n[0] and checkIntersection(bbox_s, bbox_n):
                    event_counter += 1
                    short_list.append(occ_n)
            

            # Go through the events list and check if our current occlusion sequence is already accounted for
            # If not go through all events, find the ones temporally overlapping and check if the current occlusion sequence overlap with any.
            # If there is a BBOX overlap we count each element in our short list as an occlusion event. Otherwise we count it as length of the list -1 
            assigned_event = -1
            already_in_events = False
            for event_idx in range(len(events)):
                if [x for x in short_list if x in events[event_idx]]:
                    already_in_events = True
                    break

                for occ in events[event_idx]:
                    if assigned_event != -1:
                        break

                    if occ[0] >= occ_s[0] and occ[1] <= occ_s[1]:
                        bbox_s = getBBox(df, cam_key, occ[0], occ_s[-1])
                        bbox_n = getBBox(df, cam_key, occ[0], occ[-1])   # NOTE LOOKING UP occ FRAME
                        if checkIntersection(bbox_s, bbox_n):
                            assigned_event = event_idx
            
            if assigned_event != -1:
                events[assigned_event].extend(short_list)
                event_counter += (len(short_list)-1)
            elif len(events) == 0 or not already_in_events:
                events.append(short_list)

        stats_dict["{}-events".format(cam_key)] = event_counter        


    return stats_dict, occlusions_tuple_dict


def getOcclusionIntersection(df, occ_tuple_dict):
    """
    Given the annotated dataset and the found occlusion series, this function calcualtes the Intersection Between Occlusion metric

    Input:
        df: Pandas Dataframe containing the annotated dataset
        occ_tuple_dict: A dict of dicts containing the occlusion ata per fish per camera view (output from the getOcclusions function)

    Output:
        occ_intersection_dict: Dict of dicts containing a list of intersection between cclusion values per camera view per fish
    """

    ids = df["id"].unique()

    occ_intersection_dict = {k:{"cam1": [], "cam2": []} for k in ids}
    
    for view in ["cam1", "cam2"]:
        cam_df = df[df["{}_occlusion".format(view)] == 1]
        frames = sorted(cam_df["{}_frame".format(view)].unique())

        for f in frames:
            f_df = cam_df[cam_df["{}_frame".format(view)] == f]

            if len(f_df["id"].unique()) < 2:
                print("Only {} ids with occlsuions in cam {}, ID {}, frame {}. Something is wrong".format(len(f_df["id"].unique()), view, f_df["id"].unique(), f))
                continue
            
            mask = np.zeros((1520, 2704), dtype=np.bool)

            # Generate a mask consisting of all the areas where bboxes are intersecting
            for idx1 in range(len(f_df)):
                id1_df = f_df.iloc[idx1]
                for idx2 in range(idx1+1, len(f_df)):
                    id2_df = f_df.iloc[idx2]

                    max_tl_x = int(max(id1_df["{}_tl_x".format(view)], id2_df["{}_tl_x".format(view)]))
                    max_tl_y = int(max(id1_df["{}_tl_y".format(view)], id2_df["{}_tl_y".format(view)]))
                    min_br_x = int(min(id1_df["{}_br_x".format(view)], id2_df["{}_br_x".format(view)]))
                    min_br_y = int(min(id1_df["{}_br_y".format(view)], id2_df["{}_br_y".format(view)]))

                    intersection_width = min_br_x - max_tl_x + 1
                    intersection_height = min_br_y - max_tl_y + 1

                    intersection_area = max(0, intersection_width) * max(0, intersection_height)

                    if intersection_area == 0:
                        continue

                    mask[max_tl_y:min_br_y+1, max_tl_x:min_br_x+1] = True

            # Per fish ID, determine how much of the bbox is intersected

            for idx in range(len(f_df)):
                id_df = f_df.iloc[idx]
                fish_id = int(id_df["id"])

                tl_x = int(id_df["{}_tl_x".format(view)])
                tl_y = int(id_df["{}_tl_y".format(view)])
                br_x = int(id_df["{}_br_x".format(view)])
                br_y = int(id_df["{}_br_y".format(view)])

                id_intersection = np.sum(mask[tl_y:br_y+1,tl_x:br_x+1])/((br_x-tl_x+1)*(br_y-tl_y+1))


                for occ_idx in range(len(occ_tuple_dict[fish_id][view])):
                    occ_tuple = occ_tuple_dict[fish_id][view][occ_idx]

                    if f >= occ_tuple[0] and f <= occ_tuple[1]:
                        if f == occ_tuple[0]:
                            occ_intersection_dict[fish_id][view].append([id_intersection])
                        else:
                            occ_intersection_dict[fish_id][view][occ_idx].append(id_intersection)
                        
                        if id_intersection == 0.0:
                            print("No intersection for Fish {} in frame {} - {}".format(fish_id, id_df["{}_frame".format(view)], view))

    return occ_intersection_dict



def analyzeAnnotations(args):
    """
    Analyzes the annotated dataset and calcualtes the metrics used for calculating the dataset complexity, and more

    Input:
        directory Path to the root folder containing the GT folder with the annotations
    
    Output:
        Saves a files called "dataset_metrics.txt" in the GT folder under the 'directory' path
    """

    directory = args["directory"]
    gt_directory = os.path.join(directory, "GT")

    ann_df = pd.read_csv(os.path.join(gt_directory, "annotations_full.csv"), sep=";")

    unique_ids = ann_df["id"].unique()
    
    if len(unique_ids) == 1:
        cam1_occ_events = 0
        cam2_occ_events = 0
        stats_dict = {}
        occ_intersection_cam = {}
        stats_dict["OL-macro-cam1"] = 0
        stats_dict["OL-macro-cam2"] = 0
        stats_dict["OL-micro-cam1"] = 0
        stats_dict["OL-micro-cam2"] = 0
        stats_dict["FBO-macro-cam1"] = len(ann_df["cam1_frame"].values)
        stats_dict["FBO-macro-cam2"] = len(ann_df["cam2_frame"].values)
        stats_dict["FBO-micro-cam1"] = len(ann_df["cam1_frame"].values)
        stats_dict["FBO-micro-cam2"] = len(ann_df["cam2_frame"].values)
        occ_intersection_cam["cam1"] = [0]
        occ_intersection_cam["cam2"] = [0]        
        cam1_occ_events = 0
        cam2_occ_events = 0
    else:
        # Determine amount of occlusion events
        stats_dict, occlusions_tuple_dict = analyzeOcclusions(ann_df)

        # Determine how much each bbox is intersected during an occlusion
        occ_intersection_dict = getOcclusionIntersection(ann_df, occlusions_tuple_dict)
        occ_intersection_dict_reduced = {k:{"cam1": None, "cam2": None} for k in occ_intersection_dict.keys()}

        occ_intersection_cam = {"cam1": [], "cam2": []}

        for idx in occ_intersection_dict.keys():
            for view in occ_intersection_dict[idx].keys():
                occ_intersection_dict_reduced[idx][view] = np.mean([np.mean(np.asarray(lst)) for lst in occ_intersection_dict[idx][view]])

                occ_intersection_cam[view].append(occ_intersection_dict_reduced[idx][view])
        # Count occlusoin events (Amount of occlusion series)
        cam1_occ_events = 0
        cam2_occ_events = 0

        for fid in occlusions_tuple_dict.keys():
            cam1_occ_events += len(occlusions_tuple_dict[fid]["cam1"])
            cam2_occ_events += len(occlusions_tuple_dict[fid]["cam2"])

        for cam in ["cam1", "cam2"]:
            OL_macro = [x for k, x in stats_dict["OL"].items() if cam in k]
            FBO_macro = [x for k, x in stats_dict["FBO"].items() if cam in k]
            OL_micro = []
            [OL_micro.extend(x) for k, x in stats_dict["OL-raw"].items() if cam in k]
            FBO_micro = []
            [FBO_micro.extend(x) for k, x in stats_dict["FBO-raw"].items() if cam in k]

            stats_dict["OL-micro-{}".format(cam)] = np.mean(OL_macro)
            stats_dict["OL-macro-{}".format(cam)] = np.mean(OL_micro)
            stats_dict["FBO-micro-{}".format(cam)] = np.mean(FBO_micro)
            stats_dict["FBO-macro-{}".format(cam)] = np.mean(FBO_macro)

    
    with open(os.path.join(gt_directory, "dataset_metrics.txt"), "w") as f: 
        
        f.write("Fish: {}\n".format(len(ann_df["id"].unique())))
        f.write("Frame rate: {}\n".format(60))
        f.write("Amount of frames: {}\n\n".format(len(ann_df["frame"].unique())))
        f.write("Duration (s): {}\n\n".format(len(ann_df["frame"].unique())/60))

        f.write("Complexity metric components (NOTE: THESE ARE NOT PER SECOND!)\n")
        f.write("Average Occlusion Length (frames) - cam 1: {}\n".format(stats_dict["OL-macro-cam1"]))
        f.write("Average Occlusion Length (frames) - cam 2: {}\n".format(stats_dict["OL-macro-cam2"]))
        f.write("Average Frames Between Occlusion (frames) - cam 1: {}\n".format(stats_dict["FBO-macro-cam1"]))
        f.write("Average Frames Between Occlusion (frames) - cam 2: {}\n".format(stats_dict["FBO-macro-cam2"]))
        f.write("Occlusion Count - cam 1: {}\n".format(cam1_occ_events)) 
        f.write("Occlusion Count - cam 2: {}\n".format(cam2_occ_events))
        f.write("Intersection Between Occlusions - cam 1: {}\n".format(np.mean(occ_intersection_cam["cam1"])))
        f.write("Intersection Between Occlusions - cam 2: {}\n".format(np.mean(occ_intersection_cam["cam2"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyze 3D annotations")
    ap.add_argument("-d", "--directory", help="Path to video directory", type=str)
    args = vars(ap.parse_args())

    analyzeAnnotations(args)