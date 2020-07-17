import os
import sys
import argparse
import motmetrics as mm
import numpy as np
import pandas as pd


def get3Dpos(df, cam, task):
    """
    Returns the 3D position in a dataset

    Input:
        df: Pandas dataframe
        cam: Camera view
        task: Not used

    Output:
        pos: Numpy array of size [n_ids, 3] containing the 3d position
        ids: List of IDs
    """
    ids = df["id"].unique()
    ids = [int(x) for x in ids]

    pos = np.zeros((len(ids), 3))
    for idx, identity in enumerate(ids):
        df_id = df[df["id"] == identity]
        pos[idx,0] = df_id["3d_x"]
        pos[idx,1] = df_id["3d_y"]
        pos[idx,2] = df_id["3d_z"]
    
    return pos, ids


def get2Dpos(df, cam, task):
    """
    Returns the 2D position in a dataset. Depending on the set task it either returns the positon of the designated keypoint or of the bounding box center

    Input:
        df: Pandas dataframe
        cam: Camera view
        task: if 'bbox' return the bbox center, else return the keypoint x and y coordinates

    Output:
        pos: Numpy array of size [n_ids, 3] containing the 3d position
        ids: List of IDs
    """
    ids = df["id"].unique()
    ids = [int(x) for x in ids]

    pos = np.zeros((len(ids), 2))
    for idx, identity in enumerate(ids):
        df_id = df[df["id"] == identity]

        if "bbox" in task:
            pos[idx,0] = df_id[cam+"_x_bb"]
            pos[idx,1] = df_id[cam+"_y_bb"]
        else:
            pos[idx,0] = df_id[cam+"_x"]
            pos[idx,1] = df_id[cam+"_y"]
    
    return pos, ids


def getBbox(df, cam, task):
    """
    Returns the bbox coordiantes in a dataset

    Input:
        df: Pandas dataframe
        cam: Camera view
        task: Not used

    Output:
        pos: Numpy array of size [n_ids, 4] containing the bounding box top left x/y and bottom right x/y coordinates
        ids: List of IDs
    """
    ids = df["id"].unique()
    ids = [int(x) for x in ids]

    pos = np.zeros((len(ids), 4))
    for idx, identity in enumerate(ids):
        df_id = df[df["id"] == identity]

        pos[idx,0] = df_id[cam+"_bb_tl_x"]
        pos[idx,1] = df_id[cam+"_bb_tl_y"]
        pos[idx,2] = df_id[cam+"_bb_br_x"]
        pos[idx,3] = df_id[cam+"_bb_br_y"]

    return pos, ids


def pairwiseDistance(X,Y, maxDist):
    """
    X and Y are n x d and m x d matrices, where n and m are the amount of observations, and d is the dimensionality of the observations
    """

    X_ele, X_dim = X.shape
    Y_ele, Y_dim = Y.shape

    assert X_dim == Y_dim, "The two provided matrices not have observations of the same dimensionality"

    mat = np.zeros((X_ele, Y_ele))

    for row, posX in enumerate(X):
        for col, posY in enumerate(Y):
            mat[row, col] = np.linalg.norm(posX-posY)

    mat[mat > maxDist] = np.nan

    return mat  


def IoU(X, Y):
    """
    Calculates the Intersection over Union for two bounding boxes, X and Y, which are 4D vectos containing the top left and bottom right positions of the bounding box
    """

    max_tl_x = max(X[0], Y[0])
    max_tl_y = max(X[1], Y[1])
    min_br_x = min(X[2], Y[2])
    min_br_y = min(X[3], Y[3])

    intersection_area = max(0, min_br_x - max_tl_x + 1) * max(0, min_br_y - max_tl_y + 1)

    X_area = (X[2]-X[0]+1)*(X[3]-X[1]+1)
    Y_area = (Y[2]-Y[0]+1)*(Y[3]-Y[1]+1)

    return intersection_area / (float(X_area + Y_area - intersection_area))


def IoUDistance(X, Y, maxDist):
    """
    X and Y are n x d and m x d matrices, where n and m are the amount of observations, and d is the dimensionality of the observations
    """

    X_ele, X_dim = X.shape
    Y_ele, Y_dim = Y.shape

    assert X_dim == Y_dim, "The two provided matrices not have observations of the same dimensionality"

    mat = np.zeros((X_ele, Y_ele))

    for row, bboxX in enumerate(X):
        for col, bboxY in enumerate(Y):
            mat[row, col] = 1 - IoU(bboxX, bboxY)

    mat[mat > maxDist] = np.nan

    return mat  


def writeMOTtoTXT(metrics, output_file, cat):
    '''
    Writes the provided MOT + MTBF metrics results for a given cateogry to a txt file

    Input:
        - metrics: pandas DF containing the different metrics
        - output-file: Path to the output file
        - cat: Cateogry which has been analyzed
    '''

    with open(output_file, "w") as f:
            f.write("Category:{}\n".format(cat))
            f.write("MOTA = {:.1%}\n".format(metrics["mota"].values[0]))
            f.write("MOTAL = {:.1%}\n".format(metrics["motal"].values[0]))
            f.write("MOTP = {:.3f}\n".format(metrics["motp"].values[0]))
            f.write("Precision = {:.1%}\n".format(metrics["precision"].values[0]))
            f.write("Recall = {:.1%}\n".format(metrics["recall"].values[0]))
            f.write("ID Recall = {:.1%}\n".format(metrics["idr"].values[0]))
            f.write("ID Precision = {:.1%}\n".format(metrics["idp"].values[0]))
            f.write("ID F1-score = {:.1%}\n".format(metrics["idf1"].values[0]))
            f.write("Mostly Tracked = {:d}\n".format(metrics["mostly_tracked"].values[0]))
            f.write("Partially Tracked = {:d}\n".format(metrics["partially_tracked"].values[0]))
            f.write("Mostly Lost = {:d}\n".format(metrics["mostly_lost"].values[0]))
            f.write("False Positives = {:d}\n".format(metrics["num_false_positives"].values[0]))
            f.write("False Negatives = {:d}\n".format(metrics["num_misses"].values[0]))
            f.write("Identity Swaps = {:d}\n".format(metrics["num_switches"].values[0]))
            f.write("Fragments = {:d}\n".format(metrics["num_fragmentations"].values[0]))
            f.write("MTBF-std = {:.3f}\n".format(metrics["mtbf_std"].values[0]))
            f.write("MTBF-mono = {:.3f}\n".format(metrics["mtbf_mono"].values[0]))
            f.write("Total GT points = {:d}\n".format(metrics["num_objects"].values[0]))
            f.write("Total Detected points = {:d}\n".format(metrics["num_predictions"].values[0]))
            f.write("Number of frames = {:d}\n".format(metrics["num_frames"].values[0]))
            f.write("\n\n")
    
    header = ["MOTA", "MOTAL", "MOTP", "Precision", "Recall", "ID Recall", "ID Precision", "ID F1-score", "Mostly Tracked", "Partially Tracked", "Mostly Lost", "False Positives", "False Negatives", "Identity Swaps", "Fragments", "MTBF-std", "MTBF-mono"]
    value = ["{:.1%}".format(metrics["mota"].values[0]),
             "{:.1%}".format(metrics["motal"].values[0]),
             "{:.3f}".format(metrics["motp"].values[0]),
             "{:.1%}".format(metrics["precision"].values[0]),
             "{:.1%}".format(metrics["recall"].values[0]),
             "{:.1%}".format(metrics["idr"].values[0]),
             "{:.1%}".format(metrics["idp"].values[0]),
             "{:.1%}".format(metrics["idf1"].values[0]),
             "{:d}".format(metrics["mostly_tracked"].values[0]),
             "{:d}".format(metrics["partially_tracked"].values[0]),
             "{:d}".format(metrics["mostly_lost"].values[0]),
             "{:d}".format(metrics["num_false_positives"].values[0]),
             "{:d}".format(metrics["num_misses"].values[0]),
             "{:d}".format(metrics["num_switches"].values[0]),
             "{:d}".format(metrics["num_fragmentations"].values[0]),
             "{:.3f}".format(metrics["mtbf_std"].values[0]),
             "{:.3f}".format(metrics["mtbf_mono"].values[0])]
    
    with open(output_file[:-4] + ".tex", "w") as f:
        f.write(" & ".join(header) + "\n")
        f.write(" & ".join(value))
        
    with open(output_file[:-4] + ".csv", "w") as f:
        f.write(";".join(header) + "\n")
        f.write(";".join(value) )

    
                

def calcMetrics(acc, task):
    """
    Calcualtes all the relevant metrics for the dataset for the specified task

    Input:
        acc: MOT Accumulator oject
        task: String designating the task which has been evaluated

    Output:
        summary: Pandas dataframe containing all the metrics
    """
    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc], 
        metrics=mm.metrics.motchallenge_metrics+["num_objects"]+["num_predictions"]+["num_frames"], 
        names=[task])

    summary["motal"] = MOTAL(summary)
    mtbf_metrics = MTBF(acc.mot_events)
    summary["mtbf_std"] = mtbf_metrics[0]
    summary["mtbf_mono"] = mtbf_metrics[1]
    print(summary)

    formatter = {**mh.formatters, "num_objects": '{:d}'.format, "num_predictions": '{:d}'.format, "num_frames": "{:d}".format, "motal": '{:.1%}'.format, "mtbf_std": '{:.3f}'.format, "mtbf_mono": '{:.3f}'.format}
    namemap = {**mm.io.motchallenge_metric_names, "num_objects": "# GT", "num_predictions" : "# Dets", "num_frames" : "# Frames", "motal": "MOTAL", "mtbf_std": "MTBF-std", "mtbf_mono": "MTBF-mono"}

    strsummary = mm.io.render_summary(
        summary, 
        formatters=formatter, 
        namemap= namemap
    )
    print(strsummary)
    print(acc.mot_events)

    return summary


def MOTAL(metrics):
    """
    Calcualtes the MOTA variation where the amount of id switches is attenuated by using hte log10 function
    """
    return 1 - (metrics["num_misses"] + metrics["num_false_positives"] + np.log10(metrics["num_switches"]+1)) / metrics["num_objects"]


def MTBF(events):
    """
    Calclautes the Mean Time Betwwen Failures (MTBF) Metric from the motmetric events dataframe

    Input:
        events: Pandas Dataframe structured as per the motmetrics package

    Output:
        MTBF_standard: The Standard MTBF metric proposed in the original paper
        MTBF_monotonic: The monotonic MTBF metric proposed in the original paper
    """

    unique_gt_ids = events.OId.unique()
    seqs = []
    null_seqs = []
    for gt_id in unique_gt_ids:
        gt_events = events[events.OId == gt_id]

        counter = 0
        null_counter = 0

        for _, row in gt_events.iterrows():
            if row["Type"] == "MATCH":
                counter += 1
            elif row["Type"] == "SWITCH":
                seqs.append(counter)
                counter = 1
            else:
                seqs.append(counter)
                counter = 0
                null_counter = 1

            if counter > 0:
                if null_counter > 0:
                    null_seqs.append(null_counter)
                    null_counter = 0
        
        if counter > 0:
            seqs.append(counter)
        if null_counter > 0:
            null_seqs.append(null_counter)
    
    seqs = np.asarray(seqs)
    seqs = seqs[seqs>0]

    if len(seqs) == 0:
        return (0, 0)
    else:
        return (sum(seqs)/len(seqs), sum(seqs)/(len(seqs)+len(null_seqs)))


def MOT_Evaluation(args):
    """

    Given a ground annotated dataset and a set of predictions, calcualtes the full suite of MOT metriccs + the MTBF metrics

    Input:
        detCSV: Path to tracking CSV file. 
        gtCSV: Path to ground truth CSV file
        task: What kind of tracking to investigate [3D, cam1, cam2, cam1_bbox, cam2_bbox]
        bboxCenter: Use the bbox center instead of the head position for cam1 and cam2 tasks
        thresh: Distance threshold
        outputFile: Name of the output file
        outputPath: Path to where the output file should be saved
    """
    
    gt_csv = args["gtCSV"]
    det_csv = args["detCSV"]
    task = args["task"]
    bboxCenter = args["bboxCenter"]
    useMOTFormat = args["useMOTFormat"]
    maxDist = args["thresh"]
    outputDir = args["outputPath"]
    outputFile = args["outputFile"]

    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    if useMOTFormat:
        gt_df = pd.read_csv(gt_csv, sep=",", header=None, usecols=[0,1,2,3,4], names=['frame','id','3d_x','3d_y','3d_z'])
    else:
        gt_df = pd.read_csv(gt_csv, sep=";")
    det_df = pd.read_csv(det_csv, sep=",")

    if task == "3D":
        posFunc = get3Dpos
        distFunc = pairwiseDistance
        cam = "both"
        gt_frame_col = "frame"
        det_frame_col = "frame"

        gt_df = gt_df.dropna(subset=["3d_x", "3d_y", "3d_z"])
        det_df = det_df[(det_df["3d_x"] > 0)  &  (det_df["3d_y"] > 0)  &  (det_df["3d_z"] > 0)]

    elif task == "cam1":
        posFunc = get2Dpos
        distFunc = pairwiseDistance
        cam = "cam1"
        gt_frame_col = "cam1_frame"

        if "cam1_frame" in det_df.columns:
            det_frame_col = "cam1_frame"
        else:
            det_frame_col = "frame"


        if bboxCenter:
            task += "-bbox_center"

            gt_df["cam1_x_bb"] = gt_df["cam1_tl_x"] + (gt_df["cam1_br_x"]-gt_df["cam1_tl_x"])//2
            gt_df["cam1_y_bb"] = gt_df["cam1_tl_y"] + (gt_df["cam1_br_y"]-gt_df["cam1_tl_y"])//2
            gt_df = gt_df.dropna(subset=["cam1_x_bb", "cam1_y_bb"])
            
            if "cam1_aa_tl_x" in det_df.columns:
                det_df = det_df[(det_df["cam1_aa_tl_x"] > 0)  &  (det_df["cam1_aa_w"] > 0)  &  (det_df["cam1_aa_tl_y"] > 0) &  (det_df["cam1_aa_h"] > 0)]
                det_df["cam1_x_bb"] = det_df["cam1_aa_tl_x"] + det_df["cam1_aa_w"]//2
                det_df["cam1_y_bb"] = det_df["cam1_aa_tl_y"] + det_df["cam1_aa_h"]//2
            else:
                det_df = det_df[(det_df["aa_tl_x"] > 0)  &  (det_df["aa_w"] > 0)  &  (det_df["aa_tl_y"] > 0) &  (det_df["aa_h"] > 0)]
                det_df["cam1_x_bb"] = det_df["aa_tl_x"] + det_df["aa_w"]//2
                det_df["cam1_y_bb"] = det_df["aa_tl_y"] + det_df["aa_h"]//2
            det_df = det_df.dropna(subset=["cam1_x_bb", "cam1_y_bb"])

        else:
            gt_df = gt_df.dropna(subset=["cam1_x", "cam1_y"])

            if "cam1_x" not in det_df.columns:
                det_df["cam1_x"] = det_df["x"]
                det_df["cam1_y"] = det_df["y"]
            det_df = det_df[(det_df["cam1_x"] > 0)  &  (det_df["cam1_y"] > 0)]
            det_df = det_df.dropna(subset=["cam1_x", "cam1_y"])

    elif task =="cam2":
        posFunc = get2Dpos
        distFunc = pairwiseDistance
        cam = "cam2"
        gt_frame_col = "cam2_frame"

        if "cam2_frame" in det_df.columns:
            det_frame_col = "cam2_frame"
        else:
            det_frame_col = "frame"

        if bboxCenter:    
            task += "-bbox_center"

            gt_df["cam2_x_bb"] = gt_df["cam2_tl_x"] + (gt_df["cam2_br_x"]-gt_df["cam2_tl_x"])//2
            gt_df["cam2_y_bb"] = gt_df["cam2_tl_y"] + (gt_df["cam2_br_y"]-gt_df["cam2_tl_y"])//2
            gt_df = gt_df.dropna(subset=["cam2_x_bb", "cam2_y_bb"])

            if "cam2_aa_tl_x" in det_df.columns:
                det_df = det_df[(det_df["cam2_aa_tl_x"] > 0)  &  (det_df["cam2_aa_w"] > 0)  &  (det_df["cam2_aa_tl_y"] > 0) &  (det_df["cam2_aa_h"] > 0)]
                det_df["cam2_x_bb"] = det_df["cam2_aa_tl_x"] + det_df["cam2_aa_w"]//2
                det_df["cam2_y_bb"] = det_df["cam2_aa_tl_y"] + det_df["cam2_aa_h"]//2
            else:
                det_df = det_df[(det_df["aa_tl_x"] > 0)  &  (det_df["aa_w"] > 0)  &  (det_df["aa_tl_y"] > 0) &  (det_df["aa_h"] > 0)]
                det_df["cam2_x_bb"] = det_df["aa_tl_x"] + det_df["aa_w"]//2
                det_df["cam2_y_bb"] = det_df["aa_tl_y"] + det_df["aa_h"]//2
            det_df = det_df.dropna(subset=["cam2_x_bb", "cam2_y_bb"])

        else:
            gt_df = gt_df.dropna(subset=["cam2_x", "cam2_y"])

            if "cam2_x" not in det_df.columns:
                det_df["cam2_x"] = det_df["x"]  
                det_df["cam2_y"] = det_df["y"]
            det_df = det_df[(det_df["cam2_x"] > 0)  &  (det_df["cam2_y"] > 0)]
            det_df = det_df.dropna(subset=["cam2_x", "cam2_y"])

    elif task == "cam1_bbox":
        posFunc = getBbox
        distFunc = IoUDistance
        cam = "cam1"
        gt_frame_col = "cam1_frame"
        if "cam1_frame" in det_df.columns:
            det_frame_col = "cam1_frame"
        else:
            det_frame_col = "frame"

        gt_df["cam1_bb_tl_x"]= gt_df["cam1_tl_x"]
        gt_df["cam1_bb_tl_y"]= gt_df["cam1_tl_y"]
        gt_df["cam1_bb_br_x"]= gt_df["cam1_br_x"]
        gt_df["cam1_bb_br_y"]= gt_df["cam1_br_y"]

        gt_df = gt_df.dropna(subset=["cam1_bb_tl_x", "cam1_bb_tl_y", "cam1_bb_br_x", "cam1_bb_br_y"])
        
        if "cam1_aa_tl_x" in det_df.columns:
            det_df = det_df[(det_df["cam1_aa_tl_x"] > 0)  &  (det_df["cam1_aa_tl_y"] > 0)  &  (det_df["cam1_aa_w"] > 0) &  (det_df["cam1_aa_h"] > 0)]
            det_df["cam1_bb_tl_x"] = det_df["cam1_aa_tl_x"]
            det_df["cam1_bb_tl_y"] = det_df["cam1_aa_tl_y"]
            det_df["cam1_bb_br_x"] = det_df["cam1_aa_tl_x"] + det_df["cam1_aa_w"]
            det_df["cam1_bb_br_y"] = det_df["cam1_aa_tl_y"] + det_df["cam1_aa_h"]
        else:
            det_df = det_df[(det_df["aa_tl_x"] > 0)  &  (det_df["aa_tl_y"] > 0)  &  (det_df["aa_w"] > 0) &  (det_df["aa_h"] > 0)]
            det_df["cam1_bb_tl_x"] = det_df["aa_tl_x"]
            det_df["cam1_bb_tl_y"] = det_df["aa_tl_y"]
            det_df["cam1_bb_br_x"] = det_df["aa_tl_x"] + det_df["aa_w"]
            det_df["cam1_bb_br_y"] = det_df["aa_tl_y"] + det_df["aa_h"]

        det_df = det_df.dropna(subset=["cam1_bb_tl_x", "cam1_bb_tl_y", "cam1_bb_br_x", "cam1_bb_br_y"])

    elif task =="cam2_bbox":
        posFunc = getBbox
        distFunc = IoUDistance
        cam = "cam2"
        gt_frame_col = "cam2_frame"

        if "cam2_frame" in det_df.columns:
            det_frame_col = "cam2_frame"
        else:
            det_frame_col = "frame"

        gt_df["cam2_bb_tl_x"]= gt_df["cam2_tl_x"]
        gt_df["cam2_bb_tl_y"]= gt_df["cam2_tl_y"]
        gt_df["cam2_bb_br_x"]= gt_df["cam2_br_x"]
        gt_df["cam2_bb_br_y"]= gt_df["cam2_br_y"]

        gt_df = gt_df.dropna(subset=["cam2_bb_tl_x", "cam2_bb_tl_y", "cam2_bb_br_x", "cam2_bb_br_y"])
        
        if "cam2_aa_tl_x" in det_df.columns:
            det_df = det_df[(det_df["cam2_aa_tl_x"] > 0)  &  (det_df["cam2_aa_tl_y"] > 0)  &  (det_df["cam2_aa_w"] > 0) &  (det_df["cam2_aa_h"] > 0)]
            det_df["cam2_bb_tl_x"] = det_df["cam2_aa_tl_x"]
            det_df["cam2_bb_tl_y"] = det_df["cam2_aa_tl_y"]
            det_df["cam2_bb_br_x"] = det_df["cam2_aa_tl_x"] + det_df["cam2_aa_w"]
            det_df["cam2_bb_br_y"] = det_df["cam2_aa_tl_y"] + det_df["cam2_aa_h"]
        else:
            det_df = det_df[(det_df["aa_tl_x"] > 0)  &  (det_df["aa_tl_y"] > 0)  &  (det_df["aa_w"] > 0) &  (det_df["aa_h"] > 0)]
            det_df["cam2_bb_tl_x"] = det_df["aa_tl_x"]
            det_df["cam2_bb_tl_y"] = det_df["aa_tl_y"]
            det_df["cam2_bb_br_x"] = det_df["aa_tl_x"] + det_df["aa_w"]
            det_df["cam2_bb_br_y"] = det_df["aa_tl_y"] + det_df["aa_h"]

        det_df = det_df.dropna(subset=["cam2_bb_tl_x", "cam2_bb_tl_y", "cam2_bb_br_x", "cam2_bb_br_y"])

    else:
        print("Invalid task passed: {}".format(task))
        sys.exit()


    gt_frames = gt_df[gt_frame_col].unique()
    det_frames = det_df[det_frame_col].unique()

    gt_frames = [int(x) for x in gt_frames]
    det_frames = [int(x) for x in det_frames]

    frames = list(set(gt_frames+det_frames))

    print("Amount of GT frames: {}\nAmount of det frames: {}\nSet of all frames: {}".format(len(gt_frames), len(det_frames), len(frames)))


    acc = mm.MOTAccumulator(auto_id=False)

    for frame in frames:

        # Get the df entries for this specific frame
        gts = gt_df[gt_df[gt_frame_col] == frame]
        dets = det_df[det_df[det_frame_col] == frame]

        gt_data = True
        det_data = True

        # Get ground truth positions, if any
        if len(gts) > 0:
            gt_pos, gt_ids = posFunc(gts, cam, task)
            gt_ids = ["gt_{}".format(x) for x in gt_ids]
        else:
            gt_ids = []
            gt_data = False

        # Get detections, if any
        if len(dets) > 0:
            det_pos, det_ids = posFunc(dets, cam, task)
            det_ids = ["det_{}".format(x) for x in det_ids]
        else:
            det_ids = []
            det_data = False

        # Get the L2 distance between ground truth positions, and the detections
        if gt_data and det_data:
            dist = distFunc(gt_pos, det_pos, maxDist=maxDist).tolist()
        else:
            dist = []

        
        # Update accumulator
        acc.update(gt_ids,                 # Ground truth objects in this frame
                det_ids,                # Detector hypotheses in this frame
                dist,                   # Distance between ground truths and observations
                frame)

    metrics = calcMetrics(acc, task)


    writeMOTtoTXT(metrics, os.path.join(outputDir, outputFile), "{}-{}".format(task, maxDist))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description = "Calcualtes the MOT metrics")
    ap.add_argument("-detCSV", "--detCSV", type=str, help="Path to tracking CSV file")
    ap.add_argument("-gtCSV", "--gtCSV", type=str, help="Path to ground truth CSV file")

    ap.add_argument("-task", "--task", type=str, help="What kind of tracking to investigate [3D, cam1, cam2, cam1_bbox, cam2_bbox]")
    ap.add_argument("-bboxCenter", "--bboxCenter", action="store_true", help="Use the bbox center instead of the head position for cam1 and cam2 tasks")
    ap.add_argument("-useMOTFormat", "--useMOTFormat", action="store_true", help="Use the MOT Challenge ground truth format")
    ap.add_argument("-thresh", "--thresh", type=float, help="Distance threshold")

    ap.add_argument("-outputFile", "--outputFile", type=str, help="Name of the output file", default="MOT_Metrics.txt")
    ap.add_argument("-outputPath", "--outputPath", type=str, help="Path to where the output file should be saved", default ="")
    args = vars(ap.parse_args())
    
    MOT_Evaluation(args)