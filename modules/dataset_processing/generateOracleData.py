import os
import pandas as pd
import numpy as np
import argparse

def generateOracleData(args):
    """
    Takes the annotated dataset and constructs the perfect output at different stages of the pipeline, except when an occlusion occurs.

    Input:
        gtCSV: path to the ground truth annotations sv file
        outputDir: path to where the output files are saved
    """
    tracks = args["gtCSV"]
    outputDir = args["outputDir"]

    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    
    csv_df = pd.read_csv(tracks, sep=";")

    frames = csv_df["frame"].unique()
    ids = csv_df["id"].unique()
    

    out_dfs = {"Det-cam1": pd.DataFrame(),
                "Track-cam1": pd.DataFrame(),
                "Det-cam2": pd.DataFrame(),
                "Track-cam2": pd.DataFrame(),
                "Track-3D": pd.DataFrame(),
                "Track-Full": pd.DataFrame()}
    
    id_counters = {"cam1": 1,
                    "cam2": 1,
                    "3D": 1}

    id_state = {"cam1": False,
                 "cam2": False,
                "3D": False}

    for fid in ids:
        fish_df = csv_df[csv_df["id"] == fid]
        print(fid)

        for frame in range(frames[0],frames[-1]+1):
            
            frame_df = fish_df[fish_df["frame"] == frame]

            if len(frame_df) == 0:
                id_counters["cam1"] += 1
                id_counters["cam2"] += 1
                id_counters["3D"] += 1

            
            for cam in ["cam1", "cam2"]:
                if np.isnan(frame_df["{}_x".format(cam)].values[0]):
                    if id_state[cam]:
                        id_counters[cam] += 1
                        id_state[cam] = False
                    continue
                
                if frame_df["{}_occlusion".format(cam)].values[0] == 1.0:
                    if id_state[cam]:
                        id_counters[cam] += 1
                        id_state[cam] = False
                    continue

                id_state[cam] = True


                values = {"frame" : int(frame_df["{}_frame".format(cam)].values[0]),
                                "id": float(id_counters[cam]),
                                "cam": int(cam[-1]),
                                "x": frame_df["{}_x".format(cam)].values[0],
                                "y": frame_df["{}_y".format(cam)].values[0], 
                                "tl_x": -1,
                                "tl_y":-1,
                                "c_x": frame_df["{}_x".format(cam)].values[0],
                                "c_y": frame_df["{}_y".format(cam)].values[0],
                                "w": -1,
                                "h": -1,
                                "theta": -1,
                                "l_x": frame_df["{}_x".format(cam)].values[0],
                                "l_y": frame_df["{}_y".format(cam)].values[0], 
                                "r_x": frame_df["{}_x".format(cam)].values[0],
                                "r_y": frame_df["{}_y".format(cam)].values[0],
                                "aa_tl_x": frame_df["{}_tl_x".format(cam)].values[0],
                                "aa_tl_y": frame_df["{}_tl_y".format(cam)].values[0],
                                "aa_w": (frame_df["{}_br_x".format(cam)].values[0]-frame_df["{}_tl_x".format(cam)].values[0]),
                                "aa_h": (frame_df["{}_br_y".format(cam)].values[0]-frame_df["{}_tl_y".format(cam)].values[0])}

                values = {k: [v] for k,v in values.items()}
                

                out_dfs["Det-{}".format(cam)] = pd.concat([out_dfs["Det-{}".format(cam)], pd.DataFrame.from_dict(values)])
                out_dfs["Track-{}".format(cam)] = pd.concat([out_dfs["Track-{}".format(cam)], pd.DataFrame.from_dict(values)])


            if frame_df["cam1_occlusion"].values[0] == 0.0 and frame_df["cam2_occlusion"].values[0] == 0.0 and frame_df["3d_x"].values[0] != 0:
                values = {}
                values = {"frame" : frame_df["frame"].values[0],
                        "id": float(id_counters["3D"]),
                        "err": frame_df["cam1_reproj_err"].values[0] + frame_df["cam2_reproj_err"].values[0],
                        "3d_x": frame_df["3d_x"].values[0],
                        "3d_y": frame_df["3d_y"].values[0],
                        "3d_z": frame_df["3d_z"].values[0],
                        "cam1_x": frame_df["cam1_x"].values[0],
                        "cam1_y": frame_df["cam1_y"].values[0], 
                        "cam2_x": frame_df["cam2_x"].values[0],
                        "cam2_y": frame_df["cam2_y"].values[0], 
                        "cam1_proj_x": -1,
                        "cam1_proj_y": -1, 
                        "cam2_proj_x": -1,
                        "cam2_proj_y": -1, 
                        "cam1_tl_x": -1,
                        "cam1_tl_y": -1,
                        "cam1_c_x": (frame_df["cam1_br_x"].values[0] + frame_df["cam1_tl_x"].values[0])//2,
                        "cam1_c_y": (frame_df["cam1_br_y"].values[0] + frame_df["cam1_tl_y"].values[0])//2,
                        "cam1_w": -1,
                        "cam1_h": -1,
                        "cam1_theta": -1,
                        "cam1_aa_tl_x": frame_df["cam1_tl_x"].values[0],
                        "cam1_aa_tl_y": frame_df["cam1_tl_y"].values[0],
                        "cam1_aa_w": frame_df["cam1_br_x"].values[0] - frame_df["cam1_tl_x"].values[0],
                        "cam1_aa_h": frame_df["cam1_br_y"].values[0] - frame_df["cam1_tl_y"].values[0],
                        "cam1_frame": int(frame_df["cam1_frame"].values[0]),
                        "cam2_tl_x": -1,
                        "cam2_tl_y": -1,
                        "cam2_c_x": (frame_df["cam2_br_x"].values[0] + frame_df["cam2_tl_x"].values[0])//2,
                        "cam2_c_y": (frame_df["cam2_br_y"].values[0] + frame_df["cam2_tl_y"].values[0])//2,
                        "cam2_w": -1,
                        "cam2_h": -1,
                        "cam2_theta": -1,
                        "cam2_aa_tl_x": frame_df["cam2_tl_x"].values[0],
                        "cam2_aa_tl_y": frame_df["cam2_tl_y"].values[0],
                        "cam2_aa_w": frame_df["cam2_br_x"].values[0] - frame_df["cam2_tl_x"].values[0],
                        "cam2_aa_h": frame_df["cam2_br_y"].values[0] - frame_df["cam2_tl_y"].values[0],
                        "cam2_frame": int(frame_df["cam2_frame"].values[0])}

                values = {k: [v] for k,v in values.items()}

                out_dfs["Track-3D"] = pd.concat([out_dfs["Track-3D"], pd.DataFrame.from_dict(values)])

                values["id"] = [frame_df["id"].values[0]]
                out_dfs["Track-Full"] = pd.concat([out_dfs["Track-Full"], pd.DataFrame.from_dict(values)])
                id_state["3D"] = True

            else:
                if id_state["3D"]:
                    id_counters["3D"] += 1
                    id_state["3D"] = False
                continue


    

    # Save the different dataframes to csv files according to the used format in the pipeline
    out_dfs["Det-cam1"] = out_dfs["Det-cam1"].drop(columns=["id"])
    out_dfs["Det-cam2"] = out_dfs["Det-cam2"].drop(columns=["id"])

    out_dfs["Det-cam1"]["covar"] = out_dfs["Det-cam2"]["covar"] = 0
    out_dfs["Det-cam1"]["var_x"] = out_dfs["Det-cam2"]["var_x"] = 1
    out_dfs["Det-cam1"]["var_y"] = out_dfs["Det-cam2"]["var_y"] = 1

    out_dfs["Det-cam1"].to_csv(os.path.join(outputDir,"detections_2d_cam1.csv"))
    out_dfs["Det-cam2"].to_csv(os.path.join(outputDir,"detections_2d_cam2.csv"))
    
    out_dfs["Track-cam1"].to_csv(os.path.join(outputDir,"tracklets_2d_cam1.csv"))
    out_dfs["Track-cam2"].to_csv(os.path.join(outputDir,"tracklets_2d_cam2.csv"))
    
    out_dfs["Track-3D"].to_csv(os.path.join(outputDir,"tracklets_3d.csv"))
    out_dfs["Track-Full"].to_csv(os.path.join(outputDir,"tracks_3d.csv"))



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description = "Converts the ground truth data to perfect detection/tracklet data, without occlusions")
    ap.add_argument("-gtCSV", "--gtCSV", type=str, help="Path to the ground truth csv")
    ap.add_argument("-outputDir", "--outputDir", type=str, help="The output directory")              
    args = vars(ap.parse_args())
    
    generateOracleData(args)