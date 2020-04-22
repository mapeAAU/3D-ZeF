import os
import argparse
import pandas as pd
import numpy as np


def getStats(x):
    """
    Takes an input numpy array and calcualtes the mean, standard deviation, median, min and max values of the array

    Input:
        x: numpy array
    
    Output:
        Reutnrs the mean, std.dev., median, min and max
    """
    return np.mean(x), np.std(x), np.median(x), np.min(x), np.max(x)

def getdisplacement(pos):
    """
    Takes an numpy array contatining position information, and calculates the displacement

    Input:
        pos: numpy array containing position information
    
    Output:
        diff: numpy array containing displacment information
    """
    diff = []
    for idx in range(1, pos.shape[0]):
        pos1 = pos[idx-1]
        pos2 = pos[idx]

        diff.append(np.linalg.norm(pos2-pos1))
    return np.asarray(diff)

def getSpeed(displacement, deltaT):
    """
    Takes an numpy array contatining displacement information, and calculates the speed at a given deltaT

    Input:
        displacement: numpy array containing position information
        deltaT: float indicating the time step
    
    Output:
        speed: numpy array containing speed information
    """
    speed = []
    for idx in range(1, displacement.shape[0]):
        disp1 = displacement[idx-1]
        disp2 = displacement[idx]

        speed.append(np.linalg.norm(disp2-disp1) / deltaT)
    return np.asarray(speed)

def getAcceleration(speed, deltaT):
    """
    Takes an numpy array contatining speed information, and calculates the acceletation at a given deltaT

    Input:
        speed: numpy array containing speed information
        deltaT: float indicating the time step
    
    Output:
        acceleration: numpy array containing acceleration information
    """
    acceleration = []
    for idx in range(1, speed.shape[0]):
        speed1 = speed[idx-1]
        speed2 = speed[idx]

        acceleration.append(np.linalg.norm(speed2-speed1) / deltaT)
    return np.asarray(acceleration)


def analyzeReprojectionError(df):
    """
    Collects and computes the combined reprojection error 

    Input:
        df: Pandas Dataframe containing the annotated dataset
    
    Output:
        reproj_data: Numpy array containing the combined reprojection error
    """
    df["reproj_err"] = df["cam1_reproj_err"] + df["cam2_reproj_err"]

    reproj_data = df["reproj_err"].values
    reproj_data = reproj_data[np.isfinite(reproj_data)]

    return reproj_data


def analyzeMotionCharateristics(df):
    """
    Collects and computes the different motion charateristics (displacement, speed, acceleration) from the annotated dataset

    Input:
        df: Pandas Dataframe containing the annotated dataset
    
    Output:
        total_disp: Dict containing the displacement information from all fish ID
        total_speed: Dict containing the speed information from all fish ID
        total_acc: Dict containing the acceleration information from all fish ID
    """

    total_disp = {"3D" : []}
    total_speed = {"3D" : []}
    total_acc = {"3D" : []}

    for ids in df["id"].unique():
        tmp_df = df[df["id"] == ids]

        world_pos = np.stack((tmp_df["3d_x"], tmp_df["3d_y"], tmp_df["3d_z"]),-1)
        world_pos = world_pos[ ~np.isnan(world_pos).any(axis=1)]

        coords = {"3D": world_pos}

        for key in coords:

            pos = coords[key]

            disp = getdisplacement(pos)
            speed = getSpeed(disp, 1/60)
            acc = getAcceleration(speed, 1/60)
            
            total_disp[key].extend(disp)
            total_speed[key].extend(speed)
            total_acc[key].extend(acc)

    return total_disp, total_speed, total_acc


def getDistributionSettings(args):
    """
    Calculates the distribution parameters of the reprojetion error and motion charateristics, based on the annotated dataset
    """
    rootdir = args["gtPath"]

    dirs = {"train": [os.path.join(rootdir, x) for x in ["TRN2", "TRN5"]],
            "valid": [os.path.join(rootdir, x) for x in ["VAL2", "VAL5"]],
            "test": [os.path.join(rootdir, x) for x in ["TST1", "TST2","TST5", "TST10"]]}

    for key in dirs.keys():
        
        disp = []
        speed = []
        acc = []
        reproj = []

        for directory in dirs[key]:
            gt_directory = os.path.join(directory, "GT")

            ann_df = pd.read_csv(os.path.join(gt_directory, "annotations_full.csv"), sep=";")

            reproj_data = analyzeReprojectionError(ann_df)
            disp_dict, speed_dict, acc_dict = analyzeMotionCharateristics(ann_df)


            disp.extend(disp_dict["3D"])
            speed.extend(speed_dict["3D"])
            acc.extend(acc_dict["3D"])
            reproj.extend(reproj_data)

        reproj_mean, reproj_std, reproj_median, reproj_min, reproj_max = getStats(reproj)
        disp_mean, disp_std, disp_median, disp_min, disp_max = getStats(disp)
        speed_mean, speed_std, speed_median, speed_min, speed_max = getStats(speed)
        acc_mean, acc_std, acc_median, acc_min, acc_max = getStats(acc)


        with open(os.path.join(rootdir, key+"_stats.txt"), "w") as f:
            f.write("Reproj Mean: {}\n".format(reproj_mean))
            f.write("Reproj Std.Dev: {}\n".format(reproj_std))
            f.write("Reproj Median: {}\n".format(reproj_median))
            f.write("Reproj Min: {}\n".format(reproj_min))
            f.write("Reproj Max: {}\n\n".format(reproj_max))
            
            f.write("Displacement Mean: {}\n".format(disp_mean))
            f.write("Displacement Std.Dev: {}\n".format(disp_std))
            f.write("Displacement Median: {}\n".format(disp_median))
            f.write("Displacement Min: {}\n".format(disp_min))
            f.write("Displacement Max: {}\n\n".format(disp_max))
            
            f.write("Speed Mean: {}\n".format(speed_mean))
            f.write("Speed Std.Dev: {}\n".format(speed_std))
            f.write("Speed Median: {}\n".format(speed_median))
            f.write("Speed Min: {}\n".format(speed_min))
            f.write("Speed Max: {}\n\n".format(speed_max))
            
            f.write("Acceleration Mean: {}\n".format(acc_mean))
            f.write("Acceleration Std.Dev: {}\n".format(acc_std))
            f.write("Acceleration Median: {}\n".format(acc_median))
            f.write("Acceleration Min: {}\n".format(acc_min))
            f.write("Acceleration Max: {}\n\n".format(acc_max))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description = "Calcualtes the distribution parameters for reproejction error and motion charateristics")
    ap.add_argument("-gtPath", "--gtPath", type=str, help="Path to the ground truth directory")
    args = vars(ap.parse_args())
    
    getDistributionSettings(args)