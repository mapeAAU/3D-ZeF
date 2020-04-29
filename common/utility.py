import cv2
import sys
import configparser
import re
import json
import os.path
import pickle as pkl
import pandas as pd
import numpy as np
import networkx as nx
import scipy.spatial.distance as dist
from scipy import ndimage
from common.Track import Track
from sklearn.externals import joblib


def readConfig(path):
    """
    Reads the settings.ini file located at the specified directory path
    If no file is found the system is exited
    
    Input:
        path: String path to the directory
        
    Output:
        config: A directory of dicts containing the configuration settings
    """
    
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    configFile = os.path.join(path,'settings.ini')
    if(os.path.isfile(configFile)):
        config.read(configFile)
        return config
    else:
        print("Error loading configuration file:\n{0}\n Exiting....".format(configFile))
        sys.exit(0)


def writeConfig(path, updateValues):
    """
    Writes to the settings.ini file located at the specified directory path
    If no file is found the system is exited
    
    Input:
        path: String path to the directory
        updateValues: A Dict containing the new values o be added to the config file
        
    """
    config = configparser.ConfigParser(allow_no_value=True)
    configFile = os.path.join(path,'settings.ini')
    if(os.path.isfile(configFile)):
        config.read(configFile)

        for values in updateValues:
            config.set(values[0], values[1], values[2])
        with open(configFile, 'w') as configfile:
            config.write(configfile)

        print("Updated configuration file: {}".format(configFile))
    else:
        print("Error loading configuration file:\n{0}\n Exiting....".format(configFile))
        sys.exit(0)



def findData(df_,*args):
    """
    Used to get data from .csv files
        
    Example: findData(df,'id',2) 
    - returns a dataframe with all information about id=2
    
    Example2: findData(df,'id', 2, 'camera', 1, 'frame')
    - returns a dataframe with the frames where id=2 has been located from camera=1
    
    Input:
        df_: Pandas dataframe containing data which has to be filtered
        args: A set of arguments which can be used to filter the supplied dataframe. See above examples for use case
        
    Output:
        tmp: Pandas dataframe based on the specified arguments
    
    Return: Dataframe (pandas)

    """
    
    # **************************
    # NEED A FIX, THAT VERIFIES THAT THE ARGUMENT IS REPRESENTED AS A COLUMN IN
    # THE CSV-FILE! OTHERWISE AN ERROR OCCURS
    # **************************
    
    tmp = df_
    if len(args)%2: # uneven number of arguments
        for idx, i in enumerate(args):
            if not idx%2 and len(args)-idx == 1:
                tmp = tmp[i]
            elif idx%2:
                continue
            else:
                tmp = tmp[tmp[i] == args[idx+1]]
    else: # even number of arguments
        for idx, i in enumerate(args): 
            if idx%2:
                continue
            tmp = tmp[tmp[i] == args[idx+1]]
    return tmp


def csv2Tracks(csv,offset=0,minLen=10,maxFrame=None):    
    """
    Loads all tracks in a CSV file into a dict of tracks, where the key = track id and value = track class
    The way the data is loaded depends on whether it is a 2D or 3D track.
    
    It should be noted that when loading a 3D track, if not a parent track, 
    then the bounding boxes will loaded so that both the bounding box from both views are present.
    This means the bounding box are then represented as 2D numpy arrays, instead of 1D arrays
    
    Input:
        csv: String path to csv file which has to be converted
        offset: The amount of offset applied to the frames
        minLen: The minimum lenght of a tracklet
        
    Output:
        tracks: A dict of Track objects
    """
    
    if(isinstance(csv, str)):
        if(not os.path.isfile(csv)):
            print("Error loading tracks. Could not find file: {0}".format(csv))
            return []
        csv = pd.read_csv(csv)
    if(isinstance(csv, pd.DataFrame)):
        uniqueIds = csv['id'].unique()
    else:
        print("Error loading tracks. 'csv2tracks' expects " +
              "either a Dataframe or path to a CSV file.")
        return []

    uniqueIds = csv['id'].unique()
    tracks = {}

    for trackId in uniqueIds:
        df_ = findData(csv,'id',trackId)
        
        ## Load 2D track
        if('cam' in csv.columns):
            uniqueCams = df_['cam'].unique()
            for camId in uniqueCams:
                df2_ = findData(df_,'cam',camId)

                if maxFrame:
                    df2_ = df2_[df2_["frame"] <= maxFrame]
                    if len(df2_) == 0: # If no valid detections left, go to the next tracklet
                        continue

                t = Track()
                t.cam = int(camId)
                t.x = np.array((df2_.x),dtype=float)
                t.y = np.array((df2_.y),dtype=float)
                t.cam_frame = np.array((df2_.frame),dtype=int)
                t.frame = np.array((df2_.frame),dtype=int)+offset
                t.id = trackId
                t.tl_x = np.array((df2_.tl_x), dtype=float)
                t.tl_y = np.array((df2_.tl_y), dtype=float)
                t.c_x = np.array((df2_.c_x), dtype=float)
                t.c_y = np.array((df2_.c_y), dtype=float)
                t.w = np.array((df2_.w), dtype=float)
                t.h = np.array((df2_.h), dtype=float)
                t.theta = np.array((df2_.theta), dtype=float)
                t.l_x = np.array((df2_.l_x), dtype=float)
                t.l_y = np.array((df2_.l_y), dtype=float)
                t.r_x = np.array((df2_.r_x), dtype=float)
                t.r_y = np.array((df2_.r_y), dtype=float)
                t.aa_tl_x = np.array((df2_.aa_tl_x), dtype=float)
                t.aa_tl_y = np.array((df2_.aa_tl_y), dtype=float)
                t.aa_w = np.array((df2_.aa_w), dtype=float)
                t.aa_h = np.array((df2_.aa_h), dtype=float)
                if(len(t.frame) > minLen):
                    key = "id{0}_cam{1}".format(t.id,t.cam)
                    t.id = key
                    tracks[t.id] = t
        
    
        ## Load 3D track
        else:
            ## Load all the triangulated positions as the main track
            t = Track() 
            filtered = df_[df_['err'] != -1]
            t.x = np.array((filtered['3d_x']),dtype=float)
            t.y = np.array((filtered['3d_y']),dtype=float)
            t.z = np.array((filtered['3d_z']),dtype=float)
            
            t.tl_x = np.vstack((filtered['cam1_tl_x'].values, filtered['cam2_tl_x'].values))
            t.tl_y = np.vstack((filtered['cam1_tl_y'].values, filtered['cam2_tl_y'].values))
            t.c_x = np.vstack((filtered['cam1_c_x'].values, filtered['cam2_c_x'].values))
            t.c_y = np.vstack((filtered['cam1_c_y'].values, filtered['cam2_c_y'].values))
            t.w = np.vstack((filtered['cam1_w'].values, filtered['cam2_w'].values))
            t.h = np.vstack((filtered['cam1_h'].values, filtered['cam2_h'].values))
            t.theta = np.vstack((filtered['cam1_theta'].values, filtered['cam2_theta'].values))
            t.aa_tl_x = np.vstack((filtered['cam1_aa_tl_x'].values, filtered['cam2_aa_tl_x'].values))
            t.aa_tl_y = np.vstack((filtered['cam1_aa_tl_y'].values, filtered['cam2_aa_tl_y'].values))
            t.aa_w = np.vstack((filtered['cam1_aa_w'].values, filtered['cam2_aa_w'].values))
            t.aa_h = np.vstack((filtered['cam1_aa_h'].values, filtered['cam2_aa_h'].values))
            t.cam_frame = np.vstack((filtered["cam1_frame"].values, filtered["cam2_frame"].values))
            
            t.frame = np.array((filtered.frame),dtype=int)+offset
            t.id = trackId
            if(len(t.frame) > minLen):
                tracks[t.id] = t
            
            ## Load the parent tracks (Which consist of both the triangualted positions and the frames where only one view was present)
            t.parents = {}
            for camId in [1,2]:
                filtered = df_[df_['cam{0}_x'.format(camId)] != -1]
                parent = Track()
                parent.cam = int(camId)
                parent.x = np.array((filtered['cam{0}_x'.format(camId)]),dtype=float)
                parent.y = np.array((filtered['cam{0}_y'.format(camId)]),dtype=float)
                
                parent.tl_x = np.array((filtered['cam{0}_tl_x'.format(camId)]), dtype=float)
                parent.tl_y = np.array((filtered['cam{0}_tl_y'.format(camId)]), dtype=float)
                parent.c_x = np.array((filtered['cam{0}_c_x'.format(camId)]), dtype=float)
                parent.c_y = np.array((filtered['cam{0}_c_y'.format(camId)]), dtype=float)
                parent.w = np.array((filtered['cam{0}_w'.format(camId)]), dtype=float)
                parent.h = np.array((filtered['cam{0}_h'.format(camId)]), dtype=float)
                parent.theta = np.array((filtered['cam{0}_theta'.format(camId)]), dtype=float)
                parent.aa_tl_x = np.array((filtered['cam{0}_aa_tl_x'.format(camId)]), dtype=float)
                parent.aa_tl_y = np.array((filtered['cam{0}_aa_tl_y'.format(camId)]), dtype=float)
                parent.aa_w = np.array((filtered['cam{0}_aa_w'.format(camId)]), dtype=float)
                parent.aa_h = np.array((filtered['cam{0}_aa_h'.format(camId)]), dtype=float)
                parent.cam_frame = np.array((filtered["cam{0}_frame".format(camId)]),dtype=int)

                parent.frame = np.array((filtered.frame),dtype=int)+offset
                parent.id = trackId
                t.parents[camId] = parent                            
    return tracks


def tracks2Csv(tracks, csvPath, overwriteCsv=False):
    """
    Saves list of 'Track' objects to a CSV file
    
    Input:
        tracks: List of Track objects
        csvPath: String path to csv file
        overwriteCsv: Whether to overwrite existing CSV files
        
    """
    
    df = tracks2Dataframe(tracks)
    # Write dataframe to CSV file
    if(overwriteCsv):
        fileName = csvPath
    else:
        fileName = checkFileName(csvPath)
    df.to_csv(fileName)
    print('CSV file with tracks stored in:',fileName)



def tracks2Dataframe(tracks):
    """
    Saves lsit of Track objects to pandas dataframe
    
    Input:
        tracks: List of Track objects
        
    Output:
        df: Pandas dataframe
    """
    
    if(len(tracks) == 0):
        print("Error saving to CSV. List of tracks is empty")
        return
    
    # Collect tracks into single dataframe
    df = pd.DataFrame()
    for t in tracks:
        df = df.append(t.toDataframe())
        
    df = df.sort_values(by=['frame', 'id'], ascending=[True, True])
    return df

    
def checkFileName(fn):
    """
    Return the next permutation of the specified file name if it exists.
    Next permutation is found by adding integer to file name
    
    Input:
        fn: String of filename, excluding .csv suffix
    
    Output:
        fn_: String of the new filename
    """
    
    fn_ = fn + '.csv'
    if fn_ and os.path.isfile(fn_):
        for k in range(0,99):
            fn_ = fn + str(k) + '.csv'
            if not os.path.isfile(fn_):
                print('File already exists. New file name:', fn_)
                break
    return fn_
  
        
def frameConsistencyGraph(indecies, frames, coord, verbose = False):
    '''
    Constructs a Directed Acyclic Graph, where each node represents an index.
    Each node is connected to the possible nodes corresponding to detections in the previous and next frames
    The optimal path is found through minizing the reciprocal euclidan distance
    '''
    graph = nx.DiGraph()

    list_indecies = np.asarray([x for x in range(len(indecies))])
    
    # convert lists to numpy arrays
    if type(indecies) == list:
        indecies = np.asarray(indecies)
    if type(frames) == list:
        frames = np.asarray(frames)
    if type(coord) == list:
        coord = np.asarray(coord)
    
    # sort in ascending order
    sort_order = np.argsort(frames)

    indecies = indecies[sort_order]
    frames = frames[sort_order]
    coord = coord[sort_order]

    # Create a dict where the key is the frame and the element is a numpy array with the relevant indecies for the frame    
    fDict = {}
    for f in frames:
        fDict[f] = indecies[frames == f]
    
    # Go through each element in the dict
    prevNodes = []
    zeroCounter = 0
    for idx, key in enumerate(fDict):
        # Get the indecies for the frame (key)
        occurences = fDict[key]
    
        # For each occurance (i.e. index)
        currentNodes = []
        for occ in occurences:
            cNode = occ
            cIdx = list_indecies[indecies == occ] # get the index needed to look up coordinates
            currentNodes.append((occ, cIdx))
                
            # If there are already soem nodes in the graph
            if prevNodes is not None:

                # for each of the previous nodes calculate the reciprocal euclidean distance to the current node, and use as the edge weight between the nodes
                for tpl in prevNodes:
                    pNode, pIdx = tpl
                    
                    dist = np.linalg.norm(coord[pIdx][0] - coord[cIdx][0])
                    if(dist == 0):
                        if verbose:
                            print("0 distance between frame {} and {} - sorted indexes: {} and {}".format(frames[pIdx], frames[cIdx], pIdx, cIdx))
                        zeroCounter += 1
                        weight = 1
                    else:
                        weight = 1/dist
                    graph.add_edge(pNode, cNode, weight = weight)
                    
                    if verbose:
                        print("edge: {} - {} Weight: {}  Distance {}  -  previous 3D {} - current 3D {}".format(pNode, cNode, weight, dist, coord[pIdx], coord[cIdx]))
        
        prevNodes = currentNodes
    if zeroCounter and verbose:
        print("{} instances of 0 distance".format(zeroCounter))
        
    path = nx.dag_longest_path(graph)
    length = nx.dag_longest_path_length(graph) 
    spatialDist = []
    temporalDist = []
    for idx in range(1, len(path)):
        pIdx = list_indecies[indecies == path[idx-1]]
        cIdx = list_indecies[indecies == path[idx]]
        spatialDist.append(np.linalg.norm(coord[pIdx][0] - coord[cIdx][0]))
        temporalDist.append(frames[cIdx][0] - frames[pIdx][0])
    
    if verbose:
        print()
        print("Longest Path {}".format(path))
        print("Path length {}".format(length))
        print("Spatial Distances: {}".format(spatialDist))
        print("Total spatial distance: {}".format(np.sum(spatialDist)))
        print("Mean spatial distance: {}".format(np.mean(spatialDist)))
        print("Median spatial distance: {}".format(np.median(spatialDist)))
        print("Temporal Distances: {}".format(temporalDist))
        print("Total temporal distance: {}".format(np.sum(temporalDist)))
        print("Mean temporal distance: {}".format(np.mean(temporalDist)))
        print("Median temporal distance: {}".format(np.median(temporalDist)))
        print()

    return path, length, spatialDist, temporalDist
    

def getTrackletFeatures(multi_idx, df):
    '''
    Retrieves the frame and 3d coordiantes at each of the supplied indecies, for the supplied df
    Returns a tuple of lists
    '''
    frames = []
    coords = []

    # Restructure list so we only keep unique values, in an ordered list
    multi_idx = sorted(list(set(multi_idx)))
    
    first = multi_idx[0]
    last = multi_idx[-1]
    
    # Go through each idx, get the frame number and the 3D position
    for index in multi_idx:
        row = df.iloc[index]
        
        frames.append(int(row["frame"]))
        coords.append(np.asarray([row["3d_x"], row["3d_y"], row["3d_z"]]))

    # Check the index before and after the ones in the current list, if they exists, and if it belongs to the same tracklet
    for idx in [first-1, last+1]:
        if idx > 0 and idx < len(df)-1:
            initRow = df.iloc[idx]

            if initRow["id"] == df.iloc[multi_idx[0]]["id"]:
                pos = np.asarray([initRow["3d_x"], initRow["3d_y"], initRow["3d_z"]])

                # Check whether the index has a valid 3D position
                if not np.allclose(pos, np.ones(3)*-1):        
                    multi_idx.append(idx)
                    frames.append(int(initRow.frame))
                    coords.append(pos)
    
    return multi_idx, frames, coords


def getDropIndecies(df, verbose = False):
    '''
    Goes through a dataframe, finds all cases where there are several rows for the same frame in a single Tracklet.
    A graph is constructed where the distance is minimized. The indecies which should be removed from the dataframe is returned

    It is expected that the indecies of the dataframe are unique for each row
    '''

    ids = df.id.unique() # array containing all unique tracklets ids
    drop_idx = [] # list to keep track of which indecies are not kept
    
    # Iterate over each unique ID in the dataframe
    for iID in ids:
        df_id = df[df.id == iID]  # Sub dataframe, containing all rows relevant for the current ID. Indecies are still that of the main dataframe
        
        frame_count = df_id["frame"].value_counts()  # How many times does a frame occur in the dataframe
        multi_assignment =  frame_count[frame_count >  1].sort_index() # isolating the frames with multiple assignments
        
        if len(multi_assignment) == 0:
            continue
        if verbose:
            print("ID {}".format(iID))

        counter = 0
        prevFrame = 0
        multi_idx = []
        keep_idx = []
        analyzed_frames = []
        all_frames = []
        
        # Iterate over the multiple assignments (A pandas Series)
        for idx, sIdx in enumerate(multi_assignment.items()):
            frame, count = sIdx
            all_frames.append(frame) # Keep track of all frames that we look at
            
            # If first frame with multiple assignments, then skip to the next
            if idx == 0:
                prevFrame = frame
                continue
    
            # If the current and previous frames are continous i.e. they follow each other in the video, and therefore related
            if (frame - prevFrame) == 1:
                    counter += 1
                    prevIdx = list(df_id[df_id["frame"] == prevFrame].index.values) # Save the indecies in the main dataframe, for the previous frame, as the first frame in the series will be left out otherwise
                    curIdx = list(df_id[df_id["frame"] == frame].index.values) # Save the indecies in the main dataframe, for the current frame
                    
                    multi_idx.extend(prevIdx + curIdx) # Keep track of all indecies by combining them into one list

            # If the frames are not continous                       
            else:
                # If there is a previous series of frames, then analyze it
                if counter > 0:
                    # Get the indecies needed for the graph, their corresponding frames and 3D positions
                    multi_idx, frames, coords = getTrackletFeatures(multi_idx, df)
                    
                    # Create graph and get and keep the indecies which minimizes the distance between the possible nodes
                    new_idx, _, _, _ = frameConsistencyGraph(multi_idx,frames,coords,verbose)
                    keep_idx.extend(new_idx)

                    # The frames which have been analyzed are kept
                    analyzed_frames.extend(sorted(list(set(frames))))

                    multi_idx = []
                    
                counter = 0
            prevFrame = frame
            
        # Analyzing last set of multi detections across several connected frames, if any
        if len(multi_idx) > 0:
            # Get the indecies needed for the graph, their corresponding frames and 3D positions
            multi_idx, frames, coords = getTrackletFeatures(multi_idx, df)
            
            # Create graph and get and keep the indecies which minimizes the distance between the possible nodes
            new_idx, _, _, _ = frameConsistencyGraph(multi_idx,frames,coords,verbose)
            keep_idx.extend(new_idx)
            
            # The frames which have been analyzed are kept
            analyzed_frames.extend(sorted(list(set(frames))))

            multi_idx = []

        # Analyzing single-frame multi detections
        for idx, sIdx in enumerate(multi_assignment.items()):
            frame, count = sIdx
            
            # If the current frame is not in the list of already analyzed frames
            if frame not in analyzed_frames:
                # Get the indecies for the frame
                dfIdx = list(df_id[df_id["frame"] == frame].index.values)

                # Get the indecies needed for the graph, their corresponding frames and 3D positions
                multi_idx, frames, coords = getTrackletFeatures(dfIdx, df)
                
                # Create graph and get and keep the indecies which minimizes the distance between the possible nodes
                new_idx, _, _, _ = frameConsistencyGraph(multi_idx,frames,coords,verbose)
                keep_idx.extend(new_idx)
            
                # The frames which have been analyzed are kept
                analyzed_frames.extend(sorted(list(set(frames))))
                
                multi_idx = []
                            
        all_idx = []
        # Find the indecies related to all of the frames which we have investigated
        for f in all_frames:
            all_idx.extend(list(df_id[df_id["frame"] == f].index.values))

        # Filter out all the indecies which we did not end up keeping
        drop_idx.extend([x for x in all_idx if x not in keep_idx])      

    if verbose:
        print("Dropped indecies: {}".format(drop_idx))
    return drop_idx


def extractRoi(frame, pos, dia):
    """
    Extracts a region of interest with size dia x dia in the provided frame, at the specied position
    
    Input:
        frame: Numpy array containing the frame
        pos: 2D position of center of ROI
        dia: Integer used as width and height of the ROI
        
    Output:
        patch: Numpy array containing hte extracted ROI
    """
    
    h,w = frame.shape[:2]
    xMin = max(int(pos[0]-dia/2)+1, 0)
    xMax = min(xMin + dia, w)
    yMin = max(int(pos[1]-dia/2)+1, 0)
    yMax = min(yMin + dia, h)
    patch = frame[yMin:yMax, xMin:xMax]
    return patch


def rotate(img, angle, center):
    """
    Rotates the input image by the given angle around the given center
    
    Input:
        img: Input image
        angle: Angle in degrees
        center: Tuple consisting of the center the image should be rotated around
        
    Output:
        dst: The rotated image
    """
    
    rows, cols = img.shape[:2]
    
    M = cv2.getRotationMatrix2D(center,angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst


def prepareCams(path):
    """
    Loads the camera objects stored in a pickle file at the provided path
    
    Input:
        path: Path to the folder where the camera.pkl file is located
        
    Output:
        cams: A dict containing the extracted camera objects
    """
    
    cam1Path = os.path.join(path,'cam1.pkl')
    cam2Path = os.path.join(path,'cam2.pkl')
    if(not os.path.isfile(cam1Path)):
        print("Error finding camera calibration file: \n {0}".format(cam1Path))
        sys.exit(0)
    if(not os.path.isfile(cam2Path)):
        print("Error finding camera calibration file: \n {0}".format(cam2Path))
        sys.exit(0)

    cam1ref = os.path.join(path,'cam1_references.json')
    cam2ref = os.path.join(path,'cam2_references.json')
    if(not os.path.isfile(cam1ref)):
        print("Error finding camera corner reference file: \n {0}".format(cam1ref))
        sys.exit(0)
    if(not os.path.isfile(cam2ref)):
        print("Error finding camera corner reference file: \n {0}".format(cam2ref))
        sys.exit(0)

    cams = {}
    cam1 = joblib.load(cam1Path)
    cam1.calcExtrinsicFromJson(cam1ref)
    cams[1] = cam1

    cam2 = joblib.load(cam2Path)
    cam2.calcExtrinsicFromJson(cam2ref)
    cams[2] = cam2

    print("")
    print("Camera 1:")
    print(" - position: \n" + str(cam1.getPosition()))
    print(" - rotation: \n" + str(cam1.getRotationMat()))
    print("")

    print("Camera 2:")
    print(" - position: \n" + str(cam2.getPosition()))
    print(" - rotation: \n" + str(cam2.getRotationMat()))
    print("")

    return cams



def getROI(path, camId):
    """
    Loads the JSON camera parameters and reads the Region of Interest that has been manually set
    """

    # Load json file
    with open(os.path.join(path, "cam{}_references.json".format(camId))) as f:
        data = f.read()

    # Remove comments
    pattern = re.compile('/\*.*?\*/', re.DOTALL | re.MULTILINE)
    data = re.sub(pattern, ' ', data)

    # Parse json
    data = json.loads(data)

    # Convert to numpy arrays
    x_coords = []
    y_coords = []

    for i,entry in enumerate(data):
        x_coords.append(int(entry["camera"]["x"]))
        y_coords.append(int(entry["camera"]["y"]))

    tl = np.asarray([np.min(x_coords), np.min(y_coords)], dtype=int)
    br = np.asarray([np.max(x_coords), np.max(y_coords)], dtype=int)

    return tl, br


def applyROIBBs(bboxes, tl, br):
    """
    Checks and keeps the detected bounding boxes only if they are fully within the ROI.

    Input:
        bboxes: List of bbox tuples

    Output:
        roi_bboxes: List of bbox tuples within ROI
    """

    if bboxes is None:
        return None
    
    roi_bboxes = []

    for bbox in bboxes:
        if bbox[0] >= tl[0] and bbox[1] >= tl[1] and bbox[2] <= br[0] and bbox[3] <= br[1]:
            roi_bboxes.append(bbox)
    
    if len(roi_bboxes) == 0:
        roi_bboxes = None
    
    return roi_bboxes