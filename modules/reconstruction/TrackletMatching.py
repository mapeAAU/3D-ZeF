import numpy as np
import pandas as pd
import argparse
import os.path
import configparser
import networkx as nx
import cv2
import scipy.stats
### Module imports ###
import sys
sys.path.append('../../')
from common.utility import csv2Tracks,readConfig, getDropIndecies, prepareCams
from common.Track import Track
from modules.reconstruction.Triangulate import Triangulate


class TrackletMatcher:
    """
    Class implementation for associating 2D tracklets into 3D tracklets        
    """
    
    def __init__(self, dataPath):
        """
        Initialize object
        
        Input:
            dataPath: String path to the main folder
        """
        
        # Load settings and data
        self.loadSettings(dataPath)
        self.loadTracklets(dataPath)
        self.cams = prepareCams(dataPath) # Load camera objects

        # Internal stuff
        self.graph = nx.DiGraph()
        self.camIdMap = {}
        self.triangulated = {}


    def loadSettings(self,path):
        """
        Load settings from config file in the provided path.
        
        Config file includes information on the following, which is set in the object:
            reprojection_err_mean: The mean value of a Gaussian distribution of reprojection errors
            reprojection_err_std: The standard deviation of a Gaussian distribution of reprojection errors
            movement_err_mean: The mean value of a Gaussian distribution of movement errors
            movement_err_std: The standard deviation of a Gaussian distribution of movement errors
            same_view_max_overlap: The maximum allowed frame overlap of two tracklets
            tracklet_min_length: Minimum trackelt length
            camera_1_sync_frame: Sync frame for camera 1
            camera_2_sync_frame: Sync frame for camera 2 
            
        Input:
            path: String path to the folder where the settings.ini file is located
        """
        
        config = readConfig(path)
        
        # Get tracklet matching parameters
        c = config['TrackletMatcher']
        self.reprojMeanErr = c.getfloat('reprojection_err_mean')
        self.reprojStdErr = c.getfloat('reprojection_err_std')
        self.movErrMean = c.getfloat('movement_err_mean')
        self.movErrStd = c.getfloat('movement_err_std')  
        self.sameViewMaxOverlap = c.getint('same_view_max_overlap')
        self.trackletMinLength = c.getint('tracklet_min_length')
        self.temporalPenalty = c.getint('temporal_penalty')
        self.FPS = c.getint('FPS')
        self.camera2_useHead = c.getboolean("cam2_head_detector", False)    

        # Get aquarium size
        c = config['Aquarium']
        self.maxX = c.getfloat("aquarium_width")
        self.maxY = c.getfloat("aquarium_depth")
        self.maxZ = c.getfloat("aquarium_height", np.inf)
        
        self.minX = c.getfloat("min_aquarium_width", 0.0)
        self.minY = c.getfloat("min_aquarium_depth", 0.0)
        self.minZ = c.getfloat("min_aquarium_height", 0.0)

        print("Aquarium Dimensions\n\tX: {} - {}\n\tY: {} - {}\n\tZ: {} - {}\n".format(self.minX, self.maxX, self.minY, self.maxY, self.minZ, self.maxZ))

        # Get camera synchronization parameters
        c = config['CameraSynchronization']
        cam1frame = c.getint('cam1_sync_frame')
        cam2frame = c.getint('cam2_sync_frame')
        self.camera1_offset = max(0,cam2frame-cam1frame)
        self.camera2_offset = max(0,cam1frame-cam2frame)       
        self.camera1_length = c.getint("cam1_length")
        self.camera2_length = c.getint("cam2_length")        


    def loadTracklets(self,path):
        """
        Loads the 2D tracklets extracted by TrackerVisual.py
        The tracklets are loaded as dicts, where the key is a combination of the tracklet ID and camera ID
        
        Input:
            path: String path the main folder, containing the processed folder with the 2D tracklets.   
        """
        
        self.cam1Tracks = csv2Tracks(os.path.join(path, 'processed/tracklets_2d_cam1.csv'),
                                     offset=self.camera1_offset,
                                     minLen=self.trackletMinLength,
                                     maxFrame=self.camera1_length)
        self.cam2Tracks = csv2Tracks(os.path.join(path,'processed/tracklets_2d_cam2.csv'),
                                     offset=self.camera2_offset,
                                     minLen=self.trackletMinLength,
                                     maxFrame=self.camera2_length) 

        cam1Info = "Camera 1\n\tLength: {}\n\tOffset: {}\n\tUnique IDs: {}".format(self.camera1_length, self.camera1_offset, len(self.cam1Tracks))
        cam2Info = "Camera 2\n\tLength: {}\n\tOffset: {}\n\tUnique IDs: {}".format(self.camera2_length, self.camera2_offset, len(self.cam2Tracks))
        print(cam1Info)
        print(cam2Info)
        
            

    def withinAquarium(self,x,y,z):
        """
        Checks whether the provided x,y,z coordinates are inside the aquarium.
        
        Input:
            x: x coordinate
            y: y coordinate
            z: z coordinate
        
        Output:
            Boolean value stating whether the point is inside the aquarium
        """

        if(x < self.minX or x > self.maxX):
            return False
        if(y < self.minY or y > self.maxY):
            return False
        if(z < self.minZ or z > self.maxZ):
            return False
        return True


    def findConcurrent(self,track,candidates):
        """
        Finds the concurrent tracks (i.e. within the same span of frames) between a specific track and a set of  othertracks 
        
        Input:
            track: A Track object
            candidates: List of Track objects
            
        Output:
            concurrent: List of Track objects from candidates that were concurrent with the track argument
        """
        
        concurrent = []
        for c in candidates:
            frames = np.intersect1d(track.frame, candidates[c].frame)
            if(len(frames) == 0):
                continue        
            concurrent.append(candidates[c])
        return concurrent


    def calcMatchWeight(self,track1,track2):
        """
        Calculate the weight between two tracks from different views.
        The weight is a weighted median value of the inverse CDF value of the reprojection errors between the two tracks.
        The Gaussian CDF is used, with parameters laoded in the config file, and it is inverted so value below the mean (i.e. towards 0) is trusted more than value above
        
        Input:
            track1: Track obejct from the top camera
            track2: Track object from the front camera
            
        Output:
            weight: Weight of the constructed 3D tracklet
            track3d: Track object containing the 3D tracklet
        """
        
        frames = np.intersect1d(track1.frame, track2.frame)

        # Prepare new 3d track for saving triangulated information
        track3d = Track()
        track3d.errors = []
        track3d.reproj = []
        track3d.positions3d = []
        track3d.cam1reprojections = []
        track3d.cam2reprojections = []
        track3d.cam1positions = []
        track3d.cam2positions = []
        track3d.cam1bbox = []
        track3d.cam2bbox = []
        track3d.cam1frame = []
        track3d.cam2frame = []
        track3d.cam1Parent = track1
        track3d.cam2Parent = track2
        
        frameList = []
        
        for f in sorted(frames):           
            ## Reproject the tracks
            err,pos3d,cam1reproj,cam2reproj,cam2Pt = self.calcReprojError(f,track1,track2)
            track3d.reproj.append(err)

            ## Get the weight as the inverted CDF value.
            err = 1-scipy.stats.expon.cdf(err, scale=self.reprojMeanErr)
            
            if(self.withinAquarium(*pos3d)):
                track3d.errors.append(err)
            else:
                continue
                
            track3d.positions3d.append(pos3d) 
            track3d.cam1reprojections.append(cam1reproj) 
            track3d.cam2reprojections.append(cam2reproj) 
            track3d.cam1positions.append(track1.getImagePos(f)) 
            track3d.cam2positions.append(track2.getImagePos(f, cam2Pt)) 
            track3d.cam1bbox.append(track1.getBoundingBox(f))
            track3d.cam2bbox.append(track2.getBoundingBox(f))
            track3d.cam1frame.append(track1.getVideoFrame(f))
            track3d.cam2frame.append(track2.getVideoFrame(f))
            frameList.append(f)
            
        if len(track3d.errors) > 0:
            track3d.frame = np.array(frameList)  
            weight = np.median(track3d.errors) * (len(track3d.errors)/len(list(np.union1d(track1.frame, track2.frame))))
            return weight,track3d    
        else:
            return 0, None
    
    
    def calcReprojError(self,frameNumber,track1,track2, verbose=False):
        """
        Calculates the reprojection error between the provided tracklets at the specified frame
        This is done using a Triangulate object.
        
        Input:
            frameNumber: Index of the frame to be analyzed
            track1: Track obejct containing the first tracklet
            track2: Track object containing the second tracklet
            
        Output:
            err: Reprojection error (Euclidean distance) between the actual points of the tracks, and the reprojected points
            p: 3D position of the 3D tracklet 
            p1: 2D point of p reprojected onto camera view 1
            p2: 2D point of p reprojected onto camera view 2
        """
        minErr = np.inf
        minP = None
        minP1 = None
        minP2 = None
        minPt = None

        cam2_list = ["l","c","r"]
        if self.camera2_useHead:
            cam2_list = ["kpt"]

        for pt in cam2_list:
            tr = Triangulate()
        
            t1Pt = track1.getImagePos(frameNumber, "kpt")
            t2Pt = track2.getImagePos(frameNumber, pt)
            
    
            # 1) Triangulate 3D point
            p,d = tr.triangulatePoint(t1Pt,
                                      t2Pt,
                                      self.cams[track1.cam],
                                      self.cams[track2.cam],
                                      correctRefraction=True)
        
            p1 = self.cams[track1.cam].forwardprojectPoint(*p)
            p2 = self.cams[track2.cam].forwardprojectPoint(*p)
    
            # 2) Calc re-projection errors
            pos1 = np.array(t1Pt)
            err1 = np.linalg.norm(pos1-p1)
            pos2 = np.array(t2Pt)
            err2 = np.linalg.norm(pos2-p2)
            err = err1 + err2     
            
            if err < minErr:
                minErr = err
                minP = p
                minP1 = p1
                minP2 = p2    
                minPt = pt
            
        if verbose:
            print("Min error: {}\n\t3D coords: {}\n\tTrack 1: {}\n\tTrack 2: {}\n\tPos1 {} (GT) / {}\n\tPos2 {} (GT) / {}\n\tTrack 2 pt: {}".format(minErr, minP, track1.id, track2.id, pos1, p1, pos2, p2, minPt))
        return minErr, minP, minP1, minP2, minPt


    def createNodes(self, verbose=False):
        """
        Populates the internal graph with nodes, where each node is a 3D tracklet with the weight from calcMatchWeight
        Only 2D tracklets which are concurrent are analyzed.
        
        Also stores all the 3D tracklets in a internal triagnualted structure
        
        Input:
            Verbose: Whether to print information for each node added
        """
        
        for tId in self.cam1Tracks:
            t = self.cam1Tracks[tId]
            concurrent = self.findConcurrent(t,self.cam2Tracks)
            
            for c in concurrent:
                weight,track3d = self.calcMatchWeight(t,c)
                if(weight <= 0.001) or track3d is None:
                  continue
                nodeName = "{0}-{1}".format(t.id,c.id)
                self.graph.add_node(nodeName, weight=weight,
                                    frames=track3d.frame,
                                    cam1=t.id,
                                    cam2=c.id)
                self.addToMap(nodeName)

                # Save triangulated information
                self.triangulated[nodeName] = track3d
                
                if verbose:
                    print("Added node:")
                    print(" {0}-{1} with weight: {2}".format(t.id, c.id, weight))


    def addToMap(self, nodeName):
        """
        Populates the internal camera id map, which is a dict tracking that per 2D tracklet writes all 'nodeNames' (i.e. identifiers for nodes in the internal graph)
        that the 2D tracklet is a part of.
        
        Input:
            nodeName: A string signifying the 2 2D tracklets used for a 3D tracklet
        """
        
        for key in ['cam1','cam2']:
            currId = self.graph.nodes[nodeName][key]
            if(currId not in self.camIdMap):
                self.camIdMap[currId] = []
            self.camIdMap[currId].append(nodeName)     


    def connectNodes3D(self, verbose=False):
            """
            Per tracklet goes through and calculates an edge weight between all nodes with the same trackID in its node name
            This is an attempt to combine tracklets in a view, who is associated with the same tracklet in the other view.
            This way tracklets in the same view can be associated, even though there are frames missing in between
            
            This is done if based on te average speed to travel between the two 2D tracklet positions of which is not hte same trackle
            The edge weight is based on the inverse CDF value of the distance between the first and last frames in the 2D tracklets in the same view.
            The CDF value is multiplied with the sum of the node weights for the 2 nodes being connected.
            
            Input:
                verbose: Whether to print information on the nodes connected and their weights.
            """

            for trackId in self.camIdMap:       
                elements = [e for e in self.camIdMap[trackId]]
                    
                for e1 in elements:
                    e1Track = self.triangulated[e1]
                    
                    for e2 in elements:
                        if(e1 == e2):
                            continue
                        e2Track = self.triangulated[e2]
                        
                        frameDiff = e2Track.frame[0]-e1Track.frame[-1]
                        posDiff = np.linalg.norm(e1Track.positions3d[-1]-e2Track.positions3d[0])

                        overlap3D = (e2Track.frame[0]-e1Track.frame[-1]) <= 0
                        
                        overlap2D = False
                        if "cam1" in trackId:
                            overlap2D = (e2Track.cam2frame[0]-e1Track.cam2frame[-1]) <= 0
                        if "cam2" in trackId:
                            overlap2D = (e2Track.cam1frame[0]-e1Track.cam1frame[-1]) <= 0

                        if verbose:
                            print("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(e1, e2, e1Track.frame[0], e1Track.frame[-1], e2Track.frame[0], e2Track.frame[-1], overlap3D, overlap2D, frameDiff, posDiff))
                        
                        ## If the tracklets start and ends frames differ too much, ignore it
                        if overlap3D or overlap2D or self.graph.has_edge(e1, e2) or self.graph.has_edge(e2, e1): # Check that the tracklets does not temporally overlap, and that there is not already an edge in the DAG between the two tracklets
                            continue

                        frameDiff = abs(frameDiff)

                        ## calculate Euclidean distance between the tracklet end/start points            
                        if frameDiff != 0:
                            speed = posDiff/(frameDiff/self.FPS)
                        else:
                            speed = 0.0

                        ## Calculate a weight value based on the inverse exp CDF that penalises a large distance
                        moveProb = (1.0-scipy.stats.expon.cdf(speed, scale=self.movErrMean+self.movErrStd)) * np.exp(-frameDiff/self.temporalPenalty)

                        dist = self.graph.nodes[e1]['weight'] + self.graph.nodes[e2]['weight']
                        dist *= moveProb

                        if verbose:
                            print("\nEdge: {0} to {1} with weight: {2}".format(e1,e2, dist))
                        self.graph.add_edge(e1,e2,weight=dist)
 
        
def combine2DTracklets(df, tm):
    
    ids = df.id.unique() # array containing all unique tracklets ids
    drop_idx = [] # list to keep track of which indecies are not kept
    
    # Iterate over each unique ID in the dataframe
    for iID in ids:
        df_id = df[df.id == iID]  # Sub dataframe, containing all rows relevant for the current ID. Indecies are still that of the main dataframe
        
        frame_count = df_id["frame"].value_counts()  # How many times does a frame occur in the dataframe
        dual_assignment =  frame_count[frame_count == 2].sort_index() # isolating the frames with multiple assignments

        # GO through each frame with two assignments to the same ID
        for idx, sIdx in enumerate(dual_assignment.items()):
            frame, count = sIdx

            frame_idx = list(df_id[df_id["frame"] == frame].index.values)

            rows = df.iloc[frame_idx]   

            # Check if each of the rows have a detection in a different 2D view, and if so calculate the 3D position 
            if (rows.ix[frame_idx[0]]["cam1_x"] > -1.0 and rows.ix[frame_idx[1]]["cam2_x"] > -1.0) or (rows.ix[frame_idx[0]]["cam2_x"] > -1.0 and rows.ix[frame_idx[1]]["cam1_x"] > -1.0):
                row_max = rows.max()
                drop_idx.extend(frame_idx)

                minErr = np.inf
                minP = None
                minP1 = None
                minP2 = None
                minPt = None

                cam2_list = ["l","c","r"]
                if tm.camera2_useHead:
                    cam2_list = ["kpt"]

                for pt in cam2_list:
                    tr = Triangulate()
                
                    t1Pt = np.asarray([row_max["cam1_x"], row_max["cam1_y"]])
                    t2Pt = np.asarray([row_max["cam2_x"], row_max["cam2_y"]])
                    
                    # 1) Triangulate 3D point
                    p,d = tr.triangulatePoint(t1Pt,
                                            t2Pt,
                                            tm.cams[1],
                                            tm.cams[2],
                                            correctRefraction=True)
                
                    p1 = tm.cams[1].forwardprojectPoint(*p)
                    p2 = tm.cams[2].forwardprojectPoint(*p)
            
                    # 2) Calc re-projection errors
                    pos1 = np.array(t1Pt)
                    err1 = np.linalg.norm(pos1-p1)
                    pos2 = np.array(t2Pt)
                    err2 = np.linalg.norm(pos2-p2)
                    err = err1 + err2     
                    
                    if err < minErr:
                        minErr = err
                        minP = p
                        minP1 = p1
                        minP2 = p2    
                        minPt = pt
                
                # If the calculated point is within the aquairum, add it to the df, else do nothing
                if tm.withinAquarium(*minP):
                    row_max["3d_x"] = minP[0]
                    row_max["3d_y"] = minP[1]
                    row_max["3d_z"] = minP[2]
                    row_max["err"] = 1-scipy.stats.expon.cdf(minErr, scale=tm.reprojMeanErr)
                    row_max["cam1_proj_x"] = minP1[0]
                    row_max["cam1_proj_y"] = minP1[1]
                    row_max["cam2_proj_x"] = minP2[0]
                    row_max["cam2_proj_y"] = minP2[1]

                    df = df.append(row_max,ignore_index=True)

    
    return df, drop_idx



## ---- Test stuff --- ##
if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--path", help="Path to folder")
    
    args = vars(ap.parse_args())
    
    # ARGUMENTS *************
    if args.get("path", None) is None:
        print('No path was provided. Try again!')
        sys.exit()
    else:
        dataPath = args["path"]

    tm = TrackletMatcher(dataPath)
    tm.createNodes()
    tm.connectNodes3D()

    csv = pd.DataFrame()
    mergedCount = 0

    ## While there are still nodes in the graph
    while(True):
        if(len(tm.graph.nodes) == 0):
            break
        
        ## Find the largest path through the graph
        path = nx.dag_longest_path(tm.graph)
        length = nx.dag_longest_path_length(tm.graph)

        allFrames = []
        for p in path:
            allFrames += list(tm.triangulated[p].frame)

        toBeRemoved = []
        print("Best path:")
        for p in path:
            print(" ",p)

            # Save triangulated 3D information to CSV
            track3d = tm.triangulated[p]
            
            df = pd.DataFrame({
                'frame':track3d.frame,
                'id':[mergedCount]*len(track3d.frame),
                'err':track3d.errors,
                '3d_x':[q[0] for q in track3d.positions3d],
                '3d_y':[q[1] for q in track3d.positions3d],
                '3d_z':[q[2] for q in track3d.positions3d],
                'cam1_x':[q[0] for q in track3d.cam1positions],
                'cam1_y':[q[1] for q in track3d.cam1positions],
                'cam2_x':[q[0] for q in track3d.cam2positions],
                'cam2_y':[q[1] for q in track3d.cam2positions],
                'cam1_proj_x':[q[0] for q in track3d.cam1reprojections],
                'cam1_proj_y':[q[1] for q in track3d.cam1reprojections],
                'cam2_proj_x':[q[0] for q in track3d.cam2reprojections],
                'cam2_proj_y':[q[1] for q in track3d.cam2reprojections],
                'cam1_tl_x': [q[0] for q in track3d.cam1bbox],
                'cam1_tl_y': [q[1] for q in track3d.cam1bbox],
                'cam1_c_x': [q[2] for q in track3d.cam1bbox],
                'cam1_c_y': [q[3] for q in track3d.cam1bbox],
                'cam1_w': [q[4] for q in track3d.cam1bbox],
                'cam1_h': [q[5] for q in track3d.cam1bbox],
                'cam1_theta': [q[6] for q in track3d.cam1bbox],
                'cam1_aa_tl_x': [q[7] for q in track3d.cam1bbox],
                'cam1_aa_tl_y': [q[8] for q in track3d.cam1bbox],
                'cam1_aa_w': [q[9] for q in track3d.cam1bbox],
                'cam1_aa_h': [q[10] for q in track3d.cam1bbox],
                'cam1_frame': track3d.cam1frame,
                'cam2_tl_x': [q[0] for q in track3d.cam2bbox],
                'cam2_tl_y': [q[1] for q in track3d.cam2bbox],
                'cam2_c_x': [q[2] for q in track3d.cam2bbox],
                'cam2_c_y': [q[3] for q in track3d.cam2bbox],
                'cam2_w': [q[4] for q in track3d.cam2bbox],
                'cam2_h': [q[5] for q in track3d.cam2bbox],
                'cam2_theta': [q[6] for q in track3d.cam2bbox],
                'cam2_aa_tl_x': [q[7] for q in track3d.cam2bbox],
                'cam2_aa_tl_y': [q[8] for q in track3d.cam2bbox],
                'cam2_aa_w': [q[9] for q in track3d.cam2bbox],
                'cam2_aa_h': [q[10] for q in track3d.cam2bbox],
                'cam2_frame': track3d.cam2frame})
    
            # Save information from parent tracks which are
            # not already present in the saved 3D track
            for parent in [track3d.cam1Parent, track3d.cam2Parent]:   
                for f in parent.frame:
                    if(f in allFrames):
                        continue
                    
                    newRow = pd.DataFrame({
                        'frame':[f],
                        'id':[mergedCount],
                        'err':[-1],
                        '3d_x':[-1],
                        '3d_y':[-1],
                        '3d_z':[-1],
                        'cam1_x':[-1],
                        'cam1_y':[-1],
                        'cam2_x':[-1],
                        'cam2_y':[-1],
                        'cam1_proj_x':[-1.0],
                        'cam1_proj_y':[-1.0],
                        'cam2_proj_x':[-1.0],
                        'cam2_proj_y':[-1.0],
                        'cam1_tl_x': [-1.0],
                        'cam1_tl_y': [-1.0],
                        'cam1_c_x': [-1.0],
                        'cam1_c_y': [-1.0],
                        'cam1_w': [-1.0],
                        'cam1_h': [-1.0],
                        'cam1_theta': [-1.0],
                        'cam1_aa_tl_x': [-1.0],
                        'cam1_aa_tl_y': [-1.0],
                        'cam1_aa_w': [-1.0],
                        'cam1_aa_h': [-1.0],
                        'cam1_frame': [-1],
                        'cam2_tl_x': [-1.0],
                        'cam2_tl_y': [-1.0],
                        'cam2_c_x': [-1.0],
                        'cam2_c_y': [-1.0],
                        'cam2_w': [-1.0],
                        'cam2_h': [-1.0],
                        'cam2_theta': [-1.0],
                        'cam2_aa_tl_x': [-1.0],
                        'cam2_aa_tl_y': [-1.0],
                        'cam2_aa_w': [-1.0],
                        'cam2_aa_h': [-1.0],
                        'cam2_frame': [-1]})

                    # Update cam2 with correct 2D positions
                    pointType = "kpt"
                    if parent.cam == 2 and not tm.camera2_useHead:
                        maxTemporalDiff = 10
                        indToPoint = {0:"l", 1:"c", 2:"r"}
                        track3DFrames = np.asarray(track3d.frame)
                        cam2Positions = np.asarray(track3d.cam2positions)

                        frameDiff = track3DFrames - f
                        validFrames = track3DFrames[np.abs(frameDiff) <= maxTemporalDiff]

                        hist = np.zeros((3))
                        for f_t in validFrames:
                            ftPoint = np.asarray(cam2Positions[track3DFrames == f_t])
                            points = np.zeros((3))
                            points[0] = np.linalg.norm(np.asarray(parent.getImagePos(f, "l")) - ftPoint)
                            points[1] = np.linalg.norm(np.asarray(parent.getImagePos(f, "c")) - ftPoint)
                            points[2] = np.linalg.norm(np.asarray(parent.getImagePos(f, "r")) - ftPoint)
                            hist[np.argmin(points)] += 1
                        
                        if hist.sum() > 0:
                            pointType = indToPoint[np.argmax(hist)]

                    newRow['cam{0}_x'.format(parent.cam)] = parent.getImagePos(f, pointType)[0]
                    newRow['cam{0}_y'.format(parent.cam)] = parent.getImagePos(f, pointType)[1]
                    
                    newRow['cam{0}_tl_x'.format(parent.cam)] = parent.getBoundingBox(f)[0]
                    newRow['cam{0}_tl_y'.format(parent.cam)] = parent.getBoundingBox(f)[1]
                    newRow['cam{0}_c_x'.format(parent.cam)] = parent.getBoundingBox(f)[2]                    
                    newRow['cam{0}_c_y'.format(parent.cam)] = parent.getBoundingBox(f)[3]
                    newRow['cam{0}_w'.format(parent.cam)] = parent.getBoundingBox(f)[4]
                    newRow['cam{0}_h'.format(parent.cam)] = parent.getBoundingBox(f)[5]
                    newRow['cam{0}_theta'.format(parent.cam)] = parent.getBoundingBox(f)[6]
                    newRow['cam{0}_aa_tl_x'.format(parent.cam)] = parent.getBoundingBox(f)[7]
                    newRow['cam{0}_aa_tl_y'.format(parent.cam)] = parent.getBoundingBox(f)[8]
                    newRow['cam{0}_aa_w'.format(parent.cam)] = parent.getBoundingBox(f)[9]
                    newRow['cam{0}_aa_h'.format(parent.cam)] = parent.getBoundingBox(f)[10]
                    newRow['cam{0}_frame'.format(parent.cam)] = parent.getVideoFrame(f)
                    
                    df = df.append(newRow)
            csv = csv.append(df)
                       
            # Remove used tracklets
            toBeRemoved.append(p)            
            cam1 = tm.camIdMap[tm.graph.nodes[p]["cam1"]]
            cam2 = tm.camIdMap[tm.graph.nodes[p]["cam2"]]
            for e in (cam1+cam2):
                if(e not in toBeRemoved):
                    toBeRemoved.append(e)
        for e in toBeRemoved:
           if(tm.graph.has_node(e)):
               tm.graph.remove_node(e)
        mergedCount += 1
        
    csv = csv.sort_values(by=['id', 'frame'], ascending=[True,True])

    # Drop cases with exact same frame, id, and x/y coordinates, for each camera view
    csv = csv.drop_duplicates(['frame','id','cam1_x','cam1_y'])
    csv = csv.drop_duplicates(['frame','id','cam2_x','cam2_y'])
    
    csv.reset_index(inplace=True, drop=True)
    csv, drop_idx = combine2DTracklets(csv, tm)
    csv = csv.drop(drop_idx)
    csv = csv.sort_values(by=['id', 'frame'], ascending=[True,True])
    
    csv.reset_index(inplace=True, drop=True)

    # Find cases where there are several rows for the same frame in a single Tracklet, and determines which ones minimize the 3D distance (and therefore should be kept)
    csv = csv.drop(getDropIndecies(csv, True))

    outputPath = os.path.join(dataPath, 'processed', 'tracklets_3d.csv')
    print("Saving data to: {0}".format(outputPath))
    csv.to_csv(outputPath)