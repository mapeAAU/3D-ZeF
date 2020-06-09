import scipy.stats
import cv2
import argparse
import os.path
import configparser
import numpy as np
import pandas as pd
import networkx as nx
import time
from itertools import combinations

### Module imports ###
import sys
sys.path.append('../../')
from common.utility import prepareCams, csv2Tracks, readConfig, getTrackletFeatures, frameConsistencyGraph, getDropIndecies
from common.Track import Track
from modules.reconstruction.Triangulate import Triangulate

class TrackFinalizer:
    """
    Class implementation for associating 3D tracklets and create full 3D tracks     
    
    
    Links input tracklets into a predefined number of tracks
    (i.e. the known number of objects to track)
    """
    
    def __init__(self, dataPath, bboxResize = None):
        """
        Initialize object
        
        Input:
            dataPath: String path to the main folder
        """
        
        self.loadSettings(dataPath, bboxResize)
        self.path = dataPath


    def loadSettings(self,path, bboxResize):
        """
        Load settings from config file in the provided path.
        
        Config file includes information on the following, which is set in the object:
            same_view_max_overlap: The maximum allowed frame overlap of two tracklets
            max_pixel_dist: Maximum distance between pixels
            movement_err_mean: The mean value of a Gaussian distribution of movement errors
            movement_err_std: The standard deviation of a Gaussian distribution of movement errors
            
        Input:
            path: String path to the folder where the settings.ini file is located
        """
        
        config = readConfig(path)        
        
        c = config['DEFAULT']
        # Get number of fish
        self.n_fish = c.getint("n_fish")
        
        # Random number generator seed
        self.seed = c.getint("seed", None)
        np.random.seed(self.seed)            

        # Specifc 3D tracks association parameters
        c = config['TrackletLinker']
        
        ## Main track search setting
        self.min_main_overlap = c.getfloat("min_main_track_overlap_multiplier")
        self.main_track_search_multiple = c.getint("main_track_search_multiplier")

        ## Assignment settings
        self.metric_margin = c.getfloat("metric_margin")
        
        ## Track requirement settings
        self.maxTemporalDiff = c.getint("max_frame_difference")
        self.maxSpatialDiff = c.getint("max_spatial_difference")
        self.maxIntersectingFrames = c.getint("max_intersecting_frames")
        self.maxIntersectionRatio = c.getfloat("max_intersection_ratio")

        if self.maxTemporalDiff == -1:
            self.maxTemporalDiff = np.inf
        if self.maxSpatialDiff == -1:
            self.maxSpatialDiff = np.inf
        if self.maxIntersectingFrames == -1:
            self.maxIntersectingFrames = np.inf
        if self.maxIntersectionRatio == -1:
            self.maxIntersectionRatio = np.inf


    ### Calculate distances
    def getTemporalShift(self, mainTrack, galleryTrack):
        '''
        Calculates the temporal distance between the main and gallery track. If the tracks are overlapping, a distance of -1 is returned
        
        Input:
            mainTrack: Track object of the main track
            galleryTrack: Track obejct of the gallery track
        
        Output:
            tuple: Contains the order of the tracks; 0 for gallery track first, 1 for gallery track last, and the temporal distance. Both are -1 for overlapping tracks
        '''
        ## Currently doesnt accept galleryTracks that occupy any of the mainTrack span
        
        if mainTrack.frame[0] > galleryTrack.frame[-1]:
            return (0, mainTrack.frame[0] - galleryTrack.frame[-1])
        elif mainTrack.frame[-1] < galleryTrack.frame[0]:
            return (1, galleryTrack.frame[0] - mainTrack.frame[-1])
        else:
            return (-1,-1)
        

    def getSpaitalDistance(self, mainTrack, galleryTrack, trackOrder):
        '''
        Calculates the spatial distance between two tracks, depending on the temporal order of the tracks
        If the tracks are overlapping, -1 is returned

        Input:
            mainTrack: Track object of the main track
            galleryTrack: Track obejct of the gallery track
            trackOrder: Indicator of the order of the tracks

        Output:
            float: Euclidean distance between the two tracks
        '''

        if trackOrder == 0:
            mtIdx = 0
            gtIdx = -1
            
        elif trackOrder == 1:
            mtIdx = -1
            gtIdx = 0
        else:
            return -1
        
        mt3D = self.frameTo3DCoord(mainTrack, mtIdx)
        gt3D = self.frameTo3DCoord(galleryTrack, gtIdx)
    
        return np.linalg.norm(gt3D - mt3D)


    def getInternalDistances(self, mainTrack, galleryTrack):
        '''
        Determines spatio-temporal distances when two tracks overlap (Using only the known 3D positions).
        This is done by constructing a graph, where each node is a detection in a frame, and the edge weights are the reciporal spatial distance between the adjacent nodes
        The path which minimizes the total travelled distance is then found.
        THe spatio-temporal distances are then based on the times where the graph switches between the tracks i.e. we base it on when we actually combine the tracks.

        Input:
            mainTrack: Track object of the main track
            galleryTrack: Track object of the gallery track

        Output:
            nSpatialDist: List of spatial distances from the found graph, at the points where the graph jumps between tracks
            nTemporalDist: List of temporal distances from the found graph, at the points where the graph jumps between tracks
            validIndecies: Dict contianing the indecies per track, which should be kept after solving the graph
        '''

        mtStart = mainTrack.frame[0]
        mtEnd = mainTrack.frame[-1]
        gtStart = galleryTrack.frame[0]
        gtEnd = galleryTrack.frame[-1]

        start = max(mtStart, gtStart)
        end = min(mtEnd, gtEnd)

        frames = list(np.linspace(start, end, end-start+1, dtype=np.int))

        ## Add the detections just before/after the intersection, if there are any
        for track in [mainTrack, galleryTrack]:
            if start > track.frame[0]:
                vals = start - track.frame
                vals[vals <= 0] = np.max(vals)+1 
                frames.append(track.frame[np.argmin(vals)])
            if end < track.frame[-1]:
                vals = track.frame - end
                vals[vals <= 0] = np.max(vals)+1 
                frames.append(track.frame[np.argmin(vals)])

        ## Create a dataframe containing all the relevant detections
        track_id = "{}-{}".format(mainTrack.id, galleryTrack.id)
        df_total = pd.DataFrame()
        for frame in frames:
            for track in [mainTrack, galleryTrack]:
                if frame in track.frame:
                    index = track.frame == frame
                    
                    # print(index)
                    # print(track.id)
                    # print(track.frame)
                    if track.x[index] == -1:
                        continue

                    df = pd.DataFrame({
                        'frame': frame,
                        'id': track_id,
                        '3d_x': track.x[index],
                        '3d_y': track.y[index],
                        "3d_z": track.z[index],
                        "source": track.id})
                    df_total = df_total.append(df)

        df_total.reset_index(inplace=True, drop=True)

        multi_idx = np.linspace(0, len(df_total)-1, len(df_total), dtype=np.int)

        # Get the indecies needed for the graph, their corresponding frames and 3D positions
        multi_idx, frames, coords = getTrackletFeatures(multi_idx, df_total)
        
        # Create graph and get and keep the indecies which minimizes the distance between the possible nodes
        path_idx, _, spatialDist, temporalDist = frameConsistencyGraph(multi_idx,frames,coords,False)

        # Find the indecies which from each track which should be kept when combined
        validIndecies = {}
        for track in [mainTrack, galleryTrack]:
            start_frames = list(track.frame[track.frame < start])
            end_frames = list(track.frame[track.frame > end])

            graph_idx = []
            for idx in path_idx:
                row = df_total.iloc[idx]
                if row["source"] == track.id:
                    graph_idx.append(row["frame"])  
            
            fullList = []
            for lst in [start_frames, end_frames, graph_idx]:
                fullList.extend(lst)
            fullList = sorted(list(set(fullList)))
            validIndecies[track.id] = [True if x in fullList else False for x in track.frame]

        # Only use the distances for when swapping between the two tracks
        nSpatialDist = []
        nTemporalDist = []

        for idx in range(1, len(path_idx)):
            prevRow = df_total.iloc[path_idx[idx-1]]
            curRow = df_total.iloc[path_idx[idx]]

            if prevRow["source"] != curRow["source"]:
                nSpatialDist.append(spatialDist[idx-1])
                nTemporalDist.append(temporalDist[idx-1])

        if len(nSpatialDist) == 0:
            nSpatialDist.append(0)
        if len(nTemporalDist) == 0:
            nTemporalDist.append(0)

        return nSpatialDist, nTemporalDist, validIndecies


    def getDistances(self, galleryTrackID, mainTrackID):
        '''
        Calculates all interesting distances between the provided gallery track and main track.

        Input:
            galleryTrackID: ID of the gallery track
            mainTrackID: ID of the main track

        
        Output:
            temporalShift: Temporal distance between tracks
            spatialDiff: Spatial distance between tracks
            intersecting_frames: Amount of time there is a detection at the same frame in both tracks
            intersection_ratio: intersecting_frames as a ratio of the length of the gallery track
            internalTemporalDist: Mean temporal distance when combining two overlapping tracks
            internalSpatialDist: Mean spatial distance when combining two overlapping tracks
            trackIndecies: Dict of indecies, idnicating which detections from each track should be used, in case of overlapping tracks
            trackOrder: Temporal order of tracks
            validTrack: Whether the track is valid i.e. whether all values are within user defiend thresholds
        '''

        mt = self.mainTracks[mainTrackID]            
        gt = self.tracks[galleryTrackID]
        
        # Get spatio-temporal distances
        trackOrder, temporalShift = self.getTemporalShift(mt, gt)
        spatialDiff = self.getSpaitalDistance(mt, gt, trackOrder)
        
        # Get detection overlap between main and gallerytrack, as count and ratio of full gallery track length
        intersecting_frames = len(np.intersect1d(mt.frame, gt.frame))
        intersection_ratio = intersecting_frames/len(gt.frame)

        # If tracks are overlapping, get internal spatio-temporal distances. Else set to invalid values
        if temporalShift == -1:
            internalSpatialDistM, internalTemporalDistM, trackIndecies = self.getInternalDistances(mt, gt)
            internalSpatialDist = np.mean(internalSpatialDistM)
            internalTemporalDist = np.mean(internalTemporalDistM)
        else:
            internalSpatialDist = -1
            internalTemporalDist = -1
            trackIndecies = {mt.id: [True]*len(mt.frame),
                            gt.id: [True]*len(gt.frame)}
        
        
        # Check if any of the derived values are outside of the user defiend thresholds
        if temporalShift <= self.maxTemporalDiff and spatialDiff <= self.maxSpatialDiff and intersecting_frames <= self.maxIntersectingFrames and intersection_ratio <= self.maxIntersectionRatio:
            validTrack = True
        else:
            validTrack = False
        
        return temporalShift, spatialDiff, intersecting_frames, intersection_ratio, internalTemporalDist, internalSpatialDist, trackIndecies, trackOrder, validTrack
          


    ### Cost based assignment approach
    def calcCost(self, m):
        '''
        Takes a matrix, and normalizes values into a range of [0;1], where all values sum to 1, i.e. into cost/probability values.

        Input:
            m: Matrix containing values

        Output:
            m_prob: matrix containing cost values
        '''

        return m / np.sum(m, 0)


    def getCosts(self, metric_dict):   
        '''
        Takes a metric dictionary, and converts the values into cost values (through the use of the calcCost function).

        Input:
            metric_dict: Dictionary containing all valid metric data

        Output:
            res: Dictionary cotnaining the same keys as metric_dict, but now with probabiltiy values
        '''

        res = {}
        for key in metric_dict:
            # Find all valid values in the current metric (invalid values have negative values)
            m = metric_dict[key]
            index = m >= 0

            if sum(index) > 0 :
                m_prob = np.zeros_like(m)
                m_prob[index] = self.calcCost(m[index])
            else:
                m_prob = np.ones(self.n_fish) * 1/self.n_fish
    
            ## Set invalid values to infinite
            m_prob[~index] = np.inf

            res[key] = m_prob
        return res


    def getMetrics(self, tDistm, sDistm, iFramesm, iFramesRatiom, iTDistm, iSDistm, verbose = False):
        '''
        Calcualtes all the relevant metrics and returns them in a dictionary.

        Re-Id values are always saved. If there are only a subset of main tracks that are overlapping, then these re-id values are excluded (by setting to -1)
        If all main tracks are not overlapping, then the spatio-temporal distances are used.

        If all tracks are overlapping, then the internal spatio-temporal distances, the amount of concurrent detections, and how large a % of the gallery track this is, is used.

        Input:
            tDistm: distance matrix containing temporal distances
            sDistm: distance matrix containing spatial distances
            iFramesm: matrix containing amount of concurrent frames between gallery track and each main tracks
            iFramesRatiom: matrix containing iFramesm values but as a ratio of gallery track length
            iTDistm: distance matrix of the internal temporal distances
            iSDistm: distance matrix ofhte internal spatial distances.
            verbose: Whether to print information

        Output:
            metric_dict: Dictionary containing values from all metrics that are to be used
        '''

        metric_dict = {}
        temporalData = False
        spatialData = False
        internalSpatialData = False
        internalTemporalData = False
        intersectionData = False
        intersectionRatioData = False

        if sum(tDistm == -1) < len(tDistm): # Check if there are any valid temporal data
            temporalData = True
            metric_dict["temporal"] = tDistm
        if sum(sDistm == -1) < len(sDistm): # Check if there are any valid spatial data
            spatialData = True
            metric_dict["spatial"] = sDistm

        concurrentData = not temporalData and not spatialData

        if concurrentData:
            # Check if there are any frames where the gallery and any main track intesects
            if sum(iFramesm <= 0) < len(iFramesm):
                internalSpatialData = internalTemporalData = intersectionData = True
                metric_dict["intersection"] = iFramesm
                metric_dict["internal_temporal"] = iTDistm
                metric_dict["internal_spatial"] = iSDistm

                # Check if there are any frames where gallery track and main tracks both have detections
                if np.sum(iFramesRatiom) != 0:
                    intersectionRatioData = True
                    metric_dict["intersectionRatio"] = iFramesRatiom

        if verbose:
            print("Valid metrics\n\tTemporal: {}\n\tSpatial: {}\n\tIntersection: {}\n\tIntersection Ratio: {}\n\tInternal Temporal: {}\n\tInternal Spatial: {}\n".format(temporalData, spatialData, intersectionData, intersectionRatioData, internalTemporalData, internalSpatialData))

        return metric_dict


    def costAssignment(self, tDistm, sDistm, iFramesm, iFramesRatiom, iTDistm, iSDistm, verbose=False):
        # If selected, perform intial Re-ID check. Else just get all metric probabilities

        metric_dict = self.getMetrics(tDistm, sDistm, iFramesm, iFramesRatiom, iTDistm, iSDistm)
        res = self.getCosts(metric_dict)

        ## Calculate the votes per main track. A low probabiltiy is desirable, as we are working with distances
        votes = np.zeros(self.n_fish)
        voteSum = np.zeros(self.n_fish)
        metrics = 0
        for key in res:
            votes[np.argmin(res[key])] += 1
            voteSum += res[key]
            metrics += 1
        
        if verbose:
            print(res)
            print("Votes {}".format(votes))
            print("Vote Sum {}".format(voteSum))
            print("Normalized Vote Sum {}".format(voteSum / metrics))

        votes = voteSum/metrics

        # Check if the tracks are equally propable
        if self.n_fish > 1:
            inf_counter = 0
            equal_prob = True
            for idx1 in range(self.n_fish):
                if votes[idx1] == np.inf:
                    inf_counter += 1
                    continue
                for idx2 in range(idx1, self.n_fish):

                    if np.abs(votes[idx1]-votes[idx2]) > self.metric_margin:
                        equal_prob = False
                        break
            
            if inf_counter == self.n_fish-1:
                equal_prob = False

            if equal_prob:
                print("Equal prob")
                return (False, -1)

        mtID = np.argmin(votes)

        return (True, mtID)
    

    ### Tracklet association code
    def combineTracklets(self, mainTrackID, galleryTrackID, indecies):
        '''
        Takes two Tracks and combine them into a single track. Used when having to merge a gallery track with a main track
        The 3D position, frames, and features/labels are combined.

        Input:
            mainTrackID: ID of the main track
            galleryTrackID: ID of the gallery track
            indecies: Dict containing a boolean list per track, indicating which elements from each track should be used (Used when internal spatio-temporal distances has been considered)

        Output:
            No output
        '''

        mt = self.mainTracks[mainTrackID]
        gt = self.tracks[galleryTrackID]

        mt_indecies = indecies[mt.id]
        gt_indecies = indecies[gt.id]

        mt_frames = mt.frame[mt_indecies]
        gt_frames = gt.frame[gt_indecies]
        
        mt.frame = np.concatenate((mt_frames, gt_frames))
        mt.x = np.concatenate((mt.x[mt_indecies], gt.x[gt_indecies]))
        mt.y = np.concatenate((mt.y[mt_indecies], gt.y[gt_indecies]))
        mt.z = np.concatenate((mt.z[mt_indecies], gt.z[gt_indecies]))

        mt.sort()
        

    def addMainTrack(self, mtID):
        '''
        Takes the provided track and adds to the list of main tracks as a new Track obejct. The 3D position, id and frames are copied over.

        Input:
            mtID: ID of the track we want to add as a main track
        '''
        self.mainTracks[mtID] = Track()
        self.mainTracks[mtID].id = self.tracks[mtID].id
        self.mainTracks[mtID].frame = self.tracks[mtID].frame
        self.mainTracks[mtID].x = self.tracks[mtID].x
        self.mainTracks[mtID].y = self.tracks[mtID].y
        self.mainTracks[mtID].z = self.tracks[mtID].z


    def frameTo3DCoord(self, track, frameIdx):
        '''
        Returns the 3D position at a specified frame in a track

        Input:
            track: Track object
            frameIdx: Index of the frame we want the 3D position of

        Output:
            numpy array: Array containing the 3D position
        '''

        return np.array([track.x[frameIdx], track.y[frameIdx], track.z[frameIdx]])
    

    def rankTrackletTemporally(self, galleryTracks, mainTracks, verbose = True):
        '''
        Goes through all gallery tracks, determine the distance to each main track, and sorts them based on the shortest distance. 
        If the gallery track overlaps with any main track, it is relagated to the end of the list

        Input:
            galleryTracks: List of IDs for the gallery tracks
            mainTracks: List of IDs for the main tracks
            verbose: Whether to print information about each track

        Output:
            tempDistList: List of tuples, constructed as (trackID, minimum distance to any main track, track length). The list is sorted by distance, and length for tiebreakers
            orderedGalleryTracks: List of trackIDs from tempDistList
        '''
        tempDistList = []
        
        for gTrack in galleryTracks:
            gt = self.tracks[gTrack]
            
            min_diff = np.inf
            for mTrack in mainTracks:
                mt = self.mainTracks[mTrack]
                index = self.checkTrackIntersection(mt, gt)
                
                if index == -1:
                    order, diff = self.getTemporalShift(mt, gt)
                    
                    if diff < min_diff:
                        min_diff = diff
                        
            tempDistList.append((gTrack, min_diff, len(gt.frame)))    
            
        tempDistList = sorted(tempDistList, key = lambda x: (x[1], x[2]))
        
        orderedGalleryTracks = [x[0] for x in tempDistList]
        
        if verbose:
            for mTrack in mainTracks:
                mt = self.mainTracks[mTrack]
                print("Main track {} - length {} - detections {} - consistency {} - start {} - end {}".format(mTrack, mt.frame[-1]-mt.frame[0]+1, len(mt.frame), len(mt.frame)/(mt.frame[-1]-mt.frame[0]+1), mt.frame[0], mt.frame[-1]))
            
            for lst in tempDistList:
                print("Track {} - temporal Distance {} - length {} - start {} - end {}".format(lst[0], lst[1], lst[2], self.tracks[lst[0]].frame[0], self.tracks[lst[0]].frame[-1]))
            print()
        
        return tempDistList, orderedGalleryTracks


    def checkTrackIntersection(self, mainTrack, galleryTrack):
        '''
        Checks if two tracks are overlapping, and if so returns an integer indicating how.

        -1 = No overlap
        0  = Gallery track is entirely overlapped by main track
        1  = Main track is entirely overlapped by gallery track
        2  = Gallery track end is within main track
        3  = Gallery track start is within main track

        Input:
            mainTrack: Track object of the main track
            galleryTrack: Track object of the gallery track

        Output:
            int: Indicator of how the temporal intersection of the tracks is.

        '''
        
        if mainTrack.frame[0] > galleryTrack.frame[-1] or mainTrack.frame[-1] < galleryTrack.frame[0]:
            ## Tracks don't occupy same temporal span
            return -1
        
        start = galleryTrack.frame[0] > mainTrack.frame[0] ## Gallery track starts after main track
        end = galleryTrack.frame[-1] < mainTrack.frame[-1] ## Gallery track ends before main track
    
        if start and end:
            ## Gallery track start and end are in the span of the main track
            return 0
        elif not start and not end:
            ## Main track start and end are in the span of the gallery track
            return 1
        elif end:
            ## Gallery track end before main track
            return 2
        elif start:
            ## Gallery track start after main track
            return 3


    def connectTracklets(self, galleryTracks, mainTracks):
        '''
        Takes N main track seeds, and tries to associate all other tracks to each of these, using a greedy algorithm.
        This is done using a combination of temporal, spatial, and spatio-temporal distance.
        
        All unassigned tracks are sorted in ascending order based on their closest temporal distance to any main track. If overlapping, they are relegated to the end of the list
        The closest unassigned track is then assigned. If all but 1 of the tracks overlap with the gallery track, it is assigned the non overlapping.
        If there are M < N main tracks that dont overlap with the gallery track, only compare these using spatio-temporal distances and re-id.
        
        If the track is overlapping with all main tracks, then the internal spatio-temporal distance is used, as well as the amount of overlap, and how frames have concurrent detections, and re-id.

        Input:
            galleryTracks: List of IDs of gallery tracks
            mainTracks: List of IDs of main tracks

        Output:
            paths: List of lists, consisting of associated IDs
        '''     

        assignLog = []
        
        notAssigned = []
        outputDict = {}
        assignedCounter = 0

        # Create dummy main tracks and assign data
        self.mainTracks = {}
        for mTrack in mainTracks:
            outputDict[mTrack] = [mTrack]
            self.addMainTrack(mTrack)
        
        while len(galleryTracks) > 0:

            ## Find next gallery track
            tempDistList, galleryTracks = self.rankTrackletTemporally(galleryTracks, mainTracks)
            gTrack = tempDistList[0][0]
            galleryTracks.pop(0)
            
            # Per mainTrack, get distance values
            tDistm = np.ones(self.n_fish) * -1 # temporal distance
            sDistm = np.ones(self.n_fish) * -1 # spatial distance
            iFramesm = np.ones(self.n_fish) * -1 # number of intersecting frames
            iFramesRatiom = np.ones(self.n_fish) * -1 # ratio of intersecting frames and total frames in gallery track
            iTDistm = np.ones(self.n_fish) * -1 # internal temporal distance
            iSDistm = np.ones(self.n_fish) * -1 # internal spatial distance
            validIndecies = [-1]*self.n_fish
            validm = np.zeros(self.n_fish)

            for idx, mTrack in enumerate(mainTracks):
                distTuple = self.getDistances(gTrack, mTrack)
                validm[idx] = distTuple[-1]

                if validm[idx]:
                    tDistm[idx] = distTuple[0]
                    sDistm[idx] = distTuple[1]
                    iFramesm[idx] = distTuple[2]
                    iFramesRatiom[idx] = distTuple[3]
                    iTDistm[idx] = distTuple[4]
                    iSDistm[idx] = distTuple[5]
                    validIndecies[idx] = distTuple[6]


            # Check if there are any valid tracks
            if sum(validm) == 0:
                ## No valid tracks available
                assignLog.append("Gallery {} - No valid main tracks".format(gTrack))
                notAssigned.append(gTrack)
                continue
            
            assigned, mtID = self.costAssignment(tDistm, sDistm, iFramesm, iFramesRatiom, iTDistm, iSDistm)

            if assigned:
                print("Adding gallery track {} to main track {}".format(gTrack, mainTracks[mtID]))
                assignLog.append("Gallery {} - Assigned to main track {}".format(gTrack, mainTracks[mtID]))
            else:
                assignLog.append("Gallery {} - Equal Prob".format(gTrack))
                notAssigned.append(gTrack)
                continue      
            
            self.combineTracklets(mainTracks[mtID], gTrack, validIndecies[mtID])
            outputDict[mainTracks[mtID]].append(gTrack)
            assignedCounter += 1
            print()
        
        ## Statistics of the constructed main tracks
        for mTrack in mainTracks:
            mt = self.mainTracks[mTrack]
            gap = []
            sDist = []
            for i in range(1, len(mt.frame)):
                diff = mt.frame[i] - mt.frame[i-1]
        
                t1 = self.frameTo3DCoord(mt, i)
                t2 = self.frameTo3DCoord(mt, i-1)
                
                sDist.append(np.linalg.norm(t1-t2))

                if sDist[-1] == 0:
                    if mt.frame[i] == mt.frame[i-1]:
                        print(mt.frame[i], mt.frame[i-1])

                if diff > 1:
                    gap.append(diff)
            
            print("Main Track {}:".format(mTrack))
            print("\tLength {} - Detections {} - Consistency {} - Start {} - End {}".format(mt.frame[-1]-mt.frame[0]+1, len(mt.frame), len(mt.frame)/(mt.frame[-1]-mt.frame[0]+1), mt.frame[0], mt.frame[-1]))
            if len(gap) > 0:
                print("\tLargest Gap {} - Mean Gap {} - Median Gap {} - Std Dev Gap {} - Max Gap {} - Min Gap {} - # Gaps {}".format(np.max(gap), np.mean(gap), np.median(gap), np.std(gap), np.max(gap), np.min(gap), len(gap)))
            if len(sDist) > 0:
                print("\tLargest Dist {} - Mean Dist {} - Median Dist {} - Std Dev Dist {} - Max Dist {} - Min Dist {}\n".format(np.max(sDist), np.mean(sDist), np.median(sDist), np.std(sDist), np.max(sDist), np.min(sDist)))

        with open(os.path.join(self.path, "processed", "assigned.txt"), "w") as f:
            f.write("\n".join(assignLog))

        if notAssigned:
            na_len = []
            na_ids = []
            for naTrack in notAssigned:
                na = self.tracks[naTrack]
                na_len.append(len(na.frame))
                na_ids.append(naTrack)
            print("Largest unassigned tracklet: ID {} - Mean {} - Median {} - Std Dev {} - Min {} - Max {} - # {}".format(na_ids[np.argmax(na_len)], np.mean(na_len), np.median(na_len), np.std(na_len), np.min(na_len), np.max(na_len), len(na_len)))

        paths = []
        for key in outputDict:
            paths.append(outputDict[key])
        
        return paths



    ### Initial main track selection
    def getTemporalOverlap(self, mainTrack, galleryTrack):
        '''
        Calculates the temporal overlap between the provided tracks, based on their first and last frame number.
        If there is no overlap, a value of -1 is returned

        Output:
            tuple: The first element indicates whether there is an overlap, and the second the overlap itself.
        '''

        if mainTrack.frame[0] <= galleryTrack.frame[-1] and galleryTrack.frame[0] <= mainTrack.frame[-1]:
            overlap = min(galleryTrack.frame[-1] - mainTrack.frame[0], mainTrack.frame[-1]-galleryTrack.frame[0])
            return (0, overlap)
        else:
            return (-1,-1)


    def findConccurent(self, trackIds, overlaping = True, fully_concurrent = False, verbose = True):
        """
        Finds the concurrent tracks between a specific track and a set of other tracks 
        Can find both partially overlapping and fully concurrent tracklets
        
        Input:
            track: A Track object
            candidates: List of Track objects
            
        Output:
            concurrent: List of Track objects from candidates that were overlaping with the track argument
        """
        
        tracks = self.tracks
        
        if verbose: 
            print()
            print(trackIds)
        
        concurrent = []
        for idx1 in range(len(trackIds)):
            
            track1Id = trackIds[idx1]
            track1 = tracks[track1Id]
            if verbose:
                print("Track1: {}, fStart {},  fEnd {}, # frames {}, # missing frames {}".format(track1Id, tracks[track1Id].frame[0], tracks[track1Id].frame[-1], len(tracks[track1Id].frame), (tracks[track1Id].frame[-1] - tracks[track1Id].frame[0] + 1) - len(tracks[track1Id].frame)))
            
            assigned = []
            trackList = []
            for idx2 in range(idx1+1, len(trackIds)):
                
                if idx1 == idx2:
                    continue
                
                track2Id = trackIds[idx2]    
                track2 = tracks[track2Id]
                if verbose:
                    print("Track2: {}, fStart {},  fEnd {}, # frames {}, # missing frames {}".format(track2Id, tracks[track2Id].frame[0], tracks[track2Id].frame[-1], len(tracks[track2Id].frame), (tracks[track2Id].frame[-1] - tracks[track2Id].frame[0] + 1) - len(tracks[track2Id].frame)))
      
                interCheck = self.checkTrackIntersection(track1, track2)   

                if interCheck == -1:
                    if verbose:
                        print("Not intersecting ", track1Id, track2Id)
                    ## If the tracks dont share the same time span
                    continue
                
                if not fully_concurrent:
                    if interCheck == 0:
                        ## If track2 is fully in the span of track1. Only track1 is kept
                        assigned.append(track2Id)
                        continue
                    
                    if interCheck == 1:
                        ## If track1 is fully in the span of track2. Only track2 is kept
                        assigned.append(track1Id)
                        continue
                    
                if not overlaping:
                    if interCheck == 2 or interCheck == 3:
                        continue
                if verbose:
                    print("\tKept")
                trackList.append(track2Id)
                assigned.append(track2Id)

            if track1Id not in assigned:
                trackList.append(track1Id)
                assigned.append(track1Id)
                
            if len(trackList) > 0:
                concurrent.append(trackList)
            
            if verbose:
                print()
        if verbose:
            print(concurrent)
            print(assigned)
            print()
        return concurrent
    

    def findMainTracks(self):
        """
        Finds all cases where there are N tracks that are both long and overlapping. These tracks are used as seeds for the tracklet association.
        The tracks have to be of a certain length and have a certain amount of detections in them before they are considered.
        The minimum length and number of detections are determined by taking the value of the (N*user_multiple)th top value

        All tracks are then checked for overlap, and all valid cases where N tracks overlap eachother more than a user defined value, are then returned.
            
        Output:
            main_tracks: A list of lists containing the set of main track IDs which can be used as seeds
        """

        detections = []
        length = []
        ids = []
        for tr in self.tracks:
            track = self.tracks[tr]
            ids.append(tr)
            detections.append(len(track.frame))
            length.append(len(np.linspace(track.frame[0], track.frame[-1], track.frame[-1]-track.frame[0] + 1, dtype=np.int)))

            print("Track {}: Start {}, End {}, Length {}, Detections {}".format(tr, track.frame[0], track.frame[-1], length[-1], detections[-1]))

        sorted_length = sorted(length)
        sorted_detections = sorted(detections)

        candidateN = min(len(self.tracks), self.n_fish*self.main_track_search_multiple)
        candidateN = len(self.tracks)
        
        min_length = sorted_length[-candidateN]
        min_detections = sorted_detections[-candidateN]

        min_overlap = 1#min_length * self.min_main_overlap

        print("Minimum tracklet length {}".format(min_length))
        print("Minimum tracklet detections {}".format(min_detections))
        print("Minimum tracklet overlap {}".format(min_overlap))

        main_tracks_init = []
        for idx in range(len(self.tracks)):
            if length[idx] >= min_length and detections[idx] >= min_detections:
                main_tracks_init.append(ids[idx])

        # find which of the initially selected tracks are concurrent
        concurrent_tracks = self.findConccurent(main_tracks_init, overlaping = True, fully_concurrent = True, verbose = False)

        # For all set of concurrent tracks go through and see how much they overlap
        main_tracks = []
        overlap_lst = []
        checked_sets = []
        no_overlap_dict = {x:[] for x in self.tracks}

        for conc_set in concurrent_tracks:
            if len(conc_set) < self.n_fish:
                # If fewer tracks than fish in the set, discard it
                continue
            else:
                # Go through all combinations of the tracks, where there are n_fish tracks in a set, and keep the ones where the overlap between all tracks is big enough
                # All tracks should overlap so that e.g. for 3 fish it cannot be 
                #               |----------------------------------------|
                #   |-----------------|                      |----------------------------|
                for comb_set in combinations(conc_set, self.n_fish):
                    if comb_set in checked_sets:
                        continue
                    else:
                        checked_sets.append(comb_set)

                    valid_set = True
                    for track in comb_set:
                        if track in no_overlap_dict:
                            if set(no_overlap_dict[track]).intersection(set(comb_set)):
                                valid_set = False
                                break
                    
                    if not valid_set:
                        continue

                    med_overlap = []
                    for idx1 in range(self.n_fish):
                        if not valid_set:
                            break
                        for idx2 in range(idx1+1, self.n_fish):
                            tr1 = comb_set[idx1]
                            tr2 = comb_set[idx2]

                            _, overlap = self.getTemporalOverlap( self.tracks[tr1], self.tracks[tr2])
                            med_overlap.append(overlap)

                            if overlap < min_overlap:
                                valid_set = False
                                no_overlap_dict[tr1].append(tr2)
                                no_overlap_dict[tr2].append(tr1)
                    if valid_set:
                        main_tracks.append(list(sorted(comb_set)))
                        overlap_lst.append(np.median(med_overlap))
        if len(overlap_lst) > 0:
            sort = np.argmax(overlap_lst)
            return [main_tracks[sort]]
        else:
            return []

    

    ### Main Function
    def matchAll(self, tracks, verbose=True):
        """
        Matches all tracklets by using the proposed greedy association algorithm.
        Currently uses N_fish "certain" tracklets, and then tries to associate all other trackelts to these.
        
        Input:
            tracks: Dict of Track objects
            verbose: Whether to print information  
            
        Output:
            paths: A list of lists containing the associated tracks by their IDs
        """
        
        self.tracks = tracks
        
        ## Find main and gallery tracks
        mainTracks_all = self.findMainTracks()
        
        if len(mainTracks_all) == 0:
            return [[x] for x in tracks.keys()]
        
        paths = []

        for mainTracks in mainTracks_all:
            galleryTracks = []
            for track in tracks:
                if self.tracks[track].id not in mainTracks:
                    galleryTracks.append(self.tracks[track].id)
            
            print("Main tracks: {}".format(mainTracks))
            print("Gallery tracks: {}".format(galleryTracks))
            print()
            
            print(self.findConccurent(galleryTracks, overlaping = True, fully_concurrent = True, verbose = False))
            
            full_tracks = self.connectTracklets(galleryTracks.copy(), mainTracks.copy())

            for idx in range(self.n_fish):
                full_tracks[idx] = sorted(full_tracks[idx])

            paths.append(full_tracks)
        
        print("Paths: {}".format(paths))

        if len(paths) > 1:
            for idx1 in range(self.n_fish):
                for idx2 in range(idx1, self.n_fish):
                    for idx3 in range(self.n_fish):
                        if paths[idx1][idx3] != paths[idx2][idx3]:
                            print("ERROR {}-{}-{} / {}-{}-{}".format(idx1, idx3, paths[idx1][idx3], idx2, idx3, paths[idx2][idx3]))

        return paths[0]
    

def updateIds(df, paths, verbose=True):
    """
    Updates the ids of the tracks in a pandas dataframe according to the linked paths
    
    Input:
        df: A pandas dataframe, consisting of the output of TrackletMatching.py
        paths: A list of lists containing the linked paths
        verbose: Whether to print information per path
        
    Output: 
        linked: An updated pandas dataframe with the linked paths
    
    """
    
    linked = pd.DataFrame()
    newTracksCount = 0

    # Loop through all paths
    for path in paths:
        if(verbose):
            print("Path: ", path)
            
        # Loop through each element in the current path
        for e in path:
            currTracklet = df[df['id'] == e].copy()
            currTracklet['id'] = newTracksCount
            linked = linked.append(currTracklet)        
        newTracksCount += 1
        
    linked = linked.sort_values(by=['id', 'frame'], ascending=[True,True])
    return linked


def interpolate(df, dataPath):    
    """
    In case of missing frames in the constructed 3D tracks, the positons are linearly interpolated
    
    Input:
        df: Pandas Dataframe containing a set of 3D tracks
        dataPath: String path to the main folder
        
    Output:
        interpolated: Pandas dataframe containing the interpolated 3D tracks
    """
    
    config = readConfig(dataPath)
    c = config['TrackletMatcher']
    reprojMeanErr = c.getfloat('reprojection_err_mean')

    # Prepare 'interpolated' column
    df['interpolated'] = [0]*len(df)

    # Prepare dummy row
    dummyRow = df.iloc[0].copy()
    dummyRow['3d_x'] = -1
    dummyRow['3d_y'] = -1
    dummyRow['3d_z'] = -1
    dummyRow['err'] = -1
    
    dummyRow['cam1_proj_x'] = -1
    dummyRow['cam1_proj_y'] = -1
    dummyRow['cam1_x'] = None
    dummyRow['cam1_y'] = None

    dummyRow['cam2_proj_x'] = -1
    dummyRow['cam2_proj_y'] = -1
    dummyRow['cam2_x'] = None
    dummyRow['cam2_y'] = None

    # Interpolate 2D detections
    interpolated = pd.DataFrame()
    trackIds = df['id'].unique()
    for i in trackIds:
        currTrack = df[df['id'] == i]

        # Insert dummy rows at missing frames
        frames = currTrack['frame'].unique()
        for f in range(int(frames[0]),int(frames[-1])):
            # Check if frame is missing
            if(len(currTrack[currTrack['frame'] == f]) == 0):
                dummyRow['frame'] = f
                dummyRow['id'] = i
                currTrack = currTrack.append(dummyRow)
        # Sort frames ascending
        currTrack = currTrack.sort_values(by=['frame'], ascending=[True])
        
        # Linear interpolation of cam1 and cam2 information
        currTrack = currTrack.astype(float)
        for columnHeader in currTrack.columns:
            if "cam" in columnHeader:
                tmp = currTrack[columnHeader].copy()
                tmp[tmp == -1] = np.nan
                currTrack[columnHeader] = tmp.interpolate(method="linear", limit_direction = "both", axis=0)
        interpolated = interpolated.append(currTrack,sort=False)

    # Remove rows with NaN values
    interpolated = interpolated[np.isfinite(interpolated['cam1_x'])]
    interpolated = interpolated[np.isfinite(interpolated['cam2_x'])]

    # Triangulate using interpolated 2D detections
    interpolatedRows = np.where(interpolated['err'] == -1)[0]
    tr = Triangulate()
    cams = prepareCams(dataPath)
    newRows = pd.DataFrame()
    for row in interpolatedRows:
        currRow = interpolated.iloc[row].copy()
        
        cam1Pos = (currRow['cam1_x'],currRow['cam1_y'])
        cam2Pos = (currRow['cam2_x'],currRow['cam2_y'])
        
        p,d = tr.triangulatePoint(cam1Pos,cam2Pos,
                                  cams[1],cams[2],
                                  correctRefraction=True)
    
        p1 = cams[1].forwardprojectPoint(*p)
        p2 = cams[2].forwardprojectPoint(*p)

        # Calc reprojection error and probability
        err1 = np.linalg.norm(np.array(cam1Pos)-p1)
        err2 = np.linalg.norm(np.array(cam2Pos)-p2)        
        err = err1 + err2                
        currRow['err'] = 1-scipy.stats.expon.cdf(err, scale=reprojMeanErr)
        
        # Update CSV with 3D positions
        currRow['3d_x'] = p[0]
        currRow['3d_y'] = p[1]
        currRow['3d_z'] = p[2]

        # Update CSV with reprojected 2D positions
        currRow['cam1_proj_x'] = p1[0]
        currRow['cam1_proj_y'] = p1[1]
        currRow['cam2_proj_x'] = p2[0]
        currRow['cam2_proj_y'] = p2[1]
        currRow['interpolated'] = 1
        newRows = newRows.append(currRow,sort=False)

    interpolated = interpolated[interpolated['err'] != -1]
    interpolated = interpolated.append(newRows,sort=False)
    return interpolated



## --- MAIN! --- ##    
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

    
    # Load tracklets and CSV file
    tracks = csv2Tracks(os.path.join(dataPath,'processed/tracklets_3d.csv'),offset=0,minLen=5)
    oldCsv = pd.read_csv(os.path.join(dataPath,'processed/tracklets_3d.csv'), index_col=0)
    
    tf = TrackFinalizer(dataPath) 

    # Link tracklets if necessary
    if(len(tracks) > tf.n_fish):
        linkedPaths = tf.matchAll(tracks)

        # Update ids in old csv file according to linked paths
        linkedCsv = updateIds(oldCsv,linkedPaths)
    elif len(tracks) == tf.n_fish:
        print("There are exactly {} 3D tracklets. These are saved as the final 3D tracks".format(tf.n_fish))
        linkedCsv = oldCsv
    else:
        print("Only {} 3D tracklets are present, while there are {} fish. Not enough informaiton to work with".format(len(tracks), tf.n_fish))
        sys.exit()

    # Remove duplicate frame detections in a track
    linkedCsv.reset_index(inplace=True, drop=True)
    linkedCsv = linkedCsv.drop(getDropIndecies(linkedCsv))
    
    outputPath = os.path.join(dataPath, 'processed', 'tracks_3d.csv')
    print("Saving data to: {0}".format(outputPath))
    linkedCsv.to_csv(outputPath, index=False)

    # Interpolate missing detections in linked csv file
    interpolated = interpolate(linkedCsv, dataPath)    
    interpolated.reset_index(inplace=True, drop=True)

    # Save processed CSV file
    interpolated = interpolated.sort_values(by=['id', 'frame'], ascending=[True,True])

    
    outputPath = os.path.join(dataPath, 'processed', 'tracks_3d_interpolated.csv')
    print("Saving data to: {0}".format(outputPath))
    interpolated.to_csv(outputPath, index=False)