import cv2
import numpy as np
from numpy.random import randn
from numpy.linalg import norm
import pandas as pd
import math
import sys
sys.path.append('../../')

class Track:
    """
    Class implementation for representing a track or tracklet.
    Can represent both 2D and 3D tracks, meaning that some internal parameters also carry different interpetations
    """
    
    def __init__(self):
        """
        Initialize object
        """
        
        self.posWorld = []
        self.posCam1 = []
        self.posCam2 = []
        
        self.pos = []
        self.x = []
        self.y = []
        
        self.bbox = []
        self.tl_x = []
        self.tl_y = []
        self.c_x = []
        self.c_y = []
        self.w = []
        self.h = []
        self.theta = []
        self.l_x = []
        self.l_y = []
        self.r_x = []
        self.r_y = []
        
        self.frame = []
        self.frames_since_renew = 0
        self.killCount = 0
        self.id = -1
        self.cam = 0
        self.killed = False
        self.avgVel = 0
        self.kalman = None
        self.dfRow = 0

        self.visited = []
        self.patch = None


    def getImagePos(self, frameNumber, pt = "kpt", useCamFrame = False):
        """
        Gets the 2D image coordinates at a specific frame
        If there are several positions at the specified frame, the first instance is returend
        
        Input:
            frameNumber: Int indicating which frame to look up
        
        Ouput:
            pos: Tuple containing the 2D position, or None if no position is found
        """
        
        if useCamFrame:
            compArray = self.cam_frame
        else:
            compArray = self.frame

        if pt == "kpt":
            x = self.x[compArray == frameNumber]
            y = self.y[compArray == frameNumber]
        elif pt == "l":
            x = self.l_x[compArray == frameNumber]
            y = self.l_y[compArray == frameNumber]
        elif pt == "c":
            x = self.c_x[compArray == frameNumber]
            y = self.c_y[compArray == frameNumber]
        elif pt == "r":
            x = self.r_x[compArray == frameNumber]
            y = self.r_y[compArray == frameNumber]
        else:
            raise ValueError("'{}' is not a supported key for getImagePos".format(pt))
        
        if(x.size == 0 or y.size == 0):
            return None
        return (x[0],y[0])


    def getWorldPos(self, frameNumber):
        """
        Gets the 3D world coordinates at a specific frame
        If there are several positions at the specified frame, the first instance is returend
        
        Input:
            frameNumber: Int indicating which frame to look up
        
        Ouput:
            pos: Numpy array containing the 3D position, or None if no position is found
        """
        
        x = self.x[self.frame == frameNumber]
        y = self.y[self.frame == frameNumber]
        z = self.z[self.frame == frameNumber]
        if(x.size == 0 or y.size == 0 or z.size == 0):
            return None
        return np.array([x,y,z])
    
    
    def getBoundingBox(self, frameNumber, cam = None, useCamFrame = False):
        """
        Gets the bounding box at a specific frame
        If there are several bounding boxes at the specified frame, the first instance is returend
        
        Input:
            frameNumber: Int indicating which frame to look up
            cam: What camera view to get bounding box from. Also implemented for 3D tracklets/tracks
        
        Ouput:
            bbox: Tuple containing the 11 relevant elements in the follwoing order:
                tl_x, tl_y, c_x, c_y, w, h, theta, axis aligned tl_x, axis aligned tl_y, axis aligned w, axis aligned h
        """

        if useCamFrame:
            compArray = self.cam_frame
        else:
            compArray = self.frame
        
        ndim = self.tl_x.ndim
        
        if ndim == 1 or cam is None:
            tl_x = self.tl_x[compArray == frameNumber]
            tl_y = self.tl_y[compArray == frameNumber]
            c_x = self.c_x[compArray == frameNumber]
            c_y = self.c_y[compArray == frameNumber]
            w = self.w[compArray == frameNumber]
            h = self.h[compArray == frameNumber]
            theta = self.theta[compArray == frameNumber]
            aa_tl_x = self.aa_tl_x[compArray == frameNumber]
            aa_tl_y = self.aa_tl_y[compArray == frameNumber]
            aa_w = self.aa_w[compArray == frameNumber]
            aa_h = self.aa_h[compArray == frameNumber]
        elif ndim == 2 and cam is not None:
            camInd = cam-1
            tl_x = self.tl_x[camInd, compArray == frameNumber]
            tl_y = self.tl_y[camInd, compArray == frameNumber]
            c_x = self.c_x[camInd, compArray == frameNumber]
            c_y = self.c_y[camInd, compArray == frameNumber]
            w = self.w[camInd, compArray == frameNumber]
            h = self.h[camInd, compArray == frameNumber]
            theta = self.theta[camInd, compArray == frameNumber]
            aa_tl_x = self.aa_tl_x[camInd, compArray == frameNumber]
            aa_tl_y = self.aa_tl_y[camInd, compArray == frameNumber]
            aa_w = self.aa_w[camInd, compArray == frameNumber]
            aa_h = self.aa_h[camInd, compArray == frameNumber]
        else:
            print("Internal arrays have {} dimensions, and you provided a cam value of {}".format(ndim, cam))
            return None
        
        if(tl_x.size == 0 or tl_y.size == 0 or c_x.size == 0 or c_y.size == 0 or w.size == 0 or h.size == 0 or theta.size == 0):
            return None
        
        return (tl_x[0], tl_y[0], c_x[0], c_y[0], w[0], h[0], theta[0], aa_tl_x[0], aa_tl_y[0], aa_w[0], aa_h[0])
        
    def getVideoFrame(self, frameNumber, cam = None):
        """
        Given a framenumber and a cam id, return the frame number for that camera (either framenumber or some offset framenumber)

        Input:
            frameNumber: An integer
            cam: An integer denoting the camera view

        Output:
            cam_frame: An integer representing the (potentially offfset) frameNumber of cam
        """

        ndim = self.tl_x.ndim
        
        if ndim == 1 or cam is None:
            cam_frame = self.cam_frame[self.frame == frameNumber]
        elif ndim == 2 and cam is not None:
            camInd = cam-1
            cam_frame = self.cam_frame[camInd, self.frame == frameNumber]
        else:
            print("Internal arrays have {} dimensions, and you provided a cam value of {}".format(ndim, cam))
            return None
        return (cam_frame[0])

    def sort(self):
        '''
        Sorts the frame and 3D coordinate lists of the track in ascending order
        '''

        sort_order = np.argsort(self.frame)
        self.frame = self.frame[sort_order]
        self.x = self.x[sort_order]
        self.y = self.y[sort_order]
        self.z = self.z[sort_order]
    

    def toDataframe(self):
        """
        Converts the current track into a pandas dataframe
                
        Output:
            df: Pandas dataframe
        """
        
        z = None
        if len(self.pos):
            t_pos = np.array((self.pos)) 
            x = t_pos[:,0] 
            y = t_pos[:,1]
            if(len(t_pos[0]) > 2):
               z = t_pos[:,2]
        else:
            x = self.x
            y = self.y
            
        if len(self.bbox):
            t_bbox = np.array((self.bbox))
            tl_x = t_bbox[:,0]
            tl_y = t_bbox[:,1]
            c_x = t_bbox[:,2]
            c_y = t_bbox[:,3]
            w = t_bbox[:,4]
            h = t_bbox[:,5]
            theta = t_bbox[:,6]
            l_x = t_bbox[:,7]
            l_y = t_bbox[:,8]
            r_x = t_bbox[:,9]
            r_y = t_bbox[:,10]
            aa_tl_x = t_bbox[:,11]
            aa_tl_y = t_bbox[:,12]
            aa_w = t_bbox[:,13]
            aa_h = t_bbox[:,14]
        else:
            tl_x = self.tl_x
            tl_y = self.tl_y
            c_x = self.c_x
            c_y = self.c_y
            w = self.w
            h = self.h
            theta = self.theta

        t_frame = np.array((self.frame)) 
        t_id = np.zeros((len(x)))+self.id
        t_cam = np.zeros((len(x)))+self.cam

        df = pd.DataFrame({
            'frame':t_frame,
            'id': t_id,
            'cam': t_cam,
            'x': x,
            'y': y,
            "tl_x": tl_x,
            "tl_y": tl_y,
            "c_x": c_x,
            "c_y": c_y,
            "w": w,
            "h": h,
            "theta": theta,
            "l_x": l_x,
            "l_y": l_y,
            "r_x": r_x,
            "r_y": r_y,
            "aa_tl_x": aa_tl_x,
            "aa_tl_y":aa_tl_y,
            "aa_w": aa_w,
            "aa_h": aa_h})
        if(z is not None):
               df['z'] = z
            
        return df