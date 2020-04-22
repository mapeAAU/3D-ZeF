import cv2
import os.path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
### Module imports ###
import sys
sys.path.append('../../')

class Triangulate:
    """
    Class implementation for triangulating two 2D points  
    """
   
    def rayIntersection(self, ray1Dir, ray1Point, ray2Dir, ray2Point):
        """
        Calculates the intersection between two rays
        A ray is defined as a:
        - direction vector (i.e. r1D and r2D)
        - a point on the ray (i.e. r1P and r2P)
        source: http://morroworks.com/Content/Docs/Rays%20closest%20point.pdf
        
        Input:
            ray1Dir: Numpy vector indicating direction of the first ray
            ray1Point: Numpy vector indicating point of the first ray
            ray2Dir: Numpy vector indicating direction of the second ray
            ray2Point: Numpy vector indicating point of the second ray
        
        Output:
            point: The point which is closest simultaneously to both rays and is at the same distance to them
            dist: Closest distance between the two supplied rays
        """
        
        a = ray1Dir
        b = ray2Dir
        A = ray1Point
        B = ray2Point
        c = B-A
        
        ab = np.dot(a,b)
        aa = np.dot(a,a)
        bb = np.dot(b,b)
        ac = np.dot(a,c)
        bc = np.dot(b,c)
    
        denom = aa*bb - ab*ab
        tD = (-ab*bc + ac*bb)/denom
        tE = (ab*ac - bc*aa)/denom

        D = A + a*tD
        E = B + b*tE
        point = (D+E)/2
        dist = np.linalg.norm(D-E)
        return point,dist


    def refractRay(self, rayDir, planeNormal, n1, n2, verbose=False):
        """
        Refracts an incoming ray through a specified interface
        
        Input:
            rayDir: Numpy vector of the incoming ray
            planeNOrmal: Numpy vector of the plane normal of the refracting interface
            n1: The refraction index of the medium the ray travels >FROM<
            n2: The refractio index of the medium the ray travels >TO<
            verbose: Whether to print results of the calculation
                
        Output:
            refracted: Numpy vector of the refracted ray
            c1: Cosine value of the Incidence angle
            c2: Cosine value of the Refraction angle
        """
        
        r = n1/n2
        normPlane = planeNormal/np.linalg.norm(planeNormal)
        normDir = rayDir/np.linalg.norm(rayDir)
        c1 = np.dot(-normPlane,normDir)
        c2 = np.sqrt(1.0-r**2 * (1.0-c1**2))
        refracted = r*rayDir+(r*c1-c2)*normPlane
        #refracted /= np.linalg.norm(refracted)
        if(verbose):
            print("c1: {0}".format(c1))
            print("test: {0}".format(1.0-r**2 * (1.0-c1**2)))
            print("Incidence angle: " + str(np.rad2deg(np.arccos(c1))))
            print("Refraction angle: " + str(np.rad2deg(np.arccos(c2))))
        return refracted,c1,c2


    def _triangulateRefracted(self, p1, p2, cam1, cam2, verbose=False):
        """
        Internal function - do not call directly
        
        Triangulates point while accounting for refraction
        
        Input:
            p1: 2D point in camera view 1
            p2: 2D point in camera view 2
            cam1: Camera object representing view 1
            cam2: Camera object representing view 2
            verbose: Whether the refracted rays should be plotted and shown
                
        Output:
            rayIntersection[0]: 3D Point where the distance to both ray are minimized and equal
            rayIntersection[1]: The distance between the found point and the rays
        """
        
        # 1) Backprojects points into 3D ray
        ray1 = cam1.backprojectPoint(*p1)
        ray2 = cam2.backprojectPoint(*p2)
        if(verbose):
            print("Ray1 \n -dir: {0}\n -point: {1}".format(*ray1))
            print("Ray2 \n -dir: {0}\n -point: {1}".format(*ray2))

        # 2) Find plane intersection
        p1Intersect = cam1.plane.intersectionWithRay(*ray1, verbose=verbose)
        p2Intersect = cam2.plane.intersectionWithRay(*ray2, verbose=verbose)
        if(verbose):
            print("Ray1 intersection: {0}".format(p1Intersect))
            print("Ray2 intersection: {0}".format(p2Intersect))

        # 3) Refract the backprojected rays
        n1 = 1.0 # Refraction index for air
        n2 = 1.33 # Refraction index for water
        ref1,_,_ = self.refractRay(ray1[0],cam1.plane.normal,n1,n2)
        ref2,_,_ = self.refractRay(ray2[0],cam2.plane.normal,n1,n2)
        if(verbose):
            print("Refracted ray1: {0}".format(ref1))
            print("Refracted ray2: {0}".format(ref2))

        # 4) Triangulate points the refracted rays
        rayIntersection = self.rayIntersection(ref1, p1Intersect, ref2, p2Intersect)

        # Plot stuff if enabled
        if(verbose):
            # Refracted ray 1
            cam1Pos = cam1.getPosition()
            newRay1 = 200 * ray1[0]
            newRay1 += cam1Pos[0]
            x1 = [cam1Pos[0][0], newRay1[0]]    
            y1 = [cam1Pos[0][1], newRay1[1]]
            z1 = [cam1Pos[0][2], newRay1[2]]

            ref1 /= np.linalg.norm(ref1)
            ref1 *= 200
            ref1 += p1Intersect
            x1r = [p1Intersect[0], ref1[0]]    
            y1r = [p1Intersect[1], ref1[1]]
            z1r = [p1Intersect[2], ref1[2]]    

            # Refracted ray 2
            cam2Pos = cam2.getPosition()
            newRay2 = 200 * ray2[0]
            newRay2 += cam2Pos[0]
            x2 = [cam2Pos[0][0], newRay2[0]]    
            y2 = [cam2Pos[0][1], newRay2[1]]
            z2 = [cam2Pos[0][2], newRay2[2]]

            ref2 /= np.linalg.norm(ref2)
            ref2 *= 200
            ref2 += p2Intersect
            x2r = [p2Intersect[0], ref2[0]]    
            y2r = [p2Intersect[1], ref2[1]]
            z2r = [p2Intersect[2], ref2[2]]

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot(x1, y1, z1)
            ax.plot(x1r, y1r, z1r, 'yellow')
            ax.plot(x2, y2, z2, 'red')
            ax.plot(x2r, y2r, z2r, 'green')
            ax.scatter(*cam1Pos[0], c='black')
            ax.scatter(*cam2Pos[0], c='black')
            ax.scatter(*rayIntersection[0], c='black', marker='x', s=20)
            ax.auto_scale_xyz([0, 40], [0, 40], [0, 40])
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            plt.show()        
        return rayIntersection[0], rayIntersection[1]


    def _triangulateOpenCv(self, p1, p2, cam1, cam2, verbose=False):
        """
        Internal function - do not call directly
        
        Triangulates point using OpenCV's function
        This method does not account for refraction!
        
        
        Input:
            p1: 2D point in camera view 1
            p2: 2D point in camera view 2
            cam1: Camera object representing view 1
            cam2: Camera object representing view 2
            verbose: Whether the undistorted points should be written
                
        Output:
            point: 3D Point where the distance to both ray are minimized and equal
            dist: The distance between the found point and the rays. Set to -1
        
        """
        
        # 1) Undistort points
        p1 = cv2.undistortPoints(np.array([[p1]]), cam1.K, cam1.dist)
        p2 = cv2.undistortPoints(np.array([[p2]]), cam2.K, cam2.dist)
        if(verbose):
            print("Undistorted top point: " + str(p1))
            print("Undistorted side point: " + str(p2))     

        # 2) Triangulate points using camera projection matrices
        point = cv2.triangulatePoints(cam1.getExtrinsicMat(),
                                      cam2.getExtrinsicMat(),
                                      p1,p2)
        point /= point[3]
        return point[:3].flatten(), -1.0


    def triangulatePoint(self, p1, p2, cam1, cam2, correctRefraction=True, verbose=False):    
        """
        Triangulate 3D point using 2D points from two cameras
        
        This is done projecting a ray from each of the camera through their respective 2D point, and finding the point closest to both rays
        
        Input:
            p1: 2D point in camera view 1
            p2: 2D point in camera view 2
            cam1: Camera object representing view 1
            cam2: Camera object representing view 2
            correctRefraction: Whether to correction for refraction when trasitioning through air to water
            verbose: Whether information about the triangulation process should be written.
            
        Output:
            point: 3D Point where the distance to both ray are minimized and equal
            dist: The distance between the found point and the rays.
        """
        
        if(verbose):
            print("\n\nPoint 1: {0}".format(p1))
            print("Point 2: {0}".format(p2))
        if(correctRefraction):
            point, dist = self._triangulateRefracted(p1, p2, cam1, cam2, verbose=verbose)
        else:
            point, dist = self._triangulateOpenCv(p1, p2, cam1, cam2, verbose=verbose)
        if(verbose):
            print("Triangulated point: {0} with distance: {1}".format(point, dist))    
        return point, dist


## ---- Test stuff --- ##
if __name__ == '__main__':
    #path = 'temp/wand_test_8jan_2018/'
    path = '../../../data/13sep/2700_60fps_1/'
    # 1) Check whether necessary files exists
    cam = os.path.join(path,'camera.pkl')
    if(not os.path.isfile(cam)):
        print("Error finding camera calibration file: \n {0}".format(cam))
        sys.exit(0)
    
    cam1ref = os.path.join(path,'cam1_references.json')
    cam2ref = os.path.join(path,'cam2_references.json')
    if(not os.path.isfile(cam1ref)):
        print("Error finding camera corner reference file: \n {0}".format(cam1ref))
        sys.exit(0)
    if(not os.path.isfile(cam2ref)):
        print("Error finding camera corner reference file: \n {0}".format(cam2ref))
        sys.exit(0)

    # 2) Prepare cameras
    cam1 = joblib.load(cam)
    cam1.calcExtrinsicFromJson(cam1ref)
    print("Camera 1:")
    print(" - position: \n" + str(cam1.getPosition()))
    print(" - rotation: \n" + str(cam1.getRotationMat()))
    print("")

    cam2 = joblib.load(cam)
    cam2.calcExtrinsicFromJson(cam2ref)
    print("Camera 2:")
    print(" - position: \n" + str(cam2.getPosition()))
    print(" - rotation: \n" + str(cam2.getRotationMat()))
    print("")

    print(cam2.plane.normal)
    newNormal = np.array([0.2, 1.0, -0.2])
    newNormal /= np.linalg.norm(newNormal)
    cam2.plane.normal = newNormal

    # 3) Tests
    errors = []
    depths = []


    # Corner 3 - extrinsic
    cam1det = np.array([948.0,612.0])
    cam2det = np.array([1774.0,1172.0])

    tr = Triangulate()
    p,d = tr.triangulatePoint(cam1det, cam2det, cam1, cam2,
                              correctRefraction=False, verbose=False)
    print("Triangulated point: {0} with distance: {1}".format(p, d))
    errors.append(d)
    depths.append(p[-1])

    print("Mean error: {0}".format(np.mean(errors)))
    print("Mean depth: {0}".format(np.mean(depths)))
