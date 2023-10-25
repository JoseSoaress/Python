import cv2, time
import numpy as np
import pptk
import plyfile
#import freenect



data = plyfile.PlyData.read('Truck.ply')['vertex'] #take data

xyz = np.c_[data['x'], data['y'], data['z']] #converts to numpy arrays
rgb = np.c_[data['red'], data['green'], data['blue']]
n = np.c_[data['nx'], data['ny'], data['nz']]
#xyz = pptk.rand(100, 3)
v = pptk.viewer(xyz) #show point cloud
v.attributes(rgb / 255., 0.5 * (1 + n))