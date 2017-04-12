import math
import numpy as np
from scipy.linalg import norm
from scipy.linalg import logm

def compute_acc(pred,label):
    #print 'pred',pred
    #print 'label',label    
    R_pred = angle2dcm(pred[0]/ 180.0 * math.pi, pred[1]/ 180.0 * math.pi, pred[2]/ 180.0 * math.pi)
    R_label = angle2dcm(label[0]/ 180.0 * math.pi, label[1]/ 180.0 * math.pi, label[2]/ 180.0 * math.pi)
    #print 'product',np.dot(R_pred.T,R_label)
    #print 'logm',logm(np.dot(R_pred.T,R_label))
    R_angle = norm(logm(np.dot(R_pred.T,R_label)+10**(-15)),2) / math.sqrt( 2 )
    return R_angle

def angle2dcm(z=0, y=0, x=0):
    Ms = np.zeros([3,3])
    
    cosz = math.cos(z)
    sinz = math.sin(z)
     
    cosy = math.cos(y)
    siny = math.sin(y)
      
    cosx = math.cos(x)
    sinx = math.sin(x)
       
    #  [          cy*cz,          cy*sz,            -sy]
    #  [ sy*sx*cz-sz*cx, sy*sx*sz+cz*cx,          cy*sx]
    #  [ sy*cx*cz+sz*sx, sy*cx*sz-cz*sx,          cy*cx]
    Ms[0,0] = cosy * cosz
    Ms[0,1] = cosy * sinz
    Ms[0,2] = -siny
    Ms[1,0] = siny * sinx* cosz - sinz * cosx
    Ms[1,1] = siny * sinx* sinz + cosz * cosx
    Ms[1,2] = cosy * sinx
    Ms[2,0] = siny * cosx* cosz + sinz * sinx
    Ms[2,1] = siny * cosx* sinz - cosz * sinx
    Ms[2,2] = cosy * cosx
    
    return Ms

print compute_acc([5,108,227],[350,241,37])
