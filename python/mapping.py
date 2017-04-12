import numpy as np
import cPickle as pickle
from functools import reduce
class Map :

    def __init__(self):

        #self.cls = np.load('../data/ShapeNetSSD/cls.npy')
        self.az = np.load('../data/ShapeNetSSD/az.npy')
        # self.el = np.load('./data/ShapeNetSSD/el.npy')
        # self.t = np.load('./data/ShapeNetSSD/t.npy')
        self.offset = [0, 402687, 529866, 708877, 888483, 1067546, 1242939, 1419838, 1595969, 1775589, 1955446, 2135152, 2314401]
        self.bracket = []

    # def savep(self):
    #     for cls in range(12):
    #         tmp_az = np.array(self.az[np.arange(self.offset[cls],self.offset[cls+1])]) % 360
    #         tmp_el = np.array(self.el[np.arange(self.offset[cls],self.offset[cls+1])]) % 360
    #         tmp_t = np.array(self.t[np.arange(self.offset[cls],self.offset[cls+1])]) % 360
    #         for az in range(12):
    #             tmp1 = np.where( (az * 30 <= tmp_az) & (tmp_az < ((az+1) * 30 )) )[0]
    #             for el in range(36):
    #                 tmp2 = np.where( (el * 10 <= tmp_el) & (tmp_el < ((el+1) * 10 )) )[0]
    #                 for t in range(36):
    #                     tmp3 = np.where( (t * 10 <= tmp_t) & (tmp_t < ((t+1) * 10 )) )[0]
    #                     tmp = np.array(reduce(np.intersect1d, (tmp1, tmp2, tmp3)) + self.offset[cls])
    #                     if (cls==1) and (az==2) and (el==3) and (t==4):
    #                         np.save('tmp',tmp)
    #                     self.bracket.append(tmp)
    #                     print len(self.bracket)
    #     pickle.dump(self.bracket,open("bracket.p","wb"))
    #
    # def savep(self):
    #     for cls in range(12):
    #         tmp_az = np.array(self.az[np.arange(self.offset[cls],self.offset[cls+1])]) % 360
    #
    #         for az in range(12):
    #             tmp1 = np.where( (az * 30 <= tmp_az) & (tmp_az < ((az+1) * 30 )) )[0]+self.offset[cls]
    #             np.sort(tmp1)
    #             self.bracket.append(tmp1)
    #             print len(self.bracket)
    #     pickle.dump(self.bracket,open("bracket_az.p","wb"))
    #
    def az2ind(self, cls, low, high, num):

        tmp = np.array(self.az[np.arange(self.offset[cls],self.offset[cls+1])])
        tmp = np.where( (low <= tmp) & (tmp <= high) )[0] + self.offset[cls]

        np.random.shuffle(tmp)
        if num < len(tmp) :
            return tmp[range(num)]
        else:
            return tmp

    def el2ind(self, cls, low, high, num):

        tmp = self.el[np.arange(self.offset[cls],self.offset[cls+1])]
        tmp = np.where( (low <= tmp) & (tmp <= high) )[0] + self.offset[cls]
        np.random.shuffle(tmp)
        if num < len(tmp) :
            return tmp[range(num)]
        else:
            return tmp

    def t2ind(self, cls, low, high, num):

        tmp = self.t[np.arange(self.offset[cls],self.offset[cls+1])]
        tmp = np.where( (low <= tmp) & (tmp <= high) )[0] + self.offset[cls]
        np.random.shuffle(tmp)
        if num < len(tmp) :
            return tmp[range(num)]
        else:
            return tmp
