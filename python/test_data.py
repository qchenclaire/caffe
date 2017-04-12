import caffe
import numpy as np
import random
import os, struct
from array import array
import share_data as sd
#from lmdb_reader import Read_Render4CNN
from lmdb_para import read_lmdb
import scipy.misc
import time
import pdb
import cPickle as pickle
#

class Render4CNNLayer_sub(caffe.Layer):

    def setup(self, bottom, top):

        print 'setup'

    def reshape(self, bottom, top):

        print 'reshape'

    def forward(self, bottom, top):

        print 'forward'
    def backward(self, top, propagate_down, bottom):
        print 'backward'
