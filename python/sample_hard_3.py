import sys
from trial_3_helper import *

dist = np.load(sys.argv[1])
size = int(sys.argv[2])
dirname = sys.argv[3]
epoch = int(sys.argv[4])
sample_views(dist, size, dirname, epoch)
sys.stdout.write('hard sample finish\n')
sys.stdout.flush()
