#---------------------------------------- #Create by Guiying Li, 2016.4.6
#Draw simple non-convex curve for Lenet-5
#----------------------------------------
import sys
import os
import os.path as osp
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path('./python')
import caffe
import numpy as np
import pdb


if __name__=='__main__':
   caffe.set_mode_gpu()
   caffe.set_device(0)
   solverPath='examples/mnist/lenet_solver.prototxt'
   modelPath='examples/mnist/lenet_train_test.prototxt'
   weightPath='examples/mnist/lenet_iter_10000.caffemodel'
   net = caffe.Net(modelPath,weightPath, caffe.TEST)
   net.name = os.path.splitext(os.path.basename(weightPath))[0]
   acc=np.float32(0.0)
   for i in range(50):
       net.forward_all()
       acc=acc+net.blobs['accuracy'].data
       print 'batch:{}, acc:{}'.format(i/100, tmp_acc/100)
   print 'Accuracy:{}'.format(acc/50)
   #solver = caffe.SGDSolver(solverPath)
   #solver.test_nets[0].forward()
