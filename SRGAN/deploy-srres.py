import numpy as np
import os
import cv2 

import sys
caffe_root = '/home/rongsh/caffe-master/'
sys.path.insert(0,caffe_root + 'python')
#Model_root = 'examples/SRGAN/Model/SRGAN-MSE_7mse_glw1e-3_sglt_bef_iter_210000.caffemodel'
#img_name = 'examples/SRGAN/Set5_sr/srgan-mse_glw1e-3_230kti_'
Model_root = 'examples/SRGAN/Model/SRBEGAN-MSE_iter_80000.caffemodel'
img_name = 'examples/SRGAN/Set5_sr/srbegan-mse_80kti_'
img_name2 = 'examples/SRGAN/Set14_sr/srbegan-mse_80kti_'
import caffe

import os

if os.path.isfile(caffe_root + Model_root):
    print 'caffeNet found.'
else:
    print 'caffenet not found'

caffe.set_device(3)
caffe.set_mode_gpu()

model_def = caffe_root + 'examples/SRGAN/srres8b_deploy.prototxt'
model_weights = caffe_root + Model_root
net = caffe.Net(model_def, model_weights, caffe.TEST)
for i in range(5):
    net.forward()
   # for layer_name, blob in net.blobs.iteritems(): 
   #      print layer_name + '\t' + str(blob.data.shape)
    print str(i)
    out_image = net.blobs['conv_g37'].data
    channel_swap = (0, 2, 3, 1)
    out_image = out_image.transpose(channel_swap)
    cv2.imwrite(caffe_root + img_name + str(i) + ".bmp",(out_image[0,:,:,:]).astype(np.double))
for i in range(14):
    net.forward()
   # for layer_name, blob in net.blobs.iteritems(): 
   #      print layer_name + '\t' + str(blob.data.shape)
    print str(i)
    out_image = net.blobs['conv_g37'].data
    channel_swap = (0, 2, 3, 1)
    out_image = out_image.transpose(channel_swap)
    cv2.imwrite(caffe_root + img_name2 + str(i).zfill(2) + ".bmp",(out_image[0,:,:,:]).astype(np.double))

print 'deploy done!'
