import numpy as np
import os
import cv2 

import sys
caffe_root = '/home/rongsh/caffe/'
sys.path.insert(0,caffe_root + 'python')

import caffe

import os

if os.path.isfile(caffe_root + 'examples/SRGAN/Model/SRResNet_75s_iter_45000.caffemodel'):
    print 'caffeNet found.'
else:
    print 'caffenet not found'


caffe.set_mode_gpu()

model_def = caffe_root + 'examples/SRGAN/srres8b_deploy.prototxt'
model_weights = caffe_root + 'examples/SRGAN/Model/SRResNet_75s_iter_45000.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)
for i in range(5):
    net.forward()
    for layer_name, blob in net.blobs.iteritems(): 
         print layer_name + '\t' + str(blob.data.shape)
    out_image = net.blobs['conv_g37'].data
    channel_swap = (0, 2, 3, 1)
    out_image = out_image.transpose(channel_swap)
    cv2.imwrite(caffe_root + "examples/SRGAN/Set5_sr/srres_75s_45i_" + str(i) + ".bmp",(out_image[0,:,:,:]).astype(np.double))


