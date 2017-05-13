import caffe
import numpy as np
import time
import os
import sys

if len(sys.argv) == 1:
  start_snapshot = 0

caffe_root = '/home/rongsh/caffe/'
sys.path.insert(0, caffe_root + 'python')
srgan_root = caffe_root + 'examples/srgan/'
max_iter = int(2) # maximum number of iterations
display_every = 1 # show losses every so many iterations
snapshot_every = 1
snapshot_folder = srgan_root + 'model'
batch_size = 8
flags = 'srgan-mse'
log_save = caffe_root + 'log'

caffe.set_mode_gpu()
generator = caffe.AdamSolver('solver_generator.prototxt')
discriminator = caffe.AdamSolver('solver_discriminator.prototxt')
data_reader = caffe.AdamSolver('solver_dataset.prototxt')
mse = caffe.AdamSolver('solver_mse.prototxt')

if start_snapshot:
  curr_snapshot_folder = snapshot_folder +'/' + str(start_snapshot)
  s = '\n === Starting from snapshot ' + curr_snapshot_folder + ' ===\n'
  f=open(log_save +'/'+flags +'.txt','a')
  f.write(s+'\n')
  f.close()
  generator_caffemodel = curr_snapshot_folder +'/' + 'generator.caffemodel'
  if os.path.isfile(generator_caffemodel):
    generator.net.copy_from(generator_caffemodel)
  else:
    raise Exception('File %s does not exist' % generator_caffemodel)
  #discriminator_caffemodel = curr_snapshot_folder +'/' + 'discriminator.caffemodel'
  #if os.path.isfile(discriminator_caffemodel):
  #  discriminator.net.copy_from(discriminator_caffemodel)
  #else:
  #  raise Exception('File %s does not exist' % discriminator_caffemodel)


#discr_loss_weight = discriminator.net._blob_loss_weights[discriminator.net._blob_names_index['gan_loss']]
discr_loss_weight = 1
train_discr = True
train_gen = True

for iter in range(start_snapshot,max_iter):
  # read the data
  start_time = time.time()

  data_reader.net.forward_simple()
  # feed the data to the generator and run iter
  generator.net.blobs['feed_data'].data[...] = data_reader.net.blobs['data'].data
  generator.net.forward_simple()
  generated_img = generator.net.blobs['conv_g37'].data
  
  mse.net.blobs['data'].data[...] = generated_img
  mse.net.blobs['label'].data[...] = data_reader.net.blobs['data'].label
  mse.net.forward_simple()
  mse_loss = np.copy(mse.net.blobs['mse_loss'].data)
  # run the discriminator on real data
  discriminator.net.blobs['data'].data[...] = data_reader.net.blobs['data'].label
  discriminator.net.blobs['label'].data[...] = np.ones((batch_size,1), dtype='float32')
#  discriminator.net.blobs['feat'].data[...] = feat_real
  discriminator.net.forward_simple()
  discr_real_loss = np.copy(discriminator.net.blobs['gan_loss'].data)
  if train_discr:
    discriminator.increment_iter()
    discriminator.net.clear_param_diffs()
    discriminator.net.backward_simple()

  # run the discriminator on generated data
  discriminator.net.blobs['data'].data[...] = generated_img
  discriminator.net.blobs['label'].data[...] = np.zeros((batch_size,1), dtype='float32')
#  discriminator.net.blobs['feat'].data[...] = feat_real
  discriminator.net.forward_simple()
  discr_fake_loss = np.copy(discriminator.net.blobs['gan_loss'].data)
  if train_discr:
    discriminator.net.backward_simple()
    discriminator.apply_update()

  # run the discriminator on generated data with opposite labels, to get the gradient for the generator
#  discriminator.net.blobs['data'].data[...] = generated_img
  discriminator.net.blobs['label'].data[...] = np.ones((batch_size,1), dtype='float32')
#  discriminator.net.blobs['feat'].data[...] = feat_real
  discriminator.net.forward_simple()
  discr_fake_for_generator_loss = np.copy(discriminator.net.blobs['gan_loss'].data)
  if train_gen:
    generator.increment_iter()
    generator.net.clear_param_diffs()
#    encoder.net.backward_simple()
    discriminator.net.backward_simple()

    mse.net.clear_param_diffs()
    mse.net.backward_simple()

#    generator.net.blobs['generated'].diff[...] = encoder.net.blobs['data'].diff + discriminator.net.blobs['data'].diff
    generator.net.blobs['conv_g37'].diff[...] = discriminator.net.blobs['data'].diff
    generator.net.blobs['conv_g37'].diff[...] += mse.net.blobs['data'].diff

    generator.net.backward_simple()
    generator.apply_update()

  if iter % display_every == 0 :       
      s=time.strftime('%Y-%m-%d %H:%M:%S:',time.localtime(time.time())) + ' time='+str(time.time()-start_time)
      s += ' step='+str(iter)+' disc_real_loss='+str(discr_real_loss)+' disc_fake_loss='+str(discr_fake_loss) 
      s += '\n                   gen_loss=' +str(discr_fake_for_generator_loss) + ' mse_loss' +str(mse_loss)
      f=open(log_save +'/'+flags +'.txt','a')
      f.write(s+'\n')
      f.close() 

    #snapshot
  if iter % snapshot_every == 0 :
      curr_snapshot_folder = snapshot_folder +'/' + str(iter)
        
      if not os.path.exists(curr_snapshot_folder):
          os.makedirs(curr_snapshot_folder)
      generator_caffemodel = curr_snapshot_folder + '/' + 'generator.caffemodel'
      generator.net.save(generator_caffemodel)
      discriminator_caffemodel = curr_snapshot_folder + '/' + 'discriminator.caffemodel'
      discriminator.net.save(discriminator_caffemodel)
      s = '\n === Saving snapshot to ' + curr_snapshot_folder + ' ===\n'
      f=open(log_save +'/'+flags +'.txt','a')
      f.write(s+'\n')
      f.close()

  discr_loss_ratio = (discr_real_loss + discr_fake_loss) / discr_fake_for_generator_loss
  if discr_loss_ratio < 1e-1 and train_discr:    
      train_discr = False
      train_gen = True
      s += ' step='+str(iter)+' disc_real_loss='+str(discr_real_loss)+' disc_fake_loss='+str(discr_fake_loss) 
      s += '\n                   gen_loss=' +str(discr_fake_for_generator_loss) + ' mse_loss' +str(mse_loss)
      s += ' train_discr=' + str(train_discr) + ' train_gen=' + str(train_gen)
      f=open(log_save +'/'+flags +'.txt','a')
      f.write(s+'\n')
      f.close()
  if discr_loss_ratio > 5e-1 and not train_discr:    
      train_discr = True
      train_gen = True
      s += ' step='+str(iter)+' disc_real_loss='+str(discr_real_loss)+' disc_fake_loss='+str(discr_fake_loss) 
      s += '\n                   gen_loss=' +str(discr_fake_for_generator_loss) + ' mse_loss' +str(mse_loss)
      s += ' train_discr=' + str(train_discr) + ' train_gen=' + str(train_gen)
      f=open(log_save +'/'+flags +'.txt','a')
      f.write(s+'\n')
      f.close()
  if discr_loss_ratio > 1e1 and train_gen:
      train_gen = False
      train_discr = True
      s += ' step='+str(iter)+' disc_real_loss='+str(discr_real_loss)+' disc_fake_loss='+str(discr_fake_loss) 
      s += '\n                   gen_loss=' +str(discr_fake_for_generator_loss) + ' mse_loss' +str(mse_loss)
      s += ' train_discr=' + str(train_discr) + ' train_gen=' + str(train_gen)
      f=open(log_save +'/'+flags +'.txt','a')
      f.write(s+'\n')
      f.close()
    

