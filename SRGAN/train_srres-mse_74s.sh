#########################################################################
# File Name: train_dcgan.sh
# Author: Rong Shenghai 
# mail: rsh914@mail.ustc.edu.cn
# Created Time: 2017年02月24日 星期五 09时51分39秒
#########################################################################
#!/bin/bash
DATETIME=`date +%Y-%m-%d-%R`
./build/tools/caffe train -solver examples/SRGAN/srres-mse_74s_solver.prototxt -gpu all -snapshot examples/SRGAN/Model/SRResNet-MSE_74s_p_iter_30000.solverstate 2>&1 | tee examples/SRGAN/log/SRResNet-MSE_74s_p_${DATETIME}.txt
