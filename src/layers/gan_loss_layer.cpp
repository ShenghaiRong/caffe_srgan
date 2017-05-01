//  Create on: 2016/10/19 ShanghaiTech
//  Author:    Yingying Zhang

#include <algorithm>
#include <vector>

#include "caffe/layers/gan_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void GANLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      LossLayer<Dtype>::LayerSetUp(bottom, top);
      iter_idx_ = 0;
      dis_iter_ = this->layer_param_.gan_loss_param().dis_iter();
      gen_iter_ = this->layer_param_.gan_loss_param().gen_iter();
      dlw_ = this->layer_param_.gan_loss_param().dis_lossweight();
      glw_ = this->layer_param_.gan_loss_param().gen_lossweight();
      sgl_ = this->layer_param_.gan_loss_param().simple_genloss();
      k_lr_ = this->layer_param_.gan_loss_param().k_lr();
      equil_ = this->layer_param_.gan_loss_param().equilibrium();
      CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
      CHECK_EQ(bottom[0]->shape(1), 1);
      CHECK_EQ(bottom[1]->shape(1), 1);
      gan_mode_ = 1;
      //default k_t_ = 1
      k_t_ = (k_lr_ != float(0) && equil_ != float(0)) ? float(0) : float(1);
      
}

template <typename Dtype>
void GANLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      int batch_size = bottom[0]->count();
      Dtype dlw = static_cast<Dtype>(dlw_);
      Dtype glw = static_cast<Dtype>(glw_);
      Dtype loss(0.0);
      Dtype k_t = static_cast<Dtype>(k_t_);
      //1. discriminative mode
      if (gan_mode_ != 2) {
        const Dtype* score1 = bottom[0]->cpu_data(); //D(x)
        const Dtype* score2 = bottom[1]->cpu_data(); //D(G(z))
        for(int i = 0; i<batch_size; ++i) {
          loss -= std::log(score1[i]) + k_t * std::log(1 - score2[i]);
        }
        loss *= dlw;
      }
      //2. generative mode
      if (gan_mode_ == 2) {
        const Dtype* score = bottom[1]->cpu_data();
        if(sgl_){
          for(int i = 0; i<batch_size; ++i) {
             loss -= std::log(score[i]);
          }
        }else{
           for(int i = 0; i<batch_size; ++i) {
             loss += std::log(1 - score[i]);
          }          
        }
        loss *= glw;
      }
      loss /= static_cast<Dtype>(batch_size);
      top[0]->mutable_cpu_data()[0] = loss;
      iter_idx_++;
}

template <typename Dtype>
void GANLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      int batch_size = bottom[0]->count();
      Dtype dlw = static_cast<Dtype>(dlw_);
      Dtype glw = static_cast<Dtype>(glw_);
      Dtype k_t = static_cast<Dtype>(k_t_);
      float diver = 0.0;
      //1. discriminative mode
      if (gan_mode_ != 2) {
        if (iter_idx_ % dis_iter_ == 0) {
          for (int i = 0; i<batch_size; ++i) {
            bottom[0]->mutable_cpu_diff()[i] = dlw * Dtype(-1) /
                    bottom[0]->cpu_data()[i] / static_cast<Dtype>(batch_size);
            bottom[1]->mutable_cpu_diff()[i] = dlw * k_t * Dtype(-1) /
                    (bottom[1]->cpu_data()[i] - Dtype(1))  / static_cast<Dtype>(batch_size);
            if(equil_ != float(0) && k_lr_ != float(0)){
                    diver +=  bottom[1]->cpu_data()[i] - equil_ * bottom[0]->cpu_data()[i];
            }
          }
        } else {
          for (int i = 0; i<batch_size; ++i) {
            bottom[0]->mutable_cpu_diff()[i] = Dtype(0);
            bottom[1]->mutable_cpu_diff()[i] = Dtype(0);
          }
        }
        if(k_lr_ != float(0) && equil_ != float(0) ){
            diver /= static_cast<float>(batch_size);
            k_t_ += k_lr_ * diver;
            // k_t_ is in [0,1]
            k_t_ = (k_t_ > float(0)) ? k_t_ : float(0);
            k_t_ = (k_t_ < float(1)) ? k_t_ : float(1);
        }
      }
      //2. generative mode
      if (gan_mode_ == 2) {
        if (iter_idx_ % gen_iter_ == 0) {
          if(sgl_){
            for (int i = 0; i<batch_size; ++i) {
              bottom[0]->mutable_cpu_diff()[i] = Dtype(0);  
              bottom[1]->mutable_cpu_diff()[i] = glw * Dtype(-1) /
                    bottom[0]->cpu_data()[i] / static_cast<Dtype>(batch_size);
            }
          }else{
            for (int i = 0; i<batch_size; ++i) {
               bottom[0]->mutable_cpu_diff()[i] = Dtype(0);  
               bottom[1]->mutable_cpu_diff()[i] = glw * Dtype(-1) /
                   (Dtype(1) - bottom[0]->cpu_data()[i]) / static_cast<Dtype>(batch_size);
            }
          }

        } else {
          for (int i = 0; i<batch_size; ++i) {
            bottom[0]->mutable_cpu_diff()[i] = Dtype(0);
            bottom[1]->mutable_cpu_diff()[i] = Dtype(0);
          }
        }
      }
      // update gan_mode_
      gan_mode_ = gan_mode_ == 2 ? 1 : gan_mode_ + 1;
}

INSTANTIATE_CLASS(GANLossLayer);
REGISTER_LAYER_CLASS(GANLoss);
//---------------------------------------------------------------------------------------
template <typename Dtype>
void BEGANLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      LossLayer<Dtype>::LayerSetUp(bottom, top);
      iter_idx_ = 0;
      dis_iter_ = this->layer_param_.gan_loss_param().dis_iter();
      gen_iter_ = this->layer_param_.gan_loss_param().gen_iter();
      dlw_ = this->layer_param_.gan_loss_param().dis_lossweight();
      glw_ = this->layer_param_.gan_loss_param().gen_lossweight();
      k_lr_ = this->layer_param_.gan_loss_param().k_lr();
      equil_ = this->layer_param_.gan_loss_param().equilibrium();
      nothing_ = this->layer_param_.gan_loss_param().nothing();
      CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
      CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
      CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2));
      CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3));
      CHECK_EQ(bottom[2]->shape(0), bottom[3]->shape(0));
      CHECK_EQ(bottom[2]->shape(1), bottom[3]->shape(1));
      CHECK_EQ(bottom[2]->shape(2), bottom[3]->shape(2));
      CHECK_EQ(bottom[2]->shape(3), bottom[3]->shape(3));
      gan_mode_ = 1;
      diffx_.ReshapeLike(*bottom[0]);
      diffG_.ReshapeLike(*bottom[2]);
      //default k_t_ = 1
      k_t_ = (k_lr_ != float(0) && equil_ != float(0)) ? float(0) : float(1);
      
}

template <typename Dtype>
void BEGANLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      int batch_size = bottom[0]->num();
      Dtype dlw = static_cast<Dtype>(dlw_);
      Dtype glw = static_cast<Dtype>(glw_);
      Dtype loss(0.0);
      Dtype k_t = static_cast<Dtype>(k_t_);
      int count = bottom[0]->count();
      //1. discriminative mode
      if (gan_mode_ != 2) {
        
        caffe_sub(
          count,
          bottom[0]->cpu_data(),
          bottom[1]->cpu_data(),
          diffx_.mutable_cpu_data());
        caffe_sub(
          count,
          bottom[2]->cpu_data(),
          bottom[3]->cpu_data(),
          diffG_.mutable_cpu_data());
          L_x_ = caffe_cpu_asum(count, diffx_.cpu_data());
          L_G_ = caffe_cpu_asum(count, diffG_.cpu_data());
        loss = L_x_ - k_t * L_G_;
        loss *= dlw;
      }
      //2. generative mode
      if (gan_mode_ == 2) {
        if(!nothing_){
            caffe_sub(
              count,
              bottom[0]->cpu_data(),
              bottom[1]->cpu_data(),
              diffx_.mutable_cpu_data());
            caffe_sub(
              count,
              bottom[2]->cpu_data(),
              bottom[3]->cpu_data(),
              diffG_.mutable_cpu_data());
            L_x_ = caffe_cpu_asum(count, diffx_.cpu_data());
            L_G_ = caffe_cpu_asum(count, diffG_.cpu_data());
        }
        

        loss = L_G_;
        loss *= glw;
      }
      loss /= static_cast<Dtype>(batch_size);
      top[0]->mutable_cpu_data()[0] = loss;
      iter_idx_++;
}

template <typename Dtype>
void BEGANLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      int batch_size = bottom[0]->num();
      Dtype dlw = static_cast<Dtype>(dlw_);
      Dtype glw = static_cast<Dtype>(glw_);
      Dtype k_t = static_cast<Dtype>(k_t_);
      int count = bottom[0]->count();
      float diver = 0.0;
      const Dtype* X_data = bottom[0]->cpu_data();  
      Dtype* X_diff = bottom[0]->mutable_cpu_diff();

      const Dtype* Xd_data = bottom[1]->cpu_data();  
      Dtype* Xd_diff = bottom[1]->mutable_cpu_diff();

      const Dtype* G_data = bottom[2]->cpu_data();  
      Dtype* G_diff = bottom[2]->mutable_cpu_diff();

      const Dtype* Gd_data = bottom[3]->cpu_data();  
      Dtype* Gd_diff = bottom[3]->mutable_cpu_diff();
      //1. discriminative mode
      if (gan_mode_ != 2) {
        if (iter_idx_ % dis_iter_ == 0) {
          
          caffe_cpu_sign(count, X_data, X_diff);
          caffe_scal(count, dlw, X_diff);
        
          caffe_cpu_sign(count, Xd_data, Xd_diff);
          caffe_scal(count, dlw, Xd_diff);
        
          caffe_cpu_sign(count, G_data, G_diff);
          dlw *= Dtype(-1);
          dlw *= k_t;
          caffe_scal(count, dlw, G_diff);
        
          caffe_cpu_sign(count, Gd_data, Gd_diff);
          caffe_scal(count, dlw, Gd_diff);
        } else {
          caffe_scal(count, Dtype(0), bottom[0]->mutable_cpu_diff());
          caffe_scal(count, Dtype(0), bottom[1]->mutable_cpu_diff());
          caffe_scal(count, Dtype(0), bottom[2]->mutable_cpu_diff());
          caffe_scal(count, Dtype(0), bottom[3]->mutable_cpu_diff());
        }
        if(k_lr_ != float(0) && equil_ != float(0) ){
            diver = equil_ * L_x_ - L_G_;
            diver /= static_cast<float>(batch_size);
            k_t_ += k_lr_ * diver;
            // k_t_ is in [0,1]
            k_t_ = (k_t_ > float(0)) ? k_t_ : float(0);
            k_t_ = (k_t_ < float(1)) ? k_t_ : float(1);
        }
      }
      //2. generative mode
      if (gan_mode_ == 2) {
        if (iter_idx_ % gen_iter_ == 0) {
          caffe_scal(count, Dtype(0), bottom[0]->mutable_cpu_diff());
          caffe_scal(count, Dtype(0), bottom[1]->mutable_cpu_diff());

          caffe_cpu_sign(count, G_data, G_diff);
          caffe_scal(count, glw, G_diff);

          caffe_cpu_sign(count, Gd_data, Gd_diff);
          caffe_scal(count, glw, Gd_diff);
          
        } else {
          for (int i = 0; i<batch_size; ++i) {
            caffe_scal(count, Dtype(0), bottom[0]->mutable_cpu_diff());
            caffe_scal(count, Dtype(0), bottom[1]->mutable_cpu_diff());
            caffe_scal(count, Dtype(0), bottom[2]->mutable_cpu_diff());
            caffe_scal(count, Dtype(0), bottom[3]->mutable_cpu_diff());
          }
        }
      }
      // update gan_mode_
      gan_mode_ = gan_mode_ == 2 ? 1 : gan_mode_ + 1;
}

INSTANTIATE_CLASS(BEGANLossLayer);
REGISTER_LAYER_CLASS(BEGANLoss);

}  // namespace caffe
