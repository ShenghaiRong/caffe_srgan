#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const bool gen_mode = this->layer_param_.gan_loss_param().gen_mode();
    iter_size_ = this->layer_param_.gan_loss_param().iter_size();
    if (gen_mode == false){
      int count = bottom[0]->count();
      caffe_gpu_sub(
          count,
          bottom[0]->gpu_data(),
          bottom[1]->gpu_data(),
          diff_.mutable_gpu_data());
      Dtype dot;
      caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
      Dtype loss = dot / bottom[0]->num() / Dtype(2);
      top[0]->mutable_cpu_data()[0] = loss;
    }
    else{
      top[0]->mutable_cpu_data()[0] = Dtype(0);
        if(gan_mode_ > ( 2* iter_size_)){
          int count = bottom[0]->count();
          caffe_gpu_sub(
              count,
              bottom[0]->gpu_data(),
              bottom[1]->gpu_data(),
              diff_.mutable_gpu_data());
          Dtype dot;
          caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
          Dtype loss = dot / bottom[0]->num() / Dtype(2);
          top[0]->mutable_cpu_data()[0] = loss;
        }
    }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const bool gen_mode = this->layer_param_.gan_loss_param().gen_mode();
    iter_size_ = this->layer_param_.gan_loss_param().iter_size();
    if(gen_mode == false){
      for (int i = 0; i < 2; ++i) {
        if (propagate_down[i]) {
            const Dtype sign = (i == 0) ? 1 : -1;
            const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
            caffe_gpu_axpby(
                bottom[i]->count(),              // count
                alpha,                              // alpha
                diff_.gpu_data(),                   // a
                Dtype(0),                           // beta
                bottom[i]->mutable_gpu_diff());  // b
        }
      }
    }
    else {
       for (int i = 0; i < 2; ++i) {
          if (propagate_down[i]) {
              const Dtype sign = (i == 0) ? 1 : -1;
              const Dtype alpha =  (gan_mode_ > (2* iter_size_)) ? (sign * top[0]->cpu_diff()[0] / bottom[i]->num()) : Dtype(0);
              caffe_gpu_axpby(
                bottom[i]->count(),              // count
                alpha,                              // alpha
                diff_.gpu_data(),                   // a
                Dtype(0),                           // beta
                bottom[i]->mutable_gpu_diff());  // b
            }
      }
      //update gan_mode_
      gan_mode_ = gan_mode_ == (3* iter_size_) ? 1 : gan_mode_ + 1;
    }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
