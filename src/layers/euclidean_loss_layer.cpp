#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const bool gen_mode = this->layer_param_.gan_loss_param().gen_mode();
    iter_size_ = this->layer_param_.gan_loss_param().iter_size();
    if (gen_mode == false){
      int count = bottom[0]->count();
      caffe_sub(
            count,
            bottom[0]->cpu_data(),
            bottom[1]->cpu_data(),
            diff_.mutable_cpu_data());
      Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
      Dtype loss = dot / bottom[0]->num() / Dtype(2);
      top[0]->mutable_cpu_data()[0] = loss;
    }
    else{
        top[0]->mutable_cpu_data()[0] = Dtype(0);
        if(gan_mode_ > ( 2* iter_size_)){
            int count = bottom[0]->count();
            caffe_sub(
                count,
                bottom[0]->cpu_data(),
                bottom[1]->cpu_data(),
                diff_.mutable_cpu_data());
            Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
            Dtype loss = dot / bottom[0]->num() / Dtype(2);
            top[0]->mutable_cpu_data()[0] = loss;
        }
    }

}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const bool gen_mode = this->layer_param_.gan_loss_param().gen_mode();
    iter_size_ = this->layer_param_.gan_loss_param().iter_size();
    if(gen_mode == false){
      for (int i = 0; i < 2; ++i) {
         if (propagate_down[i]) {
            const Dtype sign = (i == 0) ? 1 : -1;
            const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
            caffe_cpu_axpby(
                bottom[i]->count(),              // count
                alpha,                              // alpha
                diff_.cpu_data(),                   // a
                Dtype(0),                           // beta
                bottom[i]->mutable_cpu_diff());  // b
         }
      }
    }
    else{
      for (int i = 0; i < 2; ++i) {
        if (propagate_down[i]) {
            const Dtype sign = (i == 0) ? 1 : -1;
            const Dtype alpha =  (gan_mode_ > (2* iter_size_)) ? (sign * top[0]->cpu_diff()[0] / bottom[i]->num()) : Dtype(0);
            caffe_cpu_axpby(
                bottom[i]->count(),              // count
                alpha,                              // alpha
                diff_.cpu_data(),                   // a
                Dtype(0),                           // beta
                bottom[i]->mutable_cpu_diff());  // b
        }
      }
      //update gan_mode_
      gan_mode_ = gan_mode_ == (3* iter_size_) ? 1 : gan_mode_ + 1;
    }

}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
