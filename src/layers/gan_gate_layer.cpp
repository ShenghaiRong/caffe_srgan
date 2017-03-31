#include <vector>
#include "caffe/layers/gan_gate_layer.hpp"


namespace caffe {

  template <typename Dtype>
  void GANGateLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    gan_mode_ = 1;
    top[0]->ReshapeLike(*bottom[0]);
  }

  template <typename Dtype>
  void GANGateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    const int iter_size = this->layer_param_.gan_gate_param().iter_size();
    int index = 1;
    if(gan_mode_ < (2* iter_size) && gan_mode_%2 == 1){
        index = 0;
    }
    top[0]->ReshapeLike(*bottom[index]);
    top[0]->ShareData(*bottom[index]);
    top[0]->ShareDiff(*bottom[index]);
    gan_mode_ = gan_mode_ == (3* iter_size) ? 1 : gan_mode_ + 1;
  }

  INSTANTIATE_CLASS(GANGateLayer);
  REGISTER_LAYER_CLASS(GANGate);

}  // namespace caffe
