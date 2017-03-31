#ifndef CAFFE_GAN_GATE_LAYER_HPP_
#define CAFFE_GAN_GATE_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class GANGateLayer : public Layer<Dtype> {
 public:
  explicit GANGateLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "GANGate"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  int gan_mode_;
};

}  // namespace caffe

#endif  // CAFFE_GAN_GATE_LAYER_HPP_