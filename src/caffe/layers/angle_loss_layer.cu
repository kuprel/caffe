#include <vector>
#include <math.h>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AngleLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // int n = bottom[0]->num();
  // int d = bottom[0]->count()/n;
  // const Dtype *u, *v;
  // Dtype *t = (*top)[0]->mutable_gpu_data();
  // Dtype L = 0;
  // for (int i=0; i<n; ++i) {
  //   u = bottom[0]->gpu_data() + i*d;
  //   v = bottom[1]->gpu_data() + i*d;
  //   caffe_gpu_dot(d, u, v, &L);
  //   L = acos(L);
  // }
  // *t = L/n;
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void AngleLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  // for (int i = 0; i < 2; ++i) {
  //   if (propagate_down[i]) {
  //     caffe_gpu_scale(
  //       (*bottom)[i]->count(),
  //       top[0]->gpu_diff()[0] / (*bottom)[i]->num(),
  //       (*bottom)[1-i]->gpu_data(),
  //       (*bottom)[i]->mutable_gpu_diff()); 
  //   }
  // }
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(AngleLossLayer);

}  // namespace caffe
