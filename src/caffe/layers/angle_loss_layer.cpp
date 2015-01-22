#include <vector>
#include <math.h>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AngleLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
}

template <typename Dtype>
void AngleLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int n = bottom[0]->num();
  int d = bottom[0]->count()/n;
  const Dtype *u, *v;
  Dtype *t = (*top)[0]->mutable_cpu_data();
  Dtype L = 0, dot;
  for (int i=0; i<n; ++i) {
    u = bottom[0]->cpu_data() + i*d;
    v = bottom[1]->cpu_data() + i*d;
    dot = caffe_cpu_dot(d, u, v);
    dot = std::max(Dtype(-1),dot);
    dot = std::min(Dtype(1),dot);
    L += acos(dot);
  }
  *t = L/n;
}

template <typename Dtype>
void AngleLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  int n = (*bottom)[0]->num();
  int d = (*bottom)[0]->count()/n;
  const Dtype *u, *v;
  const Dtype dt = top[0]->cpu_diff()[0];
  Dtype a;
  for (int l=0; l<2; ++l) {
    if (propagate_down[l]) {
      for (int i=0; i<n; ++i) {
        Dtype* db = (*bottom)[l]->mutable_cpu_diff() + i*d;
        u = (*bottom)[l]->cpu_data() + i*d;
        v = (*bottom)[1-l]->cpu_data() + i*d;
        a = -dt*pow(1-pow(caffe_cpu_dot(d,u,v),2),-0.5)/n;
        caffe_cpu_scale(d, a, v, db);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AngleLossLayer);
#endif

INSTANTIATE_CLASS(AngleLossLayer);

}  // namespace caffe
