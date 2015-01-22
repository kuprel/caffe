#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class AngleLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  AngleLossLayerTest()
      : x(new Blob<Dtype>(10, 5, 1, 1)),
        y(new Blob<Dtype>(10, 5, 1, 1)),
        u(new Blob<Dtype>(10, 5, 1, 1)),
        v(new Blob<Dtype>(10, 5, 1, 1)),
        t(new Blob<Dtype>()) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->x);
    filler.Fill(this->y);
    X.push_back(x);
    Y.push_back(y);
    U.push_back(u);
    V.push_back(v);
    UV.push_back(u);
    UV.push_back(v);
    T.push_back(t);
    LayerParameter layer_param;
    NormalizeLayer<Dtype> layer(layer_param);
    layer.SetUp(this->X, &(this->U));
    layer.Forward(this->X, &(this->U));
    layer.SetUp(this->Y, &(this->V));
    layer.Forward(this->Y, &(this->V));
  }
  virtual ~AngleLossLayerTest() {
    delete x, y, u, v, t; 
  }

  Blob<Dtype> *const x, *const y, *const u, *const v, *const t;
  vector<Blob<Dtype>*> X, Y, U, V, UV, T;

};

TYPED_TEST_CASE(AngleLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(AngleLossLayerTest, TestArcCos) {
  typedef typename TypeParam::Dtype Dtype;
  EXPECT_NEAR(acos(0.5), 3.14159/3, 1e-3);
}

TYPED_TEST(AngleLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AngleLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->UV, &(this->T));
  layer.Forward(this->UV, &(this->T));
  Dtype L1 = 0;
  int n = this->x->num();
  int d = this->x->channels();
  for (int i=0; i<n; ++i) {
    Dtype dot = 0;
    for (int j=0; j<d; ++j) {
      Dtype a = this->u->data_at(i,j,0,0);
      Dtype b = this->v->data_at(i,j,0,0);
      dot += a*b;
    }
    L1 += acos(dot);
  }
  L1 /= n;
  Dtype L2 = this->t->data_at(0,0,0,0);
  EXPECT_NEAR(L1, L2, 1e-5);
}

TYPED_TEST(AngleLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AngleLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->UV), &(this->T));
}

}  // namespace caffe
