#include <algorithm>
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
class SquareLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SquareLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SquareLayerTest() { delete blob_bottom_; delete blob_top_; }

  void TestForward() {
    LayerParameter layer_param;
    SquareLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
    layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype min_precision = 1e-5;
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      Dtype expected_value = bottom_data[i] * bottom_data[i];
      if (isnan(expected_value)) {
        EXPECT_TRUE(isnan(top_data[i]));
      } else {
        Dtype precision = std::max(
          Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
        EXPECT_NEAR(expected_value, top_data[i], precision);
      }
    }
  }

  void TestBackward() {
    LayerParameter layer_param;
    SquareLayer<Dtype> layer(layer_param);
    Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701, 0., 0.01);
    checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
        &(this->blob_top_vec_));
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SquareLayerTest, TestDtypesAndDevices);

TYPED_TEST(SquareLayerTest, TestSquare) {
  this->TestForward();
}

TYPED_TEST(SquareLayerTest, TestSquareGradient) {
  this->TestBackward();
}

}  // namespace caffe
