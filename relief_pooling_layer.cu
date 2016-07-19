// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/r2cnn_layers.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ReliefROIForward(const int nthreads, const Dtype* bottom_data,
    const int channels, const int height,
    const int width,Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {}
}

template <typename Dtype>
void ReliefROILayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  ReliefROIForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>();
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ReliefROILayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
     NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(ReliefROILayer);

}  // namespace caffe
