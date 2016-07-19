// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#ifndef CAFFE_RELIEF_RCNN_LAYERS_HPP_
#define CAFFE_RELIEF_RCNN_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

struct PointNode{
public:
 int x;
 int y;
 PointNode(int x_, int y_){
   x=x_;
   y=y_;
 }
};
class WindowCandidate{
public:
  int num;
  int num_limit;
  int dim;
  int map_width;
  int map_height;
  int points_num;
  vector<PointNode*> points_inds;
  int *candidates;
  //big box
  int x_min;
  int x_max;
  int y_min;
  int y_max;
  WindowCandidate(int num_lim);
  ~WindowCandidate();
  void setup_storage(int width, int height);
  void update(int x, int y);
  void calculate_windows();
  void clear();
};

template <typename Dtype>
class ReliefROILayer: public Layer<Dtype> {
 public:
  explicit ReliefROILayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ReliefROILayer"; }

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;
  int max_rois_num_in_level_;
  int level_num;
  double total_time;
  int forward_count;
  vector<WindowCandidate*> window_candidates;
};

}  // namespace caffe

#endif  // CAFFE_RELIEF_RCNN_LAYERS_HPP_
