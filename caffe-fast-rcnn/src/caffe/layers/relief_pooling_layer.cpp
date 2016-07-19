// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------
#define LOCAL_SEARCH

#include <cfloat>
#include <vector>
#include <time.h>

#include "caffe/r2cnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;
using std::vector;

namespace caffe {

WindowCandidate::WindowCandidate(int num_lim):num(0), map_width(0), map_height(0),points_num(0),
                               x_min(0),x_max(0),y_min(0),y_max(0){
       num_limit = num_lim;
       dim = 5;//x_min y_min x_max y_max, 5 dimensions
       candidates = NULL;
}

WindowCandidate::~WindowCandidate(){
       delete candidates;
}

void WindowCandidate::clear () {
     num=0;
     map_width=0;
     map_height=0;
     points_num=0;
     x_min=0;
     x_max=0;
     y_min=0;
     y_max=0;
     points_inds.clear();
     if (candidates != NULL){
        delete candidates;
     }
}
void WindowCandidate::setup_storage(int width, int height){
       this->clear();
       map_width = width;
       map_height = height;
       candidates = new int[dim*num_limit];
       x_min=map_width;
       y_min=map_height;
       points_inds.reserve(width*height);
       //LOG(INFO)<<"resize: "<<width*height<<", points_size: "<<points_num<<", vector size: "<<points_x_inds.size();
}

void WindowCandidate::update(int x, int y){
    CHECK_GT(map_width, 0)<<"map_width must be > 0";
    CHECK_GT(map_height, 0)<<"map_height must be > 0";
    //record the point index and update point num
    points_inds.push_back(new PointNode(x,y));
    //LOG(INFO)<<"x:"<<x<<", points_x_inds:"<<points_x_inds[points_x_inds.size()-1]<<", size:"<<points_x_inds.size()<<", points_num: "<<points_num<<", points_x_inds: "<<points_x_inds[points_num];
    ++points_num;
    //update big box
    if (x < x_min) {
      x_min=x;
    } 
    if (x > x_max) {
      x_max = x;
    }
    if (y < y_min) {
      y_min=y;
    } 
    if (y > y_max) {
      y_max = y;
    }
}

void WindowCandidate::calculate_windows(){
    if (points_num == 0)
       return;
    //bool find_one = false;
    int x_select=0;
    int y_select=0;
    vector<PointNode*>::iterator iter_;
    //big box
    //LOG(INFO)<<"xmin: "<<points_inds[points_num-1]->x<<", xmax:"<<points_inds[points_num-1]->y<<", points num:"<<points_num;
    candidates[1]=x_min;
    candidates[2]=y_min;
    candidates[3]=x_max;
    candidates[4]=y_max;
    ++num;
    //small boxes
    x_min=x_max=y_min=y_max=0;
    while (points_num > 0) {
      x_min = points_inds[0]->x;
      y_min = points_inds[0]->y;
      //LOG(INFO)<<"points_num:"<<points_num<<"; x,y min=("<<x_min<<", "<<x_max<<")";
      //points_inds.pop_back();
      iter_ = points_inds.begin();
      points_inds.erase(iter_);
      x_max = x_min;
      y_max = y_min;
      points_num--;//one point has been pop out of the storage
      for (iter_ = points_inds.begin(); iter_ != points_inds.end();) {
        x_select = (*iter_)->x;
        y_select = (*iter_)->y;
        if ((x_select>=x_min) && (x_select<=x_max) && (y_select>=y_min) && (y_select<=y_max)) {
              delete *iter_;
              iter_ = points_inds.erase(iter_);
              points_num--;
              //LOG(INFO)<<"points:"<<points_inds.size()<<", record:"<<points_num;
        }else if ((x_select >= x_min-1) && (x_select <= x_max+1) && (y_select >= y_min-1) && (y_select <= y_max+1)) {
              x_min = x_min>x_select?x_select:x_min;
              x_max = x_max<x_select?x_select:x_max;
              y_min = y_min>y_select?y_select:y_min;
              y_max = y_max<y_select?y_select:y_max;
              delete *iter_;
              iter_ = points_inds.erase(iter_);
              points_num--;
              //LOG(INFO)<<"points:"<<points_inds.size()<<", record:"<<points_num;
       }else{
        iter_++;
       }
      }
      // get all other linked points with the selected one
      /*do {//evaluate all the index untill one points are linked to the selected one
        find_one = false;
        iter_ = 0;
        while (iter_ < points_num) {
           x_select = points_x_inds[iter_];
           y_select = points_y_inds[iter_];
           //LOG(INFO)<<"x,y select: ("<<x_select<<", "<<y_select<<")";
           //whether to extent the box bound
           if (x_select >= x_min) {
             if (x_select <= x_max) {
                if (y_select == (y_min - 1)) {
                    y_min--;
                    find_one = true;
                } else if (y_select == (y_max + 1)) {
                    y_max++;
                    find_one = true;
                }
             } else if (x_select == x_max+1) {
                if (y_select >= y_min){
                  if(y_select <= y_max){
                    x_max = x_select;
                    find_one = true;
                  } else if (y_select == (y_max + 1)) {
                    x_max = x_select;
                    y_max = y_select; 
                    find_one = true;
                  }
                } else if (y_select == (y_min - 1)) {
                    x_max = x_select;
                    y_min = y_select;
                    find_one = true;
                }
             }
           } else if (x_select == (x_min - 1)) {
             if (y_select >= y_min){
                if(y_select <= y_max){
                  x_min = x_select;
                  find_one = true;
                } else if (y_select == (y_max + 1)) {
                  x_min = x_select;
                  y_max = y_select;
                  find_one = true;
                }
             } else if (y_select == (y_min - 1)) {
               x_min = x_select;
               y_min = y_select;
               find_one = true;
             }
           }
           if (find_one){
              remove(points_x_inds.begin(), points_x_inds.end(), x_select);
              remove(points_y_inds.begin(), points_y_inds.end(), y_select);
              points_num--;
           }else{
             iter_++;
           }
        }
      } while (find_one);
      */
      //Adding the found box into candidates
      candidates[dim*num+1]=x_min;
      candidates[dim*num+2]=y_min;
      candidates[dim*num+3]=x_max;
      candidates[dim*num+4]=y_max;
      //LOG(INFO)<<"num: "<<num<<", points_num: "<<points_num<<" ("<<x_min<<", "<<y_min<<", "<<x_max<<", "<<y_max<<")";
      num++;
      if (num == num_limit) {
         break;
      }
      //add a new region
    }
}

template <typename Dtype>
void ReliefROILayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  max_rois_num_in_level_ = 100;
  level_num = 10;
  forward_count = 0;
  total_time = 0.0;
  for (int i=0; i<level_num; i++) {
     window_candidates.push_back(new WindowCandidate(max_rois_num_in_level_));
  }
  //ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  //CHECK_GT(roi_pool_param.pooled_h(), 0)
  //    << "pooled_h must be > 0";
  //CHECK_GT(roi_pool_param.pooled_w(), 0)
  //    << "pooled_w must be > 0";
  //pooled_height_ = roi_pool_param.pooled_h();
  //pooled_width_ = roi_pool_param.pooled_w();
  //spatial_scale_ = roi_pool_param.spatial_scale();
  //LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ReliefROILayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  for (int i=0; i<level_num; i++) {
     window_candidates[i]->setup_storage(width_, height_);
  }
  //top[0]->Reshape(1, 1, 1,
  //    5);
  //max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
  //    pooled_width_);
}

//this two function fixed the layer in pool1
//TODO: make the parameters be editable in prototxt
double map_min_back(double x_min){
int pool_size=3;
int pool_stride=2;
int conv_size=11;
int conv_pad=0;
int conv_stride=4;
return (x_min*pool_stride)*conv_stride - conv_pad;
}
double map_max_back(double x_max){
int pool_size=3;
int pool_stride=2;
int pool_max_rectify=pool_size-1;
int conv_size=11;
int conv_stride=4;
int conv_max_rectify=conv_size-1;
return (x_max*pool_stride + pool_max_rectify)*conv_stride + conv_max_rectify;
}

template <typename Dtype>
void ReliefROILayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int channels = bottom[0]->channels();
  int integral_value=10;
  int fheatmap_size = height_ * width_;
  int rois_total_num = 0;
  int tmp_rois_ind = 0;
  int tmp_dim=5;
  int start_ind_1=0;
  int start_ind_2=0;
  double tmp_max = 0.0;
  double start_time = 0.0;
  double end_time = 0.0;
  Blob<Dtype> fheatmap(1,1,height_, width_);
  Dtype* fheatmap_data = fheatmap.mutable_cpu_data(); 
  caffe_set(fheatmap_size, Dtype(0), fheatmap_data);
  //go through the images
  //integrate feature map generate
  forward_count++;
  start_time = clock();
  for (int c = 0; c < channels; ++c) {
      //get max element
      tmp_max = 0;
      for (int tmp_h = 0; tmp_h < height_; ++tmp_h) {
        for (int tmp_w = 0; tmp_w < width_; ++tmp_w) {
           if (bottom_data[tmp_h*width_ + tmp_w] > tmp_max) {
              tmp_max = bottom_data[tmp_h*width_ + tmp_w];
           }
        }
      }
      // if this channel is useless, skip it
      if (tmp_max == 0) {
        bottom_data += bottom[0]->offset(0,1);
        continue;
      } else {
        caffe_axpy<Dtype>(fheatmap_size, 1/tmp_max,
                bottom_data,
                fheatmap_data);
      }
      //next channel data
      bottom_data += bottom[0]->offset(0,1);
    }
    // normalize the integral heat map for each image
    tmp_max = 0;
    for (int tmp_h = 0; tmp_h < height_; ++tmp_h) {
     for (int tmp_w = 0; tmp_w < width_; ++tmp_w) {
         if (fheatmap_data[tmp_h*width_ + tmp_w] > tmp_max) {
            tmp_max = fheatmap_data[tmp_h*width_ + tmp_w];
         }
     }
    }
    tmp_max +=0.001;//limit all the value in [0, level_num) after the operation of fmap=(level_num/tmp_max)*fmap
    caffe_scal<Dtype>(fheatmap_size, level_num/tmp_max,
                    fheatmap_data); 
    //extract rois
    for (int tmp_h = 0; tmp_h < height_; ++tmp_h) {
     for (int tmp_w = 0; tmp_w < width_; ++tmp_w) {
        integral_value = floor(fheatmap_data[tmp_h*width_ + tmp_w]);
        //if(integral_value == 10)
        //  integral_value = 9;
        window_candidates[integral_value]->update(tmp_w, tmp_h);
     }
    }
    for (int i=0; i<level_num; i++) {
     window_candidates[i]->calculate_windows();
     rois_total_num += window_candidates[i]->num;
     //LOG(INFO)<<"i: "<<i<<", num:"<<window_candidates[i]->num;
    }
    //inital top output
    vector<int> top_reshape_vec(2);
    //each roi has 4 more rois are generated for it
#ifdef LOCAL_SEARCH
    rois_total_num = rois_total_num*5;
#endif
    top_reshape_vec[0] = rois_total_num;
    top_reshape_vec[1] = 5;
    top[0]->Reshape(top_reshape_vec);
    Dtype* top_data = top[0]->mutable_cpu_data();
    LOG(INFO)<<"rois_total_num: "<<rois_total_num;
    //each roi 5 dimensions, and 4 more rois are generated for it
    caffe_set(rois_total_num*5, Dtype(0), top_data);
    tmp_dim = 5;
    tmp_rois_ind = 0;
    start_ind_1=0;
    start_ind_2=0;
    //scale the proposal to 4 more proposals
#ifdef LOCAL_SEARCH
    double x_min=0;
    double x_max=0;
    double y_min=0;
    double y_max=0;
    const double alpha = 0.8;
    const double beta = 1.5;
    double a = alpha;
    double b = beta;
    double x_mean = 0.0;
    double y_mean = 0.0;
    double w_tmp = 0.0;
    double h_tmp = 0.0;
    double w_tmp_ = 0.0;
    double h_tmp_ = 0.0;
#endif
    for (int i=0; i<level_num; i++) {
      tmp_dim = window_candidates[i]->dim;
    //LOG(INFO)<<"Hi: "<<i<<"  num: "<<tmp_rois_ind;
      for (int j=0; j<window_candidates[i]->num; j++, tmp_rois_ind++) {
          //LOG(INFO)<<"num: "<<tmp_rois_ind;
          start_ind_1 = tmp_rois_ind * tmp_dim;
          start_ind_2 = j * tmp_dim;
#ifndef LOCAL_SEARCH
          top_data[start_ind_1 + 1] = window_candidates[i]->candidates[start_ind_2 + 1];
          top_data[start_ind_1 + 2] = window_candidates[i]->candidates[start_ind_2 + 2];
          top_data[start_ind_1 + 3] = window_candidates[i]->candidates[start_ind_2 + 3];
          top_data[start_ind_1 + 4] = window_candidates[i]->candidates[start_ind_2 + 4];
#else
          x_min = window_candidates[i]->candidates[start_ind_2 + 1];
          y_min = window_candidates[i]->candidates[start_ind_2 + 2];
          x_max = window_candidates[i]->candidates[start_ind_2 + 3];
          y_max = window_candidates[i]->candidates[start_ind_2 + 4];
          top_data[start_ind_1 + 1] = x_min;
          top_data[start_ind_1 + 2] = y_min;
          top_data[start_ind_1 + 3] = x_max;
          top_data[start_ind_1 + 4] = y_max;
          //generate new proposals
          //w_tmp = (x_max - x_min)/2.0;
          //h_tmp = (y_max - y_min)/2.0;
          x_min = map_min_back(x_min);
          y_min = map_min_back(y_min);
          x_max = map_max_back(x_max);
          y_max = map_max_back(y_max);
          w_tmp = (x_max - x_min);
          h_tmp = (y_max - y_min);
          x_mean = x_min + w_tmp/2.0;
          y_mean = y_min + h_tmp/2.0;

          ++tmp_rois_ind;
          start_ind_1 = start_ind_1 + tmp_dim;
          a=alpha;
          b=beta;
          w_tmp_ = a*w_tmp/2.0;
          h_tmp_ = b*h_tmp/2.0;
          x_min = x_mean - w_tmp_;
          y_min = y_mean - h_tmp_;
          x_max = x_mean + w_tmp_;
          y_max = y_mean + h_tmp_;
          //top_data[start_ind_1 + 1] = x_min>0?x_min:0;
          //top_data[start_ind_1 + 2] = y_min>0?y_min:0;
          //top_data[start_ind_1 + 3] = x_max<width_?x_max:width_;
          //top_data[start_ind_1 + 4] = y_max<height_?y_max:height_;
          top_data[start_ind_1 + 1] = x_min;
          top_data[start_ind_1 + 2] = y_min;
          top_data[start_ind_1 + 3] = x_max;
          top_data[start_ind_1 + 4] = y_max;

          ++tmp_rois_ind;
          start_ind_1 = start_ind_1 + tmp_dim;
          a=alpha;
          b=alpha;
          w_tmp_ = a*w_tmp/2.0;
          h_tmp_ = b*h_tmp/2.0;
          x_min = x_mean - w_tmp_;
          y_min = y_mean - h_tmp_;
          x_max = x_mean + w_tmp_;
          y_max = y_mean + h_tmp_;
          //top_data[start_ind_1 + 1] = x_min>0?x_min:0;
          //top_data[start_ind_1 + 2] = y_min>0?y_min:0;
          //top_data[start_ind_1 + 3] = x_max<width_?x_max:width_;
          //top_data[start_ind_1 + 4] = y_max<height_?y_max:height_;
          top_data[start_ind_1 + 1] = x_min;
          top_data[start_ind_1 + 2] = y_min;
          top_data[start_ind_1 + 3] = x_max;
          top_data[start_ind_1 + 4] = y_max;

          ++tmp_rois_ind;
          start_ind_1 = start_ind_1 + tmp_dim;
          a=beta;
          b=alpha;
          w_tmp_ = a*w_tmp/2.0;
          h_tmp_ = b*h_tmp/2.0;
          x_min = x_mean - w_tmp_;
          y_min = y_mean - h_tmp_;
          x_max = x_mean + w_tmp_;
          y_max = y_mean + h_tmp_;
          //top_data[start_ind_1 + 1] = x_min>0?x_min:0;
          //top_data[start_ind_1 + 2] = y_min>0?y_min:0;
          //top_data[start_ind_1 + 3] = x_max<width_?x_max:width_;
          //top_data[start_ind_1 + 4] = y_max<height_?y_max:height_;
          top_data[start_ind_1 + 1] = x_min;
          top_data[start_ind_1 + 2] = y_min;
          top_data[start_ind_1 + 3] = x_max;
          top_data[start_ind_1 + 4] = y_max;

          ++tmp_rois_ind;
          start_ind_1 = start_ind_1 + tmp_dim;
          a=beta;
          b=beta;
          w_tmp_ = a*w_tmp/2.0;
          h_tmp_ = b*h_tmp/2.0;
          x_min = x_mean - w_tmp_;
          y_min = y_mean - h_tmp_;
          x_max = x_mean + w_tmp_;
          y_max = y_mean + h_tmp_;
          //top_data[start_ind_1 + 1] = x_min>0?x_min:0;
          //top_data[start_ind_1 + 2] = y_min>0?y_min:0;
          //top_data[start_ind_1 + 3] = x_max<width_?x_max:width_;
          //top_data[start_ind_1 + 4] = y_max<height_?y_max:height_;
          top_data[start_ind_1 + 1] = x_min;
          top_data[start_ind_1 + 2] = y_min;
          top_data[start_ind_1 + 3] = x_max;
          top_data[start_ind_1 + 4] = y_max;
#endif
       }
    }
    end_time = clock();
    total_time += end_time - start_time;
    LOG(INFO)<<"ROIs generated: "<< total_time/forward_count/CLOCKS_PER_SEC<<" sec";
}

template <typename Dtype>
void ReliefROILayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ReliefROILayer);
#endif

INSTANTIATE_CLASS(ReliefROILayer);
REGISTER_LAYER_CLASS(ReliefROI);

}  // namespace caffe
