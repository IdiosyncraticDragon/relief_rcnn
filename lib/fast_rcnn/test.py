# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
from utils.nms_slow import non_max_suppression_slow as nms_slow
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import random
from sklearn.cluster import DBSCAN
import pdb
import scipy.io as sio

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
    heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def im_detect(net, im, boxes):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False))
    #-----code added by lgy at 2016.4.1------^_^
    #boxes=np.multiply(net.blobs['rois2'].data[:,1:], 8)
    #boxes=net.blobs['rois2'].data[:,1:]
    #a_boxes=net.blobs['rois2'].data[:,1:]
    #pdb.set_trace()
    #-----finished addition--------------------

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = _bbox_pred(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
        #pass

    return scores, pred_boxes

def recursive_fine_tune(net, im, boxes,start_roi_name='roi_pool5'):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False),start=start_roi_name)

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = _bbox_pred(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes
def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    roidb = imdb.roidb
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        if roidb[i]['boxes'].shape[0]==0:
        #if roidb[i].shape[0]==0:
           continue
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, roidb[i]['boxes'])
        #scores, boxes = im_detect(net, im, roidb[i])
        _t['im_detect'].toc()

        _t['misc'].tic()
        for j in xrange(1, imdb.num_classes):
            inds = np.where((scores[:, j] > thresh[j]) &
                            (roidb[i]['gt_classes'] == 0))[0]
            #inds = np.where(scores[:, j] > thresh[j])[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]

            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

            if 0:
                keep = nms(all_boxes[j][i], 0.3)
                vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    for j in xrange(1, imdb.num_classes):
        for i in xrange(num_images):
            if all_boxes[j][i] == []:
                continue
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)
def test_net_recursive(net, imdb,num_recursive=10):
    """Test a Fast R-CNN network on an image database recursively."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    roidb = imdb.roidb
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        input_proposals=roidb[i]['boxes']
        for l in xrange(num_recursive):
            _t['im_detect'].tic()
            scores, boxes = im_detect(net, im, input_proposals)
            _t['im_detect'].toc()

            _t['misc'].tic()
            for j in xrange(1, imdb.num_classes):
                #if input_proposals.shape[0]==1900:
                #inds = np.where((scores[:, j] > thresh[j]) &
                #                (roidb[i]['gt_classes'] == 0))[0]
                inds = np.where((scores[:, j] > thresh[j]))[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                top_inds = np.argsort(-cls_scores)[:max_per_image]
                cls_scores = cls_scores[top_inds]
                cls_boxes = cls_boxes[top_inds, :]
                input_proposals=np.vstack((input_proposals, cls_boxes)).astype(np.float32)
                # push new scores onto the minheap
                for val in cls_scores:
                    heapq.heappush(top_scores[j], val)
                # if we've collected more than the max number of detection,
                # then pop items off the minheap and update the class threshold
                if len(top_scores[j]) > max_per_set:
                   while len(top_scores[j]) > max_per_set:
                       heapq.heappop(top_scores[j])
                   thresh[j] = top_scores[j][0]

                if l == (num_recursive-1):
                   all_boxes[j][i] = \
                       np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                       .astype(np.float32, copy=False)

                if 0:
                    keep = nms(all_boxes[j][i], 0.3)
                    vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    for j in xrange(1, imdb.num_classes):
        for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)
def test_net_recursive_rilievo(net, imdb,num_recursive=10):
    """Test a Fast R-CNN network on an image database recursively."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer(), 'prop_gen' : Timer(), 'prop_cal' : Timer(),'totall': Timer()}
    #proposal extract net
    def map_back_from_pool1(indices, layer_param=None):
         pool_size=3
         pool_stride=2
         pool_max_rectify=pool_size - 1
         conv_size1=11
         conv_pad1=0
         conv_stride1=4
         conv_max_rectify1=conv_size1 - 1
         x_min=np.min(indices[1])
         y_min=np.min(indices[0])
         x_max=np.max(indices[1])
         y_max=np.max(indices[0])
         if layer_param != None:
            for i in range(layer_param['num']):
                layer_pad=layer_param['pad'][i]
                layer_size=layer_param['size'][i] - 1
                layer_stride=layer_param['stride'][i]
                x_min=x_min*layer_stride - layer_pad
                x_max=x_max*layer_stride + layer_size - layer_pad
                y_min=y_min*layer_stride - layer_pad
                y_max=y_max*layer_stride + layer_size - layer_pad
            if x_min<0:
               x_min=0
            if y_min<0:
               y_min=0
            if x_max>26:
               x_max=26
            if y_max>26:
               y_max=26
         x_min=x_min*pool_stride
         x_max=x_max*pool_stride + pool_max_rectify
         y_min=y_min*pool_stride
         y_max=y_max*pool_stride + pool_max_rectify
         x_min=x_min*conv_stride1 - conv_pad1
         x_max=x_max*conv_stride1 + conv_max_rectify1
         y_min=y_min*conv_stride1 - conv_pad1
         y_max=y_max*conv_stride1 + conv_max_rectify1
         return np.array([x_min,y_min,x_max,y_max])
    def map_back_small_part_from_pool1(indices,fmap,layer_param=None):
         pool_size=3
         pool_stride=2
         pool_max_rectify=pool_size - 1
         conv_size1=11
         conv_pad1=0
         conv_stride1=4
         conv_max_rectify1=conv_size1 - 1
         candidates=range(indices[0].shape[0])
         new_boxes=np.zeros([0,5])
         candidate_count=len(candidates)
         while candidate_count>0:
            startpoint=candidates.pop(0)
            x_min=indices[1][startpoint]
            x_max=x_min
            y_min=indices[0][startpoint]
            y_max=y_min
            score=fmap[x_min,y_min]
            i=0
            j=1
            while j<candidate_count:
                j=j+1
                candidate_elem=candidates[i]
                _x = indices[1][candidate_elem]
                _y = indices[0][candidate_elem]
                #in the exist detection box
                if (_x>=x_min)&(_x<=x_max)&(_y>=y_min)&(_y<=y_max):
                   candidates.pop(i)
                   score=score+fmap[_x,_y]
                #attach to the boundary of detection box
                elif (_x>=x_min-1)&(_x<=x_max+1)&(_y>=y_min-1)&(_y<=y_max+1):
                   x_min=min(x_min,_x)
                   x_max=max(x_max,_x)
                   y_min=min(y_min,_y)
                   y_max=max(y_max,_y)
                else:
                   i=i+1
            new_boxes=np.vstack((new_boxes,np.array([x_min,y_min, x_max,y_max,score])))
            candidate_count=len(candidates)
         x_min=new_boxes[:,0]
         x_max=new_boxes[:,2]
         y_min=new_boxes[:,1]
         y_max=new_boxes[:,3]
         if layer_param != None:
            for i in range(layer_param['num']):
                layer_pad=layer_param['pad'][i]
                layer_size=layer_param['size'][i] - 1
                layer_stride=layer_param['stride'][i]
                x_min=x_min*layer_stride - layer_pad
                x_max=x_max*layer_stride + layer_size - layer_pad
                y_min=y_min*layer_stride - layer_pad
                y_max=y_max*layer_stride + layer_size - layer_pad
            x_min_inx=np.where(x_min<0)
            x_max_inx=np.where(x_max>26)
            y_min_inx=np.where(y_min<0)
            y_max_inx=np.where(y_max>26)
            x_min[x_min_inx]=0
            x_max[x_max_inx]=26
            y_min[y_min_inx]=0
            y_max[y_max_inx]=26
         x_min=x_min*pool_stride
         x_max=x_max*pool_stride + pool_max_rectify
         y_min=y_min*pool_stride
         y_max=y_max*pool_stride + pool_max_rectify
         x_min=x_min*conv_stride1 - conv_pad1
         x_max=x_max*conv_stride1 + conv_max_rectify1
         y_min=y_min*conv_stride1 - conv_pad1
         y_max=y_max*conv_stride1 + conv_max_rectify1
         new_boxes[:,0]=x_min
         new_boxes[:,2]=x_max
         new_boxes[:,1]=y_min
         new_boxes[:,3]=y_max
         return new_boxes
    def map_back_boxes_from_pool1(boxes):
         pool_size=3
         pool_stride=2
         pool_max_rectify=pool_size - 1
         conv_size1=11
         conv_pad1=0
         conv_stride1=4
         conv_max_rectify1=conv_size1 - 1
         new_boxes=np.zeros([boxes.shape[0],4])
         x_min=boxes[:,0]
         x_max=boxes[:,2]
         y_min=boxes[:,1]
         y_max=boxes[:,3]
         x_min_inx=np.where(x_min<0)
         x_max_inx=np.where(x_max>26)
         y_min_inx=np.where(y_min<0)
         y_max_inx=np.where(y_max>26)
         x_min[x_min_inx]=0
         x_max[x_max_inx]=26
         y_min[y_min_inx]=0
         y_max[y_max_inx]=26
         x_min=x_min*pool_stride
         x_max=x_max*pool_stride + pool_max_rectify
         y_min=y_min*pool_stride
         y_max=y_max*pool_stride + pool_max_rectify
         x_min=x_min*conv_stride1 - conv_pad1
         x_max=x_max*conv_stride1 + conv_max_rectify1
         y_min=y_min*conv_stride1 - conv_pad1
         y_max=y_max*conv_stride1 + conv_max_rectify1
         new_boxes[:,0]=x_min
         new_boxes[:,2]=x_max
         new_boxes[:,1]=y_min
         new_boxes[:,3]=y_max
         return new_boxes
    def map_back_from_fm(indices, x_factor, y_factor,fmap,layer_param=None):
         #layer_param={'num':6,'pad':[0,1,1,1,0,2],'size':[3,3,3,3,3,5],'stride':[2,1,1,1,2,1]}
         part1=map_back_from_pool1(indices,layer_param)
         part1=np.hstack((part1, np.sum(fmap[indices]/indices[0].shape[0])))
         part2=map_back_small_part_from_pool1(indices,fmap,layer_param)
         #part3=generate_clusters(indices,fmap,layer_param)
         return np.vstack((part1,part2))
         #return part1
    def box_backtrace(indices, x_factor, y_factor):
         x_min=indices[:,0]
         y_min=indices[:,1]
         x_max=indices[:,2]
         y_max=indices[:,3]
         indices[:,0]=x_min*x_factor
         indices[:,1]=y_min*y_factor
         indices[:,2]=x_max*x_factor
         indices[:,3]=y_max*y_factor
         return indices
    net_proposal=caffe.Classifier('/home/models/bvlc_reference_caffenet/deploy1.prototxt',
                '/home/relief_rcnn/models/CaffeNet/proposalnet.caffemodel',
                #'/home/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                #'/home/fast-rcnn-master/output/default/caffenet_fast_rcnn_iter_40000.caffemodel',
                image_dims=(227,227), raw_scale=255,
                mean=np.load('/home/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), channel_swap=(2,1,0))
    all_proposals=[]
    proposals_before_recursive=[]
    _t['totall'].tic()
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        im_width=im.shape[1]*1.0
        im_height=im.shape[0]*1.0
        im_proposal = caffe.io.load_image(imdb.image_path_at(i))
        input_size_list=[227]
        layer_param=None
        #net_proposal.predict([im_proposal],oversample=False)
        input_proposals=np.zeros([0,5])
        _t['prop_gen'].tic()
        #for fi in range(net_proposal.blobs['pool1'].data.shape[0]):
        #for fi in range(1):
        for im_input_size in input_size_list:
            fi=0
            level_num=10
            net_proposal.image_dims=(im_input_size,im_input_size)
            #net_proposal.predict([im_proposal],oversample=False)
            net_proposal.extractProposals([im_proposal], forward_stop="r2layer", oversample=False)
            _t['prop_cal'].tic()
        #---for r2cnn c++
        input_proposals = net_proposal.blobs['rois2'].data[:,1:]
        input_proposals = box_backtrace(input_proposals, im_width/227, im_height/227)
        #-------end-----
        input_proposals = np.hstack((input_proposals, np.zeros([input_proposals.shape[0],1])))
        #--local search
        _t['prop_cal'].toc()
        _t['prop_gen'].toc()
        save_index_before_recursive=np.argsort(-input_proposals[:,-1])
        input_proposals=input_proposals[save_index_before_recursive]
        proposals_before_recursive.append(input_proposals[:,(1,0,3,2)]+1)
        input_proposals=input_proposals[:,0:-1]
        #---end local search
        print '{} \t\t\t\tfind {} boxes'.format(i,len(input_proposals))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, input_proposals)
        #pdb.set_trace()
        _t['im_detect'].toc()

        _t['misc'].tic()
        #backup_proposals=np.zeros([0,4])
        input_prop=np.zeros([0,5]).astype(np.float32)
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh[j])[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]

            input_prop=np.vstack((input_prop, np.hstack((cls_boxes, cls_scores[:, np.newaxis]))))
        #recursive
        temp_prop=input_prop
        temp_index=[]
        num_recursive=3
        for l in xrange(num_recursive-1):
            if temp_prop.shape[0]==0:
               print l
               break
            #scores, boxes = im_detect(net, im, temp_prop[:,0:-1])
            scores, boxes = recursive_fine_tune(net, im, temp_prop[:,0:-1])

            for j in xrange(1, imdb.num_classes):
                inds = np.where((scores[:, j]-temp_prop[:,-1] > 0))[0]
                temp_prop[inds,-1] = scores[inds, j]
                temp_prop[inds,0:4] = boxes[inds, j*4:(j+1)*4]
                temp_index.extend(inds)
            temp_index=list(set(temp_index))
            temp_prop=temp_prop[temp_index,:]
            temp_index=[]
        final_save_index=np.argsort(-input_prop[:,-1])
        scores, boxes = recursive_fine_tune(net, im, input_prop[final_save_index,0:-1])
        all_proposals.append(input_prop[:,(1,0,3,2)]+1)
        for j in xrange(1, imdb.num_classes):
                inds = np.where(scores[:, j] > thresh[j])[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                top_inds = np.argsort(-cls_scores)[:max_per_image]
                cls_scores = cls_scores[top_inds]
                cls_boxes = cls_boxes[top_inds, :]
                # push new scores onto the minheap
                for val in cls_scores:
                   heapq.heappush(top_scores[j], val)
                # if we've collected more than the max number of detection,
                # then pop items off the minheap and update the class threshold
                if len(top_scores[j]) > max_per_set:
                   while len(top_scores[j]) > max_per_set:
                      heapq.heappop(top_scores[j])
                   thresh[j] = top_scores[j][0]
                all_boxes[j][i] = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)

        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['prop_gen'].average_time, _t['prop_cal'].average_time,_t['im_detect'].average_time,
                      _t['misc'].average_time)

    #----add for r2cnn save rois
    #all_proposals=np.array([all_proposals])
    #sio.savemat('r2cnn.mat',{'boxes':all_proposals})
    #return
    #----end-------------------
    for j in xrange(1, imdb.num_classes):
        for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)
    _t['totall'].toc()
    #all_proposals=np.array([all_proposals])
    #sio.savemat('voc_2007_rilievo_10_pool1_local_search_8_15.mat',{'boxes':all_proposals})
    #proposals_before_recursive=np.array([proposals_before_recursive])
    #sio.savemat('voc_2007_rilievo_10_pool1_local_search_8_15_no_recusive_sorted.mat',{'boxes':proposals_before_recursive})
    print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s {:.3f}s' \
          .format(i + 1, num_images, _t['prop_gen'].average_time, _t['prop_cal'].average_time,_t['im_detect'].average_time,_t['misc'].average_time)
    print 'All time cost: {:.3f}s'.format(_t['totall'].average_time)
