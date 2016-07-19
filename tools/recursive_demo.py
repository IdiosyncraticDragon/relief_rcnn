#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import xml.dom.minidom as minidom
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import math
import pdb
from nms import non_max_suppression_slow as nms_slow
from sklearn.cluster import DBSCAN
import random

#CLASSES = ('__background__', # always index 0
#'n02672831','n02691156','n02219486','n02419796','n07739125','n02454379','n07718747','n02764044','n02766320','n02769748','n07693725','n02777292','n07753592','n02786058','n02787622','n02799071','n02802426','n02807133','n02815834','n02131653','n02206856','n07720875','n02828884','n02834778','n02840245','n01503061','n02870880','n02879718','n02883205','n02880940','n02892767','n07880968','n02924116','n02274259','n02437136','n02951585','n02958343','n02970849','n02402425','n02992211','n01784675','n03000684','n03001627','n03017168','n03062245','n03063338','n03085013','n03793489','n03109150','n03128519','n03134739','n03141823','n07718472','n03797390','n03188531','n03196217','n03207941','n02084071','n02121808','n02268443','n03249569','n03255030','n03271574','n02503517','n03314780','n07753113','n03337140','n03991062','n03372029','n02118333','n03394916','n01639765','n03400231','n02510455','n01443537','n03445777','n03445924','n07583066','n03467517','n03483316','n03476991','n07697100','n03481172','n02342885','n03494278','n03495258','n03124170','n07714571','n03513137','n02398521','n03535780','n02374451','n07697537','n03584254','n01990800','n01910747','n01882714','n03633091','n02165456','n03636649','n03642806','n07749582','n02129165','n03676483','n01674464','n01982650','n03710721','n03720891','n03759954','n03761084','n03764736','n03770439','n02484322','n03790512','n07734744','n03804744','n03814639','n03838899','n07747607','n02444819','n03908618','n03908714','n03916031','n00007846','n03928116','n07753275','n03942813','n03950228','n07873807','n03958227','n03961711','n07768694','n07615774','n02346627','n03995372','n07695742','n04004767','n04019541','n04023962','n04026417','n02324045','n04039381','n01495701','n02509815','n04070727','n04074963','n04116512','n04118538','n04118776','n04131690','n04141076','n01770393','n04154565','n02076196','n02411705','n04228054','n02445715','n01944390','n01726692','n04252077','n04252225','n04254120','n04254680','n04256520','n04270147','n02355227','n02317335','n04317175','n04330267','n04332243','n07745940','n04336792','n04356056','n04371430','n02395003','n04376876','n04379243','n04392985','n04409515','n01776313','n04591157','n02129604','n04442312','n06874185','n04468005','n04487394','n03110669','n01662784','n03211117','n04509417','n04517823','n04536866','n04540053','n04542943','n04554684','n04557648','n04530566','n02062744','n04591713','n02391049')
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel'),
        'caffe_imagenet': ('CaffeNet',
                     'caffenet_fast_rcnn_imagenet_iter_400000.caffemodel')}

def vis_detections(im, class_name, dets, savename, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(savename)
    plt.close()

def vis_groundtruth(im, class_name, boxes, savename):
    """Draw ground bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(class_name)):
        bbox = boxes[i, :4]
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='yellow', linewidth=3.5))
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name[i], 1.0),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title('GroundTruth Image', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(savename)
    plt.close()

def iou_cal(box1,box2):
    x1=box1[0]
    x2=box1[2]
    y1=box1[1]
    y2=box1[3]
    _x1=box2[0]
    _x2=box2[2]
    _y1=box2[1]
    _y2=box2[3]
    xlist=[x1,x2,_x1,_x2]
    ylist=[y1,y2,_y1,_y2]
    xlist.sort()
    ylist.sort()
    iou_value=0
    #judage whether boxes are overlap
    width=x2-x1
    _width=_x2-_x1
    height=y2-y1
    _height=_y2-_y1
    if (xlist[3]-xlist[0])<(width+_width) and (ylist[3]-ylist[0])<(height+_height):
       overlap=(xlist[2]-xlist[1])*(ylist[2]-ylist[1])
       area1=(x2-x1)*(y2-y1)
       area2=(_x2-_x1)*(_y2-_y1)
       iou_value=overlap*1.0/(area1+area2-overlap)
       if iou_value>1:
          print 'iou value out of range: {}, area1:{},area2:{},overlap:{}'.format(iou_value,area1,area2,overlap)
          pdb.set_trace()
    else:
       iou_value=0
    return iou_value

def vis_overlap_proposal_groundtruth(im, class_name, boxes, proposals, savename):
    """Draw ground bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(class_name)):
        bbox = boxes[i, :4]
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='yellow', linewidth=3.5))
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name[i], 1.0),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        max_overlap=0
        best_pro = -1
        for j in range(proposals.shape[0]):
            tmp_overlap=iou_cal(proposals[j,:],bbox)
            if tmp_overlap > max_overlap:
               max_overlap = tmp_overlap
               best_pro = j
        ax.text(bbox[0], bbox[1] + 2,
                'overlap {:.3f}'.format(max_overlap),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        if best_pro != -1:
           bbox = proposals[best_pro , :]
           ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5))

    ax.set_title('GroundTruth overlap Image', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(savename)
    plt.close()

def vis_featuremap(im, featuremap_count, boxes, savename, isGT=False, gt_bb=None):
    """Draw ground bounding boxes."""
    if len(im.shape) == 3:
       im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(boxes.shape[0]):
        bbox = boxes[i, :4]
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='green', linewidth=3.5))

    if isGT:
       for i in range(gt_bb.shape[0]):
           bbox = gt_bb[i, :4]
           ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='yellow', linewidth=3.5))
    ax.set_title('Featuremap {}'.format(featuremap_count), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(savename)
    plt.close()

def vis_proposals(im, boxes, savename):
    """Draw ground bounding boxes."""
    boxes=boxes.astype(np.float32)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(boxes.shape[0]):
        bbox = boxes[i, :4]
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='yellow', linewidth=3.5))

    ax.set_title('All Proposals For Image', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(savename)
    plt.close()

def vis_iou_and_object_size(x, ymin,ymax, savename):
    """Draw ground bounding boxes."""
    fig, ax = plt.subplots(figsize=(12, 12))
    #plt.plot(x,yfirst,'b.-',label='first')
    #plt.plot(x,ymean,'y.-',label='mean')
    plt.plot(x,ymin,'g.-', label='min')
    plt.plot(x,ymax,'r.-',label='max')
    plt.xlabel("iou")
    plt.xlabel("sqrt(object size)")
    plt.ylim(0.0,1.0)
    ax.set_title('iou and object size', fontsize=14)
    plt.tight_layout()
    plt.draw()
    fig.savefig(savename)
    plt.close()

def vis_static(im, gt_names, gt_boxes, names, dets, score_history, iou_history, iteration_count,savename):
    """Draw history data"""
    im = im[:, :, (2, 1, 0)]
    xlist=range(iteration_count)
    cls_count=0
    cls_i=0
    for cls in score_history:
            cls_count = cls_count+score_history[cls].shape[1]
    figure_count_col=4
    figure_count_row=int(math.ceil(cls_count*1.0/figure_count_col))+1#add 1 for showing image
    fig_size=5*figure_count_row#size 500 pixel for each row

    fig=plt.figure(figsize=(fig_size,fig_size))
    for cls in score_history:
        for i in range(score_history[cls].shape[1]):
            cls_i=cls_i+1
            plt.subplot(figure_count_row, figure_count_col, cls_i)
            plt.plot(xlist,score_history[cls][:,i].transpose(),'ko-',label=cls)
            plt.plot(xlist,iou_history[cls][:,i],'r.-',label=cls)
            plt.xlabel('Iterations')
            plt.ylabel('scores[black] & iou[red]')
            plt.ylim(0.0,1.0)
            plt.title('{} proposal num: {}'.format(cls,i+1))
    ax = plt.subplot(figure_count_row,1,figure_count_row)
    #pdb.set_trace()
    ax.imshow(im, aspect='equal')
    for i in range(len(gt_names)):
        bbox = gt_boxes[i, :4]
    
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='yellow', linewidth=1.0)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(gt_names[i], 1.0),
                bbox=dict(facecolor='green', alpha=0.5),
                fontsize=14, color='white')
    for i in range(dets.shape[0]):
        bbox = dets[i, :4]
        score = dets[i, -1]
    
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1.0)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(names[i], score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    ax.set_title('Ground Truth:Yello Box, Prediction: Red Box')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(savename)
    plt.close()

def test():
    v=iou_cal([0,0,100,100],[50,50,250,250])
    print v

def boxes_iou_cal(sourceboxes,newboxes):
   if len(sourceboxes)!=len(newboxes):
      print 'Error IOU calculation'
   results=[]
   for i in range(len(sourceboxes)):
       results.append(iou_cal(sourceboxes[i],newboxes[i]))
   return results
     
#input: index->ground-index, newindexlist[0... -> index]
#output: newindex->ground-index
def mapindex(sorcedict, newindex):
    resultdict={}
    for i in ranage(len(newindex)):
      resultdict[i]=sourcedict[newindex[i]]
    return result

def diagnoise(im, class_name, dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    print 'effect boxes counts: {}'.format(len(inds))
    if len(inds) == 0:
        return 0
    result=0
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        result = result+score
    return result/len(inds)
        
def demo(net, image_name, classes, savename,loop_counts=50):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                            image_name + '_boxes.mat')
    #mat_obj_name='boxes'
    mat_obj_name='boxes'
    obj_proposals = sio.loadmat(box_file)[mat_obj_name]

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    im = cv2.imread(im_file)

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    scorelist=[]
    #history_score_list=np.empty([1,len(obj_proposals)],np.float32)
    #history_proposal_matrix=obj_proposals
    history_score_list={}
    history_iou_list={}
    input_proposals={}
#the first forward, filter most of the proposals
    for cls in classes:
       scores, boxes = im_detect(net, im, obj_proposals)
       cls_ind = CLASSES.index(cls)
       cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
       cls_scores = scores[:, cls_ind]
       keep = np.where(cls_scores >= CONF_THRESH)[0]
       select_boxes = cls_boxes[keep, :]
       select_scores = cls_scores[keep]
       dets = np.hstack((select_boxes,
                       select_scores[:, np.newaxis])).astype(np.float32)
       #pdb.set_trace()
       keep_nms = nms(dets, NMS_THRESH)
       dets = dets[keep_nms, :]
       input_proposals[cls] = select_boxes[keep_nms,:]
    keep_map={}
    for cls in classes:
       history_score_list[cls]=np.zeros([loop_counts,len(input_proposals[cls])],np.float32)
       history_iou_list[cls]=np.zeros([loop_counts,len(input_proposals[cls])],np.float32)
       keep_map[cls]=range(len(input_proposals[cls]))
    for i in range(loop_counts):
       for cls in classes:
          if input_proposals[cls].size == 0:
             continue
          scores, boxes = im_detect(net, im, input_proposals[cls])
          cls_ind = CLASSES.index(cls)
          cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
          cls_scores = scores[:, cls_ind]
          #input_proposals = np.vstack((input_proposals,cls_boxes))
          #if i==1:
          #  pdb.set_trace()
          #history_score_list[cls] = np.vstack((history_score_list[cls], cls_scores))
          keep = np.where(cls_scores >= CONF_THRESH)[0]
          #keep = np.argsort(-cls_scores)[0:batch_count]
          select_boxes = cls_boxes[keep, :]
          select_scores = cls_scores[keep]
          temp_ious=boxes_iou_cal(input_proposals[cls][keep,:],select_boxes)
          input_proposals[cls] = select_boxes
          #map the index to the initial proposal matrix
          for j in range(len(keep)):
             keep[j]=keep_map[cls][keep[j]]
          history_score_list[cls][i, keep]=select_scores
          history_iou_list[cls][i, keep]=temp_ious
          keep_map[cls]=keep
          #pdb.set_trace()
    for class_ind in history_score_list:
        with file(savename+'_'+class_ind+'_score.txt','w') as outfile:
              np.savetxt(outfile,history_score_list[class_ind].transpose())
    for class_ind in history_iou_list:
        with file(savename+'_'+class_ind+'_iou.txt','w') as outfile:
              np.savetxt(outfile,history_iou_list[class_ind].transpose())
    #np.savetxt(savename+'.txt',history_score_list)
    #f=open(savename+'.txt','w')
    #f.write('%s' % scorelist)
    #f.close()
        #vis_detections(im, cls, dets, savename, thresh=CONF_THRESH)

def demo_bagging(net, image_name, classes, savename,loop_counts=49):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                            image_name + '_boxes.mat')
    #mat_obj_name='boxes'
    mat_obj_name='boxes'
    obj_proposals = sio.loadmat(box_file)[mat_obj_name]

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    im = cv2.imread(im_file)
    im_width=im.shape[0]
    im_height=im.shape[1]

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    scorelist=[]
    history_score_list={}
    history_iou_list={}
    input_proposals={}
    proposal_final={}
    #history_boxes_list={}
    #the first forward, filter most of the proposals
    scores, boxes = im_detect(net, im, obj_proposals)
    for cls in classes:
       cls_ind = CLASSES.index(cls)
       cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
       cls_scores = scores[:, cls_ind]
       keep = np.where(cls_scores >= CONF_THRESH)[0]
       select_boxes = cls_boxes[keep, :]
       select_scores = cls_scores[keep]
       dets = np.hstack((select_boxes,
                       select_scores[:, np.newaxis])).astype(np.float32)
       keep_nms = nms(dets, NMS_THRESH)
       dets = dets[keep_nms, :]
       input_proposals[cls] = select_boxes[keep_nms,:]
       #history_boxes_list[cls]=input_proposals[cls]
       proposal_final[cls]=np.zeros([len(input_proposals[cls]),im_width,im_height],np.float32)
       vis_detections(im, cls, dets, savename+'_'+cls+'_firstForwardPredictBox.jpg', thresh=0)
       #vote the pixels
       for k in range(len(keep_nms)):
          tmp_box=select_boxes[k,:].astype(int)
          xmin=tmp_box[0]
          ymin=tmp_box[1]
          xmax=tmp_box[2]+1#+1 for array indexing
          ymax=tmp_box[3]+1
          proposal_final[cls][k,xmin:xmax,ymin:ymax]=proposal_final[cls][k,xmin:xmax,ymin:ymax]+1
          #pdb.set_trace()
    keep_map={}
    for cls in classes:
       history_score_list[cls]=np.zeros([loop_counts+1,len(input_proposals[cls])],np.float32)
       history_iou_list[cls]=np.zeros([loop_counts+1,len(input_proposals[cls])],np.float32)
       keep_map[cls]=range(len(input_proposals[cls]))
    for i in range(loop_counts):
       for cls in classes:
          if input_proposals[cls].size == 0:
             continue
          scores, boxes = im_detect(net, im, input_proposals[cls])
          cls_ind = CLASSES.index(cls)
          cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
          cls_scores = scores[:, cls_ind]
          #keep = np.where(cls_scores >= CONF_THRESH)[0]
          keep = np.where(cls_scores >= 0)[0]
          select_boxes = cls_boxes[keep, :]
          select_scores = cls_scores[keep]
          temp_ious=boxes_iou_cal(input_proposals[cls][keep,:],select_boxes)
          input_proposals[cls] = select_boxes
          #map the index to the initial proposal matrix
          for j in range(len(keep)):
             keep[j]=keep_map[cls][keep[j]]
          history_score_list[cls][i, keep]=select_scores
          history_iou_list[cls][i, keep]=temp_ious
          keep_map[cls]=keep
          #vote the pixels
          for k in range(len(keep)):
             tmp_box=select_boxes[k,:].astype(int)
             xmin=tmp_box[0]
             ymin=tmp_box[1]
             xmax=tmp_box[2]+1#+1 for array indexing
             ymax=tmp_box[3]+1
             proposal_final[cls][keep[k],xmin:xmax,ymin:ymax]=proposal_final[cls][keep[k],xmin:xmax,ymin:ymax]+1
             #pdb.set_trace()
          if i==loop_counts-1:
             dets=np.hstack((select_boxes,select_scores[:,np.newaxis])).astype(np.float32)
             vis_detections(im, cls, dets, savename+'_'+cls+'_afterRecursivePredictBox.jpg', thresh=0)
    #filter out the
    for cls in classes:
       for i_p in range(proposal_final[cls].shape[0]):
           active_num=np.where(proposal_final[cls][i_p]>0)[0].size
           aver_vote=np.sum(proposal_final[cls][i_p])*1.0/((loop_counts+1)*active_num)
           #pdb.set_trace()
           proposal_final[cls][i_p]=proposal_final[cls][i_p]-aver_vote
           selected_indices=np.where(proposal_final[cls][i_p]>0)
           tmp_xmin=np.min(selected_indices[0])
           tmp_xmax=np.max(selected_indices[0])
           tmp_ymin=np.min(selected_indices[1])
           tmp_ymax=np.max(selected_indices[1])
           if i_p==0:
              input_proposals[cls]=np.array([[tmp_xmin,tmp_ymin,tmp_xmax,tmp_ymax]])
           else:
              input_proposals[cls]=np.vstack((input_proposals[cls],[[tmp_xmin,tmp_ymin,tmp_xmax,tmp_ymax]]))
    #test vote out proposal
    predict_boxes={}
    for cls in classes:
        if len(input_proposals[cls]) == 0:
           continue
        scores, boxes = im_detect(net, im, input_proposals[cls])
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        input_proposals[cls] = select_boxes
        #pdb.set_trace()
        history_score_list[cls][i]=cls_scores.transpose()
        dets=np.hstack((cls_boxes,cls_scores[:,np.newaxis])).astype(np.float32)
        vis_detections(im, cls, dets, savename+'_'+cls+'_votePredictBox.jpg', thresh=0)
    for class_ind in history_score_list:
        if len(history_score_list[class_ind]) == 0:
           continue
        with file(savename+'_'+class_ind+'_score_voteInEnd.txt','w') as outfile:
              np.savetxt(outfile,history_score_list[class_ind].transpose())

def refine_boxes(source_arr, predict_arr, iou_list):
    if len(iou_list)!=len(source_arr) or len(source_arr)!=len(predict_arr):
       print "ERROR in length!"
       return
    refined_boxes=np.zeros(predict_arr.shape,np.float32)
    for i in range(len(iou_list)):
        the_iou=iou_list[i]
        source_box=source_arr[i,:]
        predict_box=predict_arr[i,:]
        source_x=(source_box[2]+source_box[0])/2
        source_y=(source_box[3]+source_box[1])/2
        predict_x=(predict_box[2]+predict_box[0])/2
        predict_y=(predict_box[3]+predict_box[1])/2
        source_w=(source_box[2]-source_box[0])
        source_h=(source_box[3]-source_box[1])
        predict_w=(predict_box[2]-predict_box[0])
        predict_h=(predict_box[3]-predict_box[1])
        refined_x=source_x+iou_list[i]*(predict_x-source_x)
        refined_y=source_y+iou_list[i]*(predict_y-source_y)
        refined_w=source_w+iou_list[i]*(predict_w-source_w)
        refined_h=source_h+iou_list[i]*(predict_h-source_h)
        refined_boxes[i,:]=np.array([[refined_x-(refined_w/2),refined_y-(refined_h/2),refined_x+(refined_w/2),refined_y+(refined_h/2)]],np.float32)
    return refined_boxes
#def test():
#    source=np.array([[1,2,3,4],[10,11,12,13]])
#    target=np.array([[10,20,30,40],[20,40,50,60]])
#    iou=[0.5,0.8]
#    a=refine_boxes(source,target,iou)
#    print '{} {} {} {}'.format(a[0,0],a[0,1],a[0,2],a[0,3])
#    print '{} {} {} {}'.format(a[1,0],a[1,1],a[1,2],a[1,3])

def demo_viscous(net, image_name, classes, savename,loop_counts=49):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                            image_name + '_boxes.mat')
    #mat_obj_name='boxes'
    mat_obj_name='boxes'
    obj_proposals = sio.loadmat(box_file)[mat_obj_name]

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    im = cv2.imread(im_file)
    im_width=im.shape[0]
    im_height=im.shape[1]

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    scorelist=[]
    history_score_list={}
    history_iou_list={}
    input_proposals={}
    proposal_final={}
    #history_boxes_list={}
    #the first forward, filter most of the proposals
    scores, boxes = im_detect(net, im, obj_proposals)
    for cls in classes:
       cls_ind = CLASSES.index(cls)
       cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
       cls_scores = scores[:, cls_ind]
       keep = np.where(cls_scores >= CONF_THRESH)[0]
       select_boxes = cls_boxes[keep, :]
       select_scores = cls_scores[keep]
       dets = np.hstack((select_boxes,
                       select_scores[:, np.newaxis])).astype(np.float32)
       keep_nms = nms(dets, NMS_THRESH)
       dets = dets[keep_nms, :]
       input_proposals[cls] = select_boxes[keep_nms,:]
       #history_boxes_list[cls]=input_proposals[cls]
       proposal_final[cls]=np.zeros([len(input_proposals[cls]),im_width,im_height],np.float32)
       vis_detections(im, cls, dets, savename+'_'+cls+'_firstForwardPredictBox.jpg', thresh=0)
       #vote the pixels
       for k in range(len(keep_nms)):
          tmp_box=select_boxes[k,:].astype(int)
          xmin=tmp_box[0]
          ymin=tmp_box[1]
          xmax=tmp_box[2]+1#+1 for array indexing
          ymax=tmp_box[3]+1
          proposal_final[cls][k,xmin:xmax,ymin:ymax]=proposal_final[cls][k,xmin:xmax,ymin:ymax]+1
          #pdb.set_trace()
    keep_map={}
    for cls in classes:
       history_score_list[cls]=np.zeros([loop_counts+1,len(input_proposals[cls])],np.float32)
       history_iou_list[cls]=np.zeros([loop_counts+1,len(input_proposals[cls])],np.float32)
       keep_map[cls]=range(len(input_proposals[cls]))
    for i in range(loop_counts):
       for cls in classes:
          if input_proposals[cls].size == 0:
             continue
          scores, boxes = im_detect(net, im, input_proposals[cls])
          cls_ind = CLASSES.index(cls)
          cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
          cls_scores = scores[:, cls_ind]
          #keep = np.where(cls_scores >= CONF_THRESH)[0]
          keep = np.where(cls_scores >= 0)[0]
          select_boxes = cls_boxes[keep, :]
          select_scores = cls_scores[keep]
          temp_ious=boxes_iou_cal(input_proposals[cls][keep,:],select_boxes)
          #input_proposals[cls] = select_boxes
          input_proposals[cls] = refine_boxes(input_proposals[cls][keep,:],select_boxes,temp_ious)
          #map the index to the initial proposal matrix
          for j in range(len(keep)):
             keep[j]=keep_map[cls][keep[j]]
          history_score_list[cls][i, keep]=select_scores
          history_iou_list[cls][i, keep]=temp_ious
          keep_map[cls]=keep
          #vote the pixels
          for k in range(len(keep)):
             tmp_box=select_boxes[k,:].astype(int)
             xmin=tmp_box[0]
             ymin=tmp_box[1]
             xmax=tmp_box[2]+1#+1 for array indexing
             ymax=tmp_box[3]+1
             proposal_final[cls][keep[k],xmin:xmax,ymin:ymax]=proposal_final[cls][keep[k],xmin:xmax,ymin:ymax]+1
             #pdb.set_trace()
          if i==loop_counts-1:
             dets=np.hstack((select_boxes,select_scores[:,np.newaxis])).astype(np.float32)
             vis_detections(im, cls, dets, savename+'_'+cls+'_afterRecursivePredictBox.jpg', thresh=0)
    #filter out the
    for cls in classes:
       for i_p in range(proposal_final[cls].shape[0]):
           active_num=np.where(proposal_final[cls][i_p]>0)[0].size
           aver_vote=np.sum(proposal_final[cls][i_p])*1.0/((loop_counts+1)*active_num)
           #pdb.set_trace()
           proposal_final[cls][i_p]=proposal_final[cls][i_p]-aver_vote
           selected_indices=np.where(proposal_final[cls][i_p]>0)
           tmp_xmin=np.min(selected_indices[0])
           tmp_xmax=np.max(selected_indices[0])
           tmp_ymin=np.min(selected_indices[1])
           tmp_ymax=np.max(selected_indices[1])
           if i_p==0:
              input_proposals[cls]=np.array([[tmp_xmin,tmp_ymin,tmp_xmax,tmp_ymax]])
           else:
              input_proposals[cls]=np.vstack((input_proposals[cls],[[tmp_xmin,tmp_ymin,tmp_xmax,tmp_ymax]]))
    #test vote out proposal
    predict_boxes={}
    for cls in classes:
        if len(input_proposals[cls]) == 0:
           continue
        scores, boxes = im_detect(net, im, input_proposals[cls])
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        input_proposals[cls] = select_boxes
        #pdb.set_trace()
        history_score_list[cls][i]=cls_scores.transpose()
        dets=np.hstack((cls_boxes,cls_scores[:,np.newaxis])).astype(np.float32)
        vis_detections(im, cls, dets, savename+'_'+cls+'_votePredictBox.jpg', thresh=0)
    for class_ind in history_score_list:
        if len(history_score_list[class_ind])==0:
           continue
        with file(savename+'_'+class_ind+'_score_voteInEnd.txt','w') as outfile:
              np.savetxt(outfile,history_score_list[class_ind].transpose())

def demo_iouBoundary(net, image_name, classes, savename,loop_counts=50):
    """Detect object classes in an image using pre-computed object proposals."""
    loop_counts=loop_counts-1
    if loop_counts<0:
       print "Loop Count Error!!\n"
    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                            image_name + '.mat')
    # Load annotation file
    annotation_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                            image_name + '.xml')
    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data
    with open(annotation_file) as f:
         data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    gt_boxes = np.zeros((num_objs, 4), dtype=np.float32)
    gt_classes = []

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin')) #- 1
        y1 = float(get_data_from_tag(obj, 'ymin')) #- 1
        x2 = float(get_data_from_tag(obj, 'xmax')) #- 1
        y2 = float(get_data_from_tag(obj, 'ymax')) #- 1
        cls = str(get_data_from_tag(obj, "name")).lower().strip()
        gt_boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes.append(cls)

    #mat_obj_name='boxes'
    mat_obj_name='boxes'
    obj_proposals = sio.loadmat(box_file)[mat_obj_name]
    temp_proposals=np.zeros([obj_proposals.shape[0],1])
    temp_proposals[:,0]=obj_proposals[:,0]
    obj_proposals[:,0]=obj_proposals[:,1]
    obj_proposals[:,1]=temp_proposals[:,0]
    temp_proposals[:,0]=obj_proposals[:,2]
    obj_proposals[:,2]=obj_proposals[:,3]
    obj_proposals[:,3]=temp_proposals[:,0]

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    im = cv2.imread(im_file)
    im_width=im.shape[0]
    im_height=im.shape[1]

    #visualize groudtruth
    vis_groundtruth(im,gt_classes, gt_boxes, savename+'_groundtruth.jpg')
    # Visualize detections for each class
    EXCEPT_THRESH = 0.9
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    THRESH_STEP = (EXCEPT_THRESH-CONF_THRESH)/loop_counts
    scorelist=[]
    history_score_list={}
    input_proposals={}
    history_groundtruth_iou={}
    firstforward_score={}
    gt_box_size=[]
    groundtruth_iou_record={}
    for ground_i in range(num_objs):#select a groundtuth bounding box
        gt_box_select=gt_boxes[ground_i,:]
        temp_size=(gt_box_select[2]-gt_box_select[0])*(gt_box_select[3]-gt_box_select[1])
        gt_box_size.append(temp_size)
        groundtruth_iou_record[temp_size]=[]
    #the first forward, filter most of the proposals
    scores, boxes = im_detect(net, im, obj_proposals)
    vis_proposals(im, obj_proposals, savename+'_proposals.jpg')
    for cls in classes:
       cls_ind = CLASSES.index(cls)
       cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
       cls_scores = scores[:, cls_ind]
       keep = np.where(cls_scores >= CONF_THRESH)[0]
       select_boxes = cls_boxes[keep, :]
       select_scores = cls_scores[keep]
       dets = np.hstack((select_boxes,
                       select_scores[:, np.newaxis])).astype(np.float32)
       keep_nms = nms(dets, NMS_THRESH)
       dets = dets[keep_nms, :]
       input_proposals[cls] = select_boxes[keep_nms,:]
       #history_boxes_list[cls]=input_proposals[cls]
       vis_detections(im, cls, dets, savename+'_'+cls+'_firstForwardPredictBox.jpg', thresh=0)
       firstforward_score[cls]=select_scores[keep_nms]
    #initialization
    keep_map={}
    for cls in classes:
       if len(input_proposals[cls])==0:
          continue
       history_score_list[cls]=np.zeros([loop_counts+1,len(input_proposals[cls])],np.float32)
       history_groundtruth_iou[cls]=np.zeros([loop_counts+1,len(input_proposals[cls])],np.float32)
       keep_map[cls]=range(len(input_proposals[cls]))
       #record the first forward score
       history_score_list[cls][0,:]=firstforward_score[cls]
       #calculate the groundth iou for the first forward
       for pro_i in range(len(input_proposals[cls])):#select a proposal
           temp_iou_list=[]
           for ground_i in range(num_objs):#select a groundtuth bounding box
               temp_iou=iou_cal(input_proposals[cls][pro_i,:],gt_boxes[ground_i,:])
               temp_iou_list.append(temp_iou)
               if temp_iou>0:
                  groundtruth_iou_record[gt_box_size[ground_i]].append(temp_iou)
           history_groundtruth_iou[cls][0,pro_i]=max(temp_iou_list)
           #pdb.set_trace()
    pred_names=[]
    pred_dets=np.zeros([0,5])
    for i in range(loop_counts):
       for cls in classes:
          if input_proposals[cls].size == 0:
             continue
          scores, boxes = im_detect(net, im, input_proposals[cls])
          cls_ind = CLASSES.index(cls)
          cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
          cls_scores = scores[:, cls_ind]
          keep = np.where(cls_scores >= CONF_THRESH)[0]
          #if the new proposal are not good, out put the proposal on the boundary line
          if len(keep)==0:
             dets=np.hstack((cls_boxes,cls_scores[:,np.newaxis])).astype(np.float32)
             keep_nms = nms(dets, NMS_THRESH)
             dets = dets[keep_nms, :]
             vis_detections(im, cls, dets, savename+'_'+cls+'_'+str(i)+'_missedproposals.jpg', thresh=0)
             print 'Miss proposals!'
          #keep = np.where(cls_scores >= 0)[0]
          select_boxes = cls_boxes[keep, :]
          select_scores = cls_scores[keep]
          #map the index to the initial proposal matrix
          for j in range(len(keep)):
             keep[j]=keep_map[cls][keep[j]]
          history_score_list[cls][i+1, keep]=select_scores
          keep_map[cls]=keep
          #recursive, make the predict box as the new input
          input_proposals[cls] = select_boxes
          #calculate the groundth iou 
          for pro_i in range(len(keep)):#select a proposal
              temp_iou_list=[]
              for ground_i in range(num_objs):#select a groundtuth bounding box
                  temp_iou=iou_cal(input_proposals[cls][pro_i,:],gt_boxes[ground_i,:])
                  temp_iou_list.append(temp_iou)
                  if temp_iou>0:
                     groundtruth_iou_record[gt_box_size[ground_i]].append(temp_iou)
              history_groundtruth_iou[cls][i+1,keep[pro_i]]=max(temp_iou_list)
          #draw the map after last iter
          if i==loop_counts-1:
             dets=np.hstack((select_boxes,select_scores[:,np.newaxis])).astype(np.float32)
             keep_nms = nms(dets, NMS_THRESH)
             dets = dets[keep_nms, :]
             for ii in range(len(keep_nms)):
                 pred_names.append(cls)
             pred_dets=np.vstack((pred_dets, dets)).astype(np.float32)
             vis_detections(im, cls, dets, savename+'_'+cls+'_afterRecursivePredictBox.jpg', thresh=0)
    vis_static(im, gt_classes, gt_boxes, pred_names, pred_dets, history_score_list, history_groundtruth_iou, loop_counts+1,savename+'_afterRecursive.jpg')
    #pdb.set_trace()
    for class_ind in history_score_list:
        if len(history_score_list[class_ind])==0:
           continue
        with file(savename+'_'+class_ind+'_score_val.txt','w') as outfile:
              np.savetxt(outfile,history_score_list[class_ind].transpose())
    iou_bound_result=np.zeros([2,1])
    for class_ind in history_groundtruth_iou:
        if len(history_groundtruth_iou[class_ind])==0:
           continue
        with file(savename+'_'+class_ind+'_iou_val.txt','w') as outfile:
              np.savetxt(outfile,history_groundtruth_iou[class_ind].transpose())
        #get the frist forward iou and the max iou during iterations
        temp_iou=np.zeros([2,history_groundtruth_iou[class_ind].shape[1]])
        temp_iou[0,:]=history_groundtruth_iou[class_ind][0,:]
        temp_iou[1,:]=history_groundtruth_iou[class_ind].max(0)
        iou_bound_result=np.hstack((iou_bound_result, temp_iou)).astype(np.float32)
    iou_bound_result=iou_bound_result[:,1:]
    return (iou_bound_result,groundtruth_iou_record)

def demo_iouBoundary_rilievo(net, net_proposal, image_name, classes, savename,loop_counts=1):
    """Detect object classes in an image using pre-computed object proposals."""
    loop_counts=loop_counts-1
    if loop_counts<0:
       print "Loop Count Error!!\n"
    # Load annotation file
    annotation_file = os.path.join('/home/VOCdevkit/VOC2007/Annotations', image_name + '.xml')
    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data
    with open(annotation_file) as f:
         data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    gt_boxes = np.zeros((num_objs, 4), dtype=np.float32)
    gt_classes = []

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin')) #- 1
        y1 = float(get_data_from_tag(obj, 'ymin')) #- 1
        x2 = float(get_data_from_tag(obj, 'xmax')) #- 1
        y2 = float(get_data_from_tag(obj, 'ymax')) #- 1
        cls = str(get_data_from_tag(obj, "name")).lower().strip()
        gt_boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes.append(cls)

    # Load the demo image
    im_file = os.path.join('/home/VOCdevkit/VOC2007/JPEGImages', image_name + '.jpg')
    #im_file=os.path.join('./', 'yuzhou2.jpg')
    im = cv2.imread(im_file)
    im_proposal = caffe.io.load_image(im_file)
    im_width=im.shape[1]*1.0
    im_height=im.shape[0]*1.0

    #proposal extract net
    net_proposal.predict([im_proposal], oversample=False)
    input_proposals=np.zeros([0,4])
    def generate_clusters(indices):
         boxes_matrix=np.zeros([0,4])
         x_set=indices[1]
         y_set=indices[0]
         tmp=np.vstack((x_set,y_set))
         tmp=tmp.transpose()
         conf_eps=5
         conf_min_samples=2
         db=DBSCAN(eps=conf_eps, min_samples=conf_min_samples).fit(tmp)
         n_clusters = len(set(db.labels_)) - ( 1 if -1 in db.labels_ else 0)
         for i in range(n_clusters):
             ind = np.where(db.labels_ == i)
             points_cluster = tmp[ind[0]]
             x_min = np.min(points_cluster[0])
             x_max = np.max(points_cluster[0])
             y_min = np.min(points_cluster[1])
             y_max = np.max(points_cluster[1])
             boxes_matrix=np.vstack((boxes_matrix,np.array([x_min,y_min,x_max,y_max])))
         return boxes_matrix
    def map_gt_forward_pool1(boxes):
         tmp1 = boxes[:,(0,2)]*(227.0/im_width)
         tmp2 = boxes[:,(1,3)]*(227.0/im_height)
         results=np.zeros([0,4])
         pool_size=3
         pool_stride=2
         conv_size=11
         conv_stride=4
         for i in range(tmp1.shape[0]):
             bb_box1=tmp1[i,:]
             bb_box2=tmp2[i,:]
             x_min=int(bb_box1[0])
             y_min=int(bb_box2[0])
             x_max=int(bb_box1[1])
             y_max=int(bb_box2[1])
             if x_min%conv_stride==0:
                x_min = x_min/conv_stride
             else:
                x_min = x_min/conv_stride+1

             if y_min%conv_stride==0:
                y_min = y_min/conv_stride
             else:
                y_min = y_min/conv_stride+1

             x_max=x_max-conv_size
             if x_max%conv_stride==0:
                x_max = x_max/conv_stride
             else:
                x_max = x_max/conv_stride+1

             y_max=y_max-conv_size
             if y_max%conv_stride==0:
                y_max = y_max/conv_stride
             else:
                y_max = y_max/conv_stride+1

             if x_min%pool_stride==0:
                x_min = x_min/pool_stride
             else:
                x_min = x_min/pool_stride+1

             if y_min%pool_stride==0:
                y_min = y_min/pool_stride
             else:
                y_min = y_min/pool_stride+1

             x_max=x_max-pool_size
             if x_max%pool_stride==0:
                x_max = x_max/pool_stride
             else:
                x_max = x_max/pool_stride+1

             y_max=y_max-pool_size
             if y_max%pool_stride==0:
                y_max = y_max/pool_stride
             else:
                y_max = y_max/pool_stride+1
             results=np.vstack((results, np.array([x_min,y_min,x_max,y_max])))
         return results

    def gt_map_back_from_pool1(boxes):
         pool_size=3
         pool_stride=2
         conv_size=11
         conv_stride=4
         x_min=boxes[:,0]
         y_min=boxes[:,1]
         x_max=boxes[:,2]
         y_max=boxes[:,3]
         x_min=x_min*pool_stride
         x_max=x_max*pool_stride+pool_size-1
         y_min=y_min*pool_stride
         y_max=y_max*pool_stride+pool_size-1
         x_min=x_min*conv_stride
         x_max=x_max*conv_stride+conv_size-1
         y_min=y_min*conv_stride
         y_max=y_max*conv_stride+conv_size-1
         return np.vstack((x_min,y_min,x_max,y_max)).transpose()

    def map_back_from_pool1(indices, layer_param=None):
         pool_size=3
         pool_stride=2
         conv_size=11
         conv_stride=4
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
         x_max=x_max*pool_stride+pool_size-1
         y_min=y_min*pool_stride
         y_max=y_max*pool_stride+pool_size-1
         x_min=x_min*conv_stride
         x_max=x_max*conv_stride+conv_size-1
         y_min=y_min*conv_stride
         y_max=y_max*conv_stride+conv_size-1
         return np.array([x_min,y_min,x_max,y_max])
    def map_back_small_part_from_pool1(indices,fmap,layer_param=None):
         pool_size=3
         pool_stride=2
         conv_size=11
         conv_stride=4
         candidates=range(indices[0].shape[0])
         new_boxes=np.zeros([0,5])
         candidate_count=len(candidates)
         #print "xmin:{}, xmax:{}".format(indices[1][candidates[-1]], indices[0][candidates[-1]])
         while candidate_count>0:
            startpoint=candidates.pop(0)
            #startpoint=candidates.pop(candidate_count-1)
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
                   #print "points:{}, record:{}".format(len(candidates), candidate_count-1)
                #attach to the boundary of detection box
                elif (_x>=x_min-1)&(_x<=x_max+1)&(_y>=y_min-1)&(_y<=y_max+1):
                   x_min=min(x_min,_x)
                   x_max=max(x_max,_x)
                   y_min=min(y_min,_y)
                   y_max=max(y_max,_y)
                   candidates.pop(i)
                   score=score+fmap[_x,_y]
                   #print "points:{}, record:{}".format(len(candidates), candidate_count-1)
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
         x_max=x_max*pool_stride+pool_size-1
         y_min=y_min*pool_stride
         y_max=y_max*pool_stride+pool_size-1
         x_min=x_min*conv_stride
         x_max=x_max*conv_stride+conv_size-1
         y_min=y_min*conv_stride
         y_max=y_max*conv_stride+conv_size-1
         new_boxes[:,0]=x_min
         new_boxes[:,2]=x_max
         new_boxes[:,1]=y_min
         new_boxes[:,3]=y_max
         return new_boxes
    def map_back_from_fm(indices, x_factor, y_factor,fmap):
         #layer_param={'num':6,'pad':[0,1,1,1,0,2],'size':[3,3,3,3,3,5],'stride':[2,1,1,1,2,1]}
         layer_param=None
         part1=map_back_from_pool1(indices, layer_param)
         part1=np.hstack((part1, np.sum(fmap[indices]/indices[0].shape[0])))
         part2=map_back_small_part_from_pool1(indices,fmap, layer_param)
         return np.vstack((part1,part2))
         #return part2
         '''
         x_min=np.min(indices[1])
         y_min=np.min(indices[0])
         x_max=np.max(indices[1])
         y_max=np.max(indices[0])
         if x_min==x_max:
            x_max=x_min+1
         if y_min==y_max:
            y_max=y_min+1
         x_min=x_min*x_factor
         x_max=x_max*x_factor
         y_min=y_min*y_factor
         y_max=y_max*y_factor
         return np.array([x_min,y_min,x_max,y_max])
         results=generate_clusters(indices)
         results[:,0]=x_factor*results[:,0]
         results[:,2]=x_factor*results[:,2]
         results[:,1]=y_factor*results[:,1]
         results[:,3]=y_factor*results[:,3]
         return results
         '''
    def map_back_small_part_from_pool5(indices,fmap):
         pool_size=0
         pool_stride=1
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
                   candidates.pop(i)
                   score=score+fmap[_x,_y]
                else:
                   i=i+1
            new_boxes=np.vstack((new_boxes,np.array([x_min,y_min, x_max,y_max,score]))) 
            candidate_count=len(candidates)
         x_min=new_boxes[:,0]
         x_max=new_boxes[:,2]
         y_min=new_boxes[:,1]
         y_max=new_boxes[:,3]
         x_min=x_min*pool_stride
         x_max=x_max*pool_stride+pool_size#-1
         y_min=y_min*pool_stride
         y_max=y_max*pool_stride+pool_size#-1
         new_boxes[:,0]=x_min
         new_boxes[:,2]=x_max
         new_boxes[:,1]=y_min
         new_boxes[:,3]=y_max
         return new_boxes
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
    def get_featuremap_box(indices,fmap):
         x_min=np.min(indices[1])
         y_min=np.min(indices[0])
         x_max=np.max(indices[1])
         y_max=np.max(indices[0])
         candidates=range(indices[0].shape[0])
         new_boxes=np.zeros([0,5])
         #new_boxes=np.vstack((new_boxes,np.array([x_min,y_min,x_max,y_max,0])))
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
                   candidates.pop(i)
                   score=score+fmap[_x,_y]
                else:
                   i=i+1
            new_boxes=np.vstack((new_boxes,np.array([x_min,y_min, x_max,y_max,score]))) 
            candidate_count=len(candidates)
         return new_boxes[:,0:-1]
         '''
         x_min=np.min(indices[1])
         y_min=np.min(indices[0])
         x_max=np.max(indices[1])
         y_max=np.max(indices[0])
         return np.array([x_min,y_min,x_max,y_max])
         '''
    gt_in_fmap=map_gt_forward_pool1(gt_boxes)
    #for fi in range(net_proposal.blobs['pool1'].data.shape[0]):
    '''
    temp_input=np.zeros([0,5])
    for fi in range(1):
        for fj in range(net_proposal.blobs['pool5'].data.shape[1]):
               temp_proposals=np.zeros([0,5])
               f_proposals=np.zeros([0,4])
               fmap=net_proposal.blobs['pool5'].data[fi,fj,:,:]
               fmap=np.divide(fmap, np.max(fmap))
               fmap=np.multiply(fmap, 10)
               selected_indices=np.where(fmap>np.mean(fmap))
               part2=map_back_small_part_from_pool5(selected_indices,fmap)
               temp_input = np.vstack((temp_input , part2))
               vis_featuremap(fmap, "{}:{}".format(fi,fj), part2[:,0:-1], savename+"_featuremap_{}-{}".format(fi,fj))
    pdb.set_trace()
    '''
    '''
    fmap=np.zeros([27,27])
    k_proposals=np.zeros([0,4])
    for fi in range(1):
        for fj in range(net_proposal.blobs['pool1'].data.shape[1]):
            f1=net_proposal.blobs['pool1'].data[fi,fj,:,:]
            f1=np.divide(f1,np.max(f1))
            fmap= np.add(fmap,f1) 
    vis_featuremap(fmap, "{}:{}".format(0,0), k_proposals, savename+"_featuremap_{}-{}".format(0,0), True, k_proposals)
    '''
    k_proposals=np.zeros([0,4])
    levelmap=np.zeros([28,28])
    fmap=np.zeros([27,27])
    for fj in range(net_proposal.blobs['pool1'].data.shape[1]):
        f1=net_proposal.blobs['pool1'].data[0,fj,:,:]
        if np.max(f1)==0:
           continue
        f1=np.divide(f1,np.max(f1))
        fmap= np.add(fmap,f1)
    fmap=np.divide(fmap, np.max(fmap))
    fmap = np.hstack((fmap, np.zeros([27,10])))
    fmap = np.vstack((fmap, np.zeros([10,37])))
    input_proposals = net_proposal.blobs['rois2'].data[:,1:]
    vis_featuremap(fmap, "integral feature map", input_proposals, savename+"_featuremap_integral", False, k_proposals)
    #input_proposals = map_back_boxes_from_pool1(input_proposals[:,(1,0,3,2)])
    input_proposals = map_back_boxes_from_pool1(input_proposals)
    input_proposals = box_backtrace(input_proposals, im_width/227, im_height/227)
    for fi in range(0):
               '''
        for fj in range(net_proposal.blobs['pool1'].data.shape[1]):
               temp_proposals=np.zeros([0,5])
               f_proposals=np.zeros([0,4])
               fmap=net_proposal.blobs['pool1'].data[fi,fj,:,:]
               fmap=np.divide(fmap, np.max(fmap))
               fmap=np.multiply(fmap, 10)
               #pdb.set_trace()
               '''
               level_num=10
               f_proposals=np.zeros([0,4])
               fmap=np.zeros([27,27])
               temp_proposals=np.zeros([0,5])
               for fj in range(net_proposal.blobs['pool1'].data.shape[1]):
                  f1=net_proposal.blobs['pool1'].data[fi,fj,:,:]
                  if np.max(f1)==0:
                     continue
                  f1=np.divide(f1,np.max(f1))
                  fmap= np.add(fmap,f1)
               fmap=np.divide(fmap, np.max(fmap))
               fmap=np.multiply(fmap,level_num)
               maximal_fmap=np.max(fmap)
               vis_featuremap(fmap, "integral feature map", k_proposals, savename+"_featuremap_integral", False, k_proposals)
               for level_i in range(level_num-1):
                   selected_indices=np.where((fmap>level_i)*(fmap<(level_i+1)))
                   #print "i:{},num:{}".format(level_i, len(selected_indices[0]))
                   levelmap[selected_indices]=fmap[selected_indices]
                   levelmap[27,27]=maximal_fmap
                   vis_featuremap(levelmap, "{}:{}".format(fi,level_i), k_proposals, savename+"_featuremap_{}-{}".format(fi,level_i), False, k_proposals)
                   levelmap[:]=0
                   if selected_indices[0].shape[0]!=0:
                      f_proposals = np.vstack((f_proposals,get_featuremap_box(selected_indices,fmap)))
                      selected_indices=map_back_from_fm(selected_indices, 1, 1,fmap)
                      print "i:{},num:{}".format(level_i, selected_indices.shape[0])
                      temp_proposals = np.vstack((temp_proposals,selected_indices))
               selected_indices=np.where(fmap>level_num-1)
               #print "i:{},num:{}".format(level_num-1, len(selected_indices[0]))
               levelmap[selected_indices]=fmap[selected_indices]
               levelmap[27,27]=maximal_fmap
               vis_featuremap(levelmap, "{}:{}".format(fi,9), k_proposals, savename+"_featuremap_{}-{}".format(fi,9), False, k_proposals)
               if selected_indices[0].shape[0]!=0:
                  f_proposals = np.vstack((f_proposals,get_featuremap_box(selected_indices,fmap)))
                  selected_indices=map_back_from_fm(selected_indices, 1, 1, fmap)
                  print "i:{},num:{}".format(level_num-1, selected_indices.shape[0])
                  temp_proposals = np.vstack((temp_proposals,selected_indices))
               '''
               fmap=np.divide(fmap, np.max(fmap))
               fmap=np.multiply(fmap,10)
               selected_indices=np.where(fmap>9)
               if selected_indices[0].shape[0]!=0:
                  f_proposals = np.vstack((f_proposals,get_featuremap_box(selected_indices,fmap)))
                  selected_indices=map_back_from_fm(selected_indices, 1, 1,fmap)
                  temp_proposals = np.vstack((temp_proposals,selected_indices))
               selected_indices=np.where((fmap>8)*(fmap<9))
               if selected_indices[0].shape[0]!=0:
                  f_proposals = np.vstack((f_proposals,get_featuremap_box(selected_indices,fmap)))
                  selected_indices=map_back_from_fm(selected_indices, 1, 1,fmap)
                  temp_proposals = np.vstack((temp_proposals,selected_indices))
               selected_indices=np.where((fmap>7)*(fmap<8))
               if selected_indices[0].shape[0]!=0:
                  f_proposals = np.vstack((f_proposals,get_featuremap_box(selected_indices,fmap)))
                  selected_indices=map_back_from_fm(selected_indices,1, 1,fmap)
                  temp_proposals = np.vstack((temp_proposals,selected_indices))
               selected_indices=np.where((fmap>6)*(fmap<7))
               if selected_indices[0].shape[0]!=0:
                  f_proposals = np.vstack((f_proposals,get_featuremap_box(selected_indices,fmap)))
                  selected_indices=map_back_from_fm(selected_indices,1, 1,fmap)
                  temp_proposals = np.vstack((temp_proposals,selected_indices))
               selected_indices=np.where((fmap>5)*(fmap<6))
               if selected_indices[0].shape[0]!=0:
                  f_proposals = np.vstack((f_proposals,get_featuremap_box(selected_indices,fmap)))
                  selected_indices=map_back_from_fm(selected_indices,1, 1,fmap)
                  temp_proposals = np.vstack((temp_proposals,selected_indices))
               selected_indices=np.where((fmap>4)*(fmap<5))
               if selected_indices[0].shape[0]!=0:
                  f_proposals = np.vstack((f_proposals,get_featuremap_box(selected_indices,fmap)))
                  selected_indices=map_back_from_fm(selected_indices,1, 1,fmap)
                  temp_proposals = np.vstack((temp_proposals,selected_indices))
               #temp_proposals_index = nms(temp_proposals.astype(np.float32), 0.55)
               #vis_featuremap(fmap, "{}:{}".format(fi,fj), k_proposals, savename+"_featuremap_{}-{}".format(fi,fj), True, k_proposals)
               '''
               #vis_featuremap(fmap, "{}:{}".format(fi,fj), f_proposals, savename+"_featuremap_{}-{}".format(fi,fj), True, gt_in_fmap)
               #temp_proposals = box_backtrace(temp_proposals, im_width/fmap.shape[1], im_height/fmap.shape[0])
               #pdb.set_trace()
               temp_proposals = box_backtrace(temp_proposals[:,0:-1], im_width/227, im_height/227)
               #vis_featuremap(im, "{}:{}".format(fi,fj), temp_proposals, savename+"_proposals_{}-{}".format(fi,fj))
               #pdb.set_trace()
               #temp_proposals = nms_slow(temp_proposals,0.3)
               input_proposals = np.vstack((input_proposals,temp_proposals))
    #vis_proposals(im, input_proposals, 'records/proposal_with_integral_image.jpg')
    #BOXSIZE_THRESH=2500
    for ib in range(0):
    #for ib in range(input_proposals.shape[0]):
        bb=input_proposals[ib,:]
        new_box_list=np.zeros([0,4]).astype(np.float32)
        #select the small proposals , and use local search
        #scale_v=[1,0.5,1.5]
        scale_v=[random.uniform(0.5,0.99), random.uniform(1.1, 1.5)]
        w=bb[2]-bb[0]
        h=bb[3]-bb[1]
        x=bb[0]+w/2
        y=bb[1]+h/2
        for _w in scale_v:
         for _h in scale_v:
            n_w=w*_w
            n_h=h*_h
            new_box_list=np.vstack((new_box_list, np.array([x-n_w/2, y-n_h/2, x+n_w/2, y+n_h/2])))
        input_proposals=np.vstack((input_proposals, new_box_list))
    vis_proposals(im, input_proposals, 'records/proposal_with_localsearch.jpg')
    '''
    BOXSIZE_TRANSLATE_LIM=10
    for ib in range(input_proposals.shape[0]):
        if ib%2==0:
           continue
        bb=input_proposals[ib,:]
        new_box_list=np.zeros([0,4]).astype(np.float32)
        #select the small proposals , and use local search
        translate_v=[random.randint(1, BOXSIZE_TRANSLATE_LIM), -1*random.randint(1, BOXSIZE_TRANSLATE_LIM)]
        scale_v=[random.uniform(0.5,0.99), random.uniform(1.1, 1.5)]
        w=bb[2]-bb[0]
        h=bb[3]-bb[1]
        x=bb[0]+w/2
        y=bb[1]+h/2
        for _x in translate_v:
         for _y in translate_v:
          for _w in scale_v:
           for _h in scale_v:
            n_x=x+_x
            n_y=y+_y
            n_w=w*_w
            n_h=h*_h
            if (n_x-n_w/2)<0 or (n_y-n_h/2)<0 or (n_x+n_w/2)>im.shape[1] or (n_y+n_h/2)>im.shape[0]:
               continue
            new_box_list=np.vstack((new_box_list, np.array([n_x-n_w/2, n_y-n_h/2, n_x+n_w/2, n_y+n_h/2])))
        input_proposals=np.vstack((input_proposals, new_box_list))
    '''
    obj_proposals=input_proposals
    #tmp_gt_b = gt_map_back_from_pool1(gt_in_fmap)
    #tmp_gt_b = box_backtrace(tmp_gt_b, im_width/227, im_height/227)
    #obj_proposals=np.vstack((obj_proposals,tmp_gt_b))
    #box_file = os.path.join('/home/selective_search',
    #                        image_name + '.mat')
    #obj_proposals = sio.loadmat(box_file)['boxes'].astype(np.float32)
    #temp_proposals=np.zeros([obj_proposals.shape[0],1])
    #temp_proposals[:,0]=obj_proposals[:,0]
    #obj_proposals[:,0]=obj_proposals[:,1]
    #obj_proposals[:,1]=temp_proposals[:,0]
    #temp_proposals[:,0]=obj_proposals[:,2]
    #obj_proposals[:,2]=obj_proposals[:,3]
    #obj_proposals[:,3]=temp_proposals[:,0]
    print "proposal counts: {}".format(obj_proposals.shape[0])
    #verify the power of classifier
    #obj_proposals=np.vstack((obj_proposals, gt_boxes))
    #visualize groudtruth
    vis_overlap_proposal_groundtruth(im,gt_classes, gt_boxes, obj_proposals, savename+'_groundtruth.jpg')
    # Visualize detections for each class
    EXCEPT_THRESH = 0.9
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    #THRESH_STEP = (EXCEPT_THRESH-CONF_THRESH)/loop_counts
    scorelist=[]
    history_score_list={}
    input_proposals={}
    history_groundtruth_iou={}
    firstforward_score={}
    gt_box_size=[]
    groundtruth_iou_record={}
    for ground_i in range(num_objs):#select a groundtuth bounding box
        gt_box_select=gt_boxes[ground_i,:]
        temp_size=(gt_box_select[2]-gt_box_select[0])*(gt_box_select[3]-gt_box_select[1])
        gt_box_size.append(temp_size)
        groundtruth_iou_record[temp_size]=[]
    #the first forward, filter most of the proposals
    scores, boxes = im_detect(net, im, obj_proposals)
    vis_proposals(im, obj_proposals, savename+'_proposals.jpg')
    pred_names=[]
    pred_dets=np.zeros([0,5])
    for cls in classes:
       cls_ind = CLASSES.index(cls)
       cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
       cls_scores = scores[:, cls_ind]
       keep = np.where(cls_scores >= CONF_THRESH)[0]
       select_boxes = cls_boxes[keep, :]
       select_scores = cls_scores[keep]
       dets = np.hstack((select_boxes,
                       select_scores[:, np.newaxis])).astype(np.float32)
       keep_nms = nms(dets, NMS_THRESH)
       dets = dets[keep_nms, :]
       for ii in range(len(keep_nms)):
           pred_names.append(cls)
       pred_dets=np.vstack((pred_dets, dets)).astype(np.float32)
       input_proposals[cls] = select_boxes[keep_nms,:]
       #history_boxes_list[cls]=input_proposals[cls]
       vis_detections(im, cls, dets, savename+'_'+cls+'_firstForwardPredictBox.jpg', thresh=0)
       firstforward_score[cls]=select_scores[keep_nms]
    temp_prop=pred_dets
    temp_index=[]
    num_recursive=10
    for l in xrange(num_recursive):
        if temp_prop.shape[0]==0:
           print l
           break
        scores, boxes = im_detect(net, im, temp_prop[:,0:-1])

        for cls in classes:
              cls_ind = CLASSES.index(cls)
              inds = np.where((scores[:, cls_ind]-temp_prop[:,-1] > 0))[0]
              #temp_prop_backup[inds, 0:4] = temp_prop[inds,0:4]
              temp_prop[inds,-1] = scores[inds, cls_ind]
              temp_prop[inds,0:4] = boxes[inds, cls_ind*4:(cls_ind+1)*4]
              temp_index.extend(inds)
        temp_index=list(set(temp_index))
        temp_prop=temp_prop[temp_index,:]
        #temp_prop_backup=temp_prop_backup[temp_index,:]
        temp_index=[]
    vis_static(im, gt_classes, gt_boxes, pred_names, pred_dets, history_score_list, history_groundtruth_iou, loop_counts+1,savename+'_afterRecursive.jpg')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        #choices=NETS.keys(), default='vgg16')
                        #choices=NETS.keys(), default='caffe_imagenet')
                        choices=NETS.keys(), default='caffenet')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    #prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
    #                        'imagenet_test.prototxt')
    #caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'default',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    #network for proposals
    net_proposal=caffe.Classifier('/home/models/bvlc_reference_caffenet/deploy1.prototxt','/home/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',image_dims=(227,227), raw_scale=255,mean=np.load('/home/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), channel_swap=(2,1,0))
    #net_proposal=caffe.Classifier('/home/models/bvlc_reference_caffenet/deploy.prototxt','/home/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',image_dims=(227,227), raw_scale=255,mean=np.load('/home/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), channel_swap=(2,1,0))
    #net_proposal=caffe.Classifier('/home/models/bvlc_reference_caffenet/deploy.prototxt','/home/fast-rcnn-master/output/default/caffenet_fast_rcnn_iter_40000.caffemodel',image_dims=(227,227), raw_scale=255,mean=np.load('/home/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), channel_swap=(2,1,0))
    #net_proposal=caffe.Classifier('/home/neural-style/models/VGG_ILSVRC_19_layers_deploy.prototxt','/home/neural-style/models/VGG_ILSVRC_19_layers.caffemodel',image_dims=(227,227), raw_scale=255,mean=np.load('/home/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), channel_swap=(2,1,0))

    print '\n\nLoaded network {:s}'.format(caffemodel)
    #test()

    #if use CLASSES as input it will cast out divide zero errors in nms, 
    #I thought it would be caused by __background__ class
    threshold_boundary=0.2
    max_improvement=[0,'']
    improve_uperthen_threshold=[]
    loop_counts=1
    total_improvement=0
    total_proposalcounts=0
    groundtruth_iou_record={}
    #first_iou_record={}
    with open('voc_val.txt','r') as fd:
         i=1
         #for line in fd:
         for line in ['11','002940','009647','009443','009113','009087','008424','007786','006029','005653','005326','004585']:
             fname=line.strip()
             #fname='005653'
             #fname='000268'
             fname='009113'
             print '~~~~~~~~~~~~~~~~Image {}~~~~~~~~~~~~~~~~~~~'.format(i)
             print 'Run data/demo/{}.jpg'.format(fname)
             demo_iouBoundary_rilievo(net, net_proposal, fname, CLASSES[1:],'records/{}'.format(fname),loop_counts)
             i=i+1
             #analysis
             '''
             for (k,v) in iou_record.iteritems():
                 k_sqrt=int(math.sqrt(k))
                 if k_sqrt in groundtruth_iou_record:
                    groundtruth_iou_record[k_sqrt] += v
                 else:
                    groundtruth_iou_record[k_sqrt] = v
             temp_gap=result[1,:]-result[0,:]
             total_proposalcounts=total_proposalcounts+result.shape[1] 
             total_improvement=total_improvement+np.sum(temp_gap)
             #improve more then trheshold
             for j in range(result.shape[1]):
                 if temp_gap[j] > threshold_boundary:
                    improve_uperthen_threshold.append((fname, temp_gap[j]))
                 if temp_gap[j] > max_improvement[0]:
                    max_improvement[0]=temp_gap[j]
                    max_improvement[1]=line
             '''
             if i==2:
                break
         fd.close()
    #iou_vis_x=[]
    #iou_vis_mean=[]
    #iou_vis_min=[]
    #iou_vis_max=[]
    #sort
    #groundtruth_iou_record=sorted(groundtruth_iou_record.iteritems(), key=lambda d:d[0])
    #first_iou_record=sorted(first_iou_record.iteritems(), key=lambda d:d[0])
    '''
    for (k,v) in groundtruth_iou_record:
        #pdb.set_trace()
        iou_vis_x.append(k)
        if len(v)!=0:
           #iou_vis_first.append(sum(first_iou_record[k])/len(first_iou_record[k]))
           #iou_vis_mean.append(sum(v)/len(v))
           iou_vis_min.append(min(v))
           iou_vis_max.append(max(v))
        else:
           #iou_vis_first.append(0.0)
           #iou_vis_mean.append(0.0)
           iou_vis_min.append(0.0)
           iou_vis_max.append(0.0)
    '''
    #vis_iou_and_object_size(iou_vis_x, iou_vis_min, iou_vis_max, 'iou_and_object_size.jpg')
    #print 'average gap between first forward iou and best iou: {}'.format(total_improvement/total_proposalcounts)
    #print 'maximun gap between first forward iou and best iou: {} in pic {}.jpg'.format(max_improvement[0],max_improvement[1])
    #with open('siginificant_improvement.txt','w') as fd:
    #     fd.write('improvement more then {}\n'.format(threshold_boundary))
    #     fd.write('{} out of {}\n'.format(len(improve_uperthen_threshold),total_proposalcounts))
    #     for i in range(len(improve_uperthen_threshold)):
    #         fd.write('{}:{}\n'.format(improve_uperthen_threshold[i][0],improve_uperthen_threshold[i][1]))
    #     fd.close()

    #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #print 'Demo for data/demo/000004.jpg'
    #demo_iouBoundary(net, '000004', ('car',),'records/car1')

    #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #print 'Demo for data/demo/001551.jpg'
    #demo_viscous(net, '001551', ('sofa', 'tvmonitor'),'records/sofa1')

    #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #print 'Demo for data/demo/006230.jpg'
    #demo_viscous(net, '006230', ('dog',),'records/006230')

    #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #print 'Demo for data/demo/009865.jpg'
    #demo_viscous(net, '009865', ('cow','dog'),'records/009865')
