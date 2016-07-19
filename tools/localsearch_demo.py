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
import random
import pdb

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
        
def demo_localsearch(net, image_name, classes, savename,loop_counts=50):
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
    BOXSIZE_THRESH = 2500
    BOXSIZE_TRANSLATE_LIM = 20
    THRESH_STEP = (EXCEPT_THRESH-CONF_THRESH)/loop_counts
    scorelist=[]
    history_score_list={}
    history_groundtruth_iou={}
    firstforward_score={}
    gt_box_size=[]
    groundtruth_iou_record={}
    input_proposals=np.zeros([0,4]).astype(np.float32)
    for ground_i in range(num_objs):#select a groundtuth bounding box
        gt_box_select=gt_boxes[ground_i,:]
        temp_size=(gt_box_select[2]-gt_box_select[0])*(gt_box_select[3]-gt_box_select[1])
        gt_box_size.append(temp_size)
        groundtruth_iou_record[temp_size]=[]
    #the first forward, filter most of the proposals
    scores, boxes = im_detect(net, im, obj_proposals)
    #vis_proposals(im, obj_proposals, savename+'_proposals.jpg')
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
       select_proposals=obj_proposals[keep, :]
       dets = dets[keep_nms, :]
       #input_proposals=np.vstack((select_proposals[keep_nms,:], select_boxes[keep_nms,:]))
       input_proposals=np.vstack((input_proposals,select_proposals[keep_nms,:]))
       #history_boxes_list[cls]=input_proposals[cls]
       vis_detections(im, cls, dets, savename+'_'+cls+'_firstForwardPredictBox.jpg', thresh=0)
    for ib in range(input_proposals.shape[0]):
       bb=input_proposals[ib,:]
       new_box_list=np.zeros([0,4]).astype(np.float32)
       #select the small proposals , and use local search
       if (bb[3]-bb[1])*(bb[2]-bb[0])<=BOXSIZE_THRESH:
          translate_v=[0,random.randint(1, BOXSIZE_TRANSLATE_LIM), -1*random.randint(1, BOXSIZE_TRANSLATE_LIM)]
          scale_v=[1,random.uniform(0.5,0.99), random.uniform(1.1, 1.5)]
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
                    new_box_list=np.vstack((new_box_list, np.array([n_x-n_w/2, n_y-n_h/2, n_x+n_w/2, n_y+n_h/2])))
       input_proposals=np.vstack((input_proposals, new_box_list[1:,:]))
    vis_proposals(im, input_proposals, savename+'_localsearch_proposals.jpg')
    if input_proposals.shape[0]==0:
       print 'no suitable fine tune bounding boxes'
       return
    scores, boxes = im_detect(net, im, input_proposals)
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
       input_proposals=np.vstack((input_proposals, select_boxes[keep_nms,:]))
       #history_boxes_list[cls]=input_proposals[cls]
       vis_detections(im, cls, dets, savename+'_'+cls+'_localsearch_PredictBox.jpg', thresh=0)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
                        #choices=NETS.keys(), default='caffe_imagenet')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    #prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
    #                        'imagenet_test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
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

    print '\n\nLoaded network {:s}'.format(caffemodel)
    #test()

    #if use CLASSES as input it will cast out divide zero errors in nms, 
    #I thought it would be caused by __background__ class
    threshold_boundary=0.2
    max_improvement=[0,'']
    improve_uperthen_threshold=[]
    loop_counts=30
    total_improvement=0
    total_proposalcounts=0
    groundtruth_iou_record={}
    #first_iou_record={}
    with open('valset.txt','r') as fd:
         i=1
         for line in fd:
             fname=line.strip()
             #fname='001371'
             print '~~~~~~~~~~~~~~~~Image {}~~~~~~~~~~~~~~~~~~~'.format(i)
             print 'Run data/demo/{}.jpg'.format(fname)
             demo_localsearch(net, fname, CLASSES[1:],'records/{}'.format(fname),loop_counts)
             if i==100:
                break
             i=i+1
         fd.close()
