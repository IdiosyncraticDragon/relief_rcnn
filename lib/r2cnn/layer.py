# --------------------------------------------------------
# --------------------------------------------------------

"""The data layer used during testing to generate region proposals.

"""

import caffe
from fast_rcnn.config import cfg
import numpy as np
import yaml
from multiprocessing import Process, Queue
import random
import pdb

class ReliefDataLayer(caffe.Layer):
    """Relief region extraction layer used for testing."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""
        self._level_num=10

    def map_back_from_pool1(self, indices):
         x_min=np.min(indices[1])
         y_min=np.min(indices[0])
         x_max=np.max(indices[1])
         y_max=np.max(indices[0])
         return np.array([x_min,y_min,x_max,y_max])
    def map_back_small_part_from_pool1(self, indices, fmap):
         candidates=range(indices[0].shape[0])
         new_boxes=np.zeros([0,4])
         candidate_count=len(candidates)
         while candidate_count>0:
            startpoint=candidates.pop(0)
            x_min=indices[1][startpoint]
            x_max=x_min
            y_min=indices[0][startpoint]
            y_max=y_min
            #score=fmap[x_min,y_min]
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
                   #score=score+fmap[_x,_y]
                #attach to the boundary of detection box
                elif (_x>=x_min-1)&(_x<=x_max+1)&(_y>=y_min-1)&(_y<=y_max+1):
                   x_min=min(x_min,_x)
                   x_max=max(x_max,_x)
                   y_min=min(y_min,_y)
                   y_max=max(y_max,_y)
                else:
                   i=i+1
            #new_boxes=np.vstack((new_boxes,np.array([x_min,y_min, x_max,y_max,score])))
            new_boxes=np.vstack((new_boxes,np.array([x_min,y_min, x_max,y_max])))
            candidate_count=len(candidates)
         x_min=new_boxes[:,0]
         x_max=new_boxes[:,2]
         y_min=new_boxes[:,1]
         y_max=new_boxes[:,3]
         new_boxes[:,0]=x_min
         new_boxes[:,2]=x_max
         new_boxes[:,1]=y_min
         new_boxes[:,3]=y_max
         return new_boxes
    def generate_rois(self, indices, fmap):
        part1=self.map_back_from_pool1(indices)
        part2=self.map_back_small_part_from_pool1(indices,fmap)
        return np.vstack((part1,part2))
    def forward(self, bottom, top):
        fmap_w = bottom[0].data.shape[2]
        fmap_h = bottom[0].data.shape[3]
        fheatmap=np.zeros([fmap_w,fmap_h])
        input_proposals=np.zeros([0,4])
        temp_proposals=np.zeros([0,4])
        for i in range(bottom[0].data.shape[1]):
            fx = bottom[0].data[0,i,:,:]
            fmax_v=np.max(fx)
            if fmax_v==0:
               continue
            fheatmap = np.add(fheatmap , np.divide(fx,fmax_v))
        fheatmap = np.divide(fheatmap , np.max(fheatmap))
        fheatmap = np.multiply(fheatmap , self._level_num)
        for level_i in range(self._level_num - 1):
            selected_indices=np.where((fheatmap>level_i)*(fheatmap<(level_i+1)))
            if selected_indices[0].shape[0]!=0:
               selected_indices=self.generate_rois(selected_indices, fheatmap)
               temp_proposals = np.vstack((temp_proposals,selected_indices))
        #the highest layer
        selected_indices=np.where(fheatmap>self._level_num-1)
        if selected_indices[0].shape[0]!=0:
           selected_indices=self.generate_rois(selected_indices, fheatmap)
           temp_proposals = np.vstack((temp_proposals,selected_indices))
        input_proposals = np.vstack((input_proposals,temp_proposals))
        BOXSIZE_TRANSLATE_LIM=10
        augment_proposals=np.zeros([0,4]).astype(np.float32)
        for ib in range(input_proposals.shape[0]):
            bb=input_proposals[ib,:]
            new_box_list=np.zeros([0,4]).astype(np.float32)
            #select the small proposals , and use local search
            translate_v=[random.randint(1, BOXSIZE_TRANSLATE_LIM), -1*random.randint(1, BOXSIZE_TRANSLATE_LIM)]
            #scale_v=[random.uniform(0.5,0.9), random.uniform(1.1, 1.5)]
            #scale_v=[1,random.uniform(0.5,0.9), random.uniform(1.1, 1.5)]
            scale_v=[0.5,1.5]
            w=bb[2]-bb[0]
            h=bb[3]-bb[1]
            x=bb[0]+w/2
            y=bb[1]+h/2
            for _w in scale_v:
             for _h in scale_v:
                 n_w=w*_w
                 n_h=h*_h
                 if (x-n_w/2)<0 or (y-n_h/2)<0 or (x+n_w/2)>bottom[0].data.shape[1] or (y+n_h/2)>bottom[0].data.shape[0]:
                    continue
                 new_box_list=np.vstack((new_box_list, np.array([x-n_w/2, y-n_h/2, x+n_w/2, y+n_h/2])))
            input_proposals=np.vstack((input_proposals, new_box_list))
        top[0].reshape(len(input_proposals),5)
        top[0].data[:,1:]=input_proposals
        #pdb.set_trace()

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
