# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.imagenet_2015
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import pdb

class imagenet_2015(datasets.imdb):
    def __init__(self, image_set, devkit_path='/mnt/data/Detection/ILSVRC2015'):
        datasets.imdb.__init__(self, 'val_imagenet')#this may be the roi boxes mat
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._imagesets_path = os.path.join(self._devkit_path, 'ImageSets/DET')
        self._data_path = os.path.join(self._devkit_path, 'Data/DET')
        self._annotation_path = os.path.join(self._devkit_path, 'Annotations/DET')
        self._classes = ('__background__', # always index 0
'n02672831','n02691156','n02219486','n02419796','n07739125','n02454379','n07718747','n02764044','n02766320','n02769748','n07693725','n02777292','n07753592','n02786058','n02787622','n02799071','n02802426','n02807133','n02815834','n02131653','n02206856','n07720875','n02828884','n02834778','n02840245','n01503061','n02870880','n02879718','n02883205','n02880940','n02892767','n07880968','n02924116','n02274259','n02437136','n02951585','n02958343','n02970849','n02402425','n02992211','n01784675','n03000684','n03001627','n03017168','n03062245','n03063338','n03085013','n03793489','n03109150','n03128519','n03134739','n03141823','n07718472','n03797390','n03188531','n03196217','n03207941','n02084071','n02121808','n02268443','n03249569','n03255030','n03271574','n02503517','n03314780','n07753113','n03337140','n03991062','n03372029','n02118333','n03394916','n01639765','n03400231','n02510455','n01443537','n03445777','n03445924','n07583066','n03467517','n03483316','n03476991','n07697100','n03481172','n02342885','n03494278','n03495258','n03124170','n07714571','n03513137','n02398521','n03535780','n02374451','n07697537','n03584254','n01990800','n01910747','n01882714','n03633091','n02165456','n03636649','n03642806','n07749582','n02129165','n03676483','n01674464','n01982650','n03710721','n03720891','n03759954','n03761084','n03764736','n03770439','n02484322','n03790512','n07734744','n03804744','n03814639','n03838899','n07747607','n02444819','n03908618','n03908714','n03916031','n00007846','n03928116','n07753275','n03942813','n03950228','n07873807','n03958227','n03961711','n07768694','n07615774','n02346627','n03995372','n07695742','n04004767','n04019541','n04023962','n04026417','n02324045','n04039381','n01495701','n02509815','n04070727','n04074963','n04116512','n04118538','n04118776','n04131690','n04141076','n01770393','n04154565','n02076196','n02411705','n04228054','n02445715','n01944390','n01726692','n04252077','n04252225','n04254120','n04254680','n04256520','n04270147','n02355227','n02317335','n04317175','n04330267','n04332243','n07745940','n04336792','n04356056','n04371430','n02395003','n04376876','n04379243','n04392985','n04409515','n01776313','n04591157','n02129604','n04442312','n06874185','n04468005','n04487394','n03110669','n01662784','n03211117','n04509417','n04517823','n04536866','n04540053','n04542943','n04554684','n04557648','n04530566','n02062744','n04591713','n02391049')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_name()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'val',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_name(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._imagesets_path,
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.split(' ')[0] for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where ILSVRC2015 is expected to be installed.
        """
        return '/mnt/data/Detection/ILSVRC2015/'

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if  self._image_set == 'val':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['all_boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] )
            #box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'voc_' + self._year))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._annotation_path, 'val', index + '.xml')
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(get_data_from_tag(obj, 'xmin')) #- 1
            y1 = float(get_data_from_tag(obj, 'ymin')) #- 1
            x2 = float(get_data_from_tag(obj, 'xmax')) #- 1
            y2 = float(get_data_from_tag(obj, 'ymax')) #- 1
            cls = self._class_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_voc_results_file(self, all_boxes):
        filename = os.path.join('./', 'results_imagenet.txt')
        thresh=0.5
        with open(filename, 'wt') as f:
          for im_ind, index in enumerate(self.image_index):
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                   continue
                print 'Writing {} Imagenet results '.format(cls)
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                   continue
                # the VOCdevkit expects 1-based indices
                for k in xrange(dets.shape[0]):
                    if dets[k, -1]<=thresh:
                       continue
                    f.write('{} {} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                              format(im_ind+1, cls_ind, dets[k, -1],
                                     dets[k, 0] + 1, dets[k, 1] + 1,
                                     dets[k, 2] + 1, dets[k, 3] + 1))
          f.close()
        return filename

    def _do_matlab_eval(self, predict_file, output_dir):
        rm_results = self.config['cleanup']

        path = os.path.join(self._devkit_path,
                            'devkit','evaluation')
        ground_truth_dir=os.path.join(self._devkit_path, 'Annotations','DET','val')
        meta_file=os.path.join(self._devkit_path,'devkit','data','meta_det.mat')
        eval_file=os.path.join(self._devkit_path, 'ImageSets','DET','val.txt')
        blacklist_file=os.path.join(self._devkit_path,'devkit','data','ILSVRC2015_det_validation_blacklist.txt')
        option_cache_file='../data/ILSVRC2015_det_validation_ground_truth.mat'
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'fast_rcnn_eval_det(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(predict_file, ground_truth_dir,
                       meta_file, eval_file, blacklist_file, option_cache_file)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        predict_result_path = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(os.path.abspath(predict_result_path) , output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.imagenet_2015('val')
    res = d.roidb
    from IPython import embed; embed()
