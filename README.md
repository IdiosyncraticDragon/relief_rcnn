# Relief R-CNN

## Introduction
This project is the Relief R-CNN. Which is the scratch version, some problem have not been tuned. Further submition is on the way.

Reference to http://arxiv.org/abs/1601.06719

## Installation
This project is based on Fast R-CNN, we only edit the caffe soucre code and do some small changes on the python files. It can be installed just as the Fast R-CNN described: https://github.com/rbgirshick/fast-rcnn .

### An example of installation
Take a docker image (`docker pull kaixhin/cuda-caffe:8.0`, https://hub.docker.com/r/kaixhin/cuda-caffe/) contains caffe as the base enviroment.
The versrion of NVIDIA driver on the host is: 367.48.

Requirement: install the python packages "easydict, opencv-python, cython, numpy, sklearn"

Enter the subfolder 
```
cd ./caffe-fast-rcnn
make -j8
make pycaffe
```

Then the project is successful compiled.

If the user wants to run the demo example `voc07.sh`, there are several more things to do (In this case, we do not modify the code but just make the pathes of related files as the same as that in the code. Users can also choose to modify the pathes in the code):
1. Link the dataset VOC2007 as a subfolder in ./data. For example:

```
>> ls VOCdevkit
VOC2007  VOCdevkit2007      comp.sh                                 devkit_doc.pdf        example_detector.m  example_segmenter.m  exe.sh  output.test  tmp         viewanno.m
VOCcode  annotations_cache  create_segmentations_from_detections.m  example_classifier.m  example_layout.m    exe.log              local   results      tmp.output  viewdet.m
>> ln -s VOCdevikt ~/test/relief_rcnn/data/VOCdevkit2007
```

2. Link the file `./caffe-fast-rcnn/python/caffe/imagenet/ilsvrc_2012_mean.npy` to `/home/caffe/python/caffe/imagenet/`
3. Link the file `./caffe-fast-rcnn/models/bvlc_reference_caffenet/deploy1.prototxt` to `/home/models/bvlc_reference_caffenet/` 
4. Link the folder of project relief_rcnn to `/home/relief_rcnn`
5. run `./voc07_test.sh`

### Docker images and dataset
Download the docker image from the link below:

link: https://pan.baidu.com/s/1Vz1m1WEWcNdIBhPOwW8ksQ 

password: adfc

Unzip the the files to get the r2cnn.tar, and then use docker import to load the r2cnn.tar.
Note that the user need to modify several things which are the step 1~5 shown above. 
