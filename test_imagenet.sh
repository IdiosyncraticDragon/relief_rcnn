#!/bin/bash
start=$(date "+%s")
./tools/test_net.py --gpu 0 --def models/CaffeNet/imagenet_test.prototxt --net output/default/train_imagenet/caffenet_fast_rcnn_imagenet_iter_1600000.caffemodel --imdb imagenet_2015
#./tools/test_net.py --gpu 1 --def models/VGG16/imagenet_test.prototxt --net data/fast_rcnn_models/VGG16.v2.caffemodel --imdb imagenet_2015
now=$(date "+%s")
time=$((now-start))
echo "Total Time Consume:$time seconds."
