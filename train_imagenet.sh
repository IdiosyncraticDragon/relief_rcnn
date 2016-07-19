#!/bin/bash
start=$(date "+%s")
./tools/train_net.py --gpu 0 --solver models/CaffeNet/imagenet_solver.prototxt --weights output/default/val_imagenet/caffenet_fast_rcnn_imagenet_iter_7650000.caffemodel --imdb imagenet_2015 --iters 2000000
#./tools/train_net.py --gpu 0 --solver models/CaffeNet/imagenet_solver.prototxt --weights ../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --imdb imagenet_2015 --iters 400000
now=$(date "+%s")
time=$((now-start))
echo "Total Time Consume:$time seconds."
