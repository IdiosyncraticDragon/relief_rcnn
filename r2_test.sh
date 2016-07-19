cd caffe-fast-rcnn
make all -j8
cd ..
#./tools/test_net.py --gpu 0 --def models/CaffeNet/test3.prototxt --net output/default/caffenet_fast_rcnn_iter_40000.caffemodel
#./tools/test_net.py --def models/CaffeNet/test3.prototxt --net output/default/caffenet_fast_rcnn_iter_40000.caffemodel
./tools/test_net.py --gpu 0 --def models/CaffeNet/test.prototxt --net output/default/caffenet_fast_rcnn_iter_40000.caffemodel
#./tools/test_net.py --def models/CaffeNet/test.prototxt --net output/default/caffenet_fast_rcnn_iter_40000.caffemodel
#./tools/test_net.py --gpu 0 --def models/CaffeNet/test.prototxt --net output/default/voc_2007_trainval/caffenet_fast_rcnn_iter_40000.caffemodel
