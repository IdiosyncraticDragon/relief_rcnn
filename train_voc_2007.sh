#./tools/train_net.py --gpu 0 --solver models/CaffeNet/solver.prototxt --weights ../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --iters 160000
./tools/train_net.py --gpu 0 --solver models/CaffeNet/solver.prototxt --weights ./output/default/caffenet_fast_rcnn_iter_40000.caffemodel --iters 40000
#./tools/train_net.py --gpu 0 --solver models/CaffeNet/solver2.prototxt --weights ./output/default/caffenet_fast_rcnn_iter_40000.caffemodel --iters 60000
