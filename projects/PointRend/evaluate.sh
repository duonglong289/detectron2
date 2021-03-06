CUDA_HOME=/  python3 train_net.py \
    --config-file /root/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml \
    --eval-only \
    MODEL.WEIGHTS = /root/detectron2/projects/PointRend/log_50_50/model_0014999.pth