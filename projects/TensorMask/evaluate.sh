CUDA_HOME=/  python3 train_net.py \
    --config-file /root/detectron2/projects/TensorMask/configs/tensormask_R_50_FPN_1x.yaml \
    --eval-only \
    MODEL.WEIGHTS /root/detectron2/projects/PointRend/log_50_50/model_0014999.pth