CUDA_HOME=/  python3 train_net.py \
    --config-file /root/detectron2/projects/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml \
    --eval-only \
    MODEL.WEIGHTS /root/detectron2/projects/TridentNet/log_80_20/model_0029999.pth