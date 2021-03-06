#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TensorMask Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, verify_results

from detectron2.utils.visualizer import Visualizer

# from tridentnet import add_tridentnet_config

from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    # cfg.merge_from_file("/root/detectron2/projects/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml")
    cfg.merge_from_file("/root/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml")
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = "/root/detectron2/projects/PointRend/log_80_20/model_0064999.pth"

    return cfg


def main(args):
    # Regist own dataset.
    from detectron2.data.datasets import register_coco_instances
    folder_data = "/root/detectron2/MADS_data_train_test/50_50_tonghop"

    # train_data
    name        = "mads_train"
    json_file   = os.path.join(folder_data, "train.json")
    image_root  = os.path.join(folder_data, "train", "images")

    # test data
    name_val        = "mads_val"
    json_file_val   = os.path.join(folder_data, "val.json")
    image_root_val  = os.path.join(folder_data, "val", "images")

    # registr
    register_coco_instances(name, {}, json_file, image_root)
    register_coco_instances(name_val, {}, json_file_val, image_root_val)

    cfg = setup(args)
    predictor = DefaultPredictor(cfg)

    img1 = cv2.imread("/root/detectron2/MADS_data_train_test/50_50_tonghop/val/images/Taichi_S6_ (7).jpg")
    img2 = cv2.imread("/root/detectron2/MADS_data_train_test/50_50_tonghop/val/images/Taichi_S6_ (100).jpg")
    img3 = cv2.imread("/root/detectron2/MADS_data_train_test/50_50_tonghop/val/images/Taichi_S6_ (200).jpg")
    img4 = cv2.imread("/root/detectron2/MADS_data_train_test/50_50_tonghop/val/images/Taichi_S6_ (300).jpg")
    img5 = cv2.imread("/root/detectron2/MADS_data_train_test/50_50_tonghop/val/images/Taichi_S6_ (400).jpg")

    list_img = [img1, img2, img3, img4, img5]    

    output_folder = "predict_image"
    os.makedirs(output_folder, exist_ok=True)

    for index, img in enumerate(list_img):    
        # print(img.shape)
        outputs = predictor(img)


        np_outputs = outputs["instances"].pred_masks.cpu().numpy()

        h, w, _ = img.shape
        
        zero_mask = np.zeros((h, w), np.uint8)
        for pred_mask in tqdm(np_outputs):
            zero_mask = cv2.bitwise_or(zero_mask, pred_mask.astype(np.uint8)*255)


        if cv2.__version__.startswith("3."):
            _, contours, _ = cv2.findContours(zero_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(zero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        image_out = f"test{index}.png"
        path_out = os.path.join(output_folder, image_out)
        cv2.imwrite(path_out, img)   



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)

