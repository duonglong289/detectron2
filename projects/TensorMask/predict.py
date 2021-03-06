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

from tensormask import add_tensormask_config
from detectron2.data import MetadataCatalog, DatasetCatalog


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_tensormask_config(cfg)
    cfg.merge_from_file("/root/detectron2/projects/TensorMask/configs/tensormask_R_50_FPN_1x.yaml")
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = "/root/detectron2/projects/TensorMask/log_80_20/model_0024999.pth"

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

    img = cv2.imread("/root/detectron2/MADS_data_train_test/50_50_tonghop/train/images/HipHop1_ (1).jpg")
    
    outputs = predictor(img)


    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("test.png", out.get_image()[:, :, ::-1])

    # import ipdb; ipdb.set_trace()
    # # print(outputs["instances"].pred_classes)
    # # print(outputs["instances"].pred_boxes)

    np_outputs = outputs["instances"].pred_masks.cpu().numpy()
    h, w, _ = img.shape
    zero_mask = np.zeros((h, w), np.uint8)

    for pred_mask in tqdm(np_outputs):
        
        zero_mask = cv2.bitwise_or(zero_mask, pred_mask.astype(np.uint8)*255)

    cv2.imwrite("test.png", zero_mask)
    import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)

