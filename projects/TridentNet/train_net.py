#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator

from tridentnet import add_tridentnet_config


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """



    cfg = get_cfg()
    add_tridentnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # if args.eval_only:
    #     cfg.MODEL.WEIGHTS = "/root/detectron2/projects/TridentNet/log_80_20/model_0029999.pth"
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
        # Regist own dataset.
    from detectron2.data.datasets import register_coco_instances
    folder_data = "/root/detectron2/MADS_data_train_test/80_20_tonghop"

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

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
