# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ubteacher.engine.hooks import LossEvalHook
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.solver.build import build_lr_scheduler

import torch.distributed as dist
import math


# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

job_iter = 0
# Unbiased Teacher Trainer
class UBTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.pred_num = torch.Tensor([0 for class_i in range(7)])
        self.pred_num = self.pred_num.cuda()

        self.MU = cfg.SEMISUPNET.PARA_MU

        self.THRE = cfg.SEMISUPNET.PARA_T

        # self.valid_dcit = torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        # self.valid_dcit = self.valid_dcit.cuda()

        # self.valid_thre = {0: 0.67, 1: 0.68, 2: 0.75, 3: 0.62, 4: 0.716, 5: 0.62, 6: 0.2}

        self.valid_thre = { class_i : 0.7 for class_i in range(7)}

        self.true_score =torch.Tensor([0.119217, 0.15927, 0.570566, 0.1045, 0.04089, 0.005522, 0.005522])

        self.scale_f =[4.78, 3.58, 1, 5.45, 13.93, 103, 0.005522]
        self.scale_d = [0.119217, 0.15927, 0.570566, 0.1045, 0.04089, 0.005522, 0.005522]
        self.register_hooks(self.build_hooks())
        
        self.domain_label = {}
        self.pred_dist = {}

        self.pred_number = {}
        self.pred_number['daytime'] = {}
        self.pred_number['night'] = {}
        self.pred_number['night'] ={'overcast': {'countryroad': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'highway': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'residential': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'citystreet': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda()}, 'clear': {'countryroad': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'highway': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'residential': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'citystreet': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda()}, 'rainy': {'countryroad': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'highway': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'residential': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'citystreet': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda()}, 'snowy': {'countryroad': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'highway': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'residential': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'citystreet': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda()}}
        self.pred_number['daytime'] = {'overcast': {'countryroad': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'highway': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'residential': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'citystreet': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda()}, 'clear': {'countryroad': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'highway': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'residential': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'citystreet': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda()}, 'rainy': {'countryroad': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'highway': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'residential': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'citystreet': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda()}, 'snowy': {'countryroad': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'highway': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'residential': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda(), 'citystreet': torch.Tensor([0, 0, 0, 0, 0, 0, 0]).cuda()}}

        self.valid_dcit = {}
        self.valid_dcit['daytime'] = {}
        self.valid_dcit['night'] = {}
        self.valid_dcit['night'] ={'overcast': {'countryroad': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'highway': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'residential': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'citystreet': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])}, 'clear': {'countryroad': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'highway': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'residential': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'citystreet': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])}, 'rainy': {'countryroad': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'highway': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'residential': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'citystreet': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])}, 'snowy': {'countryroad': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'highway': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'residential': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'citystreet': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])}}
        self.valid_dcit['daytime'] = {'overcast': {'countryroad': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'highway': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'residential': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'citystreet': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])}, 'clear': {'countryroad': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'highway': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'residential': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'citystreet': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])}, 'rainy': {'countryroad': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'highway': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'residential': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'citystreet': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])}, 'snowy': {'countryroad': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'highway': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'residential': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]), 'citystreet': torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])}}

# | Pedestrian | 4901         |  Cyclist   | 6548         |    Car     | 23456        |
# |   Truck    | 4297         |    Tram    | 1681         |  Tricycle  | 227          |
        # self.mean_score = torch.Tensor([0.6356, 0.6769, 0.8365, 0.6534, 0.7421, 0.4580, 0.3000]) #0.6820, 0.7287, 0.8712, 0.6818, 0.7670, 0.6296, 0.3000
        # self.mean_score = torch.Tensor([0.6820, 0.7287, 0.8712, 0.6818, 0.7670, 0.6296, 0.3000]) #0.6820, 0.7287, 0.8712, 0.6818, 0.7670, 0.6296, 0.3000

        self.mean_score = torch.Tensor([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]) #0.6820, 0.7287, 0.8712, 0.6818, 0.7670, 0.6296, 0.3000
        self.mean_score = self.mean_score.cuda()

        self.cumulative_num = torch.Tensor([490, 654, 2345, 429, 168, 22])
        self.cumulative_num = self.cumulative_num.cuda()

        self.numbers = [0]
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            raise ValueError("Unknown test evaluator.")

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)
    
    def get_domain(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        domain_label = {}
        domain_label['period']   = [x["period"] for x in batched_inputs]
        domain_label['location'] = [x["location"] for x in batched_inputs]
        domain_label['weather']  = [x["weather"] for x in batched_inputs]
        domain_label['city']     = [x["city"] for x in batched_inputs]
        return domain_label

    def get_test_domain(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        domain0 = [0.1 if x["period"] == 'night' else 0.9 for x in batched_inputs for i in range(1000)]
        domain1 = [0.9 if x["period"] == 'night' else 0.1 for x in batched_inputs for i in range(1000)]
        return domain0, domain1

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling iteration mean score longhui 47.701==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, threshold_index, thres=0.7, proposal_type="roih"):
        if self.domain_label['period'][threshold_index] == 'night' :
            pe = 'night'
        else:
            pe = 'daytime'
        we = self.domain_label['weather'][threshold_index]
        lo = self.domain_label['location'][threshold_index]

        ########################  record num in every 2000 iterations  ########################   
        ########################  record num in every 2000 iterations  ########################   
        ########################  record num in every 2000 iterations  ########################   
        # if self.iter > 2000 and self.iter % 4000 == 0 :
        #     if self.pred_number['daytime']['clear']['citystreet'][2] > 50:
        #         domain_dist = torch.from_numpy(self.pred_dist[pe][we][lo]).cuda()
        #         domain_pred = self.pred_number[pe][we][lo] / ( torch.sum(self.pred_number[pe][we][lo]) + 1e-6 )
        #         for class_i in range(6):
        #             # temp_score = domain_pred[class_i] - domain_dist[class_i]

        #             temp_score = domain_pred[class_i] / (domain_dist[class_i] + 1e-5)
        #             self.valid_dcit[pe][we][lo][class_i] = self.THRE + self.MU * (temp_score -1)
        #             if self.valid_dcit[pe][we][lo][class_i] > 0.9:
        #                 self.valid_dcit[pe][we][lo][class_i] = 0.9
        #             if self.valid_dcit[pe][we][lo][class_i] < 0.4:
        #                 self.valid_dcit[pe][we][lo][class_i] = 0.4
            
        #     for item in self.pred_number.keys():
        #         for item1 in self.pred_number[item].keys():
        #             for item2 in self.pred_number[item][item1].keys():
        #                 self.pred_number[item][item1][item2] = 0 * self.pred_number[item][item1][item2]
        # if self.iter >= 4000 and self.iter % 4002 == 0 : 
        #     print('self.valid_dcit', self.valid_dcit)
        #     print('self.pred_number', self.pred_number)
        ########################  record num in every 2000 iterations  ########################   
        ########################  record num in every 2000 iterations  ########################   
        ########################  record num in every 2000 iterations  ########################   

        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            
            ########################  record num in every iterations  ########################   
            ########################  record num in every iterations  ########################   
            domain_dist = torch.from_numpy(self.pred_dist[pe][we][lo]).cuda()
            domain_pred = self.pred_number[pe][we][lo] / ( torch.sum(self.pred_number[pe][we][lo]) + 1e-6 )
            for class_i in range(6):
                temp_score = domain_pred[class_i] / (domain_dist[class_i] + 1e-5)
                self.valid_dcit[pe][we][lo][class_i] = self.THRE + self.MU * (temp_score)
                if self.valid_dcit[pe][we][lo][class_i] > 0.9:
                    self.valid_dcit[pe][we][lo][class_i] = 0.9
                if self.valid_dcit[pe][we][lo][class_i] < 0.4:
                    self.valid_dcit[pe][we][lo][class_i] = 0.4

            ########################  record num in every iterations  ########################   
            ########################  record num in every iterations  ########################  

            valid_map = proposal_bbox_inst.scores > 0.4
            if valid_map != torch.Size([]):
                for idx in range(valid_map.shape[0]): 
                    if proposal_bbox_inst.scores[idx] > self.valid_dcit[pe][we][lo][proposal_bbox_inst.pred_classes[idx].item()]:
                        valid_map[idx] = True
                    else:
                        valid_map[idx] = False

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            # new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

            temp_new_proposal_inst = new_proposal_inst

            if temp_new_proposal_inst.gt_classes.shape !=0:
                for item in temp_new_proposal_inst.gt_classes:
                    self.pred_number[pe][we][lo][item] +=1

            for item in self.pred_number.keys():
                for item1 in self.pred_number[item].keys():
                    for item2 in self.pred_number[item][item1].keys():
                        dist.all_reduce(self.pred_number[item][item1][item2])
                        self.pred_number[item][item1][item2] = self.pred_number[item][item1][item2] *1/8
 
            
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    
    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for threshold_index, proposal_bbox_inst in enumerate(proposals_rpn_unsup_k):
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, threshold_index, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        if self.iter == 1:
            self.pred_dist = self.compute_dist()

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)

            record_dict, _, _, _ = self.model(label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            
            self.domain_label = self.get_domain(unlabel_data_k)

            # print(self.iter,self.domain_label)
            # print(self.iter,len(unlabel_data_k) # 11, 4

            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            # print(self.iter, len(proposals_roih_unsup_k))
            # print(self.iter, self.domain_label)
            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )

            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
            
            #  add pseudo-label to unlabeled data
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )

            num_temp = 0
            if comm.is_main_process():
                for i in range(len(joint_proposal_dict["proposals_pseudo_roih"])):
                    num_temp += joint_proposal_dict["proposals_pseudo_roih"][i].gt_classes.shape[0]
                num_temp = num_temp/len(joint_proposal_dict["proposals_pseudo_roih"])
                self.numbers.append(num_temp)

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised"
            )

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            weight1 = 1.3
            weight2 = 1
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT  * weight1
                        )
                    elif key == 'loss_box_reg' or key == 'loss_rpn_loc':  # supervised location loss
                        loss_dict[key] = record_dict[key] * 1
                    else:                                                 # supervised class loss
                        loss_dict[key] = record_dict[key] * 1 * weight2

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    def compute_dist(self):
        scores = {}
        scores['daytime'] = {}
        scores['night'] = {}

        numbers = {}
        numbers['daytime'] = {}
        numbers['night'] = {}

        scores['night'] = {'overcast': {'countryroad': [0, 29.721445679664612, 33.385458052158356, 448.00276297330856, 9.983879268169403, 27.482713639736176, 0], 'highway': [0, 11.996051728725433, 7.421848237514496, 554.5970436930656, 15.603741705417633, 22.102834701538086, 0], 'residential': [0, 40.702196061611176, 45.28252685070038, 246.4668808579445, 6.263306498527527, 1.270714282989502, 0], 'citystreet': [0, 365.9523329734802, 393.14266604185104, 2912.9514278173447, 57.78581553697586, 167.92099231481552, 2.871622323989868]}, 'clear': {'countryroad': [0, 521.6920391321182, 539.4311027526855, 6213.347745537758, 279.97573828697205, 327.48872435092926, 0.6897161602973938], 'highway': [0, 117.72579818964005, 119.79884988069534, 6099.864707887173, 305.33882784843445, 244.74454373121262, 1.9133360385894775], 'residential': [0, 681.3167408108711, 299.4414690732956, 3087.3708448410034, 85.53083431720734, 83.70133405923843, 4.6106942892074585], 'citystreet': [0, 7620.287672460079, 5900.833307981491, 59125.533597290516, 1552.5014216899872, 3363.0686626434326, 29.287168502807617]}, 'rainy': {'countryroad': [0, 38.58420491218567, 43.14929014444351, 250.29872345924377, 7.500450253486633, 7.945677101612091, 0], 'highway': [0, 3.7483954429626465, 8.423958778381348, 327.30460226535797, 2.832624852657318, 7.518035888671875, 0], 'residential': [0, 33.83593267202377, 19.424758911132812, 172.17344117164612, 2.1736890077590942, 6.347582936286926, 0.5247583389282227], 'citystreet': [0, 510.25939828157425, 500.5623071193695, 3578.117887675762, 80.56670933961868, 206.7526080608368, 0.7816370725631714]}, 'snowy': {'countryroad': [0, 0, 0, 2.9443084597587585, 0.6220414042472839, 0, 0], 'highway': [0, 0, 0, 6.689682722091675, 0.6016169786453247, 0.7452204823493958, 0], 'residential': [0, 0, 0, 6.079625844955444, 0, 0, 0], 'citystreet': [0, 9.481568038463593, 1.908664345741272, 
        69.1016560792923, 1.1499841213226318, 2.51260244846344, 0]}}

        numbers['night'] ={'overcast': {'countryroad': [0, 40, 40, 507, 13, 32, 0], 'highway': [0, 18, 10, 626, 20, 29, 0], 'residential': [0, 54, 60, 288, 9, 2, 0], 'citystreet': [0, 476, 506, 3325, 74, 213, 4]}, 'clear': {'countryroad': [0, 681, 701, 7114, 362, 412, 1], 'highway': [0, 168, 170, 6972, 391, 325, 3], 'residential': [0, 897, 401, 3565, 130, 112, 7], 'citystreet': [0, 9856, 7571, 67480, 2089, 4240, 41]}, 'rainy': {'countryroad': [0, 52, 60, 290, 11, 11, 0], 'highway': [0, 
        6, 12, 371, 4, 11, 0], 'residential': [0, 46, 29, 203, 4, 10, 1], 'citystreet': [0, 685, 669, 4125, 109, 256, 1]}, 'snowy': {'countryroad': [0, 0, 0, 3, 1, 0, 0], 'highway': [0, 0, 0, 8, 1, 1, 0], 'residential': [0, 0, 0, 7, 0, 0, 0], 'citystreet': [0, 12, 3, 77, 2, 3, 0]}}

        scores['daytime'] = {'overcast': {'countryroad': [0, 3179.450203180313, 3150.499945998192, 30887.70902568102, 1732.955407857895, 1544.5778368115425, 57.32259660959244], 'highway': [0, 1603.2742991447449, 465.2879315018654, 30150.448750674725, 3490.489317417145, 1004.003116607666, 5.824916183948517], 'residential': [0, 1818.4527681469917, 800.3378636240959, 8144.444184362888, 436.57753705978394, 291.32996141910553, 14.763381719589233], 'citystreet': [0, 23463.110522150993, 20983.477319955826, 134290.87233024836, 5521.937575221062, 8642.283925116062, 306.332671046257]}, 'clear': {'countryroad': [0, 2096.672752201557, 2202.7569311857224, 20981.043084681034, 1739.0058815479279, 841.6768986582756, 46.82229417562485], 'highway': [0, 741.4121446013451, 411.9885575771332, 17577.939031362534, 
        2728.2972326278687, 560.491352379322, 4.897700846195221], 'residential': [0, 1835.157265841961, 861.1450644731522, 7566.972316622734, 531.1821922659874, 187.7119272351265, 15.810079276561737], 'citystreet': [0, 24933.84847086668, 22945.09187066555, 165953.95738840103, 6121.747785210609, 8076.945617198944, 265.5698854327202]}, 'rainy': {'countryroad': [0, 198.53658854961395, 256.683775305748, 2205.672272503376, 157.19665092229843, 110.60160291194916, 3.3259157538414], 'highway': [0, 34.330329954624176, 34.84497481584549, 2557.354231238365, 217.98271071910858, 72.1152064204216, 1.5323492288589478], 'residential': [0, 138.7300961613655, 75.70065385103226, 885.5921096205711, 28.62825495004654, 14.559546291828156, 1.481249988079071], 'citystreet': [0, 1881.68590092659, 
        2102.4951536655426, 13046.702382147312, 449.5474391579628, 809.7694026827812, 23.067989826202393]}, 'snowy': {'countryroad': [0, 0.6475703120231628, 0, 7.146092116832733, 0, 1.659706473350525, 0], 'highway': [0, 13.33048814535141, 0, 124.30727779865265, 4.827629327774048, 4.285930037498474, 0], 'residential': [0, 2.2644086480140686, 0, 20.162068963050842, 0, 0, 0], 'citystreet': [0, 37.11925387382507, 17.506975412368774, 319.08166486024857, 7.353128910064697, 21.041710793972015, 0]}}

        numbers['daytime'] = {'overcast': {'countryroad': [0, 3908, 3761, 33646, 2080, 1822, 76], 'highway': [0, 2170, 587, 32633, 4181, 1241, 8], 'residential': [0, 2205, 1017, 9099, 
        577, 406, 22], 'citystreet': [0, 28347, 24732, 146704, 6765, 9972, 383]}, 'clear': {'countryroad': [0, 2572, 2625, 22938, 2112, 1009, 64], 'highway': [0, 989, 519, 19248, 3293, 697, 6], 'residential': [0, 2206, 1083, 8393, 675, 251, 20], 'citystreet': [0, 30354, 27162, 181250, 7546, 9374, 341]}, 'rainy': {'countryroad': [0, 249, 313, 2410, 193, 133, 4], 'highway': [0, 47, 48, 2788, 263, 91, 2], 'residential': [0, 176, 98, 979, 39, 21, 2], 'citystreet': [0, 2401, 2545, 14396, 572, 947, 31]}, 'snowy': {'countryroad': [0, 1, 0, 8, 0, 2, 0], 'highway': [0, 19, 0, 133, 6, 6, 0], 'residential': [0, 3, 0, 21, 0, 0, 0], 
        'citystreet': [0, 44, 20, 344, 9, 26, 0]}}


        dist = {}
        for key1 in numbers.keys() :
            dist[key1] ={}
            for key2 in numbers[key1].keys():
                dist[key1][key2] = {}
                for key3 in numbers[key1][key2].keys():
                    dist[key1][key2][key3] = []
                    pred_t = numbers['daytime']['clear']['citystreet'][1:]
                    val_n = numbers[key1][key2][key3][1:]
                    # print(pred_t)
                    true_t = [375, 198, 2143, 259, 248, 17]

                    temp = [0,0,0,0,0,0]
                    true_n = [0,0,0,0,0,0]
                    for i in range(6):
                        temp[i] = true_t[i]/pred_t[i]
                        true_n[i] = val_n[i] * temp[i]
                    dist[key1][key2][key3] = np.array(true_n) / np.sum(np.array(true_n))
        
        return dist

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))

        if cfg.TEST.VAL_LOSS:  # default is True # save training time if not applied
            ret.append(
                LossEvalHook(
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    build_detection_test_loader(
                        self.cfg,
                        self.cfg.DATASETS.TEST[0],
                        DatasetMapper(self.cfg, True),
                    ),
                    model_output="loss_proposal",
                    model_name="student",
                )
            )

            ret.append(
                LossEvalHook(
                    cfg.TEST.EVAL_PERIOD,
                    self.model_teacher,
                    build_detection_test_loader(
                        self.cfg,
                        self.cfg.DATASETS.TEST[0],
                        DatasetMapper(self.cfg, True),
                    ),
                    model_output="loss_proposal",
                    model_name="",
                )
            )

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
