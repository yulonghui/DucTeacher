# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs,
)

import numpy as np
 
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).losses()

        return losses

    def unsup_losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).unsup_losses()

        return losses


class FastRCNNFocalLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        num_classes=80,
    ):
        super(FastRCNNFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes


    def losses(self):
        return {
            "loss_cls": self.comput_focal_loss(),
            "loss_box_reg": self.box_reg_loss(),
        }

    def unsup_losses(self):
        return {
            "loss_cls": self.comput_unsup_loss(),
            "loss_box_reg": self.box_reg_loss(),
        }

    def comput_focal_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
            # total_loss = total_loss / self.gt_classes.shape[0]

            return total_loss
    
    def comput_unsup_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            Unsup_FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = Unsup_FC_loss(input=self.pred_class_logits, target=self.gt_classes)
            # total_loss = total_loss / self.gt_classes.shape[0]

            return total_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

        prior = np.array([0.119217, 0.15927, 0.570566, 0.1045, 0.04089, 0.005522])
        self.prior = torch.tensor(prior).float().cuda()
        # self.weight_b = torch.from_numpy(np.array([0.111, 0.106, 0.101, 0.116, 0.184, 1.0, 0.10])).float().cuda()
        self.weight_b = torch.from_numpy(np.array([1.11, 1.06, 1.01, 1.16, 1.84, 10.0, 1.0])).float().cuda()

    # JOB_SSCL9
    def forward(self, input, target):

        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        
        return loss.sum()/CE.shape[0]

class SeasawLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(SeasawLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

        prior = np.array([0.119217, 0.15927, 0.570566, 0.1045, 0.04089, 0.005522])
        self.prior = torch.tensor(prior).float().cuda()
        self.div_prior = 1/self.prior
        pp = self.prior.unsqueeze(0)
        exp_prior = torch.cat((pp, pp, pp, pp, pp, pp), 0)
        self.prio_ratio = exp_prior.T * self.div_prior
        self.prio_ratio = self.prio_ratio.clamp(max =1)
        self.prio_ratio = self.prio_ratio.pow(0.8)
        self.prio_ratio = self.prio_ratio.T
    
    def _split_cls_score(self, cls_score):
        # split cls_score to cls_score_classes and cls_score_objectness
        assert cls_score.size(-1) == self.num_classes + 1
        cls_score_classes = cls_score[..., :-1]
        cls_score_objectness = cls_score[..., -1:]
        return cls_score_classes, cls_score_objectness

    def forward(self, input, target):
        obj_labels = (target == self.num_classes).long()
        pos_inds = target < self.num_classes
        
        cls_score_classes, cls_score_objectness = self._split_cls_score(input)
        
        ############# mitigation factor #############
        seesaw_score, seesaw_label = cls_score_classes[pos_inds], target[pos_inds]
        onehot_labels = F.one_hot(seesaw_label, 6)
        seesaw_weights = self.prio_ratio[seesaw_label.long(), :]

        # seesaw_weights = seesaw_score.new_ones(onehot_labels.size())
        ############# mitigation factor #############
        
        ############# compensation factor #############
        scores = F.softmax(seesaw_score.detach(), dim=1)
        self_scores = scores[torch.arange(0, len(scores)).to(scores.device).long(), seesaw_label.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=1e-6)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(2) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor
        ############# compensation factor #############
        
        seesaw_score = seesaw_score + (seesaw_weights.log() * (1 - onehot_labels))
        
        obj_labels = torch.unsqueeze(obj_labels, dim=1)
    
        loss2 = F.binary_cross_entropy_with_logits(cls_score_objectness, obj_labels.float(), reduction='none')

        CE = F.cross_entropy(seesaw_score, seesaw_label, reduction="none")
        
        loss = CE

        return loss.sum()/loss2.shape[0] +0.5 * loss2.sum()/loss2.shape[0]










































class Unsup_FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(Unsup_FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

#     def forward(self, input, target):
#         # focal loss
#         # probs = F.softmax(input, dim=-1)
#         CE = F.cross_entropy(input, target, reduction="none")
#         p = torch.exp(-CE)
#         loss = (1 - p) ** self.gamma * CE
        
#         self.prior = self.prior/ torch.max(self.prior)
#         temp_w = torch.tensor([1, 1, 1, 1, 1, 0 if torch.rand(1)>0.75 else 1 ]).float().cuda()
#         input2 = input[:,:6] * temp_w 
        
#         probs = F.softmax(input, dim=-1)
#         idx = torch.where(target <5)
        
#         input2, target2 = input2[idx], target[idx]
#         CE2 = F.cross_entropy(input2, target2, reduction="none")
#         p2 = torch.exp(-CE2)
#         loss2 = (1 - p2) ** self.gamma * CE2

#         return loss.sum()/CE.shape[0] + 0.5 * loss2.sum()/CE2.shape[0]

    # def forward(self, input, target):
    #     # focal loss
    #     CE = F.cross_entropy(input, target, reduction="none")
    #     p = torch.exp(-CE)
    #     loss = (1 - p) ** self.gamma * CE

    #     probs = F.softmax(input, dim=-1)
    #     t_index = torch.where(target !=6)
    #     max_score = torch.max(probs, dim =1)[0]

    #     loss = max_score * loss

    #     return loss.sum()


# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import torch
# from torch import nn
# from torch.nn import functional as F
# from functools import partial
# from detectron2.modeling.roi_heads.fast_rcnn import (
#     FastRCNNOutputLayers,
#     FastRCNNOutputs,
# )

# import torch.distributed as dist
# import numpy as np


# # focal loss
# class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
#     def __init__(self, cfg, input_shape):
#         super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
#         self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

#     def losses(self, predictions, proposals):
#         """
#         Args:
#             predictions: return values of :meth:`forward()`.
#             proposals (list[Instances]): proposals that match the features
#                 that were used to compute predictions.
#         """
#         scores, proposal_deltas = predictions
#         losses = FastRCNNFocalLoss(
#             self.box2box_transform,
#             scores,
#             proposal_deltas,
#             proposals,
#             self.smooth_l1_beta,
#             self.box_reg_loss_type,
#             num_classes=self.num_classes,
#         ).losses()

#         return losses


# class FastRCNNFocalLoss(FastRCNNOutputs):
#     """
#     A class that stores information about outputs of a Fast R-CNN head.
#     It provides methods that are used to decode the outputs of a Fast R-CNN head.
#     """

#     def __init__(
#             self,
#             box2box_transform,
#             pred_class_logits,
#             pred_proposal_deltas,
#             proposals,
#             smooth_l1_beta=0.0,
#             box_reg_loss_type="smooth_l1",
#             num_classes=80,
#     ):
#         super(FastRCNNFocalLoss, self).__init__(
#             box2box_transform,
#             pred_class_logits,
#             pred_proposal_deltas,
#             proposals,
#             smooth_l1_beta,
#             box_reg_loss_type,
#         )
#         self.num_classes = num_classes

#     def losses(self):
#         return {
#             "loss_cls": self.comput_focal_loss(),
#             "loss_box_reg": self.box_reg_loss(),
#         }

#     def comput_focal_loss(self):
#         if self._no_instances:
#             return 0.0 * self.pred_class_logits.sum()
#         else:
#             FC_loss = FocalLoss(
#                 gamma=1.5,
#                 num_classes=self.num_classes,
#             )
#             total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
#             total_loss = total_loss / self.gt_classes.shape[0]

#             return total_loss


# class FocalLoss(nn.Module):
#     def __init__(
#             self,
#             weight=None,
#             gamma=1.0,
#             num_classes=80,
#     ):
#         super(FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.weight = weight

#         # cfg for eqlv2
#         self.vis_grad = False
#         self.gamma1 = 12
#         self.mu = 0.8
#         self.alpha = 8

#         self.grad_softmax = np.array([0, 0, 0, 0, 0, 0, 0])
#         self.grad_softmax = torch.tensor(self.grad_softmax).float().cuda()
#         # initial variables
#         self._pos_grad = None
#         self._neg_grad = None
#         self.pos_neg = None

#         def _func(x, gamma, mu):
#             return 1 / (1 + torch.exp(-gamma * (x - mu)))

#         self.map_func = partial(_func, gamma=self.gamma1, mu=self.mu)

#         self.num_classes = num_classes
#         # prior = np.array([0.119217, 0.15927, 0.570566, 0.1045, 0.04089, 0.005522, 0.01])  #jobsscl4
#         # prior = np.array([0.3949, 0.5533, 0.7490, 0.5157, 0.4828, 0.2451, 0.01])    ##jobsscl7
#         # self.prior = torch.tensor(prior).float().cuda()
#         prior = np.array([0.119217, 0.15927, 0.570566, 0.1045, 0.04089, 0.005522, 0.01])
#         # prior = (1. - prior) / prior
#         self.prior = torch.tensor(prior).float().cuda()

#     def get_weight(self, cls_score):
#         # we do not have information about pos grad and neg grad at beginning
#         if self._pos_grad is None:
#             self._pos_grad = cls_score.new_zeros(self.num_classes)
#             self._neg_grad = cls_score.new_zeros(self.num_classes)
#             neg_w = cls_score.new_ones((self.n_i, self.n_c))
#             pos_w = cls_score.new_ones((self.n_i, self.n_c))
#         else:
#             # the negative weight for objectiveness is always 1
#             neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
#             pos_w = 1 + self.alpha * (1 - neg_w)
#             neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
#             pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
#         return pos_w, neg_w


#     def collect_grad(self, cls_score, target, weight):
#         prob = torch.sigmoid(cls_score)
#         grad = target * (prob - 1) + (1 - target) * prob
#         grad = torch.abs(grad)

#         # do not collect grad for objectiveness branch [:-1]
#         pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
#         neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

#         # dist.all_reduce(pos_grad) #longhui
#         # dist.all_reduce(neg_grad)

#         self._pos_grad += pos_grad
#         self._neg_grad += neg_grad
#         self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

#     ################ EQL LOSS V2
#     def forward(self, input, target):
#         # focal loss
#         self.n_i, self.n_c = input.size()
        
#         CE = F.cross_entropy(input, target, reduction="none")
#         def expand_label(pred, gt_classes):
#             target = pred.new_zeros(self.n_i, self.n_c)
#             target[torch.arange(self.n_i), gt_classes] = 1
#             target = target >0
#             return target
#         t_t = expand_label(input, target)
#         probs = F.softmax(input, dim=-1)
#         grad = torch.ones_like(probs) - probs
        
#         for i in range(7):
#             t_i = torch.where(target ==i)
#             self.grad_softmax[i] = self.grad_softmax[i] + torch.sum(grad[t_i])
#             print('sum',i, torch.sum(grad[t_i]))
#             print('softmax',i, self.grad_softmax[i])
        
#         p = torch.exp(-CE)
#         loss = (1 - p) ** self.gamma * CE
#         t_index = torch.where(target !=6)
#         max_score = torch.max(probs, dim =1)[0]
#         loss = max_score * loss
# #         print(loss[t_index])
# #         print('probs', probs.shape, probs)
# #         print('loss',loss.shape)
        
#         return loss.sum()
    
# #     def forward(self, input, target):
# #         # input = self.get_activation(input)
# #         self.n_i, self.n_c = input.size()

# #         def expand_label(pred, gt_classes):
# #             target = pred.new_zeros(self.n_i, self.n_c)
# #             target[torch.arange(self.n_i), gt_classes] = 1
# #             return target

# #         target = expand_label(input, target)

# #         # pos_w, neg_w = self.get_weight(input)

# #         # weight = pos_w * target + neg_w * (1 - target)

# #         cls_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

# #         # cls_loss = torch.sum(cls_loss * weight)
# #         cls_loss = torch.sum(cls_loss)
# #         # cls_loss = torch.sum(cls_loss * focal_weight)
# #         # cls_loss = 0.3 * cls_loss1 + 0.7 * cls_loss2
# #         # cls_loss = torch.sum(cls_loss * weight)
# #         # cls_loss = torch.sum(cls_loss)

# #         # self.collect_grad(input.detach(), target.detach(), weight.detach())

# #         return cls_loss

# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import torch
# from torch import nn
# from torch.nn import functional as F
# from functools import partial
# from detectron2.modeling.roi_heads.fast_rcnn import (
#     FastRCNNOutputLayers,
#     FastRCNNOutputs,
# )

# import torch.distributed as dist
# import numpy as np


# # focal loss
# class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
#     def __init__(self, cfg, input_shape):
#         super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
#         self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
#         self._pos_grad = None
#         self._neg_grad = None
#         self.pos_neg = None


#     def losses(self, predictions, proposals):
#         """
#         Args:
#             predictions: return values of :meth:`forward()`.
#             proposals (list[Instances]): proposals that match the features
#                 that were used to compute predictions.
#         """
#         scores, proposal_deltas = predictions
#         mylosses = FastRCNNFocalLoss(
#             self.box2box_transform,
#             scores,
#             proposal_deltas,
#             proposals,
#             self.smooth_l1_beta,
#             self.box_reg_loss_type,
#             num_classes=self.num_classes,
#         )

#         mylosses.get_v(self._pos_grad, self._neg_grad, self.pos_neg)

#         losses = mylosses.losses()

#         self._pos_grad, self._neg_grad, self.pos_neg = mylosses.return_v()

#         return losses

#     def unsup_losses(self, predictions, proposals):
#         """
#         Args:
#             predictions: return values of :meth:`forward()`.
#             proposals (list[Instances]): proposals that match the features
#                 that were used to compute predictions.
#         """
#         # scores, proposal_deltas = predictions
#         # mylosses = FastRCNNFocalLoss(
#         #     self.box2box_transform,
#         #     scores,
#         #     proposal_deltas,
#         #     proposals,
#         #     self.smooth_l1_beta,
#         #     self.box_reg_loss_type,
#         #     num_classes=self.num_classes,
#         # )

#         # mylosses.get_v(self._pos_grad, self._neg_grad, self.pos_neg)

#         # losses = mylosses.unsup_losses()

#         # self._pos_grad, self._neg_grad, self.pos_neg = mylosses.return_v()

#         # return losses
#         scores, proposal_deltas = predictions
#         losses = FastRCNNFocalLoss(
#             self.box2box_transform,
#             scores,
#             proposal_deltas,
#             proposals,
#             self.smooth_l1_beta,
#             self.box_reg_loss_type,
#             num_classes=self.num_classes,
#         ).unsup_losses()

#         return losses


# class FastRCNNFocalLoss(FastRCNNOutputs):
#     """
#     A class that stores information about outputs of a Fast R-CNN head.
#     It provides methods that are used to decode the outputs of a Fast R-CNN head.
#     """

#     def __init__(
#             self,
#             box2box_transform,
#             pred_class_logits,
#             pred_proposal_deltas,
#             proposals,
#             smooth_l1_beta=0.0,
#             box_reg_loss_type="smooth_l1",
#             num_classes=80,
#     ):
#         super(FastRCNNFocalLoss, self).__init__(
#             box2box_transform,
#             pred_class_logits,
#             pred_proposal_deltas,
#             proposals,
#             smooth_l1_beta,
#             box_reg_loss_type,
#         )
#         self.num_classes = num_classes
#         self._pos_grad = None
#         self._neg_grad = None
#         self.pos_neg = None

#     def losses(self):
#         return {
#             "loss_cls": self.comput_focal_loss(),
#             "loss_box_reg": self.box_reg_loss(),
#         }
#     def unsup_losses(self):
#         return {
#             "loss_cls": self.comput_unsup_loss(),
#             "loss_box_reg": self.box_reg_loss(),
#         }

#     def return_v(self):
#         return self._pos_grad, self._neg_grad, self.pos_neg

#     def get_v(self,a,b,c):
#         self._pos_grad = a
#         self._neg_grad = b
#         self.pos_neg = c

#     def comput_focal_loss(self):
#         if self._no_instances:
#             return 0.0 * self.pred_class_logits.sum()
#         else:
#             FC_loss = FocalLoss(
#                 gamma=1.5,
#                 num_classes=self.num_classes,
#             )
#             FC_loss.get_v(self._pos_grad, self._neg_grad, self.pos_neg)
#             total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
#             total_loss = total_loss / self.gt_classes.shape[0]
#             self._pos_grad, self._neg_grad, self.pos_neg = FC_loss.return_v()

#             return total_loss
    
#     def comput_unsup_loss(self):
#         if self._no_instances:
#             return 0.0 * self.pred_class_logits.sum()
#         else:
#             FC_loss = Ori_FocalLoss(
#                 gamma=1.5,
#                 num_classes=self.num_classes,
#             )
#             total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
#             total_loss = total_loss / self.gt_classes.shape[0]

#             return total_loss

# class Ori_FocalLoss(nn.Module):
#     def __init__(
#         self,
#         weight=None,
#         gamma=1.0,
#         num_classes=80,
#     ):
#         super(Ori_FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.weight = weight

#         self.num_classes = num_classes

#     def forward(self, input, target):
#         # focal loss
#         CE = F.cross_entropy(input, target, reduction="none")
#         p = torch.exp(-CE)
#         loss = (1 - p) ** self.gamma * CE

#         return loss.sum()

# class FocalLoss(nn.Module):
#     def __init__(
#             self,
#             weight=None,
#             gamma=1.0,
#             num_classes=80,
#     ):
#         super(FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.weight = weight

#         # cfg for eqlv2
#         self.vis_grad = False
#         self.gamma1 = 12
#         self.mu = 0.8
#         self.alpha = 8

#         # initial variables
#         self._pos_grad = None
#         self._neg_grad = None
#         self.pos_neg = None

#         def _func(x, gamma, mu):
#             return 1 / (1 + torch.exp(-gamma * (x - mu)))

#         self.map_func = partial(_func, gamma=self.gamma1, mu=self.mu)

#         self.num_classes = num_classes
#         # prior = np.array([0.119217, 0.15927, 0.570566, 0.1045, 0.04089, 0.005522, 0.01])  #jobsscl4
#         # prior = np.array([0.3949, 0.5533, 0.7490, 0.5157, 0.4828, 0.2451, 0.01])    ##jobsscl7
#         # self.prior = torch.tensor(prior).float().cuda()
#         prior = np.array([0.119217, 0.15927, 0.570566, 0.1045, 0.04089, 0.005522, 0.01])
#         self.prior = torch.tensor(prior).float().cuda()
    
#     def get_v(self,a,b,c):
#         self._pos_grad = a
#         self._neg_grad = b
#         self.pos_neg = c

#     def return_v(self):
#         return self._pos_grad, self._neg_grad, self.pos_neg

#     def get_weight(self, cls_score):
#         # we do not have information about pos grad and neg grad at beginning
#         if self._pos_grad is None:
#             self._pos_grad = cls_score.new_zeros(self.num_classes)
#             self._neg_grad = cls_score.new_zeros(self.num_classes)
#             neg_w = cls_score.new_ones((self.n_i, self.n_c))
#             pos_w = cls_score.new_ones((self.n_i, self.n_c))
#         else:
#             # the negative weight for objectiveness is always 1
#             neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
#             pos_w = 1 + self.alpha * (1 - neg_w)
#             neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
#             pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
#         return pos_w, neg_w


#     def collect_grad(self, cls_score, target, weight):
#         prob = torch.sigmoid(cls_score)
#         grad = target * (prob - 1) + (1 - target) * prob
#         grad = torch.abs(grad)

#         # do not collect grad for objectiveness branch [:-1]
#         pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
#         neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

#         # dist.all_reduce(pos_grad) #longhui
#         # dist.all_reduce(neg_grad)
#         # print(self._pos_grad)
#         self._pos_grad += pos_grad
#         self._neg_grad += neg_grad
#         self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)


#     def forward(self, input, target):
#         # input = self.get_activation(input)
#         self.n_i, self.n_c = input.size()

#         def expand_label(pred, gt_classes):
#             target = pred.new_zeros(self.n_i, self.n_c)
#             target[torch.arange(self.n_i), gt_classes] = 1
#             return target

#         target = expand_label(input, target)

#         pos_w, neg_w = self.get_weight(input)

#         weight = pos_w * target + neg_w * (1 - target)

#         cls_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

#         cls_loss = torch.sum(cls_loss * weight)

#         self.collect_grad(input.detach(), target.detach(), weight.detach())

#         return cls_loss

