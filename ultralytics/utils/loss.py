# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from ultralytics.utils import km

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

import math

class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

# TODO (CP/IRIT): LogicSeg derived loss to enforce consistency between the knowledge model and the prediction.

class KnowledgeBasedLoss(nn.Module):
    """Criterion class for computing losses based on relations between classes in a knowledge model."""
    
    # TODO (CP/IRIT): The model should carry all required data (inheritance and composition relations).
    def __init__(self, model, power = 3.0, 
                 decomposition_weight = 1.0, decomposition_exclusion_weight = 1.0, 
                 composition_weight = 1.0, composition_exclusion_weight = 1.0,
                 generalization_weight = 1.0, 
                 specialization_weight = 1.0, specialization_exclusion_weight = 1.0,
            ):
        """Initialize the KnowledgeModelLoss class."""
        super().__init__()
        self.model = model
        self.power = power
        self.use_scores = getattr( model.args, 'use_scores', False)
        self.use_km = self.use_scores and getattr( model.args, 'use_km', False)
        self.use_km_scores = self.use_km and getattr( model.args, 'use_km_scores', False)
        self.use_variant_selection = self.use_km_scores and getattr(model.args, 'use_variant_selection', False)
        self.use_km_losses = self.use_km and getattr( model.args, 'use_km_losses', False)
        self.use_refinement = self.use_km_losses and getattr(model.args, 'use_refinement', False)
        self.use_composition = self.use_km_losses and getattr(model.args, 'use_composition', False)
        self.decomposition_weight = decomposition_weight
        self.decomposition_exclusion_weight = decomposition_exclusion_weight
        self.composition_weight = composition_weight
        self.composition_exclusion_weight = composition_exclusion_weight
        self.generalization_weight = generalization_weight
        self.specialization_weight = specialization_weight
        self.specialization_exclusion_weight = specialization_exclusion_weight
        self.refinement = model.refinement
        self.composition = model.composition
        self.classes = set(range(model.nc))
        self.refinement_forward = self.refinement # TODO (CP/IRIT)
        self.refinement_backward = km.invert_relation( self.refinement ) # TODO (CP/IRIT)
        self.composition_forward = self.composition # TODO (CP/IRIT)
        self.composition_backward = km.invert_relation( self.composition) # TODO (CP/IRIT)
    
    # One of the targets for a given valid source must be valid : forall s in S, forall o in O, p_s(o) -> \/_{t in T(s)} p_t(o)
    # origin_indexes : class indexes for the domain
    # targets_from_source : class indexes for the co-domain
    def disjunction_loss(self, pred_scores, targets_from_source, power=3.0):
        batch_size, anchor_point_size, class_number = pred_scores.shape
        sources = list(targets_from_source )
        source_number = len(sources)
        predicate_s = torch.zeros(source_number)
        for idx_s, s in enumerate(sources):
            targets_from_s = list(targets_from_source[s])
            # p_c(o)
            source_scores = pred_scores[:, :, s:s+1]
            # p_d(o)
            indexes = torch.tensor(targets_from_s,device=pred_scores.device)
            target_scores = pred_scores.index_select( 2, indexes)
            # max(p_d(o))
            target_scores_max = target_scores.max()
            # p_s(o) - p_s(o) * max(p_t(o))
            predicate_s_o = source_scores - source_scores * target_scores_max
            # p-mean on O
            predicate_s[idx_s] = torch.pow(torch.pow(predicate_s_o,power).mean(dim=(0,1)),1/power)
        # mean for s in sources
        return torch.mean(predicate_s)
    
    # All of the targets for a given valid source must be valid : forall s in S, forall o in O, p_s(o) -> /\_{t in T(s)} p_t(o)
    # sources : indexes for the sources
    # targets_from_source : list of targets for a given source
    def conjunction_loss(self, pred_scores, targets_from_source, power=3.0):
        batch_size, anchor_point_size, class_number = pred_scores.shape
        sources = list(targets_from_source )
        source_number = len(sources)
        predicate_s = torch.zeros(source_number)
        for idx_s, s in enumerate(sources):
            # p_s(o)
            source_scores = pred_scores[:,:,s:s+1]
            targets_from_s = list(targets_from_source[s])
            indexes = torch.tensor(targets_from_s,device=pred_scores.device)
            s_targets_number = len(targets_from_s)
            # p_t(o)
            target_scores_t = pred_scores.index_select( 2, indexes)
            # p_s(o) - p_s(o) * p_t(o)
            predicate_s_t_o = source_scores - source_scores * target_scores_t
            # moyenne sur O
            predicate_s_t = torch.pow(torch.pow(predicate_s_t_o,power).mean(dim=(0,1)),1/power)
            # moyenne sur A
            predicate_s[idx_s] = torch.mean(predicate_s_t)
        # moyenne sur C \ R
        return torch.mean(predicate_s)    
    
    # Only one of the targets for a given valid source must be valid : forall s in S, forall t in T(s), forall e in T(s) \ { t }, forall o in O, p_t(o) -> /\_{e in T(s) \ { t }} ~ p_e(o)
    # sources : class indexes for the domain
    # targets_from_source : class indexes for the co-domain
    def exclusion_loss(self, pred_scores, targets_from_source, power=3.0):
        batch_size, anchor_point_size, class_number = pred_scores.shape
        sources = list(targets_from_source )
        source_number = len(sources)
        predicate_s = torch.zeros(source_number)
        for idx_s, s in enumerate(sources):
            # p_s(o)
            source_scores = pred_scores[:,:,s:s+1]
            targets_from_s = list(targets_from_source[s])
            s_targets_number = len(targets_from_s)
            predicate_s_t = torch.zeros(s_targets_number)
            if s_targets_number > 1:
                for idx_t, t in enumerate(targets_from_s):
                    targets_except_t = targets_from_s.copy()
                    targets_except_t.remove(t)
                    # p_t(o)
                    target_scores_t = pred_scores[:,:, t:t+1]
                    # p_e(o)
                    indexes = torch.tensor(targets_except_t,device=pred_scores.device)
                    target_scores_e = pred_scores.index_select( 2, indexes)
                    # p_t(o) * p_e(o)
                    predicate_s_t_e_o = target_scores_t * target_scores_e
                    # moyenne sur O
                    predicate_s_t_e = torch.pow(torch.pow(predicate_s_t_e_o,power).mean(dim=(0,1)),1/power)
                    # moyenne sur e in T(s) \ { t }
                    predicate_s_t[idx_t] = torch.mean(predicate_s_t_e)
            # moyenne sur t in T(s)
            predicate_s[idx_s] = torch.mean(predicate_s_t)
        # moyenne sur s in S
        return torch.mean(predicate_s)

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """Compute knowledge based loss for class predication score."""
        
        norm_pred_scores = pred_scores.sigmoid()
        if self.use_refinement:
            if len(self.refinement_forward) == 0:
                S_loss = 0.0
                SE_loss = 0.0
            else:
                # Call disjunction_loss for refinement and composition
                S_loss = self.disjunction_loss(norm_pred_scores, self.refinement_forward)
                # Call exclusion_loss for refinement and composition
                SE_loss = self.exclusion_loss(norm_pred_scores, self.refinement_forward)
            if len(self.refinement_backward) == 0:
                 G_loss = 0.0
            else:
                # Call conjunction_loss for refinement
                G_loss = self.conjunction_loss(norm_pred_scores, self.refinement_backward)
        else:
            S_loss = 0.0
            SE_loss = 0.0
            G_loss = 0.0
        if self.use_composition:
            if len(self.composition_forward) == 0:
                C_loss = 0.0
                CE_loss = 0.0
            else:
                # Call disjunction_loss for refinement and composition
                C_loss = self.disjunction_loss(norm_pred_scores, self.composition_forward)
                # Call exclusion_loss for refinement and composition
                CE_loss = self.exclusion_loss(norm_pred_scores, self.composition_forward)
            if len(self.composition_backward) == 0:
                D_loss = 0.0
                DE_loss = 0.0
            else:
                # Call conjunction_loss for refinement
                D_loss = self.disjunction_loss(norm_pred_scores, self.composition_backward)
                DE_loss = self.exclusion_loss(norm_pred_scores, self.composition_backward)
        else:
            C_loss = 0.0
            CE_loss = 0.0
            D_loss = 0.0
            DE_loss = 0.0
        return self.specialization_weight * S_loss + self.composition_weight * C_loss + self.specialization_exclusion_weight * SE_loss + self.composition_exclusion_weight * CE_loss + self.generalization_weight * G_loss + self.decomposition_weight * D_loss

class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h # hyperparameters
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4 # number of predicted features
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.use_scores = getattr( model.args, 'use_scores', False)
        self.use_km = self.use_scores and getattr( model.args, 'use_km', False)
        self.use_km_scores = self.use_km and getattr( model.args, 'use_km_scores', False)
        self.use_variant_selection = self.use_km_scores and getattr(model.args, 'use_variant_selection', False)
        self.use_km_losses = self.use_km and getattr( model.args, 'use_km_losses', False)
        self.use_refinement = self.use_km_losses and getattr(model.args, 'use_refinement', False)
        self.use_composition = self.use_km_losses and getattr(model.args, 'use_composition', False)
        if self.use_km_losses:
            self.km_loss = KnowledgeBasedLoss(model)
        else:
            km_loss = None

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, 
            num_classes=self.nc, 
            use_scores = self.use_scores,
            use_km_scores = self.use_km_scores, 
            use_variant_selection = self.use_variant_selection,
            alpha=0.5, 
            beta=6.0,
            )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        # TODO(CP/IRIT): Add knowledge model control
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., -4:] = xywh2xyxy(out[..., -4:].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.
        
        Args:
            anchor_points (Tensor[]):
            pred_dist (Tensor[]):
        
        Returns:
        """
        if self.use_dfl:
            # Distributed Focal Loss
            b, a, c = pred_dist.shape  # batch, anchors, channels
            assert c % 4 == 0 # must be a multiple of 4 for the following split
            # Split the channels between two dimensions, compute a softmax on the 4 size 3rd dimension, 
            pred_dist = pred_dist.view(b, a, 4, c // 4)
            pred_dist = pred_dist.softmax(3)
            pred_dist = pred_dist.matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        TODO (CP/IRIT): Should all tensors be on the same device there ?
        Args:
            preds (Union[ Tuple( Tensor, ...), Tensor]) : prediction computed by the trainee network to compute the loss with respect to the ground truth
            batch (Dict[str, torch.Tensor]): ground truth data
        
        Returns:
            scaled loss items (torch.Tensor[Float]):
            loss items (torch.Tensor[Float]):
        """
        loss_number = 4 if self.use_km_losses else 3
        box_index = 0
        cls_index = 1
        if self.use_km_losses:
            km_index = 2
        dfl_index = loss_number - 1
        
        # TODO (CP/IRIT): Adding knowledge model loss
        loss = torch.zeros(loss_number, device=self.device)  # 3 or 4 loss items: box, cls, km (if knowledge model in use), dfl
        # TODO (CP/IRIT): Why use preds[1] instead of preds[0] ?
        feats = preds[1] if isinstance(preds, tuple) else preds
        # merge all the prediction levels along dimension 2
        pred_merged = torch.cat( [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        # split between bounding box and class score features
        pred_for_bboxes, pred_scores = pred_merged.split((self.reg_max * 4, self.nc), 1)

        # reorganize dimensions for future operations
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        # TODO (CP/IRIT): Check that the class ground truth is indeed not used, and that classes are not directly predicted...
        pred_for_bboxes = pred_for_bboxes.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        # anchor points from the first stride, then the second, etc
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        
        # Merge ground truth tensors (indexes, classes, scores, bounding boxes) in a single tensor
        batch_merge = (batch["batch_idx"].view(-1, 1), batch["cls"], batch["scores"], batch["bboxes"])
        targets = torch.cat(batch_merge, 1)
        # Generate the ground truth data in a single tensor
        # TODO (CP/IRIT): Why is the preprocess done on a single tensor as it is limited to bounding boxes ?
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # Split between label and bounding box ground truth data
        # TODO (CP/IRIT) : are the ground truth labels used ?
        gt_labels, gt_scores, gt_bboxes = targets.split((1, self.nc, 4), 2)  # cls, scores, xyxy
        # Identify future positive anchor points
        # Sum the components of each bounding boxes
        gt_bboxes_sum = gt_bboxes.sum(2, keepdim=True)
        # Indicates which image in a batch contains bounding boxes
        # TODO: make it boolean
        mask_gt = gt_bboxes_sum.gt_(0.0)

        # Bounding boxes
        # Compute predicted bounding boxes according to anchor points
        pred_bboxes = self.bbox_decode(anchor_points, pred_for_bboxes)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_for_bboxes.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        smoothed_pred_scores = pred_scores.detach().sigmoid()
        # Scale predicted bounding boxes along the pyramid (stride values)
        scaled_pred_boxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        scaled_anchor_points = anchor_points * stride_tensor

        # Computer the ground truth for the bounding boxes and class scores
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            smoothed_pred_scores,
            scaled_pred_boxes,
            scaled_anchor_points,
            gt_labels,
            gt_scores,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        bce_values = self.bce(pred_scores, target_scores.to(dtype))
        loss[cls_index] = bce_values.sum() / target_scores_sum  # BCE
        
        # TODO (CP/IRIT): Adding knowledge model loss to the usual class loss
        if self.use_km_losses:
            km_loss = self.km_loss(pred_scores,target_scores)
            # print('Knowledge Model Loss = ',km_loss)
            loss[km_index] = km_loss

        # Bbox loss
        # TODO (CP/IRIT): Is the loss computed for gt_labels ?
        if fg_mask.sum():
            loss[box_index], loss[dfl_index] = self.bbox_loss(
                pred_for_bboxes,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[box_index] *= self.hyp.box  # box gain
        loss[cls_index] *= self.hyp.cls  # cls gain
        if self.use_km_losses:
            loss[km_index] *= self.hyp.km  # km gain
        loss[dfl_index] *= self.hyp.dfl  # dfl gain
        
        has_nan = math.isnan(loss[box_index]) or math.isnan(loss[cls_index]) or math.isnan(loss[dfl_index])
        has_nan = has_nan or math.isnan(loss[km_index]) if self.use_km_losses else has_nan
        
        if has_nan:
            print("NaN occured in loss computation.")

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            # TODO (CP/IRIT): should "scores" be managed in the same way ?
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, pose, kobj, cls, dfl
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        # TODO (CP/IRIT): should "scores" be managed in the same way ?
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        # TODO (CP/IRIT): should "scores" be managed in the same way ?
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            # TODO (CP/IRIT): should "scores" be managed in the same way ?
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        cls_loss = vp_loss[0][1]
        return cls_loss, vp_loss[1]

    def _get_vp_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
