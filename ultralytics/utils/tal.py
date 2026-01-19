# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywh2xyxy, xywhr2xyxyxyxy, xyxy2xywh
from .torch_utils import TORCH_1_11


class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        topk2 (int): Secondary topk value for additional filtering.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        stride (list): List of stride values for different feature levels.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list = [8, 16, 32],
        eps: float = 1e-9,
        topk2=None,
        use_scores : bool = False, 
        use_km_scores : bool = False, 
        use_variant_selection : bool = False, 
    ):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            stride (list, optional): List of stride values for different feature levels.
            eps (float, optional): A small value to prevent division by zero.
            topk2 (int, optional): Secondary topk value for additional filtering.
            use_scores (bool, optional): Use class scores instead of class labels.
            use_km_scores (bool, optional): Rely on the provided knowledge model.
            use_variant_selection (bool, optional): Select object nature based on variant instead of class.
        """
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.use_scores = use_scores
        self.use_km_scores = use_km_scores
        self.use_variant_selection = use_variant_selection
        self.alpha = alpha
        self.beta = beta
        self.stride = stride
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_scores, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            # TODO (CP/IRIT): Are the ground truth labels used ?
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            # Adding the ground truth label scores to enable multi label prediction.
            gt_scores (torch.Tensor): Ground truth class scores with shape (bs, n_max_boxes, num_classes).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            # TODO (CP/IRIT): Are the ground truth labels used ?
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_scores, gt_bboxes, mask_gt)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Move tensors to CPU, compute, then move back to original device
                LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
                cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
                result = self._forward(*cpu_tensors)
                return tuple(t.to(device) for t in result)
            raise

    # DONE (CP/IRIT): Add class scores ground truth (variant scores)
    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_scores, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            # TODO (CP/IRIT): Are the ground truth labels used ?
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            # Adding the ground truth label scores to enable multi label prediction.
            gt_scores (torch.Tensor): Ground truth class scores with shape (bs, n_max_boxes, num_classes).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        # TODO (CP/IRIT): Are the ground truth labels used ?
        pos_mask, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, single_pos_mask = self.select_highest_overlaps(
            pos_mask, overlaps, self.n_max_boxes, align_metric
        )

        # Assigned target
        # TODO (CP/IRIT): Are the ground truth labels used ?
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_scores, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize : TODO (CP/IRIT): There is probably an issue there...
        align_metric *= single_pos_mask
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * single_pos_mask).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    # TODO (CP/IRIT): Is it meaningful to rely on predicted bounding boxes to select the positive mask for ground truth data ?
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        # Positive anchor points (included in bounding boxes) for all strides  
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        # sz_mask_in_gts = torch.numel(mask_in_gts)
        # nz_mask_in_gts = torch.count_nonzero(mask_in_gts)
        # save2debug( 'mask_in_gts.txt', mask_in_gts, True)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        # TODO (CP/IRIT): Is it meaningful to rely on predicted bounding boxes to select the positive mask for ground truth data ?
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # save2debug( 'align_metric.txt', align_metric, True)
        # save2debug( 'overlaps.txt', overlaps, True)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # sz_mask_topk = torch.numel(mask_topk)
        # nz_mask_topk = torch.count_nonzero(mask_topk)
        # save2debug( 'mask_topk.txt', mask_topk, True)
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt
        # sz_mask_pos = torch.numel(mask_pos)
        # nz_mask_pos = torch.count_nonzero(mask_pos)

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            # Why are the labels used if it relates to the boxes ?
            # TODO (CP/IRIT): Use the gt_scores instead of the gt_labels
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        anchor_point_number = pd_bboxes.shape[-2] # number of anchor points h*w
        # TODO (CP/IRIT): Why not convert it earlier ?
        # Indicates if an anchor point is in a given ground truth object from a given image
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        sz_mask_gt = torch.numel(mask_gt)
        # nz_mask_gt = torch.count_nonzero(mask_gt)
        overlaps = torch.zeros([self.bs, self.n_max_boxes, anchor_point_number], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        sz_overlaps = torch.numel(overlaps)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, anchor_point_number], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, bs, max_num_obj
        # for each batch, vector of max_num_obj value of batch indexes
        # image index for each gt bounding box in the image
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # bs, max_num_obj
        # TODO (CP/IRIT): Purpose of ind[1]
        # for each batch, vector of max_num_obj class indexes
        # suppress last unused dimension
        # class index for each gt bounding box in the image
        ind[1] = gt_labels.squeeze(-1) # bs, max_num_obj
        # Get the scores of each grid for each gt cls
        # Class score for each anchor point for each ground truth object in each image 
        selected_pd_scores = pd_scores[ind[0], :, ind[1]] # bs, max_num_obj, h*w
        sz_bbox_scores = torch.numel(bbox_scores)
        sz_pd_scores_ind = torch.numel(selected_pd_scores)
        nz_pd_scores_ind = torch.count_nonzero(selected_pd_scores)
        assert (sz_pd_scores_ind == sz_mask_gt), (f"Predicted scores ({sz_pd_scores_ind}) and mask_gt ({sz_mask_gt}) tensors must have compatible size")
        # Select the scores for the anchor points in a given ground truth object from a given image
        pd_scores_masked = selected_pd_scores[mask_gt]
        sz_pd_scores_masked = torch.numel(pd_scores_masked)
        nz_pd_scores_masked = torch.count_nonzero(pd_scores_masked)
        assert ((sz_bbox_scores >= sz_pd_scores_masked) and (sz_bbox_scores == sz_mask_gt)), (f"Bbox scores ({sz_bbox_scores}), Predicted scores ({sz_pdscores_masked}) and mask_gt ({sz_mask_gt}) tensors must have compatible size")
        # The score of the bounding box is the score of the class associated to the bounding box
        # TODO (CP/IRIT): Compute a score based on the vector of class scores (derived from BCE)
        # scores for the anchor points in a given ground truth object from a given image, others are 0
        bbox_scores[mask_gt] = pd_scores_masked # b, max_num_obj, h*w
        # nz_bbox_scores = torch.count_nonzero(bbox_scores)
        
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, anchor_point_number, -1)[mask_gt]
        
        # Compare the bounding box
        iou_results = self.iou_calculation(gt_boxes, pd_boxes)
        sz_iou_results = torch.numel(iou_results)
        nz_iou_results = torch.count_nonzero(iou_results)
        assert ((sz_overlaps >= sz_iou_results) and (sz_overlaps == sz_mask_gt)), (f"Overlaps ({sz_overlaps}), IOU ({sz_iou_results}) and mask_gt ({sz_mask_gt}) tensors must have compatible size")
        overlaps[mask_gt] = iou_results
        # nz_overlaps = torch.count_nonzero(overlaps)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where topk
                is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    # DONE (CP/IRIT): add class prediction scores for multi label prediction (variant prediction).
    def get_targets(self, gt_labels, gt_scores, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        TODO (CP/IRIT): h * w is not the size of dimension 2 of the tensors. The value is extracted from the gt parameters.  

        Args:
            # TODO (CP/IRIT): Are the ground truth labels used ?
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            TODO (CP/IRIT): Make it optional
            gt_scores (torch.Tensor): Ground truth probability of being in classes of shape (b, max_num_obj, classes_number).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points (0 <= value < max_num_obj, meaningful if mask is true).
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            TODO (CP/IRIT): What's the use of target_labels as target_scores contains the information ?
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # save2debug( 'gt_labels.txt', gt_labels, True)
        # save2debug( 'gt_scores.txt', gt_scores, True)
        # save2debug( 'gt_bboxes.txt', gt_bboxes, True)
        # save2debug( 'target_gt_idx_init.txt', target_gt_idx, True)
        # save2debug( 'fg_mask.txt', fg_mask, True)
        # Assigned target labels, (b, 1)
        # Image index for each image
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx_offset = batch_ind * self.n_max_boxes # offset for ground truth object if batches are flattened
        target_gt_idx_translated = target_gt_idx + target_gt_idx_offset # (b, h*w) # offset added to each ground truth object to compute object index in flattened batches
        # Convert to integers (TODO (CP/IRIT): Should it be done much earlier ?)
        # Select the labels associated to the indices
        # TODO (CP/IRIT): Are the ground truth labels used ?
        gt_labels_flattened = gt_labels.long().flatten()
        target_labels = gt_labels_flattened[target_gt_idx_translated]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        # Select the boxes associated to the indices
        gt_bbox_number = gt_bboxes.shape[-1]
        gt_bboxes_view = gt_bboxes.view(-1, gt_bbox_number) # flatten the tensor to 2 dimension, first one is flattened image / ground truth object, last one is bounding box component index
        target_bboxes = gt_bboxes_view[target_gt_idx_translated] # TODO (CP/IRIT): May contain bad data from object 0 when no object are present...

        # Assigned target scores, minimum is 0, maximum is positive
        # TODO (CP/IRIT): is the clamp_ needed ?
        target_labels.clamp_(0) # Set the minimum to 0
        # save2debug( 'target_gt_idx.txt', target_gt_idx, True)

        if self.use_km_scores:
            # TODO (CP/IRIT): Do the same for scores (identical to target_scores_base)
            gt_score_size = gt_scores.shape[-1]
            gt_scores_view = gt_scores.view(-1, gt_score_size)
            target_scores_base  = gt_scores_view[target_gt_idx_translated] # TODO (CP/IRIT): May contain bad data from object 0 when no object are present...
        else:
            # TODO (CP/IRIT): Initialize scores from ground truth data instead of one_hot for the single class.
            # 10x faster than F.one_hot()
            # Create an int64 zero tensor of dimension batch size * anchor point number * class number
            # TODO (CP/IRIT): Are the ground truth labels used ?
            # Required to provide the tensor dimensions: batch size, inferred data (grids of predictions for each anchor points)
            batch_number = target_labels.shape[0]
            anchor_point_number = target_labels.shape[1] # anchor points number
            target_scores_base = torch.zeros(
                (batch_number, anchor_point_number, self.num_classes),
                dtype=torch.int64,
                device=target_labels.device,
                )  # (b, h*w, classes_number ) Class score prediction for each class, for each image, for each anchor point (between 0 and 1)
            # Adds a dimension at the end of target labels
            target_labels_unsqueezed = target_labels.unsqueeze(-1) # (b, h*w, 1)
            # Set the value of the tensor to 1 for indexes from target_labels_unsqueezed in the last dimension 
            target_scores_base.scatter_(2, target_labels_unsqueezed, 1)
            # neq_target_scores_base = torch.where((target_scores_base_km != target_scores_base))
            # neq_image_indexes_base, neq_anchor_point_indexes_base, neq_class_indexes_base = neq_target_scores_base
            # eq_target_scores_base = torch.where((target_scores_base_km == target_scores_base))
            # eq_image_indexes_base, eq_anchor_point_indexes_base, eq_class_indexes_base = eq_target_scores_base
            # neq_target_scores = torch.where((target_scores_km != target_scores))
            # neq_image_indexes, neq_anchor_point_indexes, neq_class_indexes = neq_target_scores
            # eq_target_scores = torch.where((target_scores_km == target_scores))
            # eq_image_indexes, eq_anchor_point_indexes, eq_class_indexes = eq_target_scores
            # torch.save(target_labels,"target_labels.save",_use_new_zipfile_serialization=False)
            # save2debug( 'fg_scores_mask.txt', fg_scores_mask, True)
            # save2debug( 'target_labels.txt', target_labels)
            # save2debug( 'target_bboxes.txt', target_bboxes)
            # save2debug( 'target_scores_base.txt', target_scores_base, True)
            # save2debug( 'target_scores.txt', target_scores, True)
            # save2debug('target_scores_base.txt', target_scores_base, True)
            # save2debug( 'target_scores_km.txt', target_scores_km, True)

        # Duplicate fg_mask class number times along the 3rd dimension
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)

        # Set to zero when fg_scores_mask is zero
        target_scores = torch.where(fg_scores_mask > 0, target_scores_base, 0)

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes, shape (b, n_boxes, 1).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]  # the smallest stride
        stride_val = torch.tensor(self.stride[1], dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device)
        gt_bboxes_xywh[..., 2:] = torch.where((wh_mask * mask_gt).bool(), stride_val, gt_bboxes_xywh[..., 2:])
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        anchors_lt = xy_centers[None] - lt # positive when center over left top 
        rb_anchors = rb - xy_centers[None] # positive when center under right bottom
        anchors_bbox_deltas_base = torch.cat((anchors_lt, rb_anchors), dim=2).view(bs, n_boxes, n_anchors, -1)
        anchors_bbox_deltas_min = anchors_bbox_deltas_base.amin(3) 
        positive_anchor_points = anchors_bbox_deltas_min.gt_(eps) # both are be positive iff the center is in the box

        # positive_anchor_points_count = positive_anchor_points.sum(1)
        # image_indexes, anchor_point_indexes = torch.where(positive_anchor_points_count > 1)
        # overlap_anchor_points = positive_anchor_points[image_indexes,:,anchor_point_indexes]
        # overlap_anchor_point_indexes, overlap_box_indexes = torch.where(overlap_anchor_points == 1) 
        
        # TODO (CP/IRIT): Eliminate overlapping bounding boxes
        # Transform x_min y_min x_max y_max to x_c y_c w h to ensure that the center of the internal box is in the external box, and to introduce an error margin 
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        gt_bboxes_unsqueeze = gt_bboxes.unsqueeze(2)
        ult, urb = gt_bboxes_unsqueeze.chunk(2, 3)
        utl = ult.transpose(1,2)
        ubr = urb.transpose(1,2)
        delta_lt = utl - ult
        delta_rb = urb - ubr
        delta_bbox =  torch.cat((delta_lt,delta_rb), dim=3)
        delta_bbox_min = delta_bbox.amin(3)
        positive_delta_bbox = delta_bbox_min.gt_(eps)
        
        return positive_anchor_points

    # TODO (CP/IRIT): Is it the best criteria when handling composition ?
    # Should select the smallest box in the composition hierarchy ?
    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.
            align_metric (torch.Tensor): Alignment metric for selecting best matches.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        # Sum along all boxes for a given anchor point in a given image, produce the number of boxes for a given point
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes (sum along the boxes > 1)
            # add a dimension between images and anchor points
            # expand this dimension to size n_max_boxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)

            # select the bounding box that maximize the overlap with ??? (according to IoU)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            # creates a zero tensor with shape (b, n_max_boxes, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            # set to 1 the index of the box that maximizes overlap
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            # replace with the selected bounding box
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            # sz_mask_pos = torch.numel(mask_pos)
            # nz_mask_pos = torch.count_nonzero(mask_pos)
            
            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos  # update overlaps
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices  # (b, n_max_boxes)
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # update mask_pos
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        # Select the bounding box with 1 (there should be a single one - all others should be 0)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt):
        """Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (b, n_boxes, 1).
            stride (list[int]): List of stride values for each feature map level.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors with shape (b, n_boxes, h*w).
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor points and stride tensors using the size feature tensors."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(relative_points, anchor_points, xywh=True, dim=-1):
    """Transform coordinates relative to anchor_points distance(ltrb) to absolute for box (xywh or xyxy)."""
    # left-top and right-bottom
    lt, rb = relative_points.chunk(2, dim)
    # the bounding box is relative to the anchor points
    # x1y1 and x2y2 are the absolute position
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    """Transform the absolute coordinates of the bbox(xyxy) to relative ones (ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle with shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        dim (int, optional): Dimension along which to split.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def rbox2dist(
    target_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_angle: torch.Tensor,
    dim: int = -1,
    reg_max: int | None = None,
):
    """Decode rotated bounding box (xywh) to distance(ltrb). This is the inverse of dist2rbox.

    Args:
        target_bboxes (torch.Tensor): Target rotated bounding boxes with shape (bs, h*w, 4), format [x, y, w, h].
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        target_angle (torch.Tensor): Target angle with shape (bs, h*w, 1).
        dim (int, optional): Dimension along which to split.
        reg_max (int, optional): Maximum regression value for clamping.

    Returns:
        (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4), format [l, t, r, b].
    """
    xy, wh = target_bboxes.split(2, dim=dim)
    offset = xy - anchor_points  # (bs, h*w, 2)
    offset_x, offset_y = offset.split(1, dim=dim)
    cos, sin = torch.cos(target_angle), torch.sin(target_angle)
    xf = offset_x * cos + offset_y * sin
    yf = -offset_x * sin + offset_y * cos

    w, h = wh.split(1, dim=dim)
    target_l = w / 2 - xf
    target_t = h / 2 - yf
    target_r = w / 2 + xf
    target_b = h / 2 + yf

    dist = torch.cat([target_l, target_t, target_r, target_b], dim=dim)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)

    return dist

def save2debug(filename,tensor,nonzero=False):
    def aux2(file,tensor,index,nonzero):
        if tensor.numel() == 1:
            if (tensor.item() != 0):
                print( index + "] = " + str(tensor.item()), file=f)
        else:
            for position in range(tensor.size(dim=0)):
                aux2(file, tensor[position], index + ", " + str(position), nonzero)
                
    def aux1(file, tensor, nonzero):
        if tensor.numel() == 1:
            if (tensor.item() != 0):
                print( str(tensor.item()), file=f)
        else:
            for position in range(tensor.size(dim=0)):
                aux2(file, tensor[position], '[ ' + str(position), nonzero)
            
    with open( filename, 'w') as f:
        if nonzero:
            aux1( f, tensor, False)
            # nonzeros = tensor.nonzero().tolist()
            # print(len(nonzeros), file=f)
            # print(nonzeros, file=f)
        # print(tensor.shape, file=f)
        # print(tensor.tolist(), file=f)

