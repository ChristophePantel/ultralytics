# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy
from .torch_utils import TORCH_1_11


class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
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
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            # TODO (CP/IRIT): Are the ground truth labels used ?
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_scores, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_scores, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            # TODO (CP/IRIT): Are the ground truth labels used ?
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
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
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        # TODO (CP/IRIT): Are the ground truth labels used ?
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_scores, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    # TODO (CP/IRIT): Is it meaningful to rely on predicted bounding boxes to select the positive mask for ground truth data ?
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt):
        """Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anchor_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        # Positive anchor points (included in bounding boxes) for all strides  
        mask_in_gts = self.select_candidates_in_gts(anchor_points, gt_bboxes)
        sz_mask_in_gts = torch.numel(mask_in_gts)
        nz_mask_in_gts = torch.count_nonzero(mask_in_gts)
        # save2debug( 'mask_in_gts.txt', mask_in_gts, True)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        # TODO (CP/IRIT): Is it meaningful to rely on predicted bounding boxes to select the positive mask for ground truth data ?
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # save2debug( 'align_metric.txt', align_metric, True)
        # save2debug( 'overlaps.txt', overlaps, True)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        sz_mask_topk = torch.numel(mask_topk)
        nz_mask_topk = torch.count_nonzero(mask_topk)
        # save2debug( 'mask_topk.txt', mask_topk, True)
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt
        sz_mask_pos = torch.numel(mask_pos)
        nz_mask_pos = torch.count_nonzero(mask_pos)

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
        try:
            sz_anchor_points = pd_bboxes.shape[-2] # number of anchor points h*w
            # TODO (CP/IRIT): Why not convert it earlier ?
            mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
            sz_mask_gt = torch.numel(mask_gt)
            nz_mask_gt = torch.count_nonzero(mask_gt)
            overlaps = torch.zeros([self.bs, self.n_max_boxes, sz_anchor_points], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
            sz_overlaps = torch.numel(overlaps)
            bbox_scores = torch.zeros([self.bs, self.n_max_boxes, sz_anchor_points], dtype=pd_scores.dtype, device=pd_scores.device)
    
            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, bs, max_num_obj
            # for each batch, vector of max_num_obj value of batch index
            ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # bs, max_num_obj
            # TODO (CP/IRIT): Purpose of ind[1]
            # for each batch, vector of max_num_obj class index
            gt_labels_squeeze = gt_labels.squeeze(-1)
            ind[1] =  gt_labels_squeeze # b, max_num_obj
            # Get the scores of each grid for each gt cls
            ind_0 = ind[0]
            ind_1 = ind[1]
            pd_scores_ind = pd_scores[ind_0, :, ind_1]
            sz_bbox_scores = torch.numel(bbox_scores)
            sz_pd_scores_ind = torch.numel(pd_scores_ind)
            nz_pd_scores_ind = torch.count_nonzero(pd_scores_ind)
            assert (sz_pd_scores_ind == sz_mask_gt), (f"Predicted scores ({sz_pd_scores_ind}) and mask_gt ({sz_mask_gt}) tensors must have compatible size")
            pd_scores_masked = pd_scores_ind[mask_gt]
            sz_pd_scores_masked = torch.numel(pd_scores_masked)
            nz_pd_scores_masked = torch.count_nonzero(pd_scores_masked)
            assert ((sz_bbox_scores >= sz_pd_scores_masked) and (sz_bbox_scores == sz_mask_gt)), (f"Bbox scores ({sz_bbox_scores}), Predicted scores ({sz_pdscores_masked}) and mask_gt ({sz_mask_gt}) tensors must have compatible size")
            bbox_scores[mask_gt] =  pd_scores_masked # b, max_num_obj, h*w
            nz_bbox_scores = torch.count_nonzero(bbox_scores)
            # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
            pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
            gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, sz_anchor_points, -1)[mask_gt]
            
            iou_results = self.iou_calculation(gt_boxes, pd_boxes)
            sz_iou_results = torch.numel(iou_results)
            nz_iou_results = torch.count_nonzero(iou_results)
            
            assert ((sz_overlaps >= sz_iou_results) and (sz_overlaps == sz_mask_gt)), (f"Overlaps ({sz_overlaps}), IOU ({sz_iou_results}) and mask_gt ({sz_mask_gt}) tensors must have compatible size")
            overlaps[mask_gt] = iou_results
            nz_overlaps = torch.count_nonzero(overlaps)
            align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
            sz_align_metrics = torch.numel(align_metric)
            nz_align_metrics = torch.count_nonzero(align_metric)
            
            return align_metric, overlaps
        except Exception as e:
            return None, None

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

    def get_targets(self, gt_labels, gt_scores, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        TODO (CP/IRIT): h * w is not the size of dimension 2 of the tensors. The value is extracted from the gt parameters.  

        Args:
            # TODO (CP/IRIT): Are the ground truth labels used ?
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            TODO (CP/IRIT): Make it optional
            gt_scores (torch.Tensor): Ground truth probability of being in classes of shape (b, max_num_obj, num_class)
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            TODO (CP/IRIT): What's the use as target_scores contains the information ?
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # save2debug( 'gt_labels.txt', gt_labels, True)
        # save2debug( 'gt_scores.txt', gt_scores, True)
        # save2debug( 'gt_bboxes.txt', gt_bboxes, True)
        # save2debug( 'target_gt_idx_init.txt', target_gt_idx, True)
        # save2debug( 'fg_mask.txt', fg_mask, True)
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        # save2debug( 'target_gt_idx.txt', target_gt_idx, True)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        # Select the boxes associated to the indices
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        
        # TODO (CP/IRIT): Do the same for scores (identical to target_scores_base)
        target_scores_km  = gt_scores.view(-1, gt_scores.shape[-1])[target_gt_idx]

        # Assigned target scores, minimum is 0, maximum is positive
        # Convert to integers (TODO (CP/IRIT): Should it be done much earlier ?)
        # TODO (CP/IRIT): is the clamp_ needed ?
        # Select the labels associated to the indices
        # TODO (CP/IRIT): Are the ground truth labels used ?
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)
        target_labels.clamp_(0)

        # TODO (CP/IRIT): Do the same for scores (wrong) -- Why is it wrong ?
        # target_scores_km = gt_scores.view(-1, gt_scores.shape[-1])[target_gt_idx]
        
        # TODO (CP/IRIT): Initialize scores from ground truth data instead of one_hot for the single class.
        # 10x faster than F.one_hot()
        # Create an int64 zero tensor of dimension batch size * anchor point number * class number
        # TODO (CP/IRIT): Are the ground truth labels used ?
        # Required to provide the tensor dimensions: batch size, inferred data (grids of predictions for each anchor points)
        batch_size =  target_labels.shape[0]
        pred_size = target_labels.shape[1]
        target_scores_base = torch.zeros(
            (batch_size, pred_size, self.num_classes),
            dtype=torch.float32,
            device=target_labels.device,
        )  # (b, h*w, 80)
        # Adds a dimension at the end of target labels
        target_labels_unsqueezed = target_labels.unsqueeze(-1) # (b, h*w, 1)
        # Set the value of the tensor to 1 for indexes from target_labels_unsqueezed in the last dimension 
        target_scores_base.scatter_(2, target_labels_unsqueezed, 1)
        # Duplicate fg_mask class number times along the 3rd dimension
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        # Set to zero when fg_scores_mask is zero
        target_scores = torch.where(fg_scores_mask > 0, target_scores_base, 0)
        compare_target_scores = torch.where((target_scores_km != target_scores))
        image_indexes, anchor_point_indexes, class_indexes = compare_target_scores
        
        # torch.save(target_labels,"target_labels.save",_use_new_zipfile_serialization=False)
        # save2debug( 'fg_scores_mask.txt', fg_scores_mask, True)
        # save2debug( 'target_labels.txt', target_labels)
        # save2debug( 'target_bboxes.txt', target_bboxes)
        # save2debug( 'target_scores_base.txt', target_scores_base, True)
        # save2debug( 'target_scores.txt', target_scores, True)
        # save2debug( 'target_scores_km.txt', target_scores_km, True)
        # TODO (CP/IRIT): use target_scores_km instead of target_scores built with a single 1 at the class position
        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        anchors_lt = xy_centers[None] - lt # positive when center over left top 
        rb_anchors = rb - xy_centers[None] # positive when center under right bottom
        bbox_deltas = torch.cat((anchors_lt, rb_anchors), dim=2).view(bs, n_boxes, n_anchors, -1)
        bbox_deltas = bbox_deltas.amin(3) 
        return bbox_deltas.gt_(eps) # both are be positive iff the center is in the box

    @staticmethod
    # TODO (CP/IRIT): Is it the best criteria when handling composition ?
    # Should select the smallest box in the composition hierarchy ?
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w): sum along all boxes for a given anchor point in a given image, produce the number of boxes for a given point
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
            sz_mask_pos = torch.numel(mask_pos)
            nz_mask_pos = torch.count_nonzero(mask_pos)
            # Convert (b, n_max_boxes, h*w) -> (b, h*w): sum along all boxes for a given anchor point in a given image, produce the number of boxes for a given point
            fg_mask = mask_pos.sum(-2)
            # None should be > 1
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
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).

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
    for i, stride in enumerate(strides):
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


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform the absolute coordinates of the bbox(xyxy) to relative ones (ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


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
            nonzeros = tensor.nonzero().tolist()
            print(len(nonzeros), file=f)
            print(nonzeros, file=f)
        print(tensor.shape, file=f)
        print(tensor.tolist(), file=f)
