# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
import time

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, box_iou
from ultralytics.utils.ops import xywh2xyxy

# TODO : test A <-> B = A -> B /\ B -> A
def scores_fuzzy_equiv(batch_scores, prediction_scores, alpha=0.9, power=3):
    """Compute fuzzy equivalence between expected scores and predicted scores.
    
    Args:
    
    Returns:
    """
    batch_range = batch_scores.shape[0]
    prediction_range = prediction_scores.shape[0]
    aligned_batch_scores = torch.unsqueeze(batch_scores, 0).expand(prediction_range,-1,-1)
    aligned_batch_scores_sum = aligned_batch_scores.sum(-1)
    aligned_prediction_scores = torch.unsqueeze(prediction_scores, 1).expand(-1,batch_range,-1)
    negated_aligned_batch_scores = 1.0 - aligned_batch_scores
    negated_aligned_prediction_scores = 1.0 - aligned_prediction_scores
    positive_component = aligned_batch_scores * aligned_prediction_scores
    negative_component = negated_aligned_batch_scores * negated_aligned_prediction_scores
    positive_component_sum = positive_component.sum(-1)
    batch_prediction_equiv = positive_component_sum  / aligned_batch_scores_sum # alpha *  positive_component + (1 - alpha) * negative_component
    return batch_prediction_equiv # batch_prediction_equiv.pow(power).mean(-1).pow(1/power)

def scores_bce(batch_scores, prediction_scores):
    """Compute binary cross entropy between expected scores and predicted scores.
    
    Args:
    
    Returns:
    """
    bce_calculator = nn.BCELoss(reduction="none")
    batch_range = batch_scores.shape[0]
    prediction_range = prediction_scores.shape[0]
    batch_scores_sum = torch.sum(batch_scores,1)
    aligned_batch_scores_sum = torch.unsqueeze(batch_scores_sum, 0).expand(prediction_range,-1)
    aligned_batch_scores = torch.unsqueeze(batch_scores, 0).expand(prediction_range,-1,-1)
    aligned_prediction_scores = torch.unsqueeze(prediction_scores, 1).expand(-1,batch_range,-1)
    bce_per_class = bce_calculator(aligned_prediction_scores,aligned_batch_scores) 
    bce = torch.sum(bce_per_class,2) / batch_scores_sum
    # detected = torch.where(bce < 1)
    return bce

def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    use_km_scores : bool = False,
    use_variant_selection : bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
    class_variants = None,
    variant_to_class = None,
):
    """Perform non-maximum suppression (NMS) on prediction results.

    Applies NMS to filter overlapping bounding boxes based on confidence and IoU thresholds. Supports multiple detection
    formats including standard boxes, rotated boxes, and masks.

    Args:
        prediction (torch.Tensor): Predictions with shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing boxes, classes, and optional masks.
            if tuple/list : first is aggregated prediction for all anchor points, second is TPN prediction for 3 level of details
        conf_thres (float): Confidence threshold for filtering detections. Valid values are between 0.0 and 1.0.
        iou_thres (float): IoU threshold for NMS filtering. Valid values are between 0.0 and 1.0.
        classes (list[int], optional): List of class indices to consider. If None, all classes are considered.
        agnostic (bool): Whether to perform class-agnostic NMS.
        multi_label (bool): Whether each box can have multiple labels.
        labels (list[list[Union[int, float, torch.Tensor]]]): A priori labels for each image.
        max_det (int): Maximum number of detections to keep per image.
        nc (int): Number of classes. Indices after this are considered masks.
        max_time_img (float): Maximum time in seconds for processing one image.
        max_nms (int): Maximum number of boxes for NMS.
        max_wh (int): Maximum box width and height in pixels.
        rotated (bool): Whether to handle Oriented Bounding Boxes (OBB).
        end2end (bool): Whether the model is end-to-end and doesn't require NMS.
        return_idxs (bool): Whether to return the indices of kept detections.

    Returns:
        output (list[torch.Tensor]): List of detections per image with shape (num_boxes, 6 + num_masks) containing (x1,
            y1, x2, y2, confidence, class, mask1, mask2, ...).
        keepi (list[torch.Tensor]): Indices of kept detections if return_idxs=True.
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output (aggregated predictions for all anchor points)
    if class_variants is not None:
        class_variants = class_variants.to(device=prediction.device,dtype=prediction.dtype)
    if variant_to_class is not None:
        variant_to_class = variant_to_class.to(device=prediction.device)
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)
    anchor_points_number = prediction.shape[-1]

    if anchor_points_number == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6) / only 6 anchor points
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    # TODO (CP/IRIT): Why is nc set to the prediction number of classes when it is 0 (for example, detection case) ?
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index / end of scores
    # TODO (CP/IRIT): Confidence is more complex when using knowledge models. A confidence should be computed for class variants (BCE with scores).
    # requires to have access to the class variants and not only the predictions
    pred_scores = prediction[:, 4:mi] # extract the score for each raw class
    
    max_pred_scores = pred_scores.amax(1) # maximum only makes sense for a single class prediction
    # permuter pour avoir BS * AP * Nc
    # calculer le BCE du score avec les scores du modèle de chaque variante de classe
    # garder le minimum qui donne le numéro de la variante sélectionnée puis en déduire le numéro de classe
    # sinon conserver plusieurs prédictions pertinentes basée sur le BCE
    anchor_point_candidates =  max_pred_scores > conf_thres  # candidates (maximum of scores over confidence threshold)
    anchor_point_indexes = torch.arange(anchor_points_number, device=prediction.device).expand(bs, -1)[..., None]  # to track idxs, associate its index to each anchor point in each image from the batch

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84) a.k.a (1,features,anchor points) -> (1,anchor points, features)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs # 6 = bounding box & class & confidence
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    for image_index, (image_prediction, image_anchor_point_indexes) in enumerate(zip(prediction, anchor_point_indexes)):  # image index, (preds, preds indices)
        # Apply constraints
        # selected_image_prediction[((selected_image_prediction[:, 2:4] < min_wh) | (selected_image_prediction[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        image_anchor_point_candidates = anchor_point_candidates[image_index]  # confidence for each anchor point in an image index
        selected_image_prediction = image_prediction[image_anchor_point_candidates] # 
        if return_idxs:
            selected_xk = image_anchor_point_indexes[image_anchor_point_candidates]

        # Cat apriori labels if autolabelling
        if labels and len(labels[image_index]) and not rotated:
            lb = labels[image_index]
            v = torch.zeros((len(lb), nc + extra + 4), device=selected_image_prediction.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            selected_image_prediction = torch.cat((selected_image_prediction, v), 0)

        # If none remain process next image
        if not selected_image_prediction.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls) 
        predicted_boxes, predicted_scores, predicted_masks = selected_image_prediction.split((4, nc, extra), 1) # bounding box, scores, additional data

        if multi_label:
            # TODO (CP/IRIT): compute BCE between predicted_scores and class_variants instead of simple predicted score
            i, j = torch.where(predicted_scores > conf_thres) # indices in predicted_scores where the values are over conf_thres, i: anchor point index, j: class index  
            # TODO (CP/IRIT): select the class based on BCE between class variants from the knowledge model and class predicted scores
            selected_boxes = predicted_boxes[i]
            selected_confidence = selected_image_prediction[i, 4 + j, None]
            selected_class = j[:, None].float()
            selected_scores = predicted_scores[i]
            selected_mask = predicted_masks[i]
            # TODO (CP/IRIT): use selected class (yolo) OR variant (km)
            if use_variant_selection:
                # TODO (CP/IRIT): Compare variants with selected predicted scores to identify
                # my_variants = class_variants
                # test_bce = scores_bce(my_variants, my_variants)
                # selected_test_bce = torch.argmin(test_bce,1)
                # bce = scores_bce(class_variants, selected_scores)
                # selected_variant_bce = torch.unsqueeze(torch.argmin(bce,1),1)
                # selected_class_from_variant_bce = variant_to_class[selected_variant_bce]
                # test_sfe = scores_fuzzy_equiv(my_variants, my_variants)
                # selected_test_sfe = torch.argmax(test_sfe,1)
                # Remove abstract variants
                class_variants[3,2] = 0.0 # Animals
                class_variants[156,58] = 0.0 # Vehicle
                variants_scores = scores_fuzzy_equiv(class_variants, selected_scores)
                selected_variants_scores, selected_variants = torch.max(variants_scores,1)
                selected_variants = selected_variants.unsqueeze(1)
                selected_variants_scores = selected_variants_scores.unsqueeze(1)
                # selected_variant_sfe = torch.unsqueeze(torch.argmax(sfe,1),1)
                selected_classes_from_variants = variant_to_class[selected_variants]
                # cpu = torch.device('cpu')
                # selected_variant = torch.unsqueeze(torch.argmin(bce,1),1)
                # variant_to_class_decoder = torch.tensor([variant_to_class[i] for i in range(len(variant_to_class))],device=selected_variant.device)
                # selected_class_from_variant = selected_class_from_variant_sfe
                # selected_class_from_variant = selected_variant.to(cpu).apply_(variant_to_class.get).to(class_variants.device)
                # neq_indexes, neq_values = torch.where(selected_class_from_variant != selected_class)
                # TODO (CP/IRIT): Duplicate bounding boxes for each class in each selected variant, keep the variant index for the fusion phase 
                selected_image_prediction = torch.cat((selected_boxes, selected_confidence, selected_classes_from_variants, selected_scores, selected_mask), 1) # box[i] box of the i-th prediction, selected_image_prediction[i, 4+j] score of the j-th class in the i-th prediction, j[:] class number, cls[i] scores of the i-th prediction, mask[i] extra data of the i-th prediction
            else:
                selected_image_prediction = torch.cat((selected_boxes, selected_confidence, selected_class, selected_scores, selected_mask), 1) # box[i] box of the i-th prediction, selected_image_prediction[i, 4+j] score of the j-th class in the i-th prediction, j[:] class number, cls[i] scores of the i-th prediction, mask[i] extra data of the i-th prediction
            if return_idxs:
                selected_xk = selected_xk[i]
        else:  # best class only
            confidence, class_index = predicted_scores.max(1, keepdim=True)
            image_anchor_point_candidates = confidence.view(-1) > conf_thres
            selected_image_prediction = torch.cat((predicted_boxes, confidence, class_index.float(), predicted_masks), 1)[image_anchor_point_candidates]
            if return_idxs:
                selected_xk = selected_xk[image_anchor_point_candidates]

        # Filter by class
        if classes is not None:
            image_anchor_point_candidates = (selected_image_prediction[:, 5:6] == classes).any(1)
            selected_image_prediction = selected_image_prediction[image_anchor_point_candidates]
            if return_idxs:
                selected_xk = selected_xk[image_anchor_point_candidates]

        # Check shape
        box_number = selected_image_prediction.shape[0]  # number of boxes
        if not box_number:  # no boxes
            continue
        if box_number > max_nms:  # excess boxes
            image_anchor_point_candidates = selected_image_prediction[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            selected_image_prediction = selected_image_prediction[image_anchor_point_candidates]
            if return_idxs:
                selected_xk = selected_xk[image_anchor_point_candidates]

        widened_class_indexes = selected_image_prediction[:, 5:6] * (0 if agnostic else max_wh)  # class index multiplied by max_wh in order to separate boxes by class
        selected_image_scores = selected_image_prediction[:, 4]  # scores de confiance pour chaque point
        if rotated:
            candidate_boxes = torch.cat((selected_image_prediction[:, :2] + widened_class_indexes, selected_image_prediction[:, 2:4], selected_image_prediction[:, -1:]), dim=-1)  # xywhr
            i = TorchNMS.fast_nms(candidate_boxes, selected_image_scores, iou_thres, iou_func=batch_probiou)
        else:
            candidate_boxes = selected_image_prediction[:, :4] + widened_class_indexes  # boxes (offset by class) 
            # Speed strategy: torchvision for val or already loaded (faster), TorchNMS for predict (lower latency)
            if "torchvision" in sys.modules:
                import torchvision  # scope as slow import

                selected_indexes = torchvision.ops.nms(candidate_boxes, selected_image_scores, iou_thres)
            else:
                selected_indexes = TorchNMS.nms(candidate_boxes, selected_image_scores, iou_thres)
        # TODO (CP/IRIT): variant fusion 
        selected_indexes = selected_indexes[:max_det]  # limit detections

        output[image_index] = selected_image_prediction[selected_indexes]
        if return_idxs:
            keepi[image_index] = selected_xk[selected_indexes].view(-1)
        if not __debug__ and (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return (output, keepi) if return_idxs else output

#===============================================================================



















class TorchNMS:
    """Ultralytics custom NMS implementation optimized for YOLO.

    This class provides static methods for performing non-maximum suppression (NMS) operations on bounding boxes,
    including both standard NMS and batched NMS for multi-class scenarios.

    Methods:
        nms: Optimized NMS with early termination that matches torchvision behavior exactly.
        batched_nms: Batched NMS for class-aware suppression.

    Examples:
        Perform standard NMS on boxes and scores
        >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> scores = torch.tensor([0.9, 0.8])
        >>> keep = TorchNMS.nms(boxes, scores, 0.5)
    """

    @staticmethod
    def fast_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        use_triu: bool = True,
        iou_func=box_iou,
        exit_early: bool = True,
    ) -> torch.Tensor:
        """Fast-NMS implementation from https://arxiv.org/pdf/1904.02689 using upper triangular matrix operations.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            use_triu (bool): Whether to use torch.triu operator for upper triangular matrix operations.
            iou_func (callable): Function to compute IoU between boxes.
            exit_early (bool): Whether to exit early if there are no boxes.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply NMS to a set of boxes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = TorchNMS.nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0 and exit_early:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            # NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            # Zeroing these scores ensures the additional indices would not affect the final results
            scores_ = scores[sorted_idx]
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            scores[sorted_idx] = scores_  # update original tensor for NMSModel
            # NOTE: return indices with fixed length to avoid TFLite reshape error
            pick = torch.topk(scores_, scores_.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Optimized NMS with early termination that matches torchvision behavior exactly.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply NMS to a set of boxes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = TorchNMS.nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Pre-allocate and extract coordinates once
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)

        # Sort by scores descending
        order = scores.argsort(0, descending=True)

        # Pre-allocate keep list with maximum possible size
        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0
        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1

            if order.numel() == 1:
                break
            # Vectorized IoU calculation for remaining boxes
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            # Fast intersection and IoU
            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            # Early exit: skip IoU calculation if no intersection
            if inter.sum() == 0:
                # No overlaps with current box, keep all remaining boxes
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter)
            # Keep boxes with IoU <= threshold
            order = rest[iou <= iou_threshold]

        return keep[:keep_idx]

    @staticmethod
    def batched_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        iou_threshold: float,
        use_fast_nms: bool = False,
    ) -> torch.Tensor:
        """Batched NMS for class-aware suppression.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            idxs (torch.Tensor): Class indices with shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            use_fast_nms (bool): Whether to use the Fast-NMS implementation.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply batched NMS across multiple classes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> idxs = torch.tensor([0, 1])
            >>> keep = TorchNMS.batched_nms(boxes, scores, idxs, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Strategy: offset boxes by class index to prevent cross-class suppression
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        return (
            TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold)
            if use_fast_nms
            else TorchNMS.nms(boxes_for_nms, scores, iou_threshold)
        )
