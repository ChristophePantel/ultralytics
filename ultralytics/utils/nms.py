# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
import time

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, box_iou
from ultralytics.utils.ops import xywh2xyxy

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
    detected = torch.where(bce < 1)
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
                bce = scores_bce(class_variants, selected_scores)
                cpu = torch.device('cpu')
                selected_variant = torch.unsqueeze(torch.argmin(bce,1),1)
                selected_class_from_variant = selected_variant.to(cpu).apply_(variant_to_class.get).to(class_variants.device)
                neq_indexes, neq_values = torch.where(selected_class_from_variant != selected_class)
                selected_image_prediction = torch.cat((selected_boxes, selected_confidence, selected_class_from_variant, selected_scores, selected_mask), 1) # box[i] box of the i-th prediction, selected_image_prediction[i, 4+j] score of the j-th class in the i-th prediction, j[:] class number, cls[i] scores of the i-th prediction, mask[i] extra data of the i-th prediction
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
        selected_indexes = selected_indexes[:max_det]  # limit detections

        output[image_index] = selected_image_prediction[selected_indexes]
        if return_idxs:
            keepi[image_index] = selected_xk[selected_indexes].view(-1)
        if not __debug__ and (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return (output, keepi) if return_idxs else output

def non_max_suppression_revised(
    batch_predictions,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    batch_labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """
    Perform non-maximum suppression (NMS) on predictions results.

    Applies NMS to filter overlapping bounding boxes based on confidence and IoU thresholds. Supports multiple
    detection formats including standard boxes, rotated boxes, and masks.

    Args:
        batch_predictions (torch.Tensor): Predictions with shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing boxes, classes, and optional masks.
        conf_thres (float): Confidence threshold for filtering detections. Valid values are between 0.0 and 1.0.
        iou_thres (float): IoU threshold for NMS filtering. Valid values are between 0.0 and 1.0.
        classes (list[int], optional): List of class indices to consider. If None, all classes are considered.
        agnostic (bool): Whether to perform class-agnostic NMS.
        multi_label (bool): Whether each pred_bboxes can have multiple labels.
        batch_labels (list[list[Union[int, float, torch.Tensor]]]): A priori labels for each image.
        max_det (int): Maximum number of detections to keep per image.
        number_classes (int): Number of classes. Indices after this are considered masks.
        max_time_img (float): Maximum time in seconds for processing one image.
        max_nms (int): Maximum number of boxes for NMS.
        max_wh (int): Maximum pred_bboxes width and height in pixels.
        rotated (bool): Whether to handle Oriented Bounding Boxes (OBB).
        end2end (bool): Whether the model is end-to-end and doesn't require NMS.
        return_idxs (bool): Whether to return the indices of kept detections.

    Returns:
        # TODO (CP/IRIT): adding class scores to the selected image / shape
        output (list[torch.Tensor]): List of detections per image with shape (num_boxes, 6 + num_classes + num_masks)
            containing (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        keepi (list[torch.Tensor]): Indices of kept detections if return_idxs=True.
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    
#===============================================================================
#     if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
#         prediction = prediction[0]  # select only inference output
#     if classes is not None:
#         classes = torch.tensor(classes, device=prediction.device)
    
    if isinstance(batch_predictions, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        # TODO (CP/IRIT): Why is there two predictions, the first one is used there, the second one to compute loss
        batch_predictions = batch_predictions[0]  # select only inference output

    used_device = batch_predictions.device

    batch_shape = batch_predictions.shape

    if classes is not None:
        classes = torch.tensor(classes, device=used_device)
        
# 
#     if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
#         output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
#         if classes is not None:
#             output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
#         return output

    # TODO (CP/IRIT): Check that the changes are valid.
    if batch_shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        outputs = [image_predictions[image_predictions[:, 4] > conf_thres][:max_det] for image_predictions in batch_predictions]
        if classes is not None:
            outputs = [image_predictions[(image_predictions[:, 5:6] == classes).any(1)] for image_predictions in outputs]
        return outputs
# 
#     bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
#     nc = nc or (prediction.shape[1] - 4)  # number of classes
#     extra = prediction.shape[1] - nc - 4  # number of extra info
#     mi = 4 + nc  # mask start index
#     xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
#     xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]  # to track idxs
# 

    batch_size = batch_shape[0]  # batch size (BCN, i.e. 1,84,6300)
    number_classes = nc or (batch_shape[1] - 4)  # number of classes
    number_masks = batch_shape[1] - 4 - number_classes  # number of extra_size info
    mask_start = 4 + number_classes  # pred_masks start index
    
    # TODO (CP/IRIT): also keep all class scores.
    batch_prediction_scores = batch_predictions[:, 4:mask_start]
    batch_candidate_anchor_points = batch_prediction_scores.amax(1) > conf_thres  # candidates
    number_anchor_points = batch_shape[-1]
    batch_anchor_point_indexes = torch.arange(number_anchor_points, device=used_device).expand(batch_size, -1)[..., None]  # to track idxs

#     # Settings
#     # min_wh = 2  # (pixels) minimum box width and height
#     time_limit = 2.0 + max_time_img * bs  # seconds to quit after
#     multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
# 
#     prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
#     if not rotated:
#         prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    # Settings
    # min_wh = 2  # (pixels) minimum pred_bboxes width and height
    time_limit = 2.0 + max_time_img * batch_size  # seconds to quit after
    multi_label &= number_classes > 1  # multiple labels per pred_bboxes (adds 0.5ms/img)

    batch_predictions = batch_predictions.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        batch_predictions[..., :4] = xywh2xyxy(batch_predictions[..., :4])  # xywh to xyxy
        
# 
#     t = time.time()
#     output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
#     keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs

    t = time.time()
    # An output per image that contains :
    # - core predicted class: size 1
    # - bounding pred_bboxes: size 4
    # - class scores (added for multi-class by CP/IRIT): size classes_number
    # - masks data: size masks_number
    # TODO (CP/IRIT): where are the masks ? in extra_size ?
    outputs = [torch.zeros((0, 6 + number_classes + number_masks), device=used_device)] * batch_size
    keepi = [torch.zeros((0, 1), device=used_device)] * batch_size  # to store the kept idxs

#     for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)

    for image_index, (image_predictions, image_anchor_point_indexes) in enumerate(zip(batch_predictions, batch_anchor_point_indexes)):  # image index, (preds, preds indices)

#         # Apply constraints
#         # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         filt = xc[xi]  # confidence
#         x = x[filt]

        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        image_candidate_anchor_points = batch_candidate_anchor_points[image_index]  # confidence
        image_candidate_predictions = image_predictions[image_candidate_anchor_points]
        
#         if return_idxs:
#             xk = xk[filt]
# 
        
        if return_idxs:
            image_candidate_anchor_point_indexes = image_anchor_point_indexes[image_candidate_anchor_points]

#         # Cat apriori labels if autolabelling
#         if labels and len(labels[xi]) and not rotated:
#             lb = labels[xi]
#             v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
#             v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
#             v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
#             x = torch.cat((x, v), 0)
# 

        # Cat apriori labels if autolabelling
        if batch_labels and len(batch_labels[image_index]) and not rotated:
            image_labels = batch_labels[image_index]
            results = torch.zeros((len(image_labels), number_classes + number_masks + 4), device=used_device)
            results[:, :4] = xywh2xyxy(image_labels[:, 1:5])  # pred_bboxes
            results[range(len(image_labels)), image_labels[:, 0].long() + 4] = 1.0  # pred_scores
            image_candidate_predictions = torch.cat((image_candidate_predictions, results), 0)

#         # If none remain process next image
#         if not x.shape[0]:
#             continue
# 

        # If none remain process next image
        if not image_candidate_predictions.shape[0]:
            continue

#         # Detections matrix nx6 (xyxy, conf, cls)
#         box, cls, mask = x.split((4, nc, extra), 1)
# 

        # Detections matrix nx6 (xyxy, conf, pred_scores)
        image_bboxes, image_scores, image_masks = image_candidate_predictions.split((4, number_classes, number_masks), 1)


#         if multi_label:
#             i, j = torch.where(cls > conf_thres)
#             x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
#             if return_idxs:
#                 xk = xk[i]
#         else:  # best class only
#             conf, j = cls.max(1, keepdim=True)
#             filt = conf.view(-1) > conf_thres
#             x = torch.cat((box, conf, j.float(), mask), 1)[filt]
#             if return_idxs:
#                 xk = xk[filt]
# 
        if multi_label:
            image_candidate_elements = torch.where(image_scores > conf_thres)
            # i, j = ...
            image_candidate_anchor_points, image_candidate_classes = image_candidate_elements
            image_candidate_bboxes = image_bboxes[image_candidate_anchor_points]
            image_candidate_score = image_predictions[image_candidate_anchor_points, 4 + image_candidate_classes, None]
            image_candidate_classes = image_candidate_classes[:, None].float()
            image_candidate_masks = image_masks[image_candidate_anchor_points]
            
            image_candidate_predictions = torch.cat((image_candidate_bboxes, image_candidate_score, image_candidate_classes, image_candidate_masks), 1)
            if return_idxs:
                anchor_point_index = anchor_point_index[image_candidate_anchor_points]
        else:  # best class only
            candidate_confidences, candidate_classes = image_scores.max(1, keepdim=True)
            candidate_filter = candidate_confidences.view(-1) > conf_thres
            image_candidate_predictions = torch.cat((pred_bboxes, conf, candidate_classes.float(), image_masks), 1)[candidate_filter]
            if return_idxs:
                anchor_point_index = anchor_point_index[candidate_filter]

#         # Filter by class
#         if classes is not None:
#             filt = (x[:, 5:6] == classes).any(1)
#             x = x[filt]
#             if return_idxs:
#                 xk = xk[filt]
# 
        # Filter by class
        if classes is not None:
            candidate_filter = (image_predictions[:, 5:6] == classes).any(1)
            resulting_predictions = image_predictions[candidate_filter]
            if return_idxs:
                anchor_point_index = anchor_point_index[candidate_filter]

#         # Check shape
#         n = x.shape[0]  # number of boxes
#         if not n:  # no boxes
#             continue
#         if n > max_nms:  # excess boxes
#             filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
#             x = x[filt]
#             if return_idxs:
#                 xk = xk[filt]
# 
        # Check shape
        candidate_number = resulting_predictions.shape[0]  # number of boxes
        if not candidate_number:  # no boxes
            continue
        
        if candidate_number > max_nms:  # excess boxes
            candidate_filter = image_predictions[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            prediction = prediction[candidate_filter]
            if return_idxs:
                anchor_point_index = anchor_point_index[candidate_filter]

#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         scores = x[:, 4]  # scores
#         if rotated:
#             boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
#             i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
#         else:
#             boxes = x[:, :4] + c  # boxes (offset by class)
#             # Speed strategy: torchvision for val or already loaded (faster), TorchNMS for predict (lower latency)
#             if "torchvision" in sys.modules:
#                 import torchvision  # scope as slow import
# 
#                 i = torchvision.ops.nms(boxes, scores, iou_thres)
#             else:
#                 i = TorchNMS.nms(boxes, scores, iou_thres)
#         i = i[:max_det]  # limit detections
# 
        c = prediction[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = prediction[:, 4]  # scores
        if rotated:
            boxes = torch.cat((prediction[:, :2] + c, prediction[:, 2:4], prediction[:, -1:]), dim=-1)  # xywhr
            selected_elements = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = prediction[:, :4] + c  # boxes (offset by class)
            # Speed strategy: torchvision for val or already loaded (faster), TorchNMS for predict (lower latency)
            if "torchvision" in sys.modules:
                import torchvision  # scope as slow import

                selected_elements = torchvision.ops.nms(boxes, scores, iou_thres)
            else:
                selected_elements = TorchNMS.nms(boxes, scores, iou_thres)
                
        selected_elements = selected_elements[:max_det]  # limit detections

#         output[xi] = x[i]
#         if return_idxs:
#             keepi[xi] = xk[i].view(-1)
#         if (time.time() - t) > time_limit:
#             LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
#             break  # time limit exceeded
# 
#     return (output, keepi) if return_idxs else output

        output[image_index] = resulting_predictions[selected_elements]
        if return_idxs:
            keepi[image_index] = anchor_point_index[selected_elements].view(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return (outputs, keepi) if return_idxs else output

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
