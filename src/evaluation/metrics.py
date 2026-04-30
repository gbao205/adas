"""
Metrics Calculator - Tính toán các chỉ số đánh giá
Hỗ trợ:
- mAP (mean Average Precision) cho Object Detection
- mIoU (mean Intersection over Union) cho Segmentation
- Precision, Recall, F1-Score
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class MetricsCalculator:
    """
    Tính toán các metrics đánh giá cho ADAS system.
    """
    
    def __init__(self, num_classes: int = 80, iou_threshold: float = 0.5):
        """
        Args:
            num_classes: Số lượng classes (COCO = 80)
            iou_threshold: Ngưỡng IoU để xác định True Positive
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        
        # Lưu trữ predictions và ground truths
        self.detections = []  # List of (boxes, scores, labels)
        self.ground_truths = []  # List of (boxes, labels)
        
        self.seg_predictions = []  # List of segmentation masks
        self.seg_ground_truths = []  # List of ground truth masks
        
    def add_detection_batch(
        self, 
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray, 
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray
    ):
        """
        Thêm một batch predictions và ground truths cho detection.
        
        Args:
            pred_boxes: (N, 4) array [x1, y1, x2, y2]
            pred_scores: (N,) array confidence scores
            pred_labels: (N,) array class labels
            gt_boxes: (M, 4) array ground truth boxes
            gt_labels: (M,) array ground truth labels
        """
        self.detections.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels
        })
        self.ground_truths.append({
            'boxes': gt_boxes,
            'labels': gt_labels
        })
    
    def add_segmentation_batch(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ):
        """
        Thêm một batch predictions và ground truths cho segmentation.
        
        Args:
            pred_mask: (H, W) array với class labels
            gt_mask: (H, W) array với ground truth labels
        """
        self.seg_predictions.append(pred_mask)
        self.seg_ground_truths.append(gt_mask)
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Tính IoU (Intersection over Union) giữa 2 boxes.
        
        Args:
            box1, box2: [x1, y1, x2, y2]
        
        Returns:
            IoU score (0.0 - 1.0)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_map(self, iou_thresholds: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Tính mAP (mean Average Precision) cho Object Detection.
        
        Args:
            iou_thresholds: List các ngưỡng IoU (mặc định [0.5])
        
        Returns:
            Dictionary chứa mAP và AP cho từng class
        """
        if iou_thresholds is None:
            iou_thresholds = [self.iou_threshold]
        
        results = {}
        
        for iou_thresh in iou_thresholds:
            # Tính AP cho từng class
            class_aps = []
            
            for class_id in range(self.num_classes):
                # Lấy tất cả predictions và GTs của class này
                all_pred_boxes = []
                all_pred_scores = []
                all_gt_boxes = []
                
                for i, det in enumerate(self.detections):
                    # Predictions của class này
                    mask = det['labels'] == class_id
                    if np.any(mask):
                        all_pred_boxes.extend(det['boxes'][mask])
                        all_pred_scores.extend(det['scores'][mask])
                    
                    # Ground truths của class này
                    gt = self.ground_truths[i]
                    gt_mask = gt['labels'] == class_id
                    if np.any(gt_mask):
                        all_gt_boxes.extend(gt['boxes'][gt_mask])
                
                if len(all_gt_boxes) == 0:
                    continue  # Không có GT cho class này
                
                # Sắp xếp predictions theo confidence giảm dần
                if len(all_pred_boxes) == 0:
                    class_aps.append(0.0)
                    continue
                
                sorted_indices = np.argsort(all_pred_scores)[::-1]
                all_pred_boxes = np.array(all_pred_boxes)[sorted_indices]
                all_pred_scores = np.array(all_pred_scores)[sorted_indices]
                
                # Tính Precision-Recall
                tp = np.zeros(len(all_pred_boxes))
                fp = np.zeros(len(all_pred_boxes))
                matched_gt = set()
                
                for i, pred_box in enumerate(all_pred_boxes):
                    max_iou = 0.0
                    max_gt_idx = -1
                    
                    for j, gt_box in enumerate(all_gt_boxes):
                        if j in matched_gt:
                            continue
                        iou = self.calculate_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            max_gt_idx = j
                    
                    if max_iou >= iou_thresh:
                        tp[i] = 1
                        matched_gt.add(max_gt_idx)
                    else:
                        fp[i] = 1
                
                # Tính cumulative TP và FP
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                
                recalls = tp_cumsum / len(all_gt_boxes)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
                
                # Tính AP (Area Under Curve)
                ap = self._compute_ap(recalls, precisions)
                class_aps.append(ap)
            
            # Tính mAP
            map_value = np.mean(class_aps) if class_aps else 0.0
            results[f'mAP@{iou_thresh:.2f}'] = map_value
        
        return results
    
    def _compute_ap(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """
        Tính Average Precision từ Precision-Recall curve.
        Sử dụng 11-point interpolation.
        """
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap
    
    def calculate_miou(self, num_classes: Optional[int] = None) -> Dict[str, float]:
        """
        Tính mIoU (mean Intersection over Union) cho Segmentation.
        
        Args:
            num_classes: Số lượng classes (nếu None, tự động detect)
        
        Returns:
            Dictionary chứa mIoU và IoU cho từng class
        """
        if not self.seg_predictions:
            return {'mIoU': 0.0}
        
        if num_classes is None:
            num_classes = max(
                np.max(self.seg_predictions[0]),
                np.max(self.seg_ground_truths[0])
            ) + 1
        
        # Khởi tạo confusion matrix
        total_intersection = np.zeros(num_classes)
        total_union = np.zeros(num_classes)
        
        for pred_mask, gt_mask in zip(self.seg_predictions, self.seg_ground_truths):
            for class_id in range(num_classes):
                pred_class = (pred_mask == class_id)
                gt_class = (gt_mask == class_id)
                
                intersection = np.logical_and(pred_class, gt_class).sum()
                union = np.logical_or(pred_class, gt_class).sum()
                
                total_intersection[class_id] += intersection
                total_union[class_id] += union
        
        # Tính IoU cho từng class
        class_ious = {}
        valid_ious = []
        
        for class_id in range(num_classes):
            if total_union[class_id] > 0:
                iou = total_intersection[class_id] / total_union[class_id]
                class_ious[f'IoU_class_{class_id}'] = iou
                valid_ious.append(iou)
            else:
                class_ious[f'IoU_class_{class_id}'] = 0.0
        
        # Tính mIoU
        miou = np.mean(valid_ious) if valid_ious else 0.0
        
        results = {'mIoU': miou}
        results.update(class_ious)
        
        return results
    
    def calculate_precision_recall_f1(self) -> Dict[str, float]:
        """
        Tính Precision, Recall, F1-Score tổng thể.
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for i, det in enumerate(self.detections):
            gt = self.ground_truths[i]
            
            pred_boxes = det['boxes']
            gt_boxes = gt['boxes']
            
            matched_gt = set()
            
            for pred_box in pred_boxes:
                matched = False
                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou >= self.iou_threshold:
                        total_tp += 1
                        matched_gt.add(j)
                        matched = True
                        break
                
                if not matched:
                    total_fp += 1
            
            # False Negatives = GT boxes không được match
            total_fn += len(gt_boxes) - len(matched_gt)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }
    
    def reset(self):
        """Reset tất cả dữ liệu đã lưu."""
        self.detections = []
        self.ground_truths = []
        self.seg_predictions = []
        self.seg_ground_truths = []
    
    def get_summary(self) -> Dict[str, float]:
        """
        Trả về tổng hợp tất cả metrics.
        """
        summary = {}
        
        # Detection metrics
        if self.detections:
            summary.update(self.calculate_map())
            summary.update(self.calculate_precision_recall_f1())
        
        # Segmentation metrics
        if self.seg_predictions:
            summary.update(self.calculate_miou())
        
        return summary
