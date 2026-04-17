import os
import torch
from torchvision.ops import box_iou

def load_yolo_labels(label_path, img_w, img_h):
    """
    Load YOLO format labels: <class_id> <x_center> <y_center> <width> <height>
    Standard classes: 0-head, 1-helmet, 2-person
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
        
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            if cls in [0, 1]:
                x1 = (x - w/2) * img_w
                y1 = (y - h/2) * img_h
                x2 = (x + w/2) * img_w
                y2 = (y + h/2) * img_h
                gt_boxes.append({"class": int(cls), "bbox": [x1, y1, x2, y2]})
    return gt_boxes

def compute_detailed_metrics(results):
    """
    SafeGVD Metric: Calculates separate P, R, Acc for 'head' and 'helmet'.
    Uses the Centroid-in-Box matching strategy.
    """
    # stats[class_id] = [tp, fp, fn]
    stats = {0: [0, 0, 0], 1: [0, 0, 0]}

    for item in results:
        gts = item['gt']
        preds = item['preds']

        for target_cls in [0, 1]:
            # Filter data for current class
            # For preds: 'not_wearing' maps to head (0), 'wearing' maps to helmet (1)
            status_map = 'wearing' if target_cls == 1 else 'not_wearing'
            curr_gts = [g for g in gts if g['class'] == target_cls]
            curr_preds = [p for p in preds if p['status'] == status_map]

            if not curr_gts:
                stats[target_cls][1] += len(curr_preds) # fp
                continue
            if not curr_preds:
                stats[target_cls][2] += len(curr_gts) # fn
                continue

            matched_gt_indices = set()
            tp_count = 0

            for p_box in curr_preds:
                px1, py1, px2, py2 = p_box['bbox']
                found_match = False
                
                for g_idx, g_obj in enumerate(curr_gts):
                    if g_idx in matched_gt_indices:
                        continue
                    
                    gx1, gy1, gx2, gy2 = g_obj['bbox']
                    g_cx, g_cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
                    
                    if px1 <= g_cx <= px2 and py1 <= g_cy <= py2:
                        tp_count += 1
                        matched_gt_indices.add(g_idx)
                        found_match = True
                        break
                
                if not found_match:
                    stats[target_cls][1] += 1 # fp
            
            stats[target_cls][0] += tp_count # tp
            stats[target_cls][2] += (len(curr_gts) - len(matched_gt_indices)) # fn

    return _format_output(stats)

def compute_detailed_baseline_metrics(results, iou_threshold=0.3):
    """
    Baseline Metric: Calculates separate P, R, Acc for 'head' and 'helmet'.
    Uses standard IoU-based matching.
    """
    stats = {0: [0, 0, 0], 1: [0, 0, 0]}

    for item in results:
        gts = item['gt']
        preds = item['preds']

        for target_cls in [0, 1]:
            status_map = 'wearing' if target_cls == 1 else 'not_wearing'
            curr_gts = [g for g in gts if g['class'] == target_cls]
            curr_preds = [p for p in preds if p['status'] == status_map]

            if not curr_gts:
                stats[target_cls][1] += len(curr_preds)
                continue
            if not curr_preds:
                stats[target_cls][2] += len(curr_gts)
                continue

            gt_boxes = torch.tensor([g['bbox'] for g in curr_gts])
            pd_boxes = torch.tensor([p['bbox'] for p in curr_preds])
            ious = box_iou(pd_boxes, gt_boxes)
            
            matched_gt = set()
            tp_count = 0
            for i in range(len(curr_preds)):
                max_iou, argmax = ious[i].max(0)
                gt_idx = argmax.item()
                
                if max_iou >= iou_threshold and gt_idx not in matched_gt:
                    tp_count += 1
                    matched_gt.add(gt_idx)
                else:
                    stats[target_cls][1] += 1 # fp
            
            stats[target_cls][0] += tp_count
            stats[target_cls][2] += (len(curr_gts) - len(matched_gt))

    return _format_output(stats)

def _format_output(stats):
    """Helper to calculate final P, R, Acc from counts."""
    output = {}
    names = {0: "head", 1: "helmet"}
    for cls_id, counts in stats.items():
        tp, fp, fn = counts
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        # For per-class metrics, accuracy is typically represented by the F1 logic or TP rate
        acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        output[names[cls_id]] = {"precision": p, "recall": r, "accuracy": acc}
    return output