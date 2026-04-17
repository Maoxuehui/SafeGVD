import argparse
import os
import json
import re
import time
from PIL import Image
from core.validator import QwenValidator 
from core.visualizer import Visualizer  
from core.utils import load_yolo_labels, compute_detailed_baseline_metrics



def parse_vlm_json(raw_response):
    """
    Extracts and standardizes JSON from VLM raw output.
    """
    try:
        # 1. Try to extract from markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            # 2. Fallback: find the first '{' and last '}'
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                content = raw_response[start_idx:end_idx + 1]
            else:
                return []
        
        data = json.loads(content)
        detections = data.get("detections", [])
        
        standardized = []
        for d in detections:
            bbox = d.get("bbox_2d") or d.get("bbox")
            status = d.get("helmet_status") or d.get("status")
            if bbox and status:
                standardized.append({
                    "bbox": bbox, 
                    "status": status
                })
        return standardized
    except Exception as e:
        print(f"Parsing error: {e}")
        return []

def run_baseline(args):
    # Initialize components
    validator = QwenValidator(args.vlm_path)
    visualizer = Visualizer() if args.visualize else None
    
    all_results = []
    total_inference_time = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(args.img_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    num_images = len(image_files)
    print(f"Starting Baseline Evaluation | Total Images: {num_images}")

    # Start total script timer (Wall Clock)
    start_wall_time = time.time()

    for img_name in image_files:
        img_path = os.path.join(args.img_dir, img_name)
        lab_path = os.path.join(args.label_dir, os.path.splitext(img_name)[0] + ".txt")
        
        with Image.open(img_path) as img:
            w, h = img.size
        
        gt = load_yolo_labels(lab_path, w, h)
        
        # --- Step 1: VLM Direct Inference with Time Calculation ---
        inf_start = time.time()
        raw_res = validator.direct_inference(img_path)
        inf_end = time.time()
        
        duration = inf_end - inf_start
        total_inference_time += duration
        # ---------------------------------------------------------
        
        preds = parse_vlm_json(raw_res)
        
        all_results.append({
            "image": img_name,
            "gt": gt,
            "preds": preds,
            "inference_time": duration
        })
        
        # Step 3: Visualization (Enabled by default)
        if visualizer:
            vis_save_path = os.path.join(args.output_dir, f"vis_{img_name}")
            visualizer.draw_and_save(img_path, preds, vis_save_path)
            print(f"Result saved: {img_name} ({duration:.2f}s)")
        else:
            print(f"Processed: {img_name} ({duration:.2f}s)")

    end_wall_time = time.time()
    avg_time = total_inference_time / num_images if num_images > 0 else 0

    # Step 4: Save raw data for records
    with open(os.path.join(args.output_dir, "baseline_raw_results.json"), 'w') as f:
        json.dump(all_results, f, indent=4)

    # Step 5: Compute and Print Per-Class Metrics
    metrics = compute_detailed_baseline_metrics(all_results, iou_threshold=args.iou_threshold)

    print("\n--- Baseline Detailed Metrics ---")
    print("\n" + "="*55)
    print(f"{'CLASS':<12} | {'PRECISION':<10} | {'RECALL':<10} | {'ACCURACY':<10}")
    print("-" * 55)
    for cls_name, m in metrics.items():
        print(f"{cls_name.upper():<12} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['accuracy']:<10.4f}")
    print("-" * 55)
    print(f"Total Inference Time:  {total_inference_time:.2f} s")
    print(f"Total Wall Clock Time: {end_wall_time - start_wall_time:.2f} s")
    print(f"Average Time per Img: {avg_time:.2f} s")
    print("="*55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline VLM Evaluation Script")
    # Path settings
    parser.add_argument("--img_dir", default="./data/samples")
    parser.add_argument("--label_dir", default="./data/labels")
    parser.add_argument("--vlm_path", default="./models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", default="./results/baseline")
    
    # Parameters
    parser.add_argument("--iou_threshold", type=float, default=0.1, help="IoU threshold (default: 0.1)")
    # Default is now True
    parser.add_argument("--visualize", type=bool, default=True, help="Whether to save visualization results")
    
    args = parser.parse_args()
    run_baseline(args)