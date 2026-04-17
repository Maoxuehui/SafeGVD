import argparse
import os
import time  
from PIL import Image
from core.download_utils import download_model
from core.detector import GroundingDinoDetector
from core.validator import QwenValidator
from core.visualizer import Visualizer  
from core.utils import load_yolo_labels, compute_detailed_metrics

def main(args):
    # Step 0: Resource Initialization
    print("--- Checking Model Resources ---")
    download_model("IDEA-Research/grounding-dino-tiny", args.detector_path)
    download_model("Qwen/Qwen2.5-VL-3B-Instruct", args.vlm_path)
    
    # Step 1: Framework Initialization
    print("--- Initializing SafeGVD Components ---")
    detector = GroundingDinoDetector(args.detector_path)
    validator = QwenValidator(args.vlm_path)
    visualizer = Visualizer() if args.visualize else None

    # Ensure output directory exists for visualization
    if args.visualize and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_results = []
    img_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    num_images = len(img_files)
    print(f"--- Starting SafeGVD Pipeline | Total Images: {num_images} ---")

    # Initialize timing statistics
    total_det_time = 0.0
    total_val_time = 0.0
    start_wall_clock = time.time()

    for img_name in img_files:
        img_path = os.path.join(args.img_dir, img_name)
        lab_path = os.path.join(args.label_dir, img_name.rsplit('.', 1)[0] + ".txt")
        
        # Load Image Metadata for coordinate conversion
        with Image.open(img_path) as img:
            w, h = img.size
        gt = load_yolo_labels(lab_path, w, h)
        
        # --- Core Pipeline: Detect -> Validate (Timed) ---
        
        # Stage 1: Coarse Localization (Detector)
        t0 = time.time()
        boxes = detector.detect_persons(img_path)
        t1 = time.time()
        
        # Stage 2: Fine-grained Validation (VLM)
        preds = []
        val_start = time.time()
        for b in boxes:
            # Check compliance for each detected box
            status = validator.check_helmet(img_path, b)
            preds.append({"bbox": b, "status": status})
        val_end = time.time()

        # Calculate time durations
        det_duration = t1 - t0
        val_duration = val_end - val_start
        img_total_duration = det_duration + val_duration
        
        total_det_time += det_duration
        total_val_time += val_duration
        # ------------------------------------------------

        all_results.append({"gt": gt, "preds": preds})

        # Save visualization if enabled
        if visualizer:
            vis_save_path = os.path.join(args.output_dir, f"vis_{img_name}")
            visualizer.draw_and_save(img_path, preds, vis_save_path)
            print(f"Done: {img_name} | Det: {det_duration:.2f}s | Val: {val_duration:.2f}s")
        else:
            print(f"Processed: {img_name} ({img_total_duration:.2f}s)")

    end_wall_clock = time.time()
    
    # Compute performance averages
    avg_total = (total_det_time + total_val_time) / num_images if num_images > 0 else 0

    # Step 4: Metrics Evaluation
    safegvd_results = compute_detailed_metrics(all_results)
    
    # Print formatted performance report
    print("\n" + "="*55)
    print(f"{'SafeGVD PERFORMANCE REPORT':^55}")
    print("="*55)
    print(f"{'CLASS':<12} | {'PRECISION':<10} | {'RECALL':<10} | {'ACCURACY':<10}")
    print("-" * 55)
    for cls_name, m in safegvd_results.items():
        print(f"{cls_name.upper():<12} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['accuracy']:<10.4f}")
    print("-" * 55)
    
    # Summary of time profiling
    print(f"Total Detector Time:  {total_det_time:.2f} s")
    print(f"Total Validator Time: {total_val_time:.2f} s")
    print(f"Total Wall Clock Time: {end_wall_clock - start_wall_clock:.2f} s")
    print(f"Average Pipeline Time: {avg_total:.2f} s / img")
    print("="*55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeGVD: Coarse-to-Fine Framework")
    # Path settings
    parser.add_argument("--img_dir", default="./data/samples", help="Path to input images")
    parser.add_argument("--label_dir", default="./data/labels", help="Path to ground truth labels")
    parser.add_argument("--output_dir", default="./results/safegvd", help="Output directory for visualizations")
    
    # Model settings
    parser.add_argument("--detector_path", default="./models/grounding-dino-tiny", help="Path to Detector weights")
    parser.add_argument("--vlm_path", default="./models/Qwen2.5-VL-3B-Instruct", help="Path to VLM weights")
    
    # Execution flags
    parser.add_argument("--visualize", type=bool, default=True, help="Toggle visualization output")
    
    main(parser.parse_args())