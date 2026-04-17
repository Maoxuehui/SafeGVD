# SafeGVD: Towards Annotation-Free Safety Monitoring by Bridging the Spatial-Semantic Gap

Official implementation of **SafeGVD**, a training-free, coarse-to-fine framework designed to bridge the "spatial-semantic gap" in industrial safety compliance monitoring. By decoupling spatial localization (Grounding DINO) from fine-grained semantic validation (Qwen2.5-VL), SafeGVD enables rigorous safety monitoring without manual annotations.

## 1. Project Structure
```text
SafeGVD/
├── core/
│   ├── detector.py      # Stage 1: Coarse Localization (Grounding DINO)
│   ├── validator.py     # Stage 2: Fine-grained Validation (Qwen2.5-VL)
│   ├── visualizer.py    # Visualization engine (Decoupled logic)
│   ├── utils.py         # Data loading and evaluation metrics
│   └── download_utils.py# Automatic model downloading with mirror support
├── data/
│   ├── samples/          # Representative test images (.jpg)
│   └── labels/           # Ground truth in YOLO format (.txt)
├── models/               # Local model weights (Auto-downloaded)
├── results/              # Output directory for visualized results
├── main.py               # Main entry for SafeGVD framework
├── baseline.py           # Baseline: Direct LVLM inference (Native zero-shot)
├── requirements.txt      # Environment dependencies
└── README.md             # Documentation
```

## 2. Hardware Requirements
| Mode | Hardware | Inference Time (per image) | Recommended |
| :--- | :--- | :--- | :--- |
| **GPU Mode** | NVIDIA RTX 3060+ (8GB+ VRAM) | ~2-5 seconds | Yes (Default) |
| **CPU Mode** | Intel/AMD with 16GB+ RAM | ~60-120 seconds | For Debugging only |

> **Note:** SafeGVD is optimized for CUDA. Running on CPU is significantly slower due to the high computational cost of visual token processing in LVLMs.

## 3. Environment Setup 
To ensure compatibility and performance, we recommend using **Conda** to manage your environment. 

### Step 1: Create a Clean Environment
```bash
conda create -n safegvd python=3.10 -y
conda activate safegvd
```

### Step 2: Install PyTorch (Choose ONE based on your hardware)

**Option A: NVIDIA GPU (Highly Recommended for RTX 3090/3060, etc.)**
> **Note:** This requires ~10GB of free disk space and a CUDA-capable driver.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Option B: CPU Only (For Debugging/Review without GPU)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Dependencies
```bash
pip install transformers accelerate qwen_vl_utils pillow numpy
```

## 4. Model Preparation
SafeGVD supports **Automatic Offline Mode**. 

By default, the system will automatically download the required weights from the [HF-Mirror](https://hf-mirror.com) and save them to the `models/` directory upon the first execution. 

If you prefer to download them manually, please ensure the following structure:
- `models/grounding-dino-tiny/`
- `models/Qwen2.5-VL-3B-Instruct/`

## 5. Data Preparation
Organize your data for evaluation:
- **Images**: Place test images in `data/samples/`.
- **Labels**: Place corresponding YOLO format `.txt` labels in `data/labels/`.
  - **Class 0**: Head
  - **Class 1**: Helmet

## 6. Usage

### 6.1 SafeGVD (Proposed Method)
Run the coarse-to-fine pipeline. You can enable the `--visualize` flag to export detection results with bounding boxes (Green: Wearing, Red: Not Wearing) to the `results/` folder.
```bash
python main.py 
```

### 6.2 Baseline (Direct LVLM Inference)
Reproduce the native zero-shot performance of LVLMs without spatial guidance to observe the "spatial-semantic gap":
```bash
python baseline.py 
```

## 7. Evaluation Metrics
The framework evaluates performance based on:
- **Head Detection**: Precision and Recall of localized heads.
- **Helmet Compliance Accuracy**: Correctness of the 'wearing' vs 'not_wearing' classification for successfully detected heads.

## 8. Data Availability
Due to proprietary restrictions of the industrial facilities, the full LNG dataset is not publicly available. However, the provided sample data in `data/` is sufficient to verify the reported pipeline.

