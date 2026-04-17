import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

class GroundingDinoDetector:
    def __init__(self, model_id, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def detect_persons(self, image_path, text_prompt="person.", box_threshold=0.3, text_threshold=0.25):
        """
        Detects persons using Grounding DINO.
        Based on the logic in hf_groundingDINO_qwen2.5vl_local.py
        """
        image = Image.open(image_path).convert("RGB")
        
        # Grounding DINO requires lowercase text prompts ending with a dot
        text_prompt = text_prompt.lower()
        if not text_prompt.endswith("."):
            text_prompt += "."

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,       
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]
        
        # Ensure results are on CPU before converting to list
        return results["boxes"].cpu().numpy().tolist()
