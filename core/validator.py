import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

class QwenValidator:
    def __init__(self, model_id, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Checking CUDA: {torch.cuda.is_available()}")
        print(f"Target Device: {self.device}")

        # Using auto device_map and auto torch_dtype for optimal performance on the available hardware
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        print(f"Model is on device: {self.model.device}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.min_size = 28  # Minimum size requirement from your qwen2.5vl_local.py

    def _prepare_image(self, image):
        """Helper to ensure image meets minimum size requirements."""
        width, height = image.size
        if width < self.min_size or height < self.min_size:
            ratio = max(self.min_size / width, self.min_size / height)
            new_width = int(width * ratio) + 1
            new_height = int(height * ratio) + 1
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image

    def check_helmet(self, image_path, box):
        """Validates safety helmet status on a cropped head region."""
        image = Image.open(image_path).convert("RGB")
        crop = image.crop(box)  # box: [x1, y1, x2, y2]
        crop = self._prepare_image(crop)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": "Is the person in this image wearing a safety helmet? Answer only 'yes' or 'no'."}
            ]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=10)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        result = response.strip().lower()
        
        if "yes" in result and "no" not in result:
            return "wearing"
        return "not_wearing"

    def direct_inference(self, image_path):
        """
        Baseline method: Performs detection and classification on the full image 
        simultaneously to demonstrate the 'spatial-semantic gap'.
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Detect all visible head in the image and determine if each is wearing a helmet. For each head, provide a JSON object with \"class\": \"head\", \"bbox_2d\": [x_min, y_min, x_max, y_max], and \"helmet_status\": \"wearing\" or \"not_wearing\". Your entire response must be a single, valid JSON object with a key \"detections\" containing a list of these head objects."}
            ]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Higher token limit for JSON output
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response