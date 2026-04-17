import os
from PIL import Image, ImageDraw, ImageFont

class Visualizer:
    def __init__(self, font_path="DejaVuSans.ttf"):
        # We will handle font sizing dynamically per box
        self.font_path = font_path

    def _get_font(self, size):
        try:
            return ImageFont.truetype(self.font_path, int(size))
        except IOError:
            return ImageFont.load_default()

    def draw_and_save(self, image_path, predictions, save_path):
        """
        Optimized visualization for academic reporting.
        Features: Dynamic font scaling and semi-transparent label backgrounds.
        """
        try:
            with Image.open(image_path).convert("RGBA") as img:
                # Create a separate layer for transparent drawings
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                for pred in predictions:
                    box = pred["bbox"]  # [x1, y1, x2, y2]
                    status = pred["status"]
                    
                    # Academic color palette (Green-Safe / Red-Alert)
                    # Using 200 alpha for the box and 255 for text
                    if status == "wearing":
                        color_main = (34, 197, 94, 255)  # Tailwind Green-500
                        bg_label = (34, 197, 94, 180)    # Semi-transparent
                    else:
                        color_main = (239, 68, 68, 255)  # Tailwind Red-500
                        bg_label = (239, 68, 68, 180)    # Semi-transparent

                    # 1. Draw Bounding Box with slight transparency
                    draw.rectangle(box, outline=color_main, width=3)

                    # 2. Dynamic Text Scaling (Font size is 12% of box height)
                    box_h = box[3] - box[1]
                    font_size = max(14, int(box_h * 0.12))
                    font = self._get_font(font_size)

                    # 3. Draw Label Header
                    label = status.upper().replace("_", " ")
                    
                    # Get text dimensions for background rectangle
                    if hasattr(draw, 'textbbox'):
                        t_x1, t_y1, t_x2, t_y2 = draw.textbbox((box[0], box[1]), label, font=font)
                        # Padding for the label
                        padding = 4
                        label_bg = [t_x1, t_y1 - padding, t_x2 + padding*2, t_y2 + padding]
                    else:
                        w, h = draw.textsize(label, font=font)
                        label_bg = [box[0], box[1] - h, box[0] + w, box[1]]

                    # Draw label background and text
                    draw.rectangle(label_bg, fill=bg_label)
                    draw.text((box[0] + 2, label_bg[1]), label, fill=(255, 255, 255, 255), font=font)

                # Composite the overlay back onto the image
                combined = Image.alpha_composite(img, overlay).convert("RGB")
                combined.save(save_path, quality=95)
                
        except Exception as e:
            print(f"Warning: Visualization failed for {image_path}. Error: {e}")