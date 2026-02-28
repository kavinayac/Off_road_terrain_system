"""
Offroad Terrain Segmentation Backend (FINAL — OBJECT NAMES)
Compatible with offroad_best_model_fast.pth (5 classes)
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp


class OffroadSegmenter:
    """Wrapper class for offroad terrain segmentation"""

    def __init__(self, model_path, device=None):
        # ===============================
        # Device selection
        # ===============================
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.num_classes = 5

        print(f"📱 Loading model on: {self.device}")

        # ===============================
        # DeepLabV3+ (MATCH TRAINING)
        # ===============================
        self.model = smp.DeepLabV3Plus(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=self.num_classes,
        )

        # ===============================
        # Load weights
        # ===============================
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print("✅ Model loaded successfully!")

        # ===============================
        # Image preprocessing
        # ===============================
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # ===============================
        # ✅ REAL OBJECT NAMES (FIXED)
        # ===============================
        self.class_names = {
            0: "Background",
            1: "Grass",
            2: "Path",
            3: "Rock",
            4: "Sky",
        }

        # ===============================
        # 🎨 Visualization colors
        # ===============================
        self.class_colors = {
            0: (0, 0, 0),        # Background
            1: (34, 139, 34),    # Grass
            2: (210, 180, 140),  # Path
            3: (128, 128, 128),  # Rock
            4: (135, 206, 235),  # Sky
        }

    # ==========================================
    # 🔍 SEGMENT IMAGE
    # ==========================================
    def segment_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Resize back if needed
        if prediction.shape != (image.size[1], image.size[0]):
            prediction = np.array(
                Image.fromarray(prediction.astype(np.uint8)).resize(
                    original_size, Image.NEAREST
                )
            )

        return {
            "prediction": prediction,
            "original_size": original_size,
        }

    # ==========================================
    # 📈 STATISTICS
    # ==========================================
    def get_statistics(self, prediction):
        unique, counts = np.unique(prediction, return_counts=True)
        total_pixels = prediction.size

        stats = []
        for class_id, count in zip(unique, counts):
            stats.append(
                {
                    "class_id": int(class_id),
                    "class_name": self.class_names[class_id],
                    "pixel_count": int(count),
                    "percentage": float(count / total_pixels * 100),
                    "color": "#{:02x}{:02x}{:02x}".format(
                        *self.class_colors[class_id]
                    ),
                }
            )

        return sorted(stats, key=lambda x: x["pixel_count"], reverse=True)

    # ==========================================
    # 🎨 COLORED MASK
    # ==========================================
    def create_colored_mask(self, prediction):
        height, width = prediction.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id, color in self.class_colors.items():
            colored_mask[prediction == class_id] = color

        return Image.fromarray(colored_mask)
