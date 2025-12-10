# classifier.py
import torch
import torch.nn.functional as F
from torchvision.models import convnext_tiny
import cv2
import numpy as np

class ShapeClassifier:
    def __init__(self, weight_path, device="cpu"):
        self.device = device

        # class order from your dataset
        self.class_names = ['cone', 'cube', 'cylinder', 'sphere']

        # build model
        self.model = convnext_tiny(weights=None)
        self.model.classifier[2] = torch.nn.Linear(768, 4)

        # load weights
        state = torch.load(weight_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()

    # single-image prediction
    def predict_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(img)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        return pred_label, confidence, probs
