
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# from ultralytics import YOLO

# from utils.image_processing import estimate_size


# # -------------------------------
# # Load EfficientNet
# # -------------------------------
# def load_classifier(path, num_classes=4):
#     from torchvision import models

#     model = models.efficientnet_v2_s(pretrained=False)
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

#     state_dict = torch.load(path, map_location="cpu")
#     model.load_state_dict(state_dict)

#     model.eval()
#     return model


# # -------------------------------
# # Load YOLO
# # -------------------------------
# def load_yolo(path):
#     return YOLO(path)


# # -------------------------------
# # Transform
# # -------------------------------
# def get_transform():
#     return transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])


# # -------------------------------
# # Classify crop
# # -------------------------------
# def classify_crop(model, crop):
#     transform = get_transform()
#     img_tensor = transform(crop).unsqueeze(0)

#     with torch.no_grad():
#         outputs = model(img_tensor)
#         probs = torch.softmax(outputs, dim=1)
#         confidence, pred = torch.max(probs, 1)

#     classes = ["Fiber", "Fragment", "Film", "Pellet"]
#     return classes[pred.item()], confidence.item()


# # -------------------------------
# # YOLO + EffNet Pipeline
# # -------------------------------
# def yolo_effnet_pipeline(yolo_model, clf_model, img_np, scale_factor=1.0):

#     results = []

#     detections = yolo_model(img_np)[0]

#     if detections.boxes is None:
#         return results

#     for det in detections.boxes.data:
#         x1, y1, x2, y2, conf, cls = det.tolist()

#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

#         crop = img_np[y1:y2, x1:x2]

#         if crop.size == 0:
#             continue

#         # Classification
#         label, confidence = classify_crop(clf_model, crop)

#         # Size estimation
#         size, _ = estimate_size(crop, scale_factor)

#         results.append({
#             "box": (x1, y1, x2, y2),
#             "label": label,
#             "confidence": confidence,
#             "size": size
#         })

#     return results

# def load_classifier_mobilenet(path, num_classes=4):
#     import torch
#     import torch.nn as nn
#     from torchvision import models

#     model = models.mobilenet_v2(pretrained=False)
#     model.classifier[1] = nn.Linear(model.last_channel, num_classes)

#     model.load_state_dict(torch.load(path, map_location="cpu"))
#     model.eval()
#     return model

from utils.image_processing import estimate_size

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from ultralytics import YOLO

from utils.image_processing import estimate_size


# -------------------------------
# Load YOLO
# -------------------------------
def load_yolo(path):
    return YOLO(path)


# -------------------------------
# Load EfficientNet
# -------------------------------
def load_effnet(path, num_classes=4):
    from torchvision import models

    model = models.efficientnet_v2_s(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# -------------------------------
# Load MobileNet
# -------------------------------
def load_mobilenet(path, num_classes=4):
    from torchvision import models

    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# -------------------------------
# Transform
# -------------------------------
def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


# -------------------------------
# Classify crop
# -------------------------------
def classify_crop(model, crop):
    transform = get_transform()
    img_tensor = transform(crop).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    classes = ["Fiber", "Fragment", "Film", "Pellet"]
    return classes[pred.item()], confidence.item()


# -------------------------------
# YOLO + Classifier Pipeline
# -------------------------------
def yolo_pipeline(yolo_model, clf_model, img_np, scale_factor=1.0):

    results = []

    detections = yolo_model(img_np, conf=0.15)[0]

    if detections.boxes is None:
        return results

    for det in detections.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        crop = img_np[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Classification
        label, confidence = classify_crop(clf_model, crop)

        # Size estimation
        # size, _ = estimate_size(crop, scale_factor)
        size, contour = estimate_size(crop, scale_factor)

        results.append({
            "box": (x1, y1, x2, y2),
            "label": label,
            "confidence": confidence,
            "size": size
        })

    return results
