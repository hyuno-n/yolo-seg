import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score

def load_mask(path, target_size=None):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if target_size and mask is not None:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return mask

def binarize_mask(mask, threshold=127):
    return (mask > threshold).astype(np.uint8)

def evaluate_pixelwise(model_path, image_dir, gt_mask_dir, conf_threshold=0.3):
    model = YOLO(model_path)
    y_true_all = []
    y_pred_all = []

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for fname in tqdm(image_files):
        image_path = os.path.join(image_dir, fname)
        gt_path = os.path.join(gt_mask_dir, fname)

        img = cv2.imread(image_path)
        if img is None or not os.path.exists(gt_path):
            continue

        # Inference
        results = model(img)[0]
        pred_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for seg in results.masks.xy:
            cv2.fillPoly(pred_mask, [seg.astype(np.int32)], 1)

        # Load GT mask
        gt_mask = load_mask(gt_path, target_size=(img.shape[1], img.shape[0]))
        gt_mask = binarize_mask(gt_mask)

        # Flatten and accumulate
        y_true_all.extend(gt_mask.flatten())
        y_pred_all.extend(pred_mask.flatten())

    precision = precision_score(y_true_all, y_pred_all)
    recall = recall_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all)

    print(f"Pixel-wise Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

if __name__ == "__main__":
    model_path = "weights/crack_seg.pt"
    image_dir = "test_images"
    gt_mask_dir = "gt_masks"
    evaluate_pixelwise(model_path, image_dir, gt_mask_dir)
