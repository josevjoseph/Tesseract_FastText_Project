from ultralytics import YOLO
import cv2
import pytesseract
from PIL import Image
import numpy as np
import fasttext

# Load YOLOv8 model
yolo_model = YOLO('../models/yolov8n.pt')

# Load FastText model
fasttext_model = fasttext.load_model("../models/fasttext_toxic.bin")

def predict_toxicity(text):
    label, confidence = fasttext_model.predict(text)
    return label[0], confidence[0]

# Load image
image_path = '../images/toxic.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# YOLOv8 inference
results = yolo_model(image_path, conf=0.25)

# Extract text and predict toxicity
all_detected_texts = []

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        roi = img[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        roi_pil = Image.fromarray(roi_thresh)

        text = pytesseract.image_to_string(roi_pil, lang='eng', config='--psm 6').strip()
        if text:
            all_detected_texts.append(text)
            label, confidence = predict_toxicity(text)
            print(f"Detected Text: {text}")
            print(f"Toxicity Prediction: {label}, Confidence: {confidence:.2f}\n")

if all_detected_texts:
    combined_text = " ".join(all_detected_texts)
    label, confidence = predict_toxicity(combined_text)
    print(f"Combined Text Prediction: {label}, Confidence: {confidence:.2f}")
else:
    print("No text detected in image.")
