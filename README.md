# Tesseract + FastText Text Detection & Toxicity Analysis

This project extracts text from images using YOLOv8 + pytesseract and predicts toxicity using FastText.

## Folder Structure

- `models/` : YOLOv8 and FastText models
- `images/` : Sample images
- `src/`    : Python scripts
- `utils/`  : Optional helper functions
- `requirements.txt` : Python dependencies

## Usage

```bash
conda create -n yolov8_env python=3.10 -y
conda activate yolov8_env
pip install -r requirements.txt
python src/text_detection.py
