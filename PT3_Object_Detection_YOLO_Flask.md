# ITST 303 – Web and Database Integration
## Group Performance Task #3
**Activity Title:** Object Detection System using Ultralytics YOLO with Flask Deployment

| | |
|---|---|
| **Name/s:** | **Group #:** |
| **Group Performance Task #3** | **Date:** |
| **Section:** | **Image Detection / Recognition** |

---

## Activity Overview

In this activity, students will transition from traditional machine learning models to computer vision using Ultralytics YOLO (You Only Look Once).

Students will:
- Train or use a pre-trained YOLO model
- Perform image recognition (object detection)
- Integrate the model into a Flask web application
- Build a system that allows users to upload images and receive real-time detection results

This activity emphasizes:
- Deep learning application
- Image processing
- Model deployment
- Real-world AI system integration

> This is an upgrade from your previous Flask ML deployment task, shifting from tabular prediction to visual intelligence systems.

---

## Learning Objectives

At the end of this activity, students should be able to:

1. Understand object detection using YOLO
2. Use Ultralytics YOLO for image recognition
3. Train or fine-tune a YOLO model *(optional but encouraged)*
4. Perform inference on images
5. Integrate YOLO into a Flask web application
6. Accept image uploads using web forms (Jinja2)
7. Display detection results (bounding boxes + labels)
8. Deploy an AI-powered web system

---

## Prerequisite

Students must use:
- **Ultralytics YOLOv8** (Python)
- A dataset (any of the following):
  - COCO pre-trained model (default)
  - Custom dataset (recommended)

---

## Task Requirements

### 1. YOLO Model Setup & Testing

Install Ultralytics:

```bash
pip install ultralytics
```

Load a YOLO model:

```python
from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model (small + fast)
model = YOLO("yolov8n.pt")
```

Perform test detection:

```python
# Load image
image_path = "dog.jpg"

# Run detection
results = model(image_path)

# Show results (with bounding boxes)
results[0].show()
```

**Output:**
- Sample detected image
- Identified objects with labels

---

### 2. Model Training / Custom Dataset

Students may use a custom dataset and train using:

```python
from ultralytics import YOLO

print("Starting training...")

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data="dataset",
    epochs=20,
    imgsz=224
)

print("Training finished!")
```

**Output:**
- Training results (loss, accuracy)
- Explanation of dataset used

---

### 3. Model Inference & Export

Students must:
- Run detection on new images
- Save output images with bounding boxes

```python
from ultralytics import YOLO

model = YOLO("runs/classify/train/weights/best.pt")

results = model("crying.png")

for r in results:
    probs = r.probs
    print("Prediction:", model.names[probs.top1])
    print("Confidence:", float(probs.top1conf))
```

**Output:**
- Detected image with labels
- Explanation of detected objects

---

### 4. Flask Web Application Development

Students must build a working web system modified for image input.

#### A. Home Page (Upload Form)

- Upload image file
- Use Jinja2 template

```html
<form action="/predict" method="post" enctype="multipart/form-data">
  <input type="file" name="image" required>
  <button type="submit">Detect</button>
</form>
```

#### B. Prediction Function (YOLO Integration)

```python
from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"

model = YOLO("yolov8n.pt")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    results = model(filepath)
    results[0].save(filename=filepath)

    return render_template("index.html", image=filepath)

if __name__ == "__main__":
    app.run(debug=True)
```

#### C. Result Display

- Show uploaded image
- Show detected image with bounding boxes

```html
{% if image %}
  <img src="{{ image }}" width="400">
{% endif %}
```

---

### 5. System Structure

```
YOLO-FLASK/
├── static/
│   └── results/
├── templates/
│   └── index.html
├── uploads/
├── app.py
└── yolov8n.pt
```

---

## Deliverables

Each group must submit:

1. Flask Web Application *(Deployment recommended)*
2. Presentation Slides *(Canva or PPT)*

---

## Evaluation Rubric

| Criteria | Excellent (5) | Good (4) | Fair (3) | Poor (2) |
|---|---|---|---|---|
| **YOLO Implementation** | Correct detection with clear explanation | Minor issues | Basic output | Not working |
| **Detection Results** | Accurate & well-labeled | Acceptable | Limited | Incorrect |
| **Flask Integration** | Fully working system | Minor bugs | Basic | Not working |
| **UI/UX** | Clean & user-friendly | Acceptable | Basic | Poor |
| **Code Quality** | Clean, organized | Minor issues | Messy | Unusable |
| **Presentation** | Clear & engaging | Good | Fair | Poor |
