# Fruit Detection System — YOLOv8

A real-time fruit detection system built with **YOLOv8**, trained on a custom-merged dataset of 12 fruit classes. The system supports both **live webcam inference** and **offline video processing**, with bounding boxes, confidence scores, real-time FPS overlay, and object-counting feature.

## 🗂️ Dataset

The dataset was built on **Roboflow** by merging two source datasets into a single custom fruit-detection dataset, then annotated and augmented for training.

| Metric | Value |
|---|---|
| Total images | 10,300 |
| Train set | 8,482 images (82%) |
| Validation set | 909 images (9%) |
| Test set | 909 images (9%) |
| Classes | 12 |


### Preprocessing
- **Resize:** Stretch to `640x640`

### Augmentations
- Outputs per training example: `2`
- Rotation: between `-15°` and `+15°`
- Blur: up to `2.5px`

**Dataset link (Roboflow Universe):** [new-fruits-wgdh2 v3](https://universe.roboflow.com/omar-husnye/new-fruits-wgdh2/model/3)

> ⚠️ The raw dataset is **not** included in this repository — use the link above to access it via Roboflow.

---

## 🧠 Model & Training

- **Architecture:** YOLOv8n (`yolov8n.pt`)
- **Framework:** Ultralytics
- **Training environment:** Kaggle Notebook, GPU (T4)
- **Export format:** YOLOv8 PyTorch (from Roboflow)
- **Model size:** 73 layers, 11,130,228 parameters, 28.5 GFLOPs

### Overall Validation Metrics

| Metric | Score |
|---|---|
| Precision (B) | 0.760 |
| Recall (B) | 0.765 |
| mAP@50 | 0.779 |
| mAP@50-95 | 0.501 |

### Per-Class Performance

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| apple | 16 | 39 | 0.774 | 1.000 | 0.952 | 0.835 |
| banana | 128 | 227 | 0.534 | 0.549 | 0.486 | 0.198 |
| cherry | 10 | 16 | 0.543 | 0.750 | 0.691 | 0.526 |
| grape | 11 | 21 | 0.799 | 0.714 | 0.824 | 0.467 |
| kiwi | 148 | 632 | 0.801 | 0.706 | 0.740 | 0.284 |
| melon | 26 | 43 | 0.909 | 0.953 | 0.978 | 0.845 |
| orange | 24 | 99 | 0.913 | 0.919 | 0.935 | 0.768 |
| pineapple | 150 | 273 | 0.685 | 0.565 | 0.656 | 0.234 |
| pomegranate | 14 | 18 | 0.647 | 0.944 | 0.726 | 0.508 |
| rambutan | 135 | 467 | 0.770 | 0.294 | 0.525 | 0.153 |
| strawberry | 156 | 274 | 0.877 | 0.825 | 0.875 | 0.384 |
| watermelon | 19 | 26 | 0.874 | 0.962 | 0.963 | 0.810 |

**Inference speed:** 0.2ms preprocess · 4.4ms inference · 2.2ms postprocess (per image)

**Roboflow model dashboard:** [View Model](https://app.roboflow.com/omar-husnye/new-fruits-wgdh2/3)

---

## 📁 Project Structure

```
Fruits Detection System/
├── main.py                 # Inference script (camera + video modes)
├── best.pt                 # Trained YOLOv8 weights
├── requirements.txt        # Python dependencies
├── assets/results          # Demo images, metrics, confusion matrix
└── README.md
```

---

## ⚙️ Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- `ultralytics`
- `opencv-python`

---

## 🚀 Usage

Make sure `best.pt` is in the same folder as `main.py` (or pass a custom path with `--weights`).

### Live Camera Mode
```bash
python main.py --mode camera --weights best.pt
```
Opens your webcam and runs real-time detection with bounding boxes, labels, confidence scores, and FPS overlay. Press `q` to quit.

### Video File Mode
```bash
python main.py --mode video --weights best.pt --source input.mp4 --output result.mp4
```
Processes an offline video frame-by-frame and saves the annotated result.

### Object Counter
Add `--count-class <fruit_name>` to either mode to count and print every detection of a specific class in real time:

```bash
python main.py --mode video --weights best.pt --source input.mp4 --output result.mp4 --count-class strawberry
```

### All Arguments

| Flag | Description | Default |
|---|---|---|
| `--mode` | `camera` or `video` (required) | — |
| `--weights` | Path to trained weights | `best.pt` |
| `--source` | Path to input video (video mode) | — |
| `--camera-id` | Webcam device index | `0` |
| `--output` | Path to save annotated video | `output_annotated.mp4` |
| `--conf` | Confidence threshold | `0.4` |
| `--count-class` | Class name to count (bonus feature) | — |

---

## 👤 Author

**Omar Hosni**
