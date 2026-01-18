# **DEVision – Driver Eye & Gaze Estimation using Multi-Modal Deep Learning**

DEVision is a multi-modal deep learning framework for **driver gaze estimation**, combining **driver images, face crops, eye images, and facial landmarks** using **PyTorch Lightning**.

The system predicts:
- **Gaze angles** (yaw, pitch)
- **Gaze location** (x, y) on the dashboard screen

This architecture is inspired by modern **Driver Monitoring Systems (DMS)** and research works such as **DGAZE / I-DGAZE–style pipelines**.

---

## 📂 Dataset

**Dataset Source (Google Drive):**  
👉 https://drive.google.com/drive/folders/10U3v5Jw78Px771UUStnWpke3AH99VoP8

The dataset consists of:
- **HDF5 file (`.h5`)** containing image tensors
- **CSV file** containing gaze labels and facial landmarks

---

### Dataset Contents

#### 🔹 HDF5 Keys
| Key | Description | Shape |
|----|------------|------|
| `img` | Driver image | `(C, H, W)` |
| `face` | Face crop | `(C, H, W)` |
| `leye` | Left eye crop | `(C, H, W)` |
| `reye` | Right eye crop | `(C, H, W)` |

#### 🔹 CSV Labels
- **Gaze Location**
  - `dash gaze x [px]`
  - `dash gaze y [px]`
- **Gaze Angles**
  - `azimuth [deg]`
  - `elevation [deg]`
- **Facial Landmarks (19 values)**
    - face x, face y,
    - leye x, leye y,
    - reye x, reye y,
    - leye x mark, leye y mark,
    - reye x mark, reye y mark,
    - nose x mark, nose y mark,
    - lmouth x mark, lmouth y mark,
    - rmouth x mark, rmouth y mark,
    - yaw_new, pitch_new, roll_new

---

## 🧠 DEVision Architecture

### Multi-Modal Feature Extraction

| Modality | Network | Output Dim |
|---------|--------|------------|
| Face Image | ResNet-18 | 512 |
| Driver Image | ResNet-18 | 512 |
| Left Eye | SmallCNN | 64 |
| Right Eye | SmallCNN | 64 |
| Landmarks | MLP | 32 |

**Total concatenated features:** `1184`

These features are fused and passed through a shared fully-connected layer followed by two prediction heads.

---

### 🔸 Output Heads
- **Gaze Angle Head** → `(yaw, pitch)`
- **Gaze Location Head** → `(x, y)` in screen pixels

---

## 📁 Project Structure

```bash
DEVision/
├── datamodule.py # Lightning DataModule
├── model.py # DEVision network + LightningModule
├── train.py # Training entry point
├── README.md
└── data/
├── devsion_data.h5
└── labels.csv

```

---

## ⚙️ Installation

```bash
pip install torch torchvision pytorch-lightning h5py pandas numpy
```

---

## Optional GPU support

```bash

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

```
---

# 🔄 Data Pipeline (DEVision)
## 1️⃣ DashGazeDataset

Loads image tensors from HDF5

Loads gaze labels & landmarks from CSV

Outputs raw numpy arrays

## 2️⃣ TransformSubset

Applies augmentation only on face images

Converts numpy → PyTorch tensors

Keeps driver & eye images unchanged

## 3️⃣ DashGazeDataModule

Dataset split:

80% Train

10% Validation

10% Test

Optimized multi-worker dataloading

---

# 🧪 Data Augmentation Strategy

Training (Face only):

Color jitter

Random resized crop → 224 × 224

Validation / Test:

Resize → 224 × 224

Driver and eye images are used in their original resolution.

---

# 🧩 Model Components
## 🔹 SmallCNN (Eye Branch)

Lightweight CNN designed for small eye crops:

Conv → ReLU → MaxPool (×3)
Flatten → Linear → ReLU

## 🔹 Landmark MLP

Processes 19 facial landmark values using a 2-layer MLP.

---

# 📉 Loss Function

Total Loss

L = MSE(gaze_angle) + λ · MSE(gaze_location)


Where:

λ = 1.0

---

# 📊 Evaluation Metrics
Angular Error (Degrees)

Converts (yaw, pitch) to 3D gaze vectors

Computes mean angular deviation

Location Error (Pixels)

Euclidean distance on the screen plane

Normalized Location Error
pixel_error / screen_width


Used for cross-device evaluation.

---

# 🚀 Training DEVision
Example Training Script
from pytorch_lightning import Trainer
from datamodule import DashGazeDataModule
from model import GazeEstimationLightningModule

datamodule = DashGazeDataModule(
    hdf5_path="data/devsion_data.h5",
    csv_path="data/labels.csv",
    batch_size=32
)

model = GazeEstimationLightningModule(
    screen_width=1920,
    screen_height=1080
)

trainer = Trainer(
    max_epochs=30,
    accelerator="gpu",
    devices=1,
    precision=16
)

trainer.fit(model, datamodule)
trainer.test(model, datamodule)

---

# 📈 Logged Metrics

train_total_loss

val_total_loss

val_angular_error (°)

val_location_error (px)

val_norm_location_error

---

# 🧠 Research Significance

Multi-modal fusion improves robustness

Landmark integration stabilizes gaze under occlusion

Normalized error enables real-world benchmarking

---

# 🚧 Future Work

Temporal modeling (LSTM / Transformer)

Lightweight eye-only inference mode

ONNX / TensorRT deployment

Personalized calibration module

---

# 📜 License

This project is intended for research and academic use only.
Dataset rights belong to the original dataset authors.


---


