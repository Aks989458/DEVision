DashGazeNet – Driver Gaze Estimation using Multi-Modal Deep Learning

This repository implements a multi-modal deep learning framework for driver gaze estimation, combining face images, driver images, eye crops, and facial landmarks using PyTorch Lightning.

The project is inspired by dashboard/driver gaze estimation pipelines (similar to DGAZE / I-DGAZE style architectures).

📂 Dataset

Dataset Source (Google Drive):
👉 https://drive.google.com/drive/folders/10U3v5Jw78Px771UUStnWpke3AH99VoP8

The dataset consists of:

HDF5 file (.h5) containing image tensors

CSV file containing gaze labels and facial landmarks

Dataset Contents

HDF5 Keys

img → Driver image (C, H, W)

face → Face crop (C, H, W)

leye → Left eye crop (C, H, W)

reye → Right eye crop (C, H, W)

CSV Columns

Gaze location:

dash gaze x [px]

dash gaze y [px]

Gaze angles:

azimuth [deg]

elevation [deg]

Facial landmarks (19 values):

face x, face y,
leye x, leye y,
reye x, reye y,
leye x mark, leye y mark,
reye x mark, reye y mark,
nose x mark, nose y mark,
lmouth x mark, lmouth y mark,
rmouth x mark, rmouth y mark,
yaw_new, pitch_new, roll_new

🧠 Model Overview
DashGazeNetMini Architecture
Modality	Network	Output Features
Face Image	ResNet-18	512
Driver Image	ResNet-18	512
Left Eye	Small CNN	64
Right Eye	Small CNN	64
Landmarks	MLP	32

Total concatenated features: 1184

These features are passed through a shared fully-connected layer and split into two prediction heads:

Gaze Angle Head: (yaw, pitch)

Gaze Location Head: (x, y) in pixels

🧩 Project Structure
.
├── datamodule.py        # PyTorch Lightning DataModule
├── model.py             # Model + LightningModule
├── train.py             # Training script (recommended)
├── README.md
└── data/
    ├── dashgaze.h5
    └── labels.csv

⚙️ Installation
pip install torch torchvision pytorch-lightning h5py pandas numpy


(Optional GPU support)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

🔄 Data Pipeline (Step-by-Step)
1. DashGazeDataset

Reads images from HDF5

Reads labels + landmarks from CSV

Returns a dictionary of raw numpy arrays

2. TransformSubset

Applies data augmentation only on face images

Converts numpy arrays → PyTorch tensors

Keeps eye & driver images unaltered (as per dataset geometry)

3. DashGazeDataModule

Splits dataset:

80% Train

10% Validation

10% Test

Uses optimized CPU workers

Supports Lightning-style dataloaders

🧪 Data Augmentation Strategy

Training (Face only):

Color jitter

Random resized crop → 224×224

Validation/Test (Face only):

Resize → 224×224

Driver and eye images are passed without augmentation.

🧠 Model Components Explained
🔹 SmallCNN (Eyes)

A lightweight CNN optimized for small eye crops:

Conv → ReLU → MaxPool (×3)
Flatten → Linear → ReLU

🔹 MLPLandmark

Processes 19 landmark values using a 2-layer MLP.

📉 Loss Functions

Total Loss

L = MSE(gaze_angle) + λ · MSE(gaze_location)


λ = 1.0

📊 Evaluation Metrics
1. Angular Error (Degrees)

Converts (yaw, pitch) → 3D vectors

Computes mean angular deviation

2. Location Error (Pixels)

Euclidean distance between predicted and GT screen coordinates

3. Normalized Location Error
pixel_error / screen_width


Useful for cross-screen comparisons.

🚀 Training Steps
Example Training Script
from pytorch_lightning import Trainer
from datamodule import DashGazeDataModule
from model import GazeEstimationLightningModule

datamodule = DashGazeDataModule(
    hdf5_path="data/dashgaze.h5",
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

📈 Logged Metrics (Lightning)

train_total_loss

val_total_loss

val_angular_error (deg)

val_location_error (px)

val_norm_location_error

🧠 Research Notes

Multi-modal fusion improves robustness over single-stream gaze models

Landmark integration stabilizes gaze prediction under occlusions

Normalized pixel error is critical for real-world evaluation

📌 Future Improvements

Temporal modeling (LSTM / Transformer)

Eye-only lightweight deployment model

Screen-aware calibration layer

ONNX / TensorRT export

🧾 License

This project is intended for research and academic use.
Dataset ownership belongs to the original dataset providers.

If you want, I can also:

✅ Convert this into a paper-style Methodology section

✅ Add training curves + ablation table

✅ Prepare a conference-ready experiment section

✅ Create a deployment README (ONNX / TorchScript)

change the project name to DEVision

Here is the updated README with the project name changed to DEVision everywhere.
You can directly replace your existing README.md with this.

DEVision – Driver Eye & Gaze Estimation using Multi-Modal Deep Learning

DEVision is a multi-modal deep learning framework for driver gaze estimation, combining driver images, face crops, eye images, and facial landmarks using PyTorch Lightning.

The system predicts both:

Gaze angles (yaw, pitch)

Gaze location (x, y) on the dashboard screen

This architecture is inspired by modern driver monitoring systems and research works such as DGAZE / I-DGAZE–style pipelines.

📂 Dataset

Dataset Source (Google Drive):
👉 https://drive.google.com/drive/folders/10U3v5Jw78Px771UUStnWpke3AH99VoP8

The dataset consists of:

HDF5 file (.h5) → image tensors

CSV file → gaze labels + facial landmarks

Dataset Contents
HDF5 Keys
Key	Description	Shape
img	Driver image	(C, H, W)
face	Face crop	(C, H, W)
leye	Left eye	(C, H, W)
reye	Right eye	(C, H, W)
CSV Labels

Gaze Location

dash gaze x [px]

dash gaze y [px]

Gaze Angles

azimuth [deg]

elevation [deg]

Facial Landmarks (19 values)

face x, face y,
leye x, leye y,
reye x, reye y,
leye x mark, leye y mark,
reye x mark, reye y mark,
nose x mark, nose y mark,
lmouth x mark, lmouth y mark,
rmouth x mark, rmouth y mark,
yaw_new, pitch_new, roll_new

🧠 DEVision Architecture
Multi-Modal Feature Extraction
Modality	Network	Output Dim
Face Image	ResNet-18	512
Driver Image	ResNet-18	512
Left Eye	SmallCNN	64
Right Eye	SmallCNN	64
Landmarks	MLP	32

Total feature vector: 1184

These features are fused and passed through a shared fully-connected layer, followed by two prediction heads.

Output Heads

Gaze Angle Head → (yaw, pitch)

Gaze Location Head → (x, y) in screen pixels

📁 Project Structure
DEVision/
├── datamodule.py        # Lightning DataModule
├── model.py             # DEVision network + LightningModule
├── train.py             # Training entry point
├── README.md
└── data/
    ├── devsion_data.h5
    └── labels.csv

⚙️ Installation
pip install torch torchvision pytorch-lightning h5py pandas numpy


(Optional GPU support)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

🔄 Data Pipeline (DEVision)
1️⃣ DashGazeDataset

Loads image tensors from HDF5

Loads gaze + landmarks from CSV

Outputs raw numpy arrays

2️⃣ TransformSubset

Applies augmentation only on face images

Converts numpy → PyTorch tensors

Keeps driver & eye images unchanged

3️⃣ DashGazeDataModule

Dataset split:

80% Train

10% Validation

10% Test

Optimized multi-worker dataloading

🧪 Data Augmentation Strategy

Training (Face only):

Color jitter

Random resized crop → 224 × 224

Validation / Test:

Resize → 224 × 224

Driver and eye images are used in their original resolution.

🧩 Model Components
🔹 SmallCNN (Eye Branch)

Lightweight CNN designed for small eye crops:

Conv → ReLU → MaxPool (×3)
Flatten → Linear → ReLU

🔹 Landmark MLP

Processes 19 facial landmark values using a 2-layer MLP.

📉 Loss Function

Total Loss

L = MSE(gaze_angle) + λ · MSE(gaze_location)


Where:

λ = 1.0

📊 Evaluation Metrics
Angular Error (Degrees)

Converts (yaw, pitch) to 3D gaze vectors

Computes mean angular deviation

Location Error (Pixels)

Euclidean distance on the screen plane

Normalized Location Error
pixel_error / screen_width


Used for cross-device evaluation.

🚀 Training DEVision
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

📈 Logged Metrics

train_total_loss

val_total_loss

val_angular_error (°)

val_location_error (px)

val_norm_location_error

🧠 Research Significance

Multi-modal fusion improves robustness

Landmarks stabilize gaze under occlusion

Normalized error enables real-world benchmarking

🚧 Future Work

Temporal modeling (LSTM / Transformer)

Lightweight eye-only inference mode

ONNX / TensorRT deployment

Personal calibration module

📜 License

This project is intended for research and academic use only.
Dataset rights belong to the original dataset authors.
