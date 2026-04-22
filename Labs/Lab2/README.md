# 🧠 Machine Learning Assignment 2
## 🔍 Interpretability Analysis using CAM (TorchCAM)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Overview

This project explores **interpretability in Convolutional Neural Networks (CNNs)** using **Class Activation Maps (CAM)**.

The goal is to understand **how a pre-trained model makes decisions** by visualizing the most important regions in input images.

I use:
- 🧠 Pre-trained **ResNet (ImageNet)**
- 🔥 **TorchCAM (LayerCAM)**
- 🖼️ Multiple image categories (dog, cat, car)
- ❌ Negative examples
- 🧠 Multi-layer analysis (**VG requirement**)
- ❓ Unknown object analysis

---

## 🧠 Method

For each image:

- Predict **Top-5 classes**
- Generate **CAM heatmap**
- Overlay CAM on original image

### Multi-layer analysis:
- `layer1` → low-level features (edges, textures)
- `layer2` → simple patterns
- `layer3` → object structure
- `layer4` → semantic understanding

---

## 📊 Results

### 🐶 Dog (Positive Example)

![Dog CAM](results/dog_positive_cam_layer4.png)

The model focuses strongly on the **face and upper body**, indicating semantic understanding.

---

### 🐱 Cat (Positive Example)

![Cat CAM](results/cat_positive_cam_layer4.png)

Activation highlights **eyes and ears**, which are key identifying features.

---

### 🚗 Car (Positive Example)

![Car CAM](results/car_positive_cam_layer4.png)

Focus is on **wheels and body structure**, showing object-part awareness.

---

### ❌ Negative Example

![Dog Negative](results/dog_negative_cam_layer4.png)

Activation is **scattered and less meaningful**, indicating uncertainty.

---

### 🧠 Multi-layer Analysis

#### Dog

![Dog Multi-layer](results/dog_positive_multilayer_cam.png)

#### Car

![Car Multi-layer](results/car_positive_multilayer_cam.png)

#### Cat

![Cat Multi-layer](results/cat_positive_multilayer_cam.png)

**Observation:**
- Early layers → edges & textures
- Middle layers → shapes
- Final layer → semantic regions

👉 The network builds understanding progressively.

---

### ❓ Unknown Object

![Unknown](results/unknown_object_cam_layer4.png)

The model fails to classify correctly.

Top predictions:
- slot
- analog_clock
- jigsaw_puzzle

All with **low confidence (~5%)**

👉 The model relies on **visual similarity**, not real understanding.

---

## 🧪 Key Findings

- CNNs rely on **patterns, not true understanding**
- CAM helps explain **model decisions**
- Deep layers capture **semantic meaning**
- Models struggle with **out-of-distribution data**

---

## ⚙️ Setup

Using **uv**:

```bash
uv venv
source .venv/Scripts/activate # Windows
uv pip install -r Labs/Lab2/requirements.txt

---

## ⚙️ Run

python Labs/Lab2/lab2_cam_analysis.py