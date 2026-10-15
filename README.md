# EdgeDepth-Lite: Depth Anything Synthetic Aperture Prototype

This repository contains a fast interactive prototype for **EdgeDepth-Lite** (monocular depth estimation and depth-of-field / synthetic aperture blur simulation).

---

## 1. Project Background
Mobile devices and low-cost robotic platforms have fixed-aperture lenses that cannot produce true physical depth of field (optical bokeh). The long-term goal of **EdgeDepth-Lite** is to train a parameter-efficient, fully convolutional model (MobileNetV3 + ASPP) capable of real-time depth estimation (<20 ms) on edge accelerators.

As a baseline exploration and proof-of-concept, this **prototype** demonstrates the end-to-end pipeline using **Depth Anything** (ViT-based foundation model) and a depth-guided synthetic aperture rendering engine with interactive UI sliders.

---

## 2. Prototype Features
- **Monocular Depth Estimation:** Powered by `Depth Anything V2` / `Depth Anything` foundation models.
- **Interactive Focus Control:**
  - **Focus Depth Slider:** Select the depth plane to keep sharply in focus.
  - **Blur Power Slider:** Dynamically scale the intensity of the out-of-focus background/foreground bokeh.
  - **Depth of Field Width Slider:** Adjust the thickness of the in-focus band.
- **Web UI & Standalone Scripts:** Interactive browser interface powered by Gradio alongside modular Python libraries.

---

## 3. Quickstart

### Virtual Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Launch Interactive Application
```bash
python src/app.py
```
Open the provided local URL in your browser to interact with the sliders.

---

## 4. Repository Structure
```
edge-depth-estimation/
├── assets/                  # Sample test images
├── src/
│   ├── app.py              # Interactive UI application (sliders & preview)
│   ├── depth_estimator.py  # Depth Anything model wrapper
│   └── bokeh_blur.py       # Depth-guided synthetic aperture blur engine
├── tests/                  # Verification scripts
├── .gitignore
├── README.md
└── requirements.txt
```
