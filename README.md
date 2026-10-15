# EdgeDepth-Lite: Monocular Depth & Synthetic Aperture Prototype

This repository contains the interactive prototype and foundational architecture for **EdgeDepth-Lite** — real-time monocular depth estimation and synthetic aperture (defocus bokeh) simulation.

---

## 1. Project Overview & Architecture

Modern mobile camera hardware is constrained by small sensors and fixed physical apertures, making it optically impossible to achieve shallow depth-of-field (bokeh) naturally. This project demonstrates an end-to-end pipeline that takes a single 2D image, predicts a continuous depth map using **Depth Anything V2**, and renders optical circle-of-confusion (CoC) defocus blur guided by user-selected focal planes and response curves.

```
+------------------+       +-------------------------+       +------------------------------+
| Input Image      | ----> | Depth Anything V2       | ----> | Normalized Relative Disparity|
| (RGB Frame)      |       | (ViT Foundation Model)  |       | Map d in [0.0, 1.0]          |
+------------------+       +-------------------------+       +------------------------------+
                                                                            │
                                                                            ▼
                                                             +------------------------------+
                                                             | Synthetic Bokeh Engine       |
                                                             | - Focus Depth Plane (0.0-1.0)|
                                                             | - Razor-thin DOF (>= 0.000)  |
                                                             | - Blur Response Curves       |
                                                             | - Multi-tier Pyramid Blend   |
                                                             +------------------------------+
                                                                            │
                                                                            ▼
                                                             +------------------------------+
                                                             | Rendered Bokeh Output        |
                                                             +------------------------------+
```

---

## 2. File Structure & Explanation

```
edge-depth-estimation/
├── .md                         # Original project specification for lightweight edge model
├── README.md                   # Comprehensive developer documentation & mobile guide
├── requirements.txt            # Python dependencies (PyTorch, Transformers, OpenCV, Gradio)
├── assets/
│   └── sample_desk.jpg         # Synthetic multi-depth test image for rapid verification
├── scripts/
│   └── create_sample_assets.py # Procedural generation script for multi-depth test scenes
├── src/
│   ├── app.py                  # Interactive Gradio web application with UI sliders
│   ├── bokeh_blur.py           # Optical depth-of-field rendering engine (curves & multi-tier blur)
│   ├── depth_estimator.py      # Depth Anything V2 model wrapper with CUDA acceleration
│   └── run_pipeline.py         # Headless CLI runner for batch image processing
└── tests/
    └── test_bokeh.py           # Unit test suite verifying monotonic blur, curves & DOF isolation
```

### File-by-File Breakdown

* **[`src/depth_estimator.py`](src/depth_estimator.py):**
  Wraps Hugging Face's `AutoModelForDepthEstimation` using `depth-anything/Depth-Anything-V2-Small-hf` (with automatic fallback to `LiheYoung/depth-anything-small-hf`). Detects CUDA automatically, normalizes raw model disparities to `[0.0, 1.0]`, and creates `inferno` colormap representations.
* **[`src/bokeh_blur.py`](src/bokeh_blur.py):**
  The core computer vision rendering engine. Features:
  - **Proportional Scaling:** Distance from the focal plane $|d - d_{\text{focus}}|$ continuously drives blur strength across the entire scene without premature flat cutoffs.
  - **Selectable Response Curves:** Six response profiles (`Aggressive / Square Root`, `Ultra Pop / 4th Root`, `Linear`, `Smoothstep`, `Quadratic`, `Exponential`) to control subject separation.
  - **Fine-Grained DOF Margin:** Supports margins down to `0.000` for microscopic, razor-thin focal planes.
  - **Strict In-Focus Indicator:** Highlights *only* the zero-blur focal plane in green without bleeding into soft regions.
  - **7-Tier Gaussian Pyramid:** Renders and interpolates between 7 blur tiers to avoid stepped quantization.
* **[`src/app.py`](src/app.py):**
  Full interactive web application built with Gradio:
  - Sliders for **Focus Depth** (`0.0 - 1.0`), **Blur Power** (`0.0 - 1.0`), and **DOF Margin** (`0.000 - 0.150`).
  - Dropdown for **Blur Response Curve**.
  - **Click-to-Focus:** Users can click anywhere on the input image to focus instantly on that object's depth plane.
  - **In-Memory Caching:** Automatically caches depth maps so moving sliders updates the image in milliseconds without re-running the neural network.
* **[`src/run_pipeline.py`](src/run_pipeline.py):**
  CLI tool for headless experimentation. Outputs the depth map, bokeh result, and a side-by-side comparison (`[Input | Depth | Bokeh]`).
* **[`tests/test_bokeh.py`](tests/test_bokeh.py):**
  Automated unit test suite testing zero-blur preservation, bound checks, monotonic blur growth with distance, curve comparisons, and razor-thin focus isolation.
* **[`scripts/create_sample_assets.py`](scripts/create_sample_assets.py):**
  Generates synthetic multi-depth scenes containing distinct foreground, midground, and background items.

---

## 3. Frameworks & Technologies Used

* **PyTorch & CUDA:** GPU-accelerated tensor computing and neural network runtime.
* **Hugging Face Transformers:** Model hub and tokenizer/processor ecosystem used to load pretrained Depth Anything V2 weights.
* **OpenCV (`opencv-python-headless`):** Fast multi-scale Gaussian filtering, layer blending, and color conversions.
* **Gradio:** High-performance web UI framework providing responsive sliders, real-time event routing, and image click events.

---

## 4. Mobile Deployment Alternatives (Without Flutter)

If you want to build a mobile app (e.g. Android APK) without using Flutter, here are the three industry-standard paths:

### Path 1: Native Android (Kotlin + CameraX + ONNX Runtime / TFLite) — **Recommended for Max Performance**
* **Language & UI:** Kotlin + Jetpack Compose.
* **Camera Feed:** Android Jetpack **CameraX** (provides zero-copy YUV/RGBA frames directly from the hardware camera).
* **ML Inference Engine:** 
  - **ONNX Runtime Android** (`com.microsoft.onnxruntime:onnxruntime-android`) with NNAPI or Qualcomm QNN execution provider.
  - Alternatively: **TensorFlow Lite (TFLite)** with GPU Delegate.
* **Bokeh Rendering:** OpenGL ES shader or RenderScript/Vulkan computing the multi-tier blur directly on the mobile GPU at 60 FPS.
* **Pros:** Maximum possible framerate, lowest battery consumption, direct hardware camera tuning (AE, AF, manual exposure).

### Path 2: React Native (VisionCamera + `react-native-fast-tflite`) — **Best for Fast JavaScript/TypeScript Mobile Dev**
* **Language & UI:** TypeScript / React Native (without Flutter).
* **Camera Feed:** **React Native VisionCamera** (high-performance camera library for iOS & Android).
* **ML Inference Engine:** **`react-native-fast-tflite`** or **`react-native-onnxruntime`** using Frame Processors written in TypeScript/C++.
* **Pros:** Single codebase for Android & iOS, familiar web/JS ecosystem, no Java/Kotlin boilerplate required.

### Path 3: Progressive Web App (PWA) / ONNX Runtime Web (WebGPU) — **Zero-Install Mobile App**
* **Language & UI:** HTML5 / React / Vue running in Chrome / Safari on mobile.
* **ML Inference Engine:** **ONNX Runtime Web** executing directly on mobile GPU via **WebGPU**.
* **Deployment:** Users open the URL in Chrome on Android and tap "Add to Home Screen". It installs and behaves like a native app without going through the Google Play Store or building an APK.
* **Pros:** Easiest cross-platform deployment, no build toolchains (Gradle, Android Studio), instant updates.

---

## 5. Quickstart & Verification

### One-Click Launch (No manual venv activation needed)
```bash
./run.sh
```

### Manual Usage
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run unit tests
python -m unittest discover tests

# 3. Launch interactive app
python src/app.py

# 4. Or run headless CLI test
python src/run_pipeline.py --image assets/sample_desk.jpg --focus-depth 0.65 --blur-power 0.8
```
