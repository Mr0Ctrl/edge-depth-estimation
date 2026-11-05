# Edge Depth Estimation & Spatial Vision Engine

A lightweight, real-time C++ engine for edge spatial vision, integrating camera capture, neural face tracking, monocular depth estimation, synthetic aperture bokeh filtering, and 3D Gaussian Splatting.

---

## Quick Start (Build & Run)

```bash
# 1. Configure and build (using all CPU cores)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. Run the real-time application
./build/bin/main
```

Press **`q`** or **`ESC`** in the application window to exit cleanly.

---

## Project Structure

```text
edge-depth-estimation/
├── CMakeLists.txt              # Root CMake configuration
├── cpp/
│   ├── CMakeLists.txt          # Library target (spatial_core) + executable (main)
│   ├── include/spatial/        # Public API headers
│   │   ├── camera.hpp          # Hardware camera capture abstraction
│   │   ├── tracker.hpp         # ITracker polymorphic tracking interface
│   │   ├── depth_engine.hpp    # Monocular depth inference engine
│   │   ├── bokeh_filter.hpp    # Synthetic aperture bokeh filter
│   │   └── gaussian_splat.hpp  # 3D Gaussian splat generation & export
│   ├── src/                    # Implementations
│   │   ├── camera.cpp          # Linux V4L2 backend with MJPG decode & buffer depth 1
│   │   └── tracker.cpp         # SSD-ResNet10 deep learning & fallback trackers
│   ├── apps/
│   │   └── main.cpp            # Main loop & real-time latency profiler HUD
│   └── data/                   # Neural network weights and model definitions
│       ├── deploy.prototxt     # SSD-ResNet10 model structure
│       └── res10_300x300...    # SSD-ResNet10 trained weights (11MB)
└── prototype/                  # Python reference prototype (on prototype branch)
```

---

## Core Concepts & Design Patterns

### 1. Mechanism vs. Policy
- **Classes (`Camera`, `Tracker`)**: Act as *mechanisms*. They perform hardware communication and mathematical processing, returning success flags and data structures without terminating the application or hardcoding UI popups.
- **Application (`main.cpp`)**: Acts as *policy*. It chooses algorithms, monitors latencies, handles fallbacks, and manages rendering.

### 2. Polymorphic Strategy Pattern
All tracking algorithms implement the abstract [`spatial::ITracker`](cpp/include/spatial/tracker.hpp) interface:
- **`DnnFaceTracker`**: Deep learning SSD-ResNet10 face detector (~15ms on CPU).
- **`HaarFaceTracker`**: Classical Viola-Jones fallback with downscaled 2x grayscale buffers.
- **`CenterTracker`**: Zero-overhead fallback targeting the center of the frame for camera benchmarking.

---

## Real-Time Latency Profiler HUD

The application displays a real-time diagnostic overlay:
```text
FPS: 22.0 (1280x720)
Cam: 45.2ms | AI: 14.1ms | GUI: 1.2ms
```
- **`Cam`**: Time spent waiting for the camera sensor to expose and transfer a frame over USB.
- **`AI`**: Execution time of the neural network forward pass on CPU.
- **`GUI`**: Window draw and event poll latency.

---

## Complete Documentation & Roadmap

For in-depth architectural details, performance optimization strategies (such as multi-threaded capture decoupling), future AI model integration guidelines, and the phase-by-phase development task list, see:
- [`architecture_and_roadmap.md`](architecture_and_roadmap.md)
