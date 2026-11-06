#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <memory>

#include "spatial/camera.hpp"
#include "spatial/tracker.hpp"
#include "spatial/profiler.hpp"

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    // 1. Hardware Initialization (Threaded Camera)
    spatial::Camera cam;
    if (!cam.open(0, 1280, 720))
    {
        std::cerr << "[ERROR] Could not open camera device 0 at 1280x720.\n";
        return -1;
    }

    // 2. Tracking Strategy Selection (Polymorphic ITracker)
    std::string protoPath = "cpp/data/deploy.prototxt";
    std::string modelPath = "cpp/data/res10_300x300_ssd_iter_140000.caffemodel";
    std::unique_ptr<spatial::ITracker> tracker = std::make_unique<spatial::DnnFaceTracker>(
        protoPath,
        modelPath,
        0.55f // confidence threshold
    );

    if (!tracker->init())
    {
        std::cout << "[WARN] Failed to load DNN model. Falling back to CenterPoint.\n";
        tracker = std::make_unique<spatial::CenterTracker>();
        tracker->init();
    }
    else
    {
        std::cout << "[INFO] Successfully initialized Deep Neural Network tracker.\n";
    }

    // 3. Diagnostic Profiler
    spatial::Profiler profiler;
    cv::Mat frame;

    std::cout << "[INFO] Spatial Engine running. Press 'q' or 'ESC' to exit.\n";

    // 4. Main Event Loop
    for (;;)
    {
        // A. Instant Frame Acquisition (from background thread buffer)
        auto tStartCam = std::chrono::steady_clock::now();
        if (!cam.read(frame) || frame.empty())
        {
            continue;
        }
        auto tEndCam = std::chrono::steady_clock::now();
        double camMs = std::chrono::duration<double, std::milli>(tEndCam - tStartCam).count();

        // B. Target Tracking & Focus Selection
        auto tStartTrack = std::chrono::steady_clock::now();
        spatial::TrackedTarget target = tracker->update(frame);
        auto tEndTrack = std::chrono::steady_clock::now();
        double trackMs = std::chrono::duration<double, std::milli>(tEndTrack - tStartTrack).count();

        // C. Visual Overlays
        tracker->drawDebug(frame, target);

        // D. Profiling Tick & Diagnostics HUD
        profiler.tick(camMs, trackMs, 0.0);
        profiler.drawOverlay(frame);

        // E. Screen Display & User Input
        auto tStartGui = std::chrono::steady_clock::now();
        cv::imshow("Spatial Vision Engine", frame);
        int key = cv::waitKey(1);
        auto tEndGui = std::chrono::steady_clock::now();
        double guiMs = std::chrono::duration<double, std::milli>(tEndGui - tStartGui).count();

        // Update GUI latency slice in profiler
        profiler.tick(0.0, 0.0, guiMs);

        if (key == 'q' || key == 'Q' || key == 27)
        {
            break;
        }
    }

    cam.close();
    cv::destroyAllWindows();
    return 0;
}
