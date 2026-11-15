#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

#include "spatial/camera.hpp"
#include "spatial/tracker.hpp"
#include "spatial/profiler.hpp"
#include "spatial/logger.hpp"

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    spatial::Logger::instance().setLevel(spatial::LogLevel::Debug);

    // 1. Hardware Initialization (Threaded Camera with Fresh-Frame Sync)
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

    spatial::Profiler profiler;
    cv::Mat frame;
    spatial::TrackedTarget target;

    std::cout << "[INFO] Spatial Engine running. Press 'q' or 'ESC' to exit.\n";

    // 3. Main Event Loop
    for (;;)
    {
        // A. True Hardware Sync: Blocks until the physical sensor delivers a genuinely NEW frame
        {
            spatial::ScopedTimer timmer(profiler, "CameraMs");
            if (!cam.read(frame) || frame.empty())
            {
                continue;
            }
        }

        // B. Target Tracking & Focus Selection
        {
            spatial::ScopedTimer timmer(profiler, "TargetMs");
            target = tracker->update(frame);
        }

        // C. Visual Overlays

        tracker->drawDebug(frame, target);

        { // D. Screen Display & User Input
            spatial::ScopedTimer timmer(profiler, "RenderMs");

            cv::imshow("Spatial Vision Engine", frame);
            int key = cv::waitKey(1);

            if (key == 'q' || key == 'Q' || key == 27)
            {
                break;
            }
        }

        // E. Single Profiling Tick per True Frame
        profiler.tickFrame();
        auto summary = profiler.summary();
        SPATIAL_LOG_INFO(summary);
    }

    cam.close();
    cv::destroyAllWindows();
    return 0;
}
