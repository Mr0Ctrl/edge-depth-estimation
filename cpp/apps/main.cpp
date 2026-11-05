#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <memory>

#include "spatial/camera.hpp"
#include "spatial/tracker.hpp"

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    spatial::Camera cam;
    if (!cam.open(0, 1280, 720))
    {
        std::cerr << "[ERROR] Could not open camera device 0 at 1280x720.\n";
        return -1;
    }

    // Tracker Selection:
    // Change to spatial::CenterTracker() if you want to test raw camera FPS!
    std::string protoPath = "cpp/data/deploy.prototxt";
    std::string modelPath = "cpp/data/res10_300x300_ssd_iter_140000.caffemodel";
    std::unique_ptr<spatial::ITracker> tracker = std::make_unique<spatial::DnnFaceTracker>(
        protoPath,
        modelPath,
        0.55f // confidence threshold
    );

    if (!tracker->init())
    {
        std::cout << "[WARN] Failed to load DNN model. Falling back to Haar Cascade.\n";
        tracker = std::make_unique<spatial::HaarFaceTracker>("cpp/data/haarcascade_frontalface_default.xml");
        if (!tracker->init())
        {
            std::cout << "[WARN] Haar cascade failed. Falling back to CenterPoint.\n";
            tracker = std::make_unique<spatial::CenterTracker>();
            tracker->init();
        }
    }
    else
    {
        std::cout << "[INFO] Successfully initialized Deep Neural Network tracker (SSD-ResNet10).\n";
    }

    cv::Mat frame;
    double fps = 0.0;
    auto prevTotalTime = std::chrono::steady_clock::now();

    // Latency averages (in milliseconds)
    double avgCamMs = 0.0;
    double avgTrackMs = 0.0;
    double avgGuiMs = 0.0;

    std::cout << "[INFO] Spatial Focus Tracker running. Press 'q' or 'ESC' to exit.\n";

    for (;;)
    {
        // 1. Measure Camera Read Time
        auto tStartCam = std::chrono::steady_clock::now();
        if (!cam.read(frame) || frame.empty())
        {
            continue;
        }
        auto tEndCam = std::chrono::steady_clock::now();
        double camMs = std::chrono::duration<double, std::milli>(tEndCam - tStartCam).count();

        // 2. Measure Tracking / AI Inference Time
        auto tStartTrack = std::chrono::steady_clock::now();
        spatial::TrackedTarget target = tracker->update(frame);
        auto tEndTrack = std::chrono::steady_clock::now();
        double trackMs = std::chrono::duration<double, std::milli>(tEndTrack - tStartTrack).count();

        // 3. Draw Debug Overlays & Diagnostics
        tracker->drawDebug(frame, target);

        // Exponential moving average for smooth readout
        avgCamMs = 0.9 * avgCamMs + 0.1 * camMs;
        avgTrackMs = 0.9 * avgTrackMs + 0.1 * trackMs;

        // Calculate total frame rate
        auto currTotalTime = std::chrono::steady_clock::now();
        double totalElapsedSec = std::chrono::duration<double>(currTotalTime - prevTotalTime).count();
        prevTotalTime = currTotalTime;
        if (totalElapsedSec > 0.0)
        {
            fps = 0.9 * fps + 0.1 * (1.0 / totalElapsedSec);
        }

        // Render Latency HUD onto frame
        std::string hudTop = cv::format("FPS: %.1f (%dx%d)", fps, frame.cols, frame.rows);
        std::string hudTiming = cv::format("Cam: %.1fms | AI: %.1fms | GUI: %.1fms", avgCamMs, avgTrackMs, avgGuiMs);

        cv::putText(frame, hudTop, cv::Point(20, frame.rows - 45),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::putText(frame, hudTiming, cv::Point(20, frame.rows - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        // 4. Measure GUI / Render Time
        auto tStartGui = std::chrono::steady_clock::now();
        cv::imshow("Spatial Profiler", frame);
        int key = cv::waitKey(1);
        auto tEndGui = std::chrono::steady_clock::now();
        double guiMs = std::chrono::duration<double, std::milli>(tEndGui - tStartGui).count();
        avgGuiMs = 0.9 * avgGuiMs + 0.1 * guiMs;

        if (key == 'q' || key == 'Q' || key == 27)
        {
            break;
        }
    }

    cam.close();
    cv::destroyAllWindows();
    return 0;
}
