#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

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

    spatial::FocusTracker tracker;
    std::string cascadePath = "cpp/data/haarcascade_frontalface_default.xml";
    if (!tracker.init(spatial::FocusTracker::Method::HaarCascade, cascadePath))
    {
        std::cout << "[WARN] Could not load " << cascadePath << ", attempting fallback...\n";
        cascadePath = "cpp/data/haarcascade_frontalcatface.xml";
        if (!tracker.init(spatial::FocusTracker::Method::HaarCascade, cascadePath))
        {
            std::cout << "[WARN] Cascade load failed. Falling back to CenterPoint focus.\n";
            tracker.init(spatial::FocusTracker::Method::CenterPoint);
        }
    }

    cv::Mat frame;
    auto prevTime = std::chrono::steady_clock::now();
    double fps = 0.0;

    std::cout << "[INFO] Focus Tracker running. Press 'q' or 'ESC' to exit.\n";

    for (;;)
    {
        if (!cam.read(frame) || frame.empty())
        {
            continue;
        }

        // Run tracking on frame
        spatial::TrackedTarget target = tracker.update(frame);

        // Draw visualization overlays
        tracker.drawDebug(frame, target);

        // Calculate and render FPS
        auto currTime = std::chrono::steady_clock::now();
        double elapsedSec = std::chrono::duration<double>(currTime - prevTime).count();
        prevTime = currTime;
        if (elapsedSec > 0.0)
        {
            fps = 0.9 * fps + 0.1 * (1.0 / elapsedSec);
        }

        std::string fpsText = cv::format("FPS: %.1f (%dx%d)", fps, frame.cols, frame.rows);
        cv::putText(frame, fpsText, cv::Point(20, frame.rows - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

        cv::imshow("Spatial Focus Tracker", frame);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) // 'q' or ESC
        {
            break;
        }
    }

    cam.close();
    cv::destroyAllWindows();
    return 0;
}
