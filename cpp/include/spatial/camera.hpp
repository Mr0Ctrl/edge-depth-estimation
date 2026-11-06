#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

namespace spatial {

/**
 * @brief Threaded, asynchronous camera hardware capture engine.
 * 
 * Uses a dedicated background thread to ingest hardware frames continuously
 * via Linux V4L2/MJPG, decoupling camera sensor exposure time from downstream
 * processing pipelines (AI tracking, depth estimation, UI rendering).
 */
class Camera {
public:
    Camera();
    ~Camera();

    // Prevent accidental copying (threads and hardware streams are non-copyable)
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    /**
     * @brief Opens camera hardware and starts the background capture thread.
     * @param deviceIndex Index of webcam (/dev/videoX).
     * @param width Desired stream width.
     * @param height Desired stream height.
     * @return true if hardware opened and capture thread started.
     */
    bool open(int deviceIndex = 0, int width = 1280, int height = 720);

    /**
     * @brief Opens a video file stream.
     */
    bool openFile(const std::string& filepath);

    /**
     * @brief Retrieves the latest available frame from the background buffer (non-blocking).
     * @param[out] frame Destination OpenCV Mat.
     * @return true if a valid frame was copied, false if stream ended or stopped.
     */
    bool read(cv::Mat& frame);

    /**
     * @brief Checks if camera stream and capture thread are active.
     */
    bool isOpened() const;

    /**
     * @brief Stops capture thread and releases camera hardware.
     */
    void close();

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }

private:
    void captureWorker();

    cv::VideoCapture m_cap;
    int m_width{0};
    int m_height{0};

    // Threading & Synchronization
    std::thread m_thread;
    std::mutex m_mutex;
    std::atomic<bool> m_running{false};
    cv::Mat m_latestFrame;
    bool m_hasNewFrame{false};
};

} // namespace spatial
