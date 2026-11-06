#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

namespace spatial {

/**
 * @brief Threaded, asynchronous camera hardware capture engine.
 * 
 * Uses a producer-consumer model with std::condition_variable to synchronize
 * the consumer to the camera's true hardware cadence without busy-spinning.
 */
class Camera {
public:
    Camera();
    ~Camera();

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    /**
     * @brief Opens camera hardware and starts the capture worker thread.
     */
    bool open(int deviceIndex = 0, int width = 1280, int height = 720);

    /**
     * @brief Opens a video file stream.
     */
    bool openFile(const std::string& filepath);

    /**
     * @brief Waits for the next FRESH hardware frame and copies it (true hardware pacing).
     * @param[out] frame Destination OpenCV Mat.
     * @param timeoutMs Maximum wait time before returning false.
     * @return true if a fresh frame arrived, false if timeout or stream stopped.
     */
    bool read(cv::Mat& frame, int timeoutMs = 100);

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
    std::condition_variable m_cv;
    std::atomic<bool> m_running{false};
    cv::Mat m_latestFrame;
    bool m_hasNewFrame{false};
};

} // namespace spatial
