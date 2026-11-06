#include "spatial/camera.hpp"
#include <iostream>

namespace spatial {

Camera::Camera() {}

Camera::~Camera() {
    close();
}

bool Camera::open(int deviceIndex, int width, int height) {
    close();

#if defined(__linux__)
    m_cap.open(deviceIndex, cv::CAP_V4L2);
#else
    m_cap.open(deviceIndex);
#endif

    if (!m_cap.isOpened()) {
        m_cap.open(deviceIndex);
    }

    if (!m_cap.isOpened()) {
        return false;
    }

    // Set high-speed MJPG compressed stream format
    m_cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    m_cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    m_cap.set(cv::CAP_PROP_FPS, 30.0);
    m_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // Standard Auto-Exposure with Auto-Gain enabled
    m_cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 3);

    m_width = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    m_height = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Prime the camera by reading first frame synchronously
    cv::Mat firstFrame;
    if (m_cap.read(firstFrame) && !firstFrame.empty()) {
        m_latestFrame = firstFrame.clone();
        m_hasNewFrame = true;
    }

    // Launch background capture worker thread
    m_running = true;
    m_thread = std::thread(&Camera::captureWorker, this);

    return true;
}

bool Camera::openFile(const std::string& filepath) {
    close();

    m_cap.open(filepath);
    if (!m_cap.isOpened()) {
        return false;
    }

    m_width = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    m_height = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Mat firstFrame;
    if (m_cap.read(firstFrame) && !firstFrame.empty()) {
        m_latestFrame = firstFrame.clone();
        m_hasNewFrame = true;
    }

    m_running = true;
    m_thread = std::thread(&Camera::captureWorker, this);

    return true;
}

void Camera::captureWorker() {
    cv::Mat tempFrame;

    while (m_running) {
        if (!m_cap.isOpened()) {
            break;
        }

        // Hardware blocking read happens here on background thread
        if (m_cap.read(tempFrame) && !tempFrame.empty()) {
            std::lock_guard<std::mutex> lock(m_mutex);
            // Deep copy to buffer for thread safety
            tempFrame.copyTo(m_latestFrame);
            m_hasNewFrame = true;
        } else {
            // Short backoff on driver frame hiccup
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
}

bool Camera::read(cv::Mat& frame) {
    if (!m_running) {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_latestFrame.empty()) {
        return false;
    }

    // Fast memory copy to caller
    m_latestFrame.copyTo(frame);
    return true;
}

bool Camera::isOpened() const {
    return m_running && m_cap.isOpened();
}

void Camera::close() {
    // 1. Signal background worker to stop
    m_running = false;

    // 2. Wait for worker thread to finish
    if (m_thread.joinable()) {
        m_thread.join();
    }

    // 3. Release camera device handle
    if (m_cap.isOpened()) {
        m_cap.release();
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_latestFrame.release();
        m_hasNewFrame = false;
    }

    m_width = 0;
    m_height = 0;
}

} // namespace spatial
