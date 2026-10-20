#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace spatial {

/**
 * @brief Manages camera hardware input and frame capture using OpenCV.
 */
class Camera {
public:
    Camera();
    ~Camera();

    /**
     * @brief Opens a video capture device or video file.
     * @param deviceIndex Index of webcam (default: 0).
     * @param width Desired frame width (default: 640).
     * @param height Desired frame height (default: 480).
     * @return true if camera opened successfully, false otherwise.
     */
    bool open(int deviceIndex = 0, int width = 640, int height = 480);

    /**
     * @brief Opens a video file or test image sequence.
     * @param filepath Path to video file.
     */
    bool openFile(const std::string& filepath);

    /**
     * @brief Captures the next available frame.
     * @param[out] frame Destination OpenCV Mat (BGR format).
     * @return true if frame read successfully, false if stream ended.
     */
    bool read(cv::Mat& frame);

    /**
     * @brief Checks if camera is currently opened.
     */
    bool isOpened() const;

    /**
     * @brief Releases the camera hardware device.
     */
    void close();

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }

private:
    cv::VideoCapture m_cap;
    int m_width;
    int m_height;
};

} // namespace spatial
