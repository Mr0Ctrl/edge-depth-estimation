#include "spatial/camera.hpp"
#include <iostream>

namespace spatial
{

    Camera::Camera() {}

    Camera::~Camera()
    {
        close();
    }

    bool Camera::open(int deviceIndex, int width, int height)
    {
        close();

#if defined(__linux__)
        m_cap.open(deviceIndex, cv::CAP_V4L2);
#else
        m_cap.open(deviceIndex);
#endif

        if (!m_cap.isOpened())
        {
            m_cap.open(deviceIndex);
        }

        if (!m_cap.isOpened())
        {
            return false;
        }

        // High-framerate MJPG compressed format
        m_cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        m_cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

        m_cap.set(cv::CAP_PROP_FPS, 30.0);
        m_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        m_cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 3);

        m_width = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
        m_height = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        // Launch background capture worker thread
        m_running = true;
        m_hasNewFrame = false;
        m_thread = std::thread(&Camera::captureWorker, this);

        return true;
    }

    bool Camera::openFile(const std::string &filepath)
    {
        close();

        m_cap.open(filepath);
        if (!m_cap.isOpened())
        {
            return false;
        }

        m_width = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
        m_height = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        m_running = true;
        m_hasNewFrame = false;
        m_thread = std::thread(&Camera::captureWorker, this);

        return true;
    }

    void Camera::captureWorker()
    {
        cv::Mat tempFrame;

        while (m_running)
        {
            if (!m_cap.isOpened())
            {
                break;
            }

            // Hardware blocking read on background thread
            if (m_cap.read(tempFrame) && !tempFrame.empty())
            {
                {
                    std::lock_guard<std::mutex> lock(m_mutex);

                    tempFrame.copyTo(m_latestFrame);
                    m_hasNewFrame = true;
                }
                // Notify consumer that a fresh frame is ready
                m_cv.notify_one();
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }

    bool Camera::read(cv::Mat &frame, int timeoutMs)
    {
        if (!m_running)
        {
            return false;
        }

        std::unique_lock<std::mutex> lock(m_mutex);

        // Wait until a genuinely fresh frame is produced or timeout occurs
        bool success = m_cv.wait_for(
            lock,
            std::chrono::milliseconds(timeoutMs),
            [this]()
            { return m_hasNewFrame || !m_running; });

        if (!success || !m_running || m_latestFrame.empty())
        {
            return false;
        }

        // Copy the fresh frame and consume it
        m_latestFrame.copyTo(frame);
        m_hasNewFrame = false; // Flag consumed: next read() must wait for next hardware frame!
        return true;
    }

    bool Camera::isOpened() const
    {
        return m_running && m_cap.isOpened();
    }

    void Camera::close()
    {
        m_running = false;
        m_cv.notify_all(); // Unblock any waiting read() calls

        if (m_thread.joinable())
        {
            m_thread.join();
        }

        if (m_cap.isOpened())
        {
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
