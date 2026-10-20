#include "spatial/camera.hpp"

namespace spatial
{

    Camera::Camera()
        : m_width(0), m_height(0)
    {
    }

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
            // Fallback to default backend if V4L2 fails
            m_cap.open(deviceIndex);
        }

        if (!m_cap.isOpened())
        {
            return false;
        }

        // Request MJPG codec for high-framerate USB stream
        m_cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        m_cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        m_cap.set(cv::CAP_PROP_FPS, 30.0);

        m_width = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
        m_height = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));

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

        return true;
    }

    bool Camera::read(cv::Mat &frame)
    {
        if (!m_cap.isOpened())
            return false;

        return m_cap.read(frame) && !frame.empty();
    }

    bool Camera::isOpened() const
    {
        return m_cap.isOpened();
    }

    void Camera::close()
    {
        if (m_cap.isOpened())
        {
            m_cap.release();
        }
        m_width = 0;
        m_height = 0;
    }

}
