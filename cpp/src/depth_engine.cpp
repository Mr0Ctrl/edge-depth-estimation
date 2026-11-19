#include "spatial/depth_engine.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "spatial/logger.hpp"

namespace spatial
{
    struct DepthEngine::Impl
    {
        cv::dnn::Net net;
        cv::Mat blob;
        cv::Mat rawOutput;
    };

    DepthEngine::DepthEngine() : m_isLoaded(false), m_inputWidth(256), m_inputHeight(256), m_impl(std::make_unique<Impl>())
    {
    }

    DepthEngine::~DepthEngine() = default;

    bool DepthEngine::loadModel(const std::string &modelPath, bool useGPU)
    {
        m_isLoaded = false;
        try
        {
            m_impl->net = cv::dnn::readNetFromONNX(modelPath);
        }
        catch (const cv::Exception &e)
        {
            SPATIAL_LOG_ERROR(e.what());
            return false;
        }
        if (m_impl->net.empty())
        {
            SPATIAL_LOG_ERROR("Model empty");
            return false;
        }

        if (useGPU)
        {
            if (cv::cuda::getCudaEnabledDeviceCount())
            {
                m_impl->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                m_impl->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                SPATIAL_LOG_INFO("CUDA backend selected");
            }
            else
            {
                m_impl->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                m_impl->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                SPATIAL_LOG_WARN("CPU backend fallback");
            }
        }
        else
        {
            m_impl->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_impl->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            SPATIAL_LOG_INFO("CPU backend selected");
        }

        m_isLoaded = true;
        return true;
    }

    bool DepthEngine::estimate(const cv::Mat &frame, cv::Mat &depthMap, cv::Mat *coloredDepth)
    {
        if (!m_isLoaded || frame.empty())
            return false;

        cv::dnn::blobFromImage(frame, m_impl->blob, 1.0 / 255.0, cv::Size(m_inputWidth, m_inputHeight), cv::Scalar(123.675, 116.28, 103.53), true, false);

        // model run
        m_impl->net.setInput(m_impl->blob);
        m_impl->rawOutput = m_impl->net.forward();

        if (!(m_impl->rawOutput.total() == static_cast<size_t>(m_inputWidth * m_inputHeight)))
        {
            SPATIAL_LOG_ERROR("rawOutput dimension miss match");
            return false;
        }

        cv::Mat disparity2D(m_inputHeight, m_inputWidth, CV_32F, m_impl->rawOutput.ptr<float>());
        cv::resize(disparity2D, depthMap, frame.size(), 0, 0, cv::INTER_LINEAR);

        (void)coloredDepth;
        return true;
    }
} // namespace spatial
