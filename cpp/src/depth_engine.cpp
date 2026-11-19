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
} // namespace spatial
