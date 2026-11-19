#pragma once
#include <string>
#include <memory>
#include <opencv2/core.hpp>

namespace spatial
{
    class DepthEngine
    {
    private:
        bool m_isLoaded;
        int m_inputWidth;
        int m_inputHeight;
        struct Impl;
        std::unique_ptr<Impl> m_impl;

    public:
        DepthEngine();
        ~DepthEngine();

        bool loadModel(const std::string &modelPath, bool useGPU);

        bool estimate(const cv::Mat &frame, cv::Mat &depthMap, cv::Mat *coloredDepth = nullptr);
    };

} // namespace spatial
