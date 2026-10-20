#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace spatial {

/**
 * @brief Configuration for depth processing and linearization.
 */
struct DepthConfig {
    float skyFloor = 0.04f;      // Clamps optical infinity asymptote
    float zNear = 1.0f;          // Normalized near clipping distance
    float zFar = 8.0f;           // Normalized far clipping distance
    bool linearize = true;       // If true, applies 1/m linear depth recovery
    bool invertNearBright = true;// If true, near=1.0 (bright), far=0.0 (dark)
};

/**
 * @brief High-performance Depth Estimation Interface.
 * Loads ONNX models and converts disparity to linearized metric depth.
 */
class DepthEngine {
public:
    DepthEngine();
    ~DepthEngine();

    /**
     * @brief Loads an ONNX depth model (e.g. Depth Anything, MobileNet, FastDepth).
     * @param modelPath Path to .onnx file.
     * @param useGPU Whether to request CUDA / NPU acceleration.
     * @return true if model loaded successfully.
     */
    bool loadModel(const std::string& modelPath, bool useGPU = true);

    /**
     * @brief Estimates dense relative depth from an input BGR frame.
     * @param frame Input BGR frame (H x W x 3).
     * @param[out] depthMap Output normalized float depth map (H x W, CV_32FC1) in [0.0, 1.0].
     * @param[out] coloredDepth Optional false-color heatmap (CV_8UC3).
     * @return true on success.
     */
    bool estimate(const cv::Mat& frame, cv::Mat& depthMap, cv::Mat* coloredDepth = nullptr);

    /**
     * @brief Linearizes raw disparity m using projective z = 1/m inversion.
     * Guarantees uniform sensitivity ratio (1.0x) across near and far distances.
     */
    cv::Mat linearizeDisparity(const cv::Mat& rawDisparity) const;

    /**
     * @brief Samples the depth value at a 2D coordinate.
     */
    float getDepthAt(const cv::Mat& depthMap, int x, int y) const;

    void setConfig(const DepthConfig& config) { m_config = config; }
    const DepthConfig& getConfig() const { return m_config; }

private:
    DepthConfig m_config;
    bool m_isLoaded;
    int m_inputWidth;
    int m_inputHeight;
};

} // namespace spatial
