#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace spatial {

enum class BlurCurve {
    SquareRoot,    // Aggressive: t^0.5 (sharp subject pop)
    Linear,        // Direct distance: t
    Smoothstep,    // S-curve: 3t^2 - 2t^3
    Quadratic      // Gentle near focus: t^2
};

struct BokehParams {
    float focusDepth = 0.5f;             // Focal plane in [0.0, 1.0]
    float blurPower = 0.7f;              // Aperture strength in [0.0, 1.0]
    float dofMargin = 0.02f;             // Sharp depth-of-field window thickness
    int maxKernelSize = 41;              // Max Gaussian blur kernel
    BlurCurve curve = BlurCurve::SquareRoot;
    bool highlightFocus = false;         // Overlays green tint on in-focus plane
};

/**
 * @brief Optical Defocus Bokeh filter guided by per-pixel depth.
 */
class BokehFilter {
public:
    BokehFilter();
    ~BokehFilter();

    /**
     * @brief Renders synthetic depth-of-field bokeh onto an input image.
     * @param image Input BGR image (CV_8UC3).
     * @param depthMap Normalized depth map (CV_32FC1) in [0.0, 1.0].
     * @param params Focal plane, aperture size, and curve parameters.
     * @return Rendered bokeh image (CV_8UC3).
     */
    cv::Mat apply(const cv::Mat& image, const cv::Mat& depthMap, const BokehParams& params);

    /**
     * @brief Computes continuous per-pixel blur weight map in [0.0, 1.0].
     */
    cv::Mat computeBlurWeights(const cv::Mat& depthMap, const BokehParams& params) const;

private:
    float applyCurve(float t, BlurCurve curve) const;
};

} // namespace spatial
