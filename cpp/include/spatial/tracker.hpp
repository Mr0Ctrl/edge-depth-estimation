#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <memory>
#include <string>

namespace spatial {

/**
 * @brief Represents a 2D bounding box, focal target point, and metadata.
 */
struct TrackedTarget {
    cv::Rect bbox;                        // Bounding box [x, y, width, height]
    cv::Point2f center;                   // Focal target point (u, v) for autofocus/bokeh
    float confidence{0.0f};               // Detection/tracking confidence in [0.0, 1.0]
    bool isLocked{false};                 // True if target is actively locked
};

/**
 * @brief Abstract tracking interface (Polymorphic Strategy Pattern).
 */
class ITracker {
public:
    virtual ~ITracker() = default;
    virtual bool init() = 0;
    virtual TrackedTarget update(const cv::Mat& frame) = 0;
    virtual void reset() = 0;
    virtual void drawDebug(cv::Mat& frame, const TrackedTarget& target) const;
};

/**
 * @brief Default fallback tracker focusing on the center of the frame.
 */
class CenterTracker : public ITracker {
public:
    bool init() override { return true; }
    TrackedTarget update(const cv::Mat& frame) override;
    void reset() override {}
};

/**
 * @brief Deep Learning Face Detector based on OpenCV's native SSD-ResNet10 architecture.
 * Rock-solid, fully compatible with OpenCV 4.x, and fast on CPU (~50-80 FPS).
 */
class DnnFaceTracker : public ITracker {
public:
    DnnFaceTracker(std::string configPath, std::string modelPath, float confidenceThreshold = 0.6f);
    bool init() override;
    TrackedTarget update(const cv::Mat& frame) override;
    void reset() override;

private:
    std::string m_configPath;
    std::string m_modelPath;
    float m_confidenceThreshold;
    cv::dnn::Net m_net;
    bool m_isLoaded{false};
    TrackedTarget m_lastTarget;
};

/**
 * @brief Classical Viola-Jones Haar Cascade face tracker (Secondary fallback).
 */
class HaarFaceTracker : public ITracker {
public:
    explicit HaarFaceTracker(std::string cascadePath);
    bool init() override;
    TrackedTarget update(const cv::Mat& frame) override;
    void reset() override;

private:
    std::string m_cascadePath;
    cv::CascadeClassifier m_cascade;
    bool m_isLoaded{false};
    cv::Mat m_grayBuffer;
    cv::Mat m_smallGray;
    TrackedTarget m_lastTarget;
};

} // namespace spatial
