#pragma once

#include <opencv2/opencv.hpp>
#include <optional>

namespace spatial {

/**
 * @brief Represents a 2D bounding box and focal target point.
 */
struct TrackedTarget {
    cv::Rect bbox;           // Bounding box [x, y, width, height]
    cv::Point2f center;      // Center point (u, v) for autofocus sampling
    float confidence;        // Detection/tracking confidence in [0.0, 1.0]
    bool isLocked;           // True if tracker is actively locked on target
};

/**
 * @brief Classical tracking system for automated camera focal target selection.
 * Uses template matching, Haar cascades, or center-weighted fallback.
 */
class FocusTracker {
public:
    enum class Method {
        CenterPoint,        // Default: Center of frame
        TemplateMatching,   // Classical: cv::matchTemplate
        HaarCascade         // Classical: cv::CascadeClassifier (Face/Object)
    };

    FocusTracker();
    ~FocusTracker();

    /**
     * @brief Initializes tracker with a specific method.
     * @param method Tracking algorithm.
     * @param cascadePath Path to XML cascade if using HaarCascade (optional).
     */
    bool init(Method method = Method::CenterPoint, const std::string& cascadePath = "");

    /**
     * @brief Sets a custom template from a bounding box for template matching.
     */
    void setTemplate(const cv::Mat& frame, const cv::Rect& roi);

    /**
     * @brief Processes the current frame and returns the active focus target.
     * @param frame BGR frame from camera.
     * @return TrackedTarget with bounding box and center focal point.
     */
    TrackedTarget update(const cv::Mat& frame);

    /**
     * @brief Renders tracking visualizer (crosshairs, bounding box) onto the frame.
     */
    void drawDebug(cv::Mat& frame, const TrackedTarget& target) const;

private:
    Method m_method;
    cv::Mat m_template;
    cv::CascadeClassifier m_cascade;
    bool m_hasCascade;
    TrackedTarget m_lastTarget;
};

} // namespace spatial
