#include "spatial/tracker.hpp"
#include <algorithm>
#include <iostream>

namespace spatial {

// ----------------------------------------------------------------------------
// ITracker Base Debug Drawing
// ----------------------------------------------------------------------------
void ITracker::drawDebug(cv::Mat& frame, const TrackedTarget& target) const {
    if (frame.empty()) return;

    if (target.isLocked) {
        // Draw green bounding box around detected target
        cv::rectangle(frame, target.bbox, cv::Scalar(0, 255, 0), 2);

        // Draw autofocus center point & crosshairs
        cv::circle(frame, target.center, 5, cv::Scalar(0, 0, 255), -1);
        cv::line(frame, cv::Point2f(target.center.x - 12, target.center.y),
                 cv::Point2f(target.center.x + 12, target.center.y), cv::Scalar(0, 255, 255), 1);
        cv::line(frame, cv::Point2f(target.center.x, target.center.y - 12),
                 cv::Point2f(target.center.x, target.center.y + 12), cv::Scalar(0, 255, 255), 1);

        // Draw confidence badge
        std::string label = cv::format("AF Lock [%.0f%%]", target.confidence * 100.0f);
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int textY = std::max(labelSize.height + 5, target.bbox.y - 6);
        cv::putText(frame, label, cv::Point(target.bbox.x, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    } else {
        cv::putText(frame, "AF Searching...", cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
    }
}

// ----------------------------------------------------------------------------
// CenterTracker Implementation
// ----------------------------------------------------------------------------
TrackedTarget CenterTracker::update(const cv::Mat& frame) {
    TrackedTarget target;
    if (frame.empty()) return target;

    target.bbox = cv::Rect(frame.cols / 4, frame.rows / 4, frame.cols / 2, frame.rows / 2);
    target.center = cv::Point2f(frame.cols * 0.5f, frame.rows * 0.5f);
    target.confidence = 1.0f;
    target.isLocked = true;
    return target;
}

// ----------------------------------------------------------------------------
// DnnFaceTracker (Hybrid Detection-and-Tracking)
// ----------------------------------------------------------------------------
DnnFaceTracker::DnnFaceTracker(std::string configPath,
                               std::string modelPath,
                               float confidenceThreshold,
                               int detectInterval)
    : m_configPath(std::move(configPath)),
      m_modelPath(std::move(modelPath)),
      m_confidenceThreshold(confidenceThreshold),
      m_detectInterval(detectInterval) {}

bool DnnFaceTracker::init() {
    if (m_configPath.empty() || m_modelPath.empty()) {
        return false;
    }

    try {
        m_net = cv::dnn::readNetFromCaffe(m_configPath, m_modelPath);
        m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        m_isLoaded = !m_net.empty();
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] DnnFaceTracker failed to load: " << e.what() << '\n';
        m_isLoaded = false;
    }

    reset();
    return m_isLoaded;
}

void DnnFaceTracker::reset() {
    m_lastTarget.isLocked = false;
    m_lastTarget.confidence = 0.0f;
    m_templatePatch.release();
    m_frameCounter = 0;
}

TrackedTarget DnnFaceTracker::update(const cv::Mat& frame) {
    if (frame.empty() || !m_isLoaded) {
        m_lastTarget.isLocked = false;
        return m_lastTarget;
    }

    bool needDnnDetection = (!m_lastTarget.isLocked) ||
                            (m_frameCounter % m_detectInterval == 0) ||
                            m_templatePatch.empty();

    if (needDnnDetection) {
        // Heavy pass: Deep Neural Network Face Detection (~15-20ms)
        TrackedTarget dnnResult;
        if (detectDnn(frame, dnnResult)) {
            m_lastTarget = dnnResult;

            // Cache template patch for fast tracking in intermediate frames
            cv::Rect frameBounds(0, 0, frame.cols, frame.rows);
            cv::Rect safeBbox = m_lastTarget.bbox & frameBounds;
            if (safeBbox.width >= 10 && safeBbox.height >= 10) {
                // Downsample template to max 64x64 for ultra-fast normalized matching
                cv::Mat patch = frame(safeBbox);
                cv::resize(patch, m_templatePatch, cv::Size(64, 64), 0, 0, cv::INTER_LINEAR);
            }
        } else {
            // Target lost in detector
            m_lastTarget.isLocked = false;
            m_lastTarget.confidence = 0.0f;
            m_templatePatch.release();
        }
    } else {
        // Fast pass: Localized Correlation Tracking (~1ms)
        TrackedTarget trackedResult;
        if (trackFast(frame, trackedResult)) {
            m_lastTarget = trackedResult;
        } else {
            // Fast track failed (e.g. rapid head turn); trigger DNN immediately next frame
            m_lastTarget.isLocked = false;
            m_templatePatch.release();
        }
    }

    m_frameCounter++;
    return m_lastTarget;
}

bool DnnFaceTracker::detectDnn(const cv::Mat& frame, TrackedTarget& outTarget) {
    cv::Mat inputBlob = cv::dnn::blobFromImage(
        frame,
        1.0,
        cv::Size(300, 300),
        cv::Scalar(104.0, 177.0, 123.0),
        false,
        false
    );

    m_net.setInput(inputBlob, "data");
    cv::Mat detection = m_net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    float bestConfidence = m_confidenceThreshold;
    int bestRow = -1;
    float maxArea = 0.0f;

    for (int i = 0; i < detectionMat.rows; ++i) {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > m_confidenceThreshold) {
            float x1 = detectionMat.at<float>(i, 3);
            float y1 = detectionMat.at<float>(i, 4);
            float x2 = detectionMat.at<float>(i, 5);
            float y2 = detectionMat.at<float>(i, 6);

            float w = (x2 - x1);
            float h = (y2 - y1);
            float area = w * h;

            if (area > maxArea) {
                maxArea = area;
                bestConfidence = confidence;
                bestRow = i;
            }
        }
    }

    if (bestRow >= 0) {
        int x1 = static_cast<int>(detectionMat.at<float>(bestRow, 3) * frame.cols);
        int y1 = static_cast<int>(detectionMat.at<float>(bestRow, 4) * frame.rows);
        int x2 = static_cast<int>(detectionMat.at<float>(bestRow, 5) * frame.cols);
        int y2 = static_cast<int>(detectionMat.at<float>(bestRow, 6) * frame.rows);

        x1 = std::max(0, std::min(x1, frame.cols - 1));
        y1 = std::max(0, std::min(y1, frame.rows - 1));
        x2 = std::max(0, std::min(x2, frame.cols - 1));
        y2 = std::max(0, std::min(y2, frame.rows - 1));

        int width = std::max(0, x2 - x1);
        int height = std::max(0, y2 - y1);

        outTarget.bbox = cv::Rect(x1, y1, width, height);
        outTarget.center = cv::Point2f(x1 + width * 0.5f, y1 + height * 0.5f);
        outTarget.confidence = bestConfidence;
        outTarget.isLocked = true;
        return true;
    }

    return false;
}

bool DnnFaceTracker::trackFast(const cv::Mat& frame, TrackedTarget& outTarget) {
    if (m_templatePatch.empty() || m_lastTarget.bbox.area() <= 0) {
        return false;
    }

    // Expand search ROI around previous target by 40% in each direction
    int padX = m_lastTarget.bbox.width * 2 / 5;
    int padY = m_lastTarget.bbox.height * 2 / 5;

    cv::Rect searchRoi(
        m_lastTarget.bbox.x - padX,
        m_lastTarget.bbox.y - padY,
        m_lastTarget.bbox.width + 2 * padX,
        m_lastTarget.bbox.height + 2 * padY
    );

    cv::Rect frameBounds(0, 0, frame.cols, frame.rows);
    searchRoi = searchRoi & frameBounds;

    if (searchRoi.width <= m_lastTarget.bbox.width || searchRoi.height <= m_lastTarget.bbox.height) {
        return false;
    }

    // Downsample search area to match template scale (64x64 reference)
    float scaleX = 64.0f / static_cast<float>(m_lastTarget.bbox.width);
    float scaleY = 64.0f / static_cast<float>(m_lastTarget.bbox.height);

    cv::Size scaledSearchSize(
        static_cast<int>(searchRoi.width * scaleX),
        static_cast<int>(searchRoi.height * scaleY)
    );

    if (scaledSearchSize.width < m_templatePatch.cols || scaledSearchSize.height < m_templatePatch.rows) {
        return false;
    }

    cv::Mat searchImg;
    cv::resize(frame(searchRoi), searchImg, scaledSearchSize, 0, 0, cv::INTER_LINEAR);

    // Fast normalized correlation match on tiny image
    cv::Mat matchResult;
    cv::matchTemplate(searchImg, m_templatePatch, matchResult, cv::TM_CCOEFF_NORMED);

    double minVal = 0.0, maxVal = 0.0;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(matchResult, &minVal, &maxVal, &minLoc, &maxLoc);

    // Correlation threshold: 0.50 ensures stable lock while allowing lighting shifts
    if (maxVal > 0.50) {
        // Map peak location back to original camera coordinate space
        float invScaleX = 1.0f / scaleX;
        float invScaleY = 1.0f / scaleY;

        int newX = searchRoi.x + static_cast<int>(maxLoc.x * invScaleX);
        int newY = searchRoi.y + static_cast<int>(maxLoc.y * invScaleY);

        outTarget.bbox = cv::Rect(newX, newY, m_lastTarget.bbox.width, m_lastTarget.bbox.height) & frameBounds;
        outTarget.center = cv::Point2f(outTarget.bbox.x + outTarget.bbox.width * 0.5f,
                                      outTarget.bbox.y + outTarget.bbox.height * 0.5f);
        outTarget.confidence = static_cast<float>(maxVal);
        outTarget.isLocked = true;
        return true;
    }

    return false;
}

// ----------------------------------------------------------------------------
// HaarFaceTracker Implementation
// ----------------------------------------------------------------------------
HaarFaceTracker::HaarFaceTracker(std::string cascadePath)
    : m_cascadePath(std::move(cascadePath)) {}

bool HaarFaceTracker::init() {
    if (m_cascadePath.empty()) return false;
    m_isLoaded = m_cascade.load(m_cascadePath);
    return m_isLoaded;
}

void HaarFaceTracker::reset() {
    m_lastTarget.isLocked = false;
    m_lastTarget.confidence = 0.0f;
}

TrackedTarget HaarFaceTracker::update(const cv::Mat& frame) {
    if (frame.empty() || !m_isLoaded) {
        m_lastTarget.isLocked = false;
        return m_lastTarget;
    }

    if (frame.channels() == 3) {
        cv::cvtColor(frame, m_grayBuffer, cv::COLOR_BGR2GRAY);
    } else {
        m_grayBuffer = frame;
    }

    constexpr float scale = 0.5f;
    cv::resize(m_grayBuffer, m_smallGray, cv::Size(), scale, scale, cv::INTER_LINEAR);

    std::vector<cv::Rect> detections;
    m_cascade.detectMultiScale(m_smallGray, detections, 1.15, 4,
                               0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    if (!detections.empty()) {
        size_t bestIndex = 0;
        int maxArea = detections[0].area();

        for (size_t i = 1; i < detections.size(); ++i) {
            int area = detections[i].area();
            if (area > maxArea) {
                maxArea = area;
                bestIndex = i;
            }
        }

        constexpr float invScale = 1.0f / scale;
        const auto& bestDet = detections[bestIndex];
        cv::Rect scaledBbox(
            static_cast<int>(bestDet.x * invScale),
            static_cast<int>(bestDet.y * invScale),
            static_cast<int>(bestDet.width * invScale),
            static_cast<int>(bestDet.height * invScale));

        cv::Rect frameBounds(0, 0, frame.cols, frame.rows);
        m_lastTarget.bbox = scaledBbox & frameBounds;
        m_lastTarget.center = cv::Point2f(
            m_lastTarget.bbox.x + m_lastTarget.bbox.width * 0.5f,
            m_lastTarget.bbox.y + m_lastTarget.bbox.height * 0.5f);
        m_lastTarget.confidence = 0.85f;
        m_lastTarget.isLocked = true;
    } else {
        m_lastTarget.isLocked = false;
        m_lastTarget.confidence = 0.0f;
    }

    return m_lastTarget;
}

} // namespace spatial
