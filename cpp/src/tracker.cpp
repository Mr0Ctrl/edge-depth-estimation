#include "spatial/tracker.hpp"

namespace spatial
{
    FocusTracker::FocusTracker() {}
    FocusTracker::~FocusTracker() {}

    bool FocusTracker::init(Method method, const std::string &cascadePath)
    {
        m_method = method;
        m_hasCascade = false;

        if (m_method == Method::HaarCascade)
        {
            if (cascadePath.empty())
            {
                return false;
            }

            if (!m_cascade.load(cascadePath))
            {
                return false;
            }

            m_hasCascade = true;
        }

        return true;
    }

    void FocusTracker::setTemplate(const cv::Mat &frame, const cv::Rect &roi)
    {
        if (frame.empty() || roi.width <= 0 || roi.height <= 0)
        {
            return;
        }
        cv::Rect imageBounds(0, 0, frame.cols, frame.rows); // idk is that a definition
        cv::Rect safeRoi = roi & imageBounds;

        if (safeRoi.area() > 0)
        {
            m_template = frame(safeRoi).clone();
        }
    }

    TrackedTarget FocusTracker::update(const cv::Mat &frame)
    {
        if (frame.empty())
        {
            m_lastTarget.isLocked = false;
            return m_lastTarget;
        }

        if (m_method == Method::HaarCascade && m_hasCascade)
        {
            // Convert to grayscale reusing internal buffer
            if (frame.channels() == 3)
            {
                cv::cvtColor(frame, m_grayBuffer, cv::COLOR_BGR2GRAY);
            }
            else
            {
                m_grayBuffer = frame;
            }

            // Downscale grayscale frame by 2x for massive speedup (~4x-8x faster on CPU)
            constexpr float scale = 0.5f;
            cv::resize(m_grayBuffer, m_smallGray, cv::Size(), scale, scale, cv::INTER_LINEAR);

            std::vector<cv::Rect> detections;
            // minNeighbors=4 filters out most false positives while keeping real faces
            m_cascade.detectMultiScale(m_smallGray, detections, 1.15, 4,
                                       0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

            if (!detections.empty())
            {
                // Select detection with largest area (closest/primary face)
                size_t bestIndex = 0;
                int maxArea = detections[0].area();

                for (size_t i = 1; i < detections.size(); ++i)
                {
                    int area = detections[i].area();
                    if (area > maxArea)
                    {
                        maxArea = area;
                        bestIndex = i;
                    }
                }

                // Scale bounding box back to full resolution
                const auto &bestDet = detections[bestIndex];
                constexpr float invScale = 1.0f / scale;
                cv::Rect scaledBbox(
                    static_cast<int>(bestDet.x * invScale),
                    static_cast<int>(bestDet.y * invScale),
                    static_cast<int>(bestDet.width * invScale),
                    static_cast<int>(bestDet.height * invScale));

                // Clamp to frame boundaries
                cv::Rect frameBounds(0, 0, frame.cols, frame.rows);
                m_lastTarget.bbox = scaledBbox & frameBounds;
                m_lastTarget.center = cv::Point2f(
                    m_lastTarget.bbox.x + m_lastTarget.bbox.width * 0.5f,
                    m_lastTarget.bbox.y + m_lastTarget.bbox.height * 0.5f);
                m_lastTarget.confidence = 0.90f;
                m_lastTarget.isLocked = true;
            }
            else
            {
                m_lastTarget.isLocked = false;
                m_lastTarget.confidence = 0.0f;
            }
        }
        else if (m_method == Method::TemplateMatching && !m_template.empty())
        {
            // Guided Template Matching:
            // If already locked on previous frame, search only in a local window around previous target.
            // This speeds up execution and avoids false matches elsewhere in the frame.
            cv::Rect searchRoi;
            if (m_lastTarget.isLocked)
            {
                // Expand previous bbox by 50% in all directions
                int padX = m_lastTarget.bbox.width / 2;
                int padY = m_lastTarget.bbox.height / 2;
                cv::Rect expanded(
                    m_lastTarget.bbox.x - padX,
                    m_lastTarget.bbox.y - padY,
                    m_lastTarget.bbox.width + 2 * padX,
                    m_lastTarget.bbox.height + 2 * padY);
                cv::Rect frameBounds(0, 0, frame.cols, frame.rows);
                searchRoi = expanded & frameBounds;
            }
            else
            {
                // Full frame search if not currently locked
                searchRoi = cv::Rect(0, 0, frame.cols, frame.rows);
            }

            if (searchRoi.width >= m_template.cols && searchRoi.height >= m_template.rows)
            {
                cv::Mat searchArea = frame(searchRoi);
                cv::Mat results;
                cv::matchTemplate(searchArea, m_template, results, cv::TM_CCOEFF_NORMED);

                double minVal = 0.0, maxVal = 0.0;
                cv::Point minLoc, maxLoc;
                cv::minMaxLoc(results, &minVal, &maxVal, &minLoc, &maxLoc);

                if (maxVal > 0.60)
                {
                    cv::Rect matchedBbox(
                        searchRoi.x + maxLoc.x,
                        searchRoi.y + maxLoc.y,
                        m_template.cols,
                        m_template.rows);

                    m_lastTarget.bbox = matchedBbox;
                    m_lastTarget.center = cv::Point2f(
                        matchedBbox.x + matchedBbox.width * 0.5f,
                        matchedBbox.y + matchedBbox.height * 0.5f);
                    m_lastTarget.confidence = static_cast<float>(maxVal);
                    m_lastTarget.isLocked = true;

                    // Adaptive template update: blend current appearance to handle gradual lighting changes
                    cv::Mat currentPatch = frame(matchedBbox);
                    if (currentPatch.size() == m_template.size() && currentPatch.type() == m_template.type())
                    {
                        cv::addWeighted(m_template, 0.90, currentPatch, 0.10, 0.0, m_template);
                    }
                }
                else
                {
                    m_lastTarget.isLocked = false;
                    m_lastTarget.confidence = 0.0f;
                }
            }
            else
            {
                m_lastTarget.isLocked = false;
                m_lastTarget.confidence = 0.0f;
            }
        }
        else
        {
            // Center-point fallback
            m_lastTarget.bbox = cv::Rect(frame.cols / 4, frame.rows / 4, frame.cols / 2, frame.rows / 2);
            m_lastTarget.center = cv::Point2f(frame.cols * 0.5f, frame.rows * 0.5f);
            m_lastTarget.confidence = 1.0f;
            m_lastTarget.isLocked = true;
        }

        return m_lastTarget;
    }

    void FocusTracker::drawDebug(cv::Mat &frame, const TrackedTarget &target) const
    {
        if (frame.empty())
        {
            return;
        }

        if (target.isLocked)
        {
            // Bounding box
            cv::rectangle(frame, target.bbox, cv::Scalar(0, 255, 0), 2);

            // Focus point circle and crosshair
            cv::circle(frame, target.center, 4, cv::Scalar(0, 0, 255), -1);
            cv::line(frame, cv::Point2f(target.center.x - 10, target.center.y),
                     cv::Point2f(target.center.x + 10, target.center.y), cv::Scalar(0, 255, 255), 1);
            cv::line(frame, cv::Point2f(target.center.x, target.center.y - 10),
                     cv::Point2f(target.center.x, target.center.y + 10), cv::Scalar(0, 255, 255), 1);

            // HUD text: Focus target locked
            std::string label = cv::format("AF Lock [%.0f%%]", target.confidence * 100.0f);
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int textY = std::max(labelSize.height + 5, target.bbox.y - 6);
            cv::putText(frame, label, cv::Point(target.bbox.x, textY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
        else
        {
            // Subtle indicator that tracker is scanning
            cv::putText(frame, "AF Searching...", cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
        }
    }

} // namespace spatial
