#include "spatial/profiler.hpp"

namespace spatial {

Profiler::Profiler()
    : m_lastTickTime(std::chrono::steady_clock::now()) {}

void Profiler::tick(double camMs, double aiMs, double guiMs) {
    auto now = std::chrono::steady_clock::now();
    double elapsedSec = std::chrono::duration<double>(now - m_lastTickTime).count();
    m_lastTickTime = now;

    if (elapsedSec > 0.0) {
        double instantFps = 1.0 / elapsedSec;
        m_fps = 0.9 * m_fps + 0.1 * instantFps;
    }

    // Exponential moving average for smooth diagnostic readout
    m_avgCamMs = 0.9 * m_avgCamMs + 0.1 * camMs;
    m_avgAiMs  = 0.9 * m_avgAiMs  + 0.1 * aiMs;
    m_avgGuiMs = 0.9 * m_avgGuiMs + 0.1 * guiMs;
}

void Profiler::drawOverlay(cv::Mat& frame) const {
    if (frame.empty()) return;

    // Semi-transparent background box for crisp contrast
    cv::Rect hudBox(10, frame.rows - 65, 380, 55);
    cv::Mat roi = frame(hudBox);
    cv::Mat overlay;
    roi.copyTo(overlay);
    cv::rectangle(overlay, cv::Rect(0, 0, hudBox.width, hudBox.height), cv::Scalar(20, 20, 20), -1);
    cv::addWeighted(overlay, 0.6, roi, 0.4, 0.0, roi);

    // FPS text
    std::string line1 = cv::format("FPS: %.1f (%dx%d)", m_fps, frame.cols, frame.rows);
    cv::putText(frame, line1, cv::Point(20, frame.rows - 42),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

    // Phase latencies
    std::string line2 = cv::format("Cam: %.1fms | AI: %.1fms | GUI: %.1fms",
                                   m_avgCamMs, m_avgAiMs, m_avgGuiMs);
    cv::putText(frame, line2, cv::Point(20, frame.rows - 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
}

} // namespace spatial
