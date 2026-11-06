#pragma once

#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>

namespace spatial {

/**
 * @brief Performance profiler and diagnostic HUD overlay.
 * 
 * Tracks rolling frames-per-second (FPS) and phase latencies (Camera, AI, GUI)
 * without polluting core business logic.
 */
class Profiler {
public:
    Profiler();

    /**
     * @brief Records a tick with individual phase durations in milliseconds.
     */
    void tick(double camMs, double aiMs, double guiMs);

    /**
     * @brief Renders the diagnostic HUD onto the frame.
     */
    void drawOverlay(cv::Mat& frame) const;

    double getFps() const { return m_fps; }
    double getCamMs() const { return m_avgCamMs; }
    double getAiMs() const { return m_avgAiMs; }
    double getGuiMs() const { return m_avgGuiMs; }

private:
    std::chrono::steady_clock::time_point m_lastTickTime;
    double m_fps{0.0};
    double m_avgCamMs{0.0};
    double m_avgAiMs{0.0};
    double m_avgGuiMs{0.0};
};

} // namespace spatial
