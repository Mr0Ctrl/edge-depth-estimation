#pragma once
#include <chrono>
#include <string>
#include <unordered_map>

namespace spatial
{

    /**
     * @brief Performance profiler and diagnostic HUD overlay.
     *
     * Tracks rolling frames-per-second (FPS) and phase latencies (Camera, AI, GUI)
     * without polluting core business logic.
     */
    class Profiler
    {
    public:
        Profiler();
        ~Profiler() = default;

        void recordStage(const std::string &name, double ms);

        void tickFrame();

        std::string summary() const;

        double getFps() const { return m_fps; }
        const std::unordered_map<std::string, double> getStageMs() const { return m_stageMs; }

    private:
        std::chrono::steady_clock::time_point m_lastTickTime;
        std::unordered_map<std::string, double> m_stageMs;

        double m_fps{0.0};
    };

    class ScopedTimer
    {
    private:
        Profiler &m_profiler;
        std::string m_name;
        std::chrono::steady_clock::time_point m_start;

        ScopedTimer(const ScopedTimer &) = delete;
        ScopedTimer &operator=(const ScopedTimer &) = delete;

    public:
        // Doğru Initializer List: Değişkenler parantez içinde eşleştirilir
        ScopedTimer(Profiler &profiller, std::string name)
            : m_profiler(profiller), m_name(std::move(name)), m_start(std::chrono::steady_clock::now())
        {
            // Gövde içi boş kalabilir çünkü tüm atamalar ve zaman başlatma yukarıda bitti
        }

        ~ScopedTimer()
        {
            auto now = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - m_start).count();
            m_profiler.recordStage(m_name, ms);
        }
    };

} // namespace spatial
