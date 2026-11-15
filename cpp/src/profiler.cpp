#include "spatial/profiler.hpp"
#include <sstream>

namespace spatial
{

    Profiler::Profiler() : m_lastTickTime(std::chrono::steady_clock::now()) {}

    void Profiler::recordStage(const std::string &name, double ms)
    {
        auto it = m_stageMs.find(name);
        if (it != m_stageMs.end())
        {
            it->second = 0.9 * it->second + 0.1 * ms;
        }
        else
        {
            m_stageMs[name] = ms;
        }
    }

    void Profiler::tickFrame()
    {
        auto tickTime = std::chrono::steady_clock::now();
        auto dTime = std::chrono::duration<double, std::milli>(tickTime - m_lastTickTime).count();
        if (0.0 < dTime)
            m_fps = 0.9 * m_fps + 0.1 * (1000.0 / dTime);
        m_lastTickTime = tickTime;
    }

    std::string Profiler::summary() const
    {
        std::ostringstream oss;
        oss << "FPS: " << m_fps;

        // Haritadaki tüm aşamaları döngüyle ekleyelim:
        for (const auto &pair : m_stageMs)
        {
            oss << " | " << pair.first << ": " << pair.second << "ms"; // pair.first anahtar (ad), pair.second değer (süre)
        }

        return oss.str(); // İçinde biriken tüm metni std::string olarak dışarı verir
    }

} // namespace spatial
