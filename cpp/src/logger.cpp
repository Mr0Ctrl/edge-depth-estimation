#include "spatial/logger.hpp"
#include <iostream>

namespace spatial
{
    Logger &Logger::instance()
    {
        static Logger inst;
        return inst;
    }

    void Logger::setLevel(LogLevel level)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_minLevel = level;
    }

    void Logger::setSink(LogSink sink)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_sink = sink;
    }

    void Logger::log(LogLevel level, const std::string &message)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (level < m_minLevel)
        {
            return;
        }
        if (!m_sink)
        {
            if (level != spatial::LogLevel::Error)
            {
                std::cout << level << " " << message << std::endl;
            }
            else
            {
                std::cerr << level << " " << message << std::endl;
            }
            return;
        }
        m_sink(level, message);
    }

} // namespace spatial
