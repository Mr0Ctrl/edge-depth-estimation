// singleton
#pragma once
#include <functional>
#include <mutex>
#include <string>
#include <ostream>

namespace spatial
{
    enum class LogLevel
    {
        Debug = 0,
        Info = 1,
        Warn = 2,
        Error = 3,
    };

    inline std::ostream &operator<<(std::ostream &os, LogLevel level)
    {
        switch (level)
        {
        case LogLevel::Debug:
            os << "[DEBUG]";
            break;
        case LogLevel::Info:
            os << "[INFO]";
            break;
        case LogLevel::Warn:
            os << "[WARN]";
            break;
        case LogLevel::Error:
            os << "[ERROR]";
            break;
        }
        return os;
    }

    using LogSink = std::function<void(LogLevel, const std::string &)>;

    class Logger
    {
    private:
        Logger() = default;

        LogLevel m_minLevel = LogLevel::Info;
        LogSink m_sink;
        std::mutex m_mutex;

    public:
        static Logger &instance();

        Logger(const Logger &) = delete;
        Logger &operator=(const Logger &) = delete;

        void setLevel(LogLevel level);
        void setSink(LogSink sink);
        void log(LogLevel level, const std::string &message);
    };

} // namespace spatial

#define SPATIAL_LOG_INFO(msg) spatial::Logger::instance().log(spatial::LogLevel::Info, (msg))
#define SPATIAL_LOG_WARN(msg) spatial::Logger::instance().log(spatial::LogLevel::Warn, (msg))
#define SPATIAL_LOG_ERROR(msg) spatial::Logger::instance().log(spatial::LogLevel::Error, (msg))

#ifdef NDEBUG
#define SPATIAL_LOG_DEBUG(msg) ((void)0)
#else
#define SPATIAL_LOG_DEBUG(msg) spatial::Logger::instance().log(spatial::LogLevel::Debug, (msg))
#endif