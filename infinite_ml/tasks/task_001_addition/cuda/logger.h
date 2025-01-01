#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>

// Define log levels
enum LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

class Logger {
private:
    std::ofstream logFile;

    // Convert log level to string
    std::string levelToString(LogLevel level);

public:
    // Constructor and destructor
    Logger(const std::string& filename);
    ~Logger();

    // Log a message with a specific level
    void log(LogLevel level, const std::string& message);

    // Convenience methods for different log levels
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    void critical(const std::string& message);

    // Overloaded operators for convenience
    Logger& operator<<(const std::string& message);
};

#endif // LOGGER_H
