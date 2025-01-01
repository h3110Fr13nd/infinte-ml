#include "logger.h"
#include <ctime>
#include <iostream>
#include <sstream>
using namespace std;

// Constructor: Opens the log file in append mode
Logger::Logger(const string& filename)
{
    logFile.open(filename, ios::app);
    if (!logFile.is_open()) {
        cerr << "Error opening log file." << endl;
    }
}

// Destructor: Closes the log file
Logger::~Logger() { logFile.close(); }

// Logs a message with a given log level
void Logger::log(LogLevel level, const string& message)
{
    // Get current timestamp
    time_t now = time(0);
    tm* timeinfo = localtime(&now);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp),
             "%Y-%m-%d %H:%M:%S", timeinfo);

    // Create log entry
    ostringstream logEntry;
    logEntry << "[" << timestamp << "] "
             << levelToString(level) << ": " << message
             << endl;

    // Output to console
    cout << logEntry.str();

    // Output to log file
    if (logFile.is_open()) {
        logFile << logEntry.str();
        logFile
            .flush(); // Ensure immediate write to file
    }
}

void Logger::debug(const string& message)
{
    log(DEBUG, message);
}

void Logger::info(const string& message)
{
    log(INFO, message);
}

void Logger::warning(const string& message)
{
    log(WARNING, message);
}

void Logger::error(const string& message)
{
    log(ERROR, message);
}

void Logger::critical(const string& message)
{
    log(CRITICAL, message);
}

// Overloaded operators for convenience
Logger& Logger::operator<<(const string& message)
{
    log(INFO, message);
    return *this;
}

// Converts log level to a string for output
string Logger::levelToString(LogLevel level)
{
    switch (level) {
    case DEBUG:
        return "DEBUG";
    case INFO:
        return "INFO";
    case WARNING:
        return "WARNING";
    case ERROR:
        return "ERROR";
    case CRITICAL:
        return "CRITICAL";
    default:
        return "UNKNOWN";
    }
}