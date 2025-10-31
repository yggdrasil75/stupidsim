#pragma once

#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>
#include <functional>

class FunctionTimer {
public:
    enum class Mode { BASIC, ENHANCED };
    
    struct TimingStats {
        size_t call_count = 0;
        double total_time = 0.0;
        std::vector<double> timings;
        
        double avg_time() const {
            return call_count > 0 ? total_time / call_count : 0.0;
        }
    };
    
    struct PercentileStats {
        double p99_9 = 0.0;
        double p99 = 0.0;
        double p95 = 0.0;
        double p90 = 0.0;
        double max = 0.0;
        double min = 0.0;
        double median = 0.0;
    };
    
    // Record timing for a function
    static void recordTiming(const std::string& func_name, double elapsed_seconds);
    
    // Get statistics
    static std::unordered_map<std::string, TimingStats> getStats(Mode mode = Mode::BASIC);
    
    // Print statistics
    static void printStats(Mode mode = Mode::ENHANCED);
    
    // Clear all statistics
    static void clearStats();

private:
    static std::unordered_map<std::string, TimingStats> stats_;
    
    static PercentileStats calculatePercentiles(const std::vector<double>& timings);
};

// Macro to easily time functions - similar to Python decorator
#define TIME_FUNCTION auto function_timer_scoped_ = ScopedFunctionTimer(__func__)

// Scoped timer for RAII-style timing
class ScopedFunctionTimer {
public:
    ScopedFunctionTimer(const std::string& func_name) 
        : func_name_(func_name), start_(std::chrono::steady_clock::now()) {}
    
    ~ScopedFunctionTimer() {
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        FunctionTimer::recordTiming(func_name_, elapsed.count() / 1000000.0);
    }

private:
    std::string func_name_;
    std::chrono::steady_clock::time_point start_;
};

// Template decorator for functions (similar to Python)
template<typename Func, typename... Args>
auto time_function_decorator(const std::string& func_name, Func&& func, Args&&... args) {
    auto start = std::chrono::steady_clock::now();
    
    if constexpr (std::is_void_v<std::invoke_result_t<Func, Args...>>) {
        // Void return type
        std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        FunctionTimer::recordTiming(func_name, elapsed.count() / 1000000.0);
    } else {
        // Non-void return type
        auto result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        FunctionTimer::recordTiming(func_name, elapsed.count() / 1000000.0);
        return result;
    }
}

// Macro to create decorated functions
#define DECORATE_FUNCTION(func) [&](auto&&... args) { \
    return time_function_decorator(#func, func, std::forward<decltype(args)>(args)...); \
}
