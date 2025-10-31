#include "timing_decorator.hpp"
#include <cmath>

std::unordered_map<std::string, FunctionTimer::TimingStats> FunctionTimer::stats_;

void FunctionTimer::recordTiming(const std::string& func_name, double elapsed_seconds) {
    auto& stat = stats_[func_name];
    stat.call_count++;
    stat.total_time += elapsed_seconds;
    stat.timings.push_back(elapsed_seconds);
}

FunctionTimer::PercentileStats FunctionTimer::calculatePercentiles(const std::vector<double>& timings) {
    PercentileStats result;
    if (timings.empty()) return result;
    
    std::vector<double> sorted_timings = timings;
    std::sort(sorted_timings.begin(), sorted_timings.end());
    
    auto percentile_index = [&](double p) -> size_t {
        return std::min(sorted_timings.size() - 1, 
                       static_cast<size_t>(p * sorted_timings.size() / 100.0));
    };
    
    result.min = sorted_timings.front();
    result.max = sorted_timings.back();
    result.median = sorted_timings[percentile_index(50.0)];
    result.p90 = sorted_timings[percentile_index(90.0)];
    result.p95 = sorted_timings[percentile_index(95.0)];
    result.p99 = sorted_timings[percentile_index(99.0)];
    result.p99_9 = sorted_timings[percentile_index(99.9)];
    
    return result;
}

std::unordered_map<std::string, FunctionTimer::TimingStats> FunctionTimer::getStats(Mode mode) {
    return stats_; // In C++ we return all data, filtering happens in print
}

void FunctionTimer::printStats(Mode mode) {
    if (stats_.empty()) {
        std::cout << "No timing statistics available." << std::endl;
        return;
    }

    // Determine column widths
    size_t func_col_width = 0;
    for (const auto& [name, _] : stats_) {
        func_col_width = std::max(func_col_width, name.length());
    }
    func_col_width = std::max(func_col_width, size_t(8)); // "Function"
    
    const int num_width = 12;

    if (mode == Mode::BASIC) {
        std::cout << "\nBasic Function Timing Statistics:" << std::endl;
        std::cout << std::string(func_col_width + 3 * num_width + 8, '-') << std::endl;
        
        std::cout << std::left << std::setw(func_col_width) << "Function"
                  << std::setw(num_width) << "Calls"
                  << std::setw(num_width) << "Total (s)"
                  << std::setw(num_width) << "Avg (s)" << std::endl;
                  
        std::cout << std::string(func_col_width + 3 * num_width + 8, '-') << std::endl;
        
        for (const auto& [func_name, data] : stats_) {
            if (data.call_count > 0) {
                std::cout << std::left << std::setw(func_col_width) << func_name
                          << std::setw(num_width) << data.call_count
                          << std::setw(num_width) << std::fixed << std::setprecision(6) << data.total_time
                          << std::setw(num_width) << std::fixed << std::setprecision(6) << data.avg_time() << std::endl;
            }
        }
        
        std::cout << std::string(func_col_width + 3 * num_width + 8, '-') << std::endl;
    } else { // ENHANCED mode
        std::cout << "\nEnhanced Function Timing Statistics:" << std::endl;
        size_t total_width = func_col_width + 8 * num_width + 8;
        std::cout << std::string(total_width, '-') << std::endl;
        
        std::cout << std::left << std::setw(func_col_width) << "Function"
                  << std::setw(num_width) << "Calls"
                  << std::setw(num_width) << "Total (s)"
                  << std::setw(num_width) << "Avg (s)"
                  << std::setw(num_width) << "Min (s)"
                  << std::setw(num_width) << "Median (s)"
                  << std::setw(num_width) << "P99 (s)"
                  << std::setw(num_width) << "P99.9 (s)"
                  << std::setw(num_width) << "Max (s)" << std::endl;
                  
        std::cout << std::string(total_width, '-') << std::endl;
        
        for (const auto& [func_name, data] : stats_) {
            if (data.call_count > 0) {
                auto percentiles = calculatePercentiles(data.timings);
                
                std::cout << std::left << std::setw(func_col_width) << func_name
                          << std::setw(num_width) << data.call_count
                          << std::setw(num_width) << std::fixed << std::setprecision(6) << data.total_time
                          << std::setw(num_width) << std::fixed << std::setprecision(6) << data.avg_time()
                          << std::setw(num_width) << std::fixed << std::setprecision(6) << percentiles.min
                          << std::setw(num_width) << std::fixed << std::setprecision(6) << percentiles.median
                          << std::setw(num_width) << std::fixed << std::setprecision(6) << percentiles.p99
                          << std::setw(num_width) << std::fixed << std::setprecision(6) << percentiles.p99_9
                          << std::setw(num_width) << std::fixed << std::setprecision(6) << percentiles.max << std::endl;
            }
        }
        
        std::cout << std::string(total_width, '-') << std::endl;
    }
}

void FunctionTimer::clearStats() {
    stats_.clear();
}