#include <stdexcept>

struct NotImplementedException : public std::logic_error {
	NotImplementedException(const std::string& msg = "Function Not Implemented") : std::logic_error(msg) {}
};
