# Compiler settings
CXX := g++
CXXFLAGS := -std=c++17 -Wall -O3

# Detect Python venv (if active)
VENV_PATH := $(shell python -c "import sys; print(sys.prefix)" 2>/dev/null)

# Python and Eigen paths
PYTHON_VERSION := $(shell python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDE := $(shell python$(PYTHON_VERSION)-config --includes)
PYTHON_LDFLAGS := $(shell python$(PYTHON_VERSION)-config --ldflags)

# Eigen path (default on Ubuntu/Debian)
EIGEN_INCLUDE := /usr/include/eigen3

# If venv is active, use its matplotlibcpp.h
ifneq ($(VENV_PATH),)
    MATPLOTLIBCPP_INCLUDE := $(VENV_PATH)/lib/python$(PYTHON_VERSION)/site-packages
else
    # Fallback to system-wide matplotlibcpp.h (if exists)
    MATPLOTLIBCPP_INCLUDE := /usr/local/include
endif

# Target executable
TARGET := icosphere

# Source files
SRC := main.cpp

# Build rules
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) \
		-I$(EIGEN_INCLUDE) \
		-I$(MATPLOTLIBCPP_INCLUDE) \
		$(PYTHON_INCLUDE) \
		$(PYTHON_LDFLAGS) \
		-o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean