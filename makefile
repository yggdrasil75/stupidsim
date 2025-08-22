# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Platform-specific flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CXXFLAGS += -D_LINUX
    LIBS = -lvulkan -lglfw -lpthread
endif
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -D_MACOS
    LIBS = -lvulkan -lglfw -lpthread -framework Cocoa -framework IOKit -framework CoreVideo
endif
ifeq ($(OS),Windows_NT)
    CXXFLAGS += -D_WINDOWS
    LIBS = -lvulkan-1 -lglfw3 -lws2_32
endif

# Source files
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = cube_renderer

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps-ubuntu:
	sudo apt update
	sudo apt install -y libvulkan-dev libglfw3-dev libglm-dev build-essential

# Install dependencies (macOS with Homebrew)
install-deps-macos:
	brew update
	brew install vulkan-headers vulkan-tools glfw

# Install dependencies (Windows with vcpkg)
install-deps-windows:
	vcpkg install vulkan glfw3

# Run the application
run: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean install-deps-ubuntu install-deps-macos install-deps-windows run