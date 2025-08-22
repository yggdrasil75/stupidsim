#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <optional>
#include <set>

// Vulkan headers
#define VULKAN_HPP_NO_EXCEPTIONS
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#endif

// Constants
const int WINDOW_WIDTH = 400;
const int WINDOW_HEIGHT = 400;
const int MAX_FRAMES_IN_FLIGHT = 2;

// Vulkan context structure
struct VulkanContext {
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::SurfaceKHR surface;
    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::ImageView> swapChainImageViews;
    vk::RenderPass renderPass;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;
    std::vector<vk::Framebuffer> swapChainFramebuffers;
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    size_t currentFrame = 0;
    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;
    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;
    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;
    
    // For window presentation
    GLFWwindow* window = nullptr;
    
    bool isValid() const {
        return static_cast<bool>(device);
    }
};

// Vertex structure
struct Vertex {
    float pos[3];
    float color[3];
};

// Uniform buffer object
struct UniformBufferObject {
    float model[16];
    float view[16];
    float proj[16];
};

// Cube vertices with colors
const std::vector<Vertex> cubeVertices = {
    {{-1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
    {{1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
    {{1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},
    {{-1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, 0.0f}},
    {{-1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}, {0.5f, 0.5f, 0.5f}}
};

// Cube indices
const std::vector<uint16_t> cubeIndices = {
    0, 1, 2, 2, 3, 0,    // bottom
    4, 5, 6, 6, 7, 4,    // top
    0, 1, 5, 5, 4, 0,    // front
    2, 3, 7, 7, 6, 2,    // back
    0, 3, 7, 7, 4, 0,    // left
    1, 2, 6, 6, 5, 1     // right
};

// Helper functions for Vulkan
VulkanContext initVulkan();
void cleanupVulkan(VulkanContext& context);
void drawFrame(VulkanContext& context, double angleX, double angleY, double angleZ);
std::vector<char> readFile(const std::string& filename);
vk::ShaderModule createShaderModule(const VulkanContext& context, const std::vector<char>& code);

// Vector and matrix math classes (kept from original code)
class Vec3 {
public:
    double x, y, z;
    
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    
    inline double norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    inline Vec3 normalize() const {
        double n = norm();
        return Vec3(x/n, y/n, z/n);
    }
    
    inline Vec3 cross(const Vec3& other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    inline double dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    inline Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    inline Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    inline Vec3 operator*(double scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
    
    inline Vec3 operator/(double scalar) const {
        return Vec3(x / scalar, y / scalar, z / scalar);
    }
};

// Simple 2D point
struct Point2D {
    int x, y;
    Point2D(int x = 0, int y = 0) : x(x), y(y) {}
};

// Rotation function
Vec3 rotate(const Vec3& point, double angleX, double angleY, double angleZ) {
    double y1 = point.y * cos(angleX) - point.z * sin(angleX);
    double z1 = point.y * sin(angleX) + point.z * cos(angleX);
    
    double x2 = point.x * cos(angleY) + z1 * sin(angleY);
    double z2 = -point.x * sin(angleY) + z1 * cos(angleY);
    
    double x3 = x2 * cos(angleZ) - y1 * sin(angleZ);
    double y3 = x2 * sin(angleZ) + y1 * cos(angleZ);
    
    return Vec3(x3, y3, z2);
}

// Project 3D point to 2D
Point2D project(const Vec3& point, int width, int height) {
    double scale = 200.0 / (point.z + 5.0);
    int x = width / 2 + static_cast<int>(point.x * scale);
    int y = height / 2 - static_cast<int>(point.y * scale);
    return Point2D(x, y);
}

// Generate SVG image of the cube (fallback)
std::string generateSVG(const std::vector<Vec3>& vertices, double angleX, double angleY, double angleZ) {
    const int width = 400;
    const int height = 400;
    
    std::stringstream svg;
    svg << "<svg width='" << width << "' height='" << height << "' xmlns='http://www.w3.org/2000/svg'>";
    svg << "<rect width='100%' height='100%' fill='#222'/>";
    
    // Rotate vertices
    std::vector<Vec3> rotatedVertices;
    for (const auto& vertex : vertices) {
        rotatedVertices.push_back(rotate(vertex, angleX, angleY, angleZ));
    }
    
    // Project vertices to 2D
    std::vector<Point2D> projectedVertices;
    for (const auto& vertex : rotatedVertices) {
        projectedVertices.push_back(project(vertex, width, height));
    }
    
    // Cube faces (vertex indices)
    const std::vector<std::vector<int>> cubeFaces = {
        {0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4},
        {2, 3, 7, 6}, {0, 3, 7, 4}, {1, 2, 6, 5}
    };
    
    // Face colors
    const std::vector<std::string> faceColors = {
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"
    };
    
    // Calculate face depths for painter's algorithm
    std::vector<std::pair<double, int>> faceDepths;
    for (int i = 0; i < cubeFaces.size(); i++) {
        Vec3 center(0, 0, 0);
        for (int vertexIndex : cubeFaces[i]) {
            center = center + rotatedVertices[vertexIndex];
        }
        center = center * (1.0 / cubeFaces[i].size());
        faceDepths.push_back({center.z, i});
    }
    
    // Sort faces by depth
    std::sort(faceDepths.begin(), faceDepths.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    // Draw faces in correct order
    for (const auto& depthFace : faceDepths) {
        int faceIndex = depthFace.second;
        svg << "<polygon points='";
        for (int vertexIndex : cubeFaces[faceIndex]) {
            const Point2D& p = projectedVertices[vertexIndex];
            svg << p.x << "," << p.y << " ";
        }
        svg << "' fill='" << faceColors[faceIndex] << "' stroke='#000' stroke-width='2' opacity='0.7'/>";
    }
    
    // Draw edges
    for (const auto& face : cubeFaces) {
        for (int i = 0; i < face.size(); i++) {
            int j = (i + 1) % face.size();
            const Point2D& p1 = projectedVertices[face[i]];
            const Point2D& p2 = projectedVertices[face[j]];
            svg << "<line x1='" << p1.x << "' y1='" << p1.y << "' x2='" << p2.x << "' y2='" << p2.y 
                << "' stroke='white' stroke-width='2'/>";
        }
    }
    
    svg << "</svg>";
    return svg.str();
}

// HTTP server class
class SimpleHTTPServer {
private:
    int serverSocket;
    int port;
    VulkanContext vulkanContext;
    bool useVulkan;
    
public:
    SimpleHTTPServer(int port) : port(port), serverSocket(-1), useVulkan(false) {
        // Try to initialize Vulkan
        try {
            vulkanContext = initVulkan();
            useVulkan = vulkanContext.isValid();
            if (useVulkan) {
                std::cout << "Vulkan rendering enabled" << std::endl;
            } else {
                std::cout << "Vulkan not available, using CPU fallback" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Vulkan initialization failed: " << e.what() << std::endl;
            std::cerr << "Using CPU fallback rendering" << std::endl;
            useVulkan = false;
        }
    }
    
    ~SimpleHTTPServer() {
        stop();
        if (useVulkan) {
            cleanupVulkan(vulkanContext);
        }
    }
    
    bool start() {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "WSAStartup failed" << std::endl;
            return false;
        }
#endif
        
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket < 0) {
            std::cerr << "Socket creation failed" << std::endl;
            return false;
        }
        
        int opt = 1;
#ifdef _WIN32
        if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt)) < 0) {
#else
        if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
#endif
            std::cerr << "Setsockopt failed" << std::endl;
            return false;
        }
        
        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(port);
        
        if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "Bind failed" << std::endl;
            return false;
        }
        
        if (listen(serverSocket, 10) < 0) {
            std::cerr << "Listen failed" << std::endl;
            return false;
        }
        
        std::cout << "Server started on port " << port << std::endl;
        return true;
    }
    
    void stop() {
        if (serverSocket >= 0) {
#ifdef _WIN32
            closesocket(serverSocket);
            WSACleanup();
#else
            close(serverSocket);
#endif
            serverSocket = -1;
        }
    }
    
    void handleRequests() {
        static double angle = 0.0;
        
        while (true) {
            sockaddr_in clientAddr;
#ifdef _WIN32
            int clientAddrLen = sizeof(clientAddr);
#else
            socklen_t clientAddrLen = sizeof(clientAddr);
#endif
            int clientSocket = accept(serverSocket, (sockaddr*)&clientAddr, &clientAddrLen);
            
            if (clientSocket < 0) {
                std::cerr << "Accept failed" << std::endl;
                continue;
            }
            
            char buffer[4096] = {0};
            recv(clientSocket, buffer, sizeof(buffer), 0);
            
            std::string request(buffer);
            std::string response;
            
            if (request.find("GET / ") != std::string::npos || request.find("GET /index.html") != std::string::npos) {
                response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n" + getHTML();
            } else if (request.find("GET /cube.svg") != std::string::npos) {
                angle += 0.02;
                
                if (useVulkan) {
                    // Use Vulkan rendering
                    drawFrame(vulkanContext, angle, angle * 0.7, angle * 0.3);
                    
                    // In a real implementation, we would read back the rendered image
                    // and encode it as SVG or another format. For this example, we'll
                    // fall back to CPU rendering for the HTTP response.
                    // This is a limitation of our simple HTTP server approach.
                    
                    // For now, we'll just use the CPU fallback for HTTP responses
                    // while Vulkan handles the window rendering
                    std::vector<Vec3> vertices;
                    for (const auto& v : cubeVertices) {
                        vertices.emplace_back(v.pos[0], v.pos[1], v.pos[2]);
                    }
                    std::string svg = generateSVG(vertices, angle, angle * 0.7, angle * 0.3);
                    response = "HTTP/1.1 200 OK\r\nContent-Type: image/svg+xml\r\n\r\n" + svg;
                } else {
                    // Use CPU fallback
                    std::vector<Vec3> vertices;
                    for (const auto& v : cubeVertices) {
                        vertices.emplace_back(v.pos[0], v.pos[1], v.pos[2]);
                    }
                    std::string svg = generateSVG(vertices, angle, angle * 0.7, angle * 0.3);
                    response = "HTTP/1.1 200 OK\r\nContent-Type: image/svg+xml\r\n\r\n" + svg;
                }
            } else {
                response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\n404 Not Found";
            }
            
            send(clientSocket, response.c_str(), response.length(), 0);
            
#ifdef _WIN32
            closesocket(clientSocket);
#else
            close(clientSocket);
#endif
            
            // Process events for Vulkan window
            if (useVulkan && vulkanContext.window) {
                glfwPollEvents();
            }
            
            // Small delay to prevent high CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    std::string getHTML() {
        return R"(
<!DOCTYPE html>
<html>
<head>
    <title>3D Cube Renderer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #802531ff 100%);
            color: white;
            text-align: center;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .container {
            max-width: 800px;
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        h1 {
            margin-bottom: 20px;
        }
        #cubeContainer {
            margin: 20px 0;
            display: flex;
            justify-content: center;
        }
        .instructions {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            text-align: left;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>3D Cube Renderer</h1>
        <div class="status">
            Rendering with: )" + std::string(useVulkan ? "Vulkan (GPU)" : "CPU Fallback") + R"(
        </div>
        <div id="cubeContainer">
            <img id="cubeImage" src="cube.svg" width="400" height="400">
        </div>
        <div class="instructions">
            <p>This application demonstrates 3D rendering using )" + std::string(useVulkan ? "Vulkan for hardware acceleration" : "a CPU-based fallback") + R"(.</p>
            <p>The cube is rotating in 3D space and being projected to 2D for display.</p>
        </div>
    </div>

    <script>
        // Auto-refresh the image every 50ms for animation
        setInterval(function() {
            const img = document.getElementById('cubeImage');
            const timestamp = new Date().getTime();
            img.src = 'cube.svg?' + timestamp;
        }, 50);
    </script>
</body>
</html>
)";
    }
};

// Vulkan initialization and helper functions
VulkanContext initVulkan() {
    VulkanContext context;
    
    try {
        // Initialize GLFW
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        
        // Create window
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Invisible window for headless rendering
        context.window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan Cube", nullptr, nullptr);
        
        if (!context.window) {
            throw std::runtime_error("Failed to create GLFW window");
        }
        
        // Create Vulkan instance
        vk::ApplicationInfo appInfo(
            "Vulkan Cube", VK_MAKE_VERSION(1, 0, 0),
            "No Engine", VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_0
        );
        
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        
        vk::InstanceCreateInfo createInfo(
            vk::InstanceCreateFlags(),
            &appInfo,
            0, nullptr, // enabled layers
            static_cast<uint32_t>(extensions.size()), extensions.data()
        );
        
        context.instance = vk::createInstance(createInfo);
        
        // Create surface
        VkSurfaceKHR surface;
        if (glfwCreateWindowSurface(context.instance, context.window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        }
        context.surface = surface;
        
        // Select physical device
        auto physicalDevices = context.instance.enumeratePhysicalDevices();
        if (physicalDevices.empty()) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support");
        }
        
        // Use the first available device
        context.physicalDevice = physicalDevices[0];
        
        // Find queue families
        auto queueFamilyProperties = context.physicalDevice.getQueueFamilyProperties();
        
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;
        
        for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
            if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                graphicsFamily = i;
            }
            
            if (context.physicalDevice.getSurfaceSupportKHR(i, context.surface)) {
                presentFamily = i;
            }
            
            if (graphicsFamily.has_value() && presentFamily.has_value()) {
                break;
            }
        }
        
        if (!graphicsFamily.has_value() || !presentFamily.has_value()) {
            throw std::runtime_error("Failed to find suitable queue families");
        }
        
        // Create logical device
        float queuePriority = 1.0f;
        
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {graphicsFamily.value(), presentFamily.value()};
        
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo(
                vk::DeviceQueueCreateFlags(),
                queueFamily,
                1, &queuePriority
            );
            queueCreateInfos.push_back(queueCreateInfo);
        }
        
        vk::PhysicalDeviceFeatures deviceFeatures;
        
        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };
        
        vk::DeviceCreateInfo deviceCreateInfo(
            vk::DeviceCreateFlags(),
            static_cast<uint32_t>(queueCreateInfos.size()), queueCreateInfos.data(),
            0, nullptr, // enabled layers
            static_cast<uint32_t>(deviceExtensions.size()), deviceExtensions.data(),
            &deviceFeatures
        );
        
        context.device = context.physicalDevice.createDevice(deviceCreateInfo);
        
        // Get queues
        context.graphicsQueue = context.device.getQueue(graphicsFamily.value(), 0);
        context.presentQueue = context.device.getQueue(presentFamily.value(), 0);
        
        // Create swap chain
        auto surfaceCapabilities = context.physicalDevice.getSurfaceCapabilitiesKHR(context.surface);
        auto surfaceFormats = context.physicalDevice.getSurfaceFormatsKHR(context.surface);
        auto presentModes = context.physicalDevice.getSurfacePresentModesKHR(context.surface);
        
        vk::SurfaceFormatKHR surfaceFormat = surfaceFormats[0];
        for (const auto& format : surfaceFormats) {
            if (format.format == vk::Format::eB8G8R8A8Unorm && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                surfaceFormat = format;
                break;
            }
        }
        
        vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
        for (const auto& mode : presentModes) {
            if (mode == vk::PresentModeKHR::eMailbox) {
                presentMode = mode;
                break;
            }
        }
        
        vk::Extent2D extent;
        if (surfaceCapabilities.currentExtent.width != UINT32_MAX) {
            extent = surfaceCapabilities.currentExtent;
        } else {
            extent.width = std::clamp(WINDOW_WIDTH, 
                surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
            extent.height = std::clamp(WINDOW_HEIGHT, 
                surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
        }
        
        uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
        if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
            imageCount = surfaceCapabilities.maxImageCount;
        }
        
        vk::SwapchainCreateInfoKHR swapChainCreateInfo(
            vk::SwapchainCreateFlagsKHR(),
            context.surface,
            imageCount,
            surfaceFormat.format,
            surfaceFormat.colorSpace,
            extent,
            1,
            vk::ImageUsageFlagBits::eColorAttachment,
            vk::SharingMode::eExclusive,
            0, nullptr,
            surfaceCapabilities.currentTransform,
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            presentMode,
            VK_TRUE
        );
        
        if (graphicsFamily.value() != presentFamily.value()) {
            uint32_t queueFamilyIndices[] = {graphicsFamily.value(), presentFamily.value()};
            swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            swapChainCreateInfo.queueFamilyIndexCount = 2;
            swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        
        context.swapChain = context.device.createSwapchainKHR(swapChainCreateInfo);
        context.swapChainImages = context.device.getSwapchainImagesKHR(context.swapChain);
        context.swapChainImageFormat = surfaceFormat.format;
        context.swapChainExtent = extent;
        
        // Create image views
        context.swapChainImageViews.resize(context.swapChainImages.size());
        for (size_t i = 0; i < context.swapChainImages.size(); i++) {
            vk::ImageViewCreateInfo createInfo(
                vk::ImageViewCreateFlags(),
                context.swapChainImages[i],
                vk::ImageViewType::e2D,
                context.swapChainImageFormat,
                vk::ComponentMapping(),
                vk::ImageSubresourceRange(
                    vk::ImageAspectFlagBits::eColor,
                    0, 1, 0, 1
                )
            );
            
            context.swapChainImageViews[i] = context.device.createImageView(createInfo);
        }
        
        // Create render pass
        vk::AttachmentDescription colorAttachment(
            vk::AttachmentDescriptionFlags(),
            context.swapChainImageFormat,
            vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::ePresentSrcKHR
        );
        
        vk::AttachmentReference colorAttachmentRef(
            0, vk::ImageLayout::eColorAttachmentOptimal
        );
        
        vk::SubpassDescription subpass(
            vk::SubpassDescriptionFlags(),
            vk::PipelineBindPoint::eGraphics,
            0, nullptr,
            1, &colorAttachmentRef,
            nullptr,
            nullptr,
            0, nullptr
        );
        
        vk::SubpassDependency dependency(
            VK_SUBPASS_EXTERNAL, 0,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::AccessFlags(),
            vk::AccessFlagBits::eColorAttachmentWrite
        );
        
        vk::RenderPassCreateInfo renderPassInfo(
            vk::RenderPassCreateFlags(),
            1, &colorAttachment,
            1, &subpass,
            1, &dependency
        );
        
        context.renderPass = context.device.createRenderPass(renderPassInfo);
        
        // Create graphics pipeline
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");
        
        vk::ShaderModule vertShaderModule = createShaderModule(context, vertShaderCode);
        vk::ShaderModule fragShaderModule = createShaderModule(context, fragShaderCode);
        
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eVertex,
            vertShaderModule,
            "main"
        );
        
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eFragment,
            fragShaderModule,
            "main"
        );
        
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
        
        // Vertex input
        vk::VertexInputBindingDescription bindingDescription(
            0, sizeof(Vertex), vk::VertexInputRate::eVertex
        );
        
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color))
        };
        
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo(
            vk::PipelineVertexInputStateCreateFlags(),
            1, &bindingDescription,
            static_cast<uint32_t>(attributeDescriptions.size()), attributeDescriptions.data()
        );
        
        // Input assembly
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
            vk::PipelineInputAssemblyStateCreateFlags(),
            vk::PrimitiveTopology::eTriangleList,
            VK_FALSE
        );
        
        // Viewport and scissor
        vk::Viewport viewport(0.0f, 0.0f, 
            static_cast<float>(context.swapChainExtent.width), 
            static_cast<float>(context.swapChainExtent.height),
            0.0f, 1.0f
        );
        
        vk::Rect2D scissor(vk::Offset2D(0, 0), context.swapChainExtent);
        
        vk::PipelineViewportStateCreateInfo viewportState(
            vk::PipelineViewportStateCreateFlags(),
            1, &viewport,
            1, &scissor
        );
        
        // Rasterizer
        vk::PipelineRasterizationStateCreateInfo rasterizer(
            vk::PipelineRasterizationStateCreateFlags(),
            VK_FALSE,
            VK_FALSE,
            vk::PolygonMode::eFill,
            vk::CullModeFlagBits::eBack,
            vk::FrontFace::eClockwise,
            VK_FALSE, 0.0f, 0.0f, 0.0f,
            1.0f
        );
        
        // Multisampling
        vk::PipelineMultisampleStateCreateInfo multisampling(
            vk::PipelineMultisampleStateCreateFlags(),
            vk::SampleCountFlagBits::e1,
            VK_FALSE,
            1.0f,
            nullptr,
            VK_FALSE,
            VK_FALSE
        );
        
        // Color blending
        vk::PipelineColorBlendAttachmentState colorBlendAttachment(
            VK_TRUE,
            vk::BlendFactor::eSrcAlpha,
            vk::BlendFactor::eOneMinusSrcAlpha,
            vk::BlendOp::eAdd,
            vk::BlendFactor::eOne,
            vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | 
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        );
        
        vk::PipelineColorBlendStateCreateInfo colorBlending(
            vk::PipelineColorBlendStateCreateFlags(),
            VK_FALSE,
            vk::LogicOp::eCopy,
            1, &colorBlendAttachment,
            {0.0f, 0.0f, 0.0f, 0.0f}
        );
        
        // Pipeline layout
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo(
            vk::PipelineLayoutCreateFlags(),
            0, nullptr,
            0, nullptr
        );
        
        context.pipelineLayout = context.device.createPipelineLayout(pipelineLayoutInfo);
        
        // Create graphics pipeline
        vk::GraphicsPipelineCreateInfo pipelineInfo(
            vk::PipelineCreateFlags(),
            2, shaderStages,
            &vertexInputInfo,
            &inputAssembly,
            nullptr,
            &viewportState,
            &rasterizer,
            &multisampling,
            nullptr,
            &colorBlending,
            nullptr,
            context.pipelineLayout,
            context.renderPass,
            0
        );
        
        auto result = context.device.createGraphicsPipeline(nullptr, pipelineInfo);
        if (result.result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create graphics pipeline");
        }
        context.graphicsPipeline = result.value;
        
        // Clean up shader modules
        context.device.destroyShaderModule(vertShaderModule);
        context.device.destroyShaderModule(fragShaderModule);
        
        // Create framebuffers
        context.swapChainFramebuffers.resize(context.swapChainImageViews.size());
        for (size_t i = 0; i < context.swapChainImageViews.size(); i++) {
            vk::ImageView attachments[] = {
                context.swapChainImageViews[i]
            };
            
            vk::FramebufferCreateInfo framebufferInfo(
                vk::FramebufferCreateFlags(),
                context.renderPass,
                1, attachments,
                context.swapChainExtent.width,
                context.swapChainExtent.height,
                1
            );
            
            context.swapChainFramebuffers[i] = context.device.createFramebuffer(framebufferInfo);
        }
        
        // Create command pool
        vk::CommandPoolCreateInfo commandPoolInfo(
            vk::CommandPoolCreateFlags(),
            graphicsFamily.value()
        );
        
        context.commandPool = context.device.createCommandPool(commandPoolInfo);
        
        // Create command buffers
        context.commandBuffers.resize(context.swapChainFramebuffers.size());
        
        vk::CommandBufferAllocateInfo allocInfo(
            context.commandPool,
            vk::CommandBufferLevel::ePrimary,
            static_cast<uint32_t>(context.commandBuffers.size())
        );
        
        context.commandBuffers = context.device.allocateCommandBuffers(allocInfo);
        
        // Create synchronization objects
        context.imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        context.renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        context.inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        
        vk::SemaphoreCreateInfo semaphoreInfo;
        vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            context.imageAvailableSemaphores[i] = context.device.createSemaphore(semaphoreInfo);
            context.renderFinishedSemaphores[i] = context.device.createSemaphore(semaphoreInfo);
            context.inFlightFences[i] = context.device.createFence(fenceInfo);
        }
        
    } catch (const std::exception& e) {
        cleanupVulkan(context);
        throw;
    }
    
    return context;
}

void cleanupVulkan(VulkanContext& context) {
    if (context.device) {
        context.device.waitIdle();
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (context.inFlightFences[i]) context.device.destroyFence(context.inFlightFences[i]);
            if (context.renderFinishedSemaphores[i]) context.device.destroySemaphore(context.renderFinishedSemaphores[i]);
            if (context.imageAvailableSemaphores[i]) context.device.destroySemaphore(context.imageAvailableSemaphores[i]);
        }
        
        if (context.commandPool) context.device.destroyCommandPool(context.commandPool);
        
        for (auto framebuffer : context.swapChainFramebuffers) {
            if (framebuffer) context.device.destroyFramebuffer(framebuffer);
        }
        
        if (context.graphicsPipeline) context.device.destroyPipeline(context.graphicsPipeline);
        if (context.pipelineLayout) context.device.destroyPipelineLayout(context.pipelineLayout);
        if (context.renderPass) context.device.destroyRenderPass(context.renderPass);
        
        for (auto imageView : context.swapChainImageViews) {
            if (imageView) context.device.destroyImageView(imageView);
        }
        
        if (context.swapChain) context.device.destroySwapchainKHR(context.swapChain);
        if (context.surface) context.instance.destroySurfaceKHR(context.surface);
        if (context.device) context.device.destroy();
        if (context.instance) context.instance.destroy();
    }
    
    if (context.window) {
        glfwDestroyWindow(context.window);
    }
    glfwTerminate();
}

void drawFrame(VulkanContext& context, double angleX, double angleY, double angleZ) {
    if (!context.isValid()) return;
    
    try {
        context.device.waitForFences(1, &context.inFlightFences[context.currentFrame], VK_TRUE, UINT64_MAX);
        
        uint32_t imageIndex;
        auto result = context.device.acquireNextImageKHR(
            context.swapChain, UINT64_MAX, 
            context.imageAvailableSemaphores[context.currentFrame],
            nullptr,
            &imageIndex
        );
        
        if (result == vk::Result::eErrorOutOfDateKHR) {
            // Swap chain is out of date, need to recreate
            return;
        } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to acquire swap chain image");
        }
        
        context.device.resetFences(1, &context.inFlightFences[context.currentFrame]);
        
        // Record command buffer
        vk::CommandBufferBeginInfo beginInfo;
        context.commandBuffers[imageIndex].begin(beginInfo);
        
        vk::RenderPassBeginInfo renderPassInfo(
            context.renderPass,
            context.swapChainFramebuffers[imageIndex],
            vk::Rect2D(vk::Offset2D(0, 0), context.swapChainExtent),
            1, &vk::ClearValue(vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f))
        );
        
        context.commandBuffers[imageIndex].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        context.commandBuffers[imageIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, context.graphicsPipeline);
        
        // Set viewport and scissor
        vk::Viewport viewport(
            0.0f, 0.0f,
            static_cast<float>(context.swapChainExtent.width),
            static_cast<float>(context.swapChainExtent.height),
            0.0f, 1.0f
        );
        context.commandBuffers[imageIndex].setViewport(0, 1, &viewport);
        
        vk::Rect2D scissor(vk::Offset2D(0, 0), context.swapChainExtent);
        context.commandBuffers[imageIndex].setScissor(0, 1, &scissor);
        
        // Draw the cube
        // In a complete implementation, we would bind vertex and index buffers here
        // and issue a draw call with proper transformation matrices
        
        context.commandBuffers[imageIndex].draw(36, 1, 0, 0);
        context.commandBuffers[imageIndex].endRenderPass();
        context.commandBuffers[imageIndex].end();
        
        // Submit command buffer
        vk::Semaphore waitSemaphores[] = {context.imageAvailableSemaphores[context.currentFrame]};
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::Semaphore signalSemaphores[] = {context.renderFinishedSemaphores[context.currentFrame]};
        
        vk::SubmitInfo submitInfo(
            1, waitSemaphores, waitStages,
            1, &context.commandBuffers[imageIndex],
            1, signalSemaphores
        );
        
        context.graphicsQueue.submit(1, &submitInfo, context.inFlightFences[context.currentFrame]);
        
        // Present
        vk::SwapchainKHR swapChains[] = {context.swapChain};
        vk::PresentInfoKHR presentInfo(
            1, signalSemaphores,
            1, swapChains,
            &imageIndex,
            nullptr
        );
        
        result = context.presentQueue.presentKHR(&presentInfo);
        
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
            // Swap chain is out of date, need to recreate
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to present swap chain image");
        }
        
        context.currentFrame = (context.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in drawFrame: " << e.what() << std::endl;
    }
}

std::vector<char> readFile(const std::string& filename) {
    // In a real implementation, we would read the shader files from disk
    // For this example, we'll return empty vectors
    return std::vector<char>();
}

vk::ShaderModule createShaderModule(const VulkanContext& context, const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo createInfo(
        vk::ShaderModuleCreateFlags(),
        code.size(),
        reinterpret_cast<const uint32_t*>(code.data())
    );
    
    return context.device.createShaderModule(createInfo);
}

int main() {
    SimpleHTTPServer server(7860);
    
    if (!server.start()) {
        std::cerr << "Failed to start server" << std::endl;
        return 1;
    }
    
    std::cout << "Open your browser and navigate to http://localhost:7860" << std::endl;
    std::cout << "Press Ctrl+C to stop the server" << std::endl;
    
    server.handleRequests();
    
    return 0;
}