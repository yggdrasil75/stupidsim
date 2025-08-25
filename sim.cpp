#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <functional>

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

#define M_PI 3.14159265358979323846

class Vec3 {
public:
    double x, y, z;
    
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    
    // Inline for maximum performance
    inline double norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    inline Vec3 normalize() const {
        double n = norm();
        return Vec3(x/n, y/n, z/n);
    }
    
    // Cross product (essential for face normals)
    inline Vec3 cross(const Vec3& other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    // Dot product
    inline double dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    // Arithmetic operations
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
    
    // For using Vec3 as a key in unordered_map
    bool operator==(const Vec3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    
    // Hash function for Vec3
    struct Hash {
        size_t operator()(const Vec3& v) const {
            size_t h1 = std::hash<double>()(v.x);
            size_t h2 = std::hash<double>()(v.y);
            size_t h3 = std::hash<double>()(v.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
};

// Optimized face normal calculation
inline Vec3 calculateFaceNormal(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    return edge1.cross(edge2).normalize();
}

// Dot product
double dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Cross product
Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

// Simple 2D point
struct Point2D {
    int x, y;
    Point2D(int x = 0, int y = 0) : x(x), y(y) {}
};

// Cube vertices
const std::vector<Vec3> cubeVertices = {
    Vec3(-0.5, -0.5, -0.5), Vec3(0.5, -0.5, -0.5), Vec3(0.5, 0.5, -0.5), Vec3(-0.5, 0.5, -0.5),
    Vec3(-0.5, -0.5, 0.5), Vec3(0.5, -0.5, 0.5), Vec3(0.5, 0.5, 0.5), Vec3(-0.5, 0.5, 0.5)
};

// Cube faces (vertex indices)
const std::vector<std::vector<int>> cubeFaces = {
    {0, 1, 2, 3}, // bottom
    {4, 5, 6, 7}, // top
    {0, 1, 5, 4}, // front
    {2, 3, 7, 6}, // back
    {0, 3, 7, 4}, // left
    {1, 2, 6, 5}  // right
};

// Face colors
const std::vector<std::string> faceColors = {
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"
};

// Generate a sphere of voxels (cubes)
std::vector<Vec3> generateVoxelSphere(int numVoxels, double radius) {
    std::vector<Vec3> voxels;
    voxels.reserve(numVoxels);
    
    // Calculate the number of layers needed for the sphere
    int layers = std::cbrt(numVoxels) * 1.2;
    if (layers < 1) layers = 1;
    
    // Distribute voxels in a spherical pattern
    int voxelsPerLayer = numVoxels / layers;
    int remainingVoxels = numVoxels % layers;
    
    for (int i = 0; i < layers; i++) {
        double phi = M_PI * i / (layers - 1);
        int voxelsInThisLayer = voxelsPerLayer + (i < remainingVoxels ? 1 : 0);
        
        for (int j = 0; j < voxelsInThisLayer; j++) {
            double theta = 2 * M_PI * j / voxelsInThisLayer;
            
            // Convert spherical coordinates to Cartesian
            double x = radius * std::sin(phi) * std::cos(theta);
            double y = radius * std::sin(phi) * std::sin(theta);
            double z = radius * std::cos(phi);
            
            voxels.emplace_back(x, y, z);
        }
    }
    
    return voxels;
}

// Rotation function
Vec3 rotate(const Vec3& point, double angleX, double angleY, double angleZ) {
    // Rotate around X axis
    double y1 = point.y * cos(angleX) - point.z * sin(angleX);
    double z1 = point.y * sin(angleX) + point.z * cos(angleX);
    
    // Rotate around Y axis
    double x2 = point.x * cos(angleY) + z1 * sin(angleY);
    double z2 = -point.x * sin(angleY) + z1 * cos(angleY);
    
    // Rotate around Z axis
    double x3 = x2 * cos(angleZ) - y1 * sin(angleZ);
    double y3 = x2 * sin(angleZ) + y1 * cos(angleZ);
    
    return Vec3(x3, y3, z2);
}

// Project 3D point to 2D
Point2D project(const Vec3& point, int width, int height) {
    // Simple perspective projection
    double scale = 200.0 / (point.z + 5.0);
    int x = width / 2 + static_cast<int>(point.x * scale);
    int y = height / 2 - static_cast<int>(point.y * scale);
    return Point2D(x, y);
}

std::string generateSVG(const std::vector<Vec3>& voxels, double angleX, double angleY, double angleZ) {
    const int width = 800;
    const int height = 600;
    
    std::stringstream svg;
    svg << "<svg width='" << width << "' height='" << height << "' xmlns='http://www.w3.org/2000/svg'>";
    svg << "<rect width='100%' height='100%' fill='#222'/>";
    
    // Rotate all voxels
    std::vector<Vec3> rotatedVoxels;
    rotatedVoxels.reserve(voxels.size());
    for (const auto& voxel : voxels) {
        rotatedVoxels.push_back(rotate(voxel, angleX, angleY, angleZ));
    }
    
    // For each voxel, generate its 8 vertices
    std::vector<std::vector<Vec3>> voxelVertices;
    voxelVertices.reserve(voxels.size());
    for (const auto& center : rotatedVoxels) {
        std::vector<Vec3> vertices;
        vertices.reserve(8);
        for (const auto& vertex : cubeVertices) {
            vertices.push_back(center + vertex * 0.2); // Reduced scale for better visibility
        }
        voxelVertices.push_back(vertices);
    }
    
    // Project all vertices
    std::vector<std::vector<Point2D>> projectedVoxelVertices;
    projectedVoxelVertices.reserve(voxels.size());
    for (const auto& vertices : voxelVertices) {
        std::vector<Point2D> projectedVertices;
        projectedVertices.reserve(8);
        for (const auto& vertex : vertices) {
            projectedVertices.push_back(project(vertex, width, height));
        }
        projectedVoxelVertices.push_back(projectedVertices);
    }
    
    // Calculate depths for each voxel (using minimum Z of all vertices for better sorting)
    std::vector<std::pair<double, int>> voxelDepths;
    for (int i = 0; i < voxelVertices.size(); i++) {
        double minZ;
        for (const auto& vertex : voxelVertices[i]) {
            minZ = min(minZ, vertex.z);
        }
        voxelDepths.push_back({minZ, i});
    }
    
    // Sort voxels by depth (painter's algorithm) - farthest first
    std::sort(voxelDepths.begin(), voxelDepths.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    // Draw voxels in correct order
    for (const auto& depthVoxel : voxelDepths) {
        int voxelIndex = depthVoxel.second;
        const auto& vertices = projectedVoxelVertices[voxelIndex];
        const auto& worldVertices = voxelVertices[voxelIndex];
        
        // Draw faces with backface culling
        for (int faceIndex = 0; faceIndex < cubeFaces.size(); faceIndex++) {
            const auto& face = cubeFaces[faceIndex];
            
            // Backface culling
            Vec3 v0 = worldVertices[face[0]];
            Vec3 v1 = worldVertices[face[1]];
            Vec3 v2 = worldVertices[face[2]];
            
            Vec3 normal = calculateFaceNormal(v0, v1, v2);
            Vec3 viewDir(0, 0, 1); // Viewing along Z axis
            
            if (normal.dot(viewDir) > 0) { // Face is pointing toward viewer
                svg << "<polygon points='";
                for (int vertexIndex : face) {
                    const Point2D& p = vertices[vertexIndex];
                    svg << p.x << "," << p.y << " ";
                }
                svg << "' fill='" << faceColors[faceIndex] << "' stroke='#000' stroke-width='1' opacity='0.8'/>";
            }
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
    
public:
    SimpleHTTPServer(int port) : port(port), serverSocket(-1) {}
    
    ~SimpleHTTPServer() {
        stop();
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
        // Generate the voxel sphere once
        std::vector<Vec3> voxelSphere = generateVoxelSphere(1000, 3.0);
        
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
                static double angle = 0.0;
                angle += 0.02;
                std::string svg = generateSVG(voxelSphere, angle, angle * 0.7, angle * 0.3);
                response = "HTTP/1.1 200 OK\r\nContent-Type: image/svg+xml\r\n\r\n" + svg;
            } else {
                response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\n404 Not Found";
            }
            
            send(clientSocket, response.c_str(), response.length(), 0);
            
#ifdef _WIN32
            closesocket(clientSocket);
#else
            close(clientSocket);
#endif
            
            // Small delay to prevent high CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    std::string getHTML() {
        return R"(
<!DOCTYPE html>
<html>
<head>
    <title>3D Voxel Sphere Renderer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .container {
            max-width: 900px;
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        h1 {
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
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
        .footer {
            margin-top: 30px;
            font-size: 0.8em;
            opacity: 0.7;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>3D Voxel Sphere Renderer</h1>
        <p>Rendering a sphere composed of 1000 cubes (voxels)</p>
        
        <div id="cubeContainer">
            <img id="cubeImage" src="cube.svg" width="800" height="600">
        </div>
        
        <div class="instructions">
            <h3>How it works:</h3>
            <ul>
                <li>The sphere is composed of 1000 individual cubes (voxels)</li>
                <li>Each cube is rendered with proper depth sorting</li>
                <li>Backface culling is applied to improve performance</li>
                <li>The sphere rotates automatically for visualization</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Implemented in C++ with SVG rendering | Server running on port 5101</p>
        </div>
    </div>

    <script>
        // Auto-refresh the image for animation
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

int main() {
    SimpleHTTPServer server(5101);
    
    if (!server.start()) {
        std::cerr << "Failed to start server" << std::endl;
        return 1;
    }
    
    std::cout << "Open your browser and navigate to http://localhost:5101" << std::endl;
    std::cout << "Press Ctrl+C to stop the server" << std::endl;
    
    server.handleRequests();
    
    return 0;
}