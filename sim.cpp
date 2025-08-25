#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>

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
    Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(-1, 1, -1),
    Vec3(-1, -1, 1), Vec3(1, -1, 1), Vec3(1, 1, 1), Vec3(-1, 1, 1)
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

// Generate SVG image of the cube
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
    
    // Calculate face depths for painter's algorithm
    std::vector<std::pair<double, int>> faceDepths;
    for (int i = 0; i < cubeFaces.size(); i++) {
        // Calculate face center for depth sorting
        Vec3 center(0, 0, 0);
        for (int vertexIndex : cubeFaces[i]) {
            center = center + rotatedVertices[vertexIndex];
        }
        center = center * (1.0 / cubeFaces[i].size());
        faceDepths.push_back({center.z, i});
    }
    
    // Sort faces by depth (painter's algorithm)
    std::sort(faceDepths.begin(), faceDepths.end(), [](const auto& a, const auto& b) {
        return a.first > b.first; // Further faces first
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
                std::string svg = generateSVG(cubeVertices, angle, angle * 0.7, angle * 0.3);
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
        <div id="cubeContainer">
            <img id="cubeImage" src="cube.svg" width="400" height="400">
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