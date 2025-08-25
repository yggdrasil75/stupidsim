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
#include <map>

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
#define M_PHI 1.61803398874989484820458683436563811772030917980576286213544862270526046281890
#define M_TAU 6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696506842341359

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
    
    bool operator==(const Vec3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    
    bool operator<(const Vec3& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
    
    struct Hash {
        size_t operator()(const Vec3& v) const {
            size_t h1 = std::hash<double>()(std::round(v.x * 1000.0));
            size_t h2 = std::hash<double>()(std::round(v.y * 1000.0));
            size_t h3 = std::hash<double>()(std::round(v.z * 1000.0));
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
};

class Triangle {
public:
    Vec3 v0, v1, v2;
    
    Triangle(const Vec3& v0, const Vec3& v1, const Vec3& v2) : v0(v0), v1(v1), v2(v2) {}
    
    Vec3 normal() const {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        return edge1.cross(edge2).normalize();
    }
};

std::vector<Vec3> fibsphere(int numPoints, float radius) {
    std::vector<Vec3> points;
    points.reserve(numPoints);
        
    for (int i = 0; i < numPoints; ++i) {
        double y = 1.0 - (i / (double)(numPoints - 1)) * 2.0;
        double radius_at_y = std::sqrt(1.0 - y * y);
        
        double theta = 2.0 * M_PI * i / M_PHI;
        
        double x = std::cos(theta) * radius_at_y;
        double z = std::sin(theta) * radius_at_y;
        
        points.emplace_back(x * radius, y * radius, z * radius);
    }
    
    return points;
}

// Create proper triangulation for Fibonacci sphere
std::vector<Triangle> createFibonacciSphereMesh(const std::vector<Vec3>& points) {
    std::vector<Triangle> triangles;
    int n = points.size();
    
    // Create a map to quickly find points by their spherical coordinates
    std::map<std::pair<double, double>, int> pointMap;
    for (int i = 0; i < n; i++) {
        double phi = std::acos(points[i].y / 3.0); // theta = acos(y/R)
        double theta = std::atan2(points[i].z, points[i].x);
        if (theta < 0) theta += 2.0 * M_PI;
        pointMap[{phi, theta}] = i;
    }
    
    // Create triangles by connecting neighboring points
    for (int i = 0; i < n - 1; i++) {
        // For each point, connect it to its neighbors
        if (i > 0) {
            triangles.emplace_back(points[i-1], points[i], points[(i+1) % n]);
        }
    }
    
    // Add cap triangles
    for (int i = 1; i < n/2 - 1; i++) {
        triangles.emplace_back(points[0], points[i], points[i+1]);
        triangles.emplace_back(points[n-1], points[n-1-i], points[n-2-i]);
    }
    
    return triangles;
}

// Alternative: Use Delaunay triangulation on the sphere (simplified)
std::vector<Triangle> createSphereMeshDelaunay(const std::vector<Vec3>& points) {
    std::vector<Triangle> triangles;
    int n = points.size();
    
    // Simple approach: connect each point to its nearest neighbors
    // This is a simplified version - for production use a proper Delaunay triangulation
    
    for (int i = 0; i < n; i++) {
        // Find nearest neighbors (simplified)
        std::vector<std::pair<double, int>> distances;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dist = (points[i] - points[j]).norm();
                distances.emplace_back(dist, j);
            }
        }
        
        // Sort by distance and take 6 nearest neighbors
        std::sort(distances.begin(), distances.end());
        int numNeighbors = min(6, (int)distances.size());
        
        // Create triangles with nearest neighbors
        for (int k = 0; k < numNeighbors - 1; k++) {
            triangles.emplace_back(points[i], points[distances[k].second], points[distances[k+1].second]);
        }
    }
    
    return triangles;
}

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

std::string generateSVG(const std::vector<Vec3>& points, const std::vector<Triangle>& mesh, 
                       double angleX, double angleY, double angleZ) {
    std::stringstream svg;
    int width = 800;
    int height = 600;
    
    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    svg << "<svg width=\"" << width << "\" height=\"" << height << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"#000000\"/>\n";
    
    // Project 3D to 2D
    auto project = [&](const Vec3& point) -> std::pair<double, double> {
        Vec3 rotated = rotate(point, angleX, angleY, angleZ);
        // Perspective projection
        double scale = 300.0 / (5.0 + rotated.z);
        double x = width / 2 + rotated.x * scale;
        double y = height / 2 + rotated.y * scale;
        return {x, y};
    };
    
    // Draw triangles with shading
    for (const auto& triangle : mesh) {
        Vec3 normal = triangle.normal();
        Vec3 lightDir = Vec3(0.5, 0.7, 1.0).normalize();
        double intensity = max(0.0, normal.dot(lightDir));
        
        // Calculate color based on intensity
        int r = static_cast<int>(50 + intensity * 200);
        int g = static_cast<int>(100 + intensity * 150);
        int b = static_cast<int>(200 + intensity * 55);
        
        auto [x0, y0] = project(triangle.v0);
        auto [x1, y1] = project(triangle.v1);
        auto [x2, y2] = project(triangle.v2);
        
        // Only draw triangles facing the camera
        Vec3 viewDir(0, 0, 1);
        if (normal.dot(viewDir) > 0.1) {
            svg << "<polygon points=\"" 
                << x0 << "," << y0 << " "
                << x1 << "," << y1 << " "
                << x2 << "," << y2
                << "\" fill=\"rgb(" << r << "," << g << "," << b << ")\" "
                << "stroke=\"rgba(0,0,0,0.3)\" stroke-width=\"1\"/>\n";
        }
    }
    
    // Draw points for debugging
    for (const auto& point : points) {
        auto [x, y] = project(point);
        svg << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"2\" fill=\"white\"/>\n";
    }
    
    svg << "</svg>";
    return svg.str();
}

// HTTP server class (keep your existing server implementation)
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
        // Generate proper Fibonacci sphere
        std::vector<Vec3> spherePoints = fibsphere(200, 3.0); // Reduced for performance
        std::vector<Triangle> sphereMesh = createSphereMeshDelaunay(spherePoints);
        
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
            } else if (request.find("GET /mesh.svg") != std::string::npos) {
                static double angle = 0.0;
                angle += 0.02;
                std::string svg = generateSVG(sphereMesh, angle, angle * 0.7, angle * 0.3);
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
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    std::string getHTML() {
        return R"(
<!DOCTYPE html>
<html>
<head>
    <title>3D Sphere Mesh Renderer</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
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
            font-size: 2.5em;
        }
        #sphereContainer {
            margin: 20px 0;
            display: flex;
            justify-content: center;
        }
        .instructions {
            margin-top: 20px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            text-align: left;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .feature {
            background: rgba(255, 255, 255, 0.15);
            padding: 15px;
            border-radius: 8px;
        }
        .footer {
            margin-top: 30px;
            font-size: 0.9em;
            opacity: 0.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>3D Sphere Mesh Renderer</h1>
        
        <div id="sphereContainer">
            <img id="sphereImage" src="mesh.svg" width="600" height="450" alt="3D Sphere">
        </div>
        <--- DO NOT EDIT --->
        <div class="footer">
            <p>Built with C++ HTTP Server and SVG Graphics</p>
        </div>
    </div>

    <script>
        setInterval(function() {
            const img = document.getElementById('sphereImage');
            const timestamp = new Date().getTime();
            img.src = 'mesh.svg?' + timestamp;
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