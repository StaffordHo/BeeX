// Copyright by BeeX [2025].

#include <arpa/inet.h>
#include <fcntl.h>
#include <pty.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Add missing includes for socket programming
#include <netinet/in.h>
#include <sys/socket.h>

using namespace cv;
using namespace std;

#define STR_LEN         100
#define CELL_RESOLUTION 0.2F    // Each cell is 0.2m x 0.2m
#define MAX_GAME_AREA   100.0F  // 100m x 100m
#define TCP_PORT        8'091

class GameWorld {
private:
    std::atomic<char> m_incomingCmd;

    /* Window Settings */
    const int M_WINDOW_SIZE      = 400;
    const int M_GRID_SIZE        = 20;  // Number of grid cells along one axis
    const std::string M_WIN_NAME = "BeeX Assignment Grid World";
    const uint64_t M_GAME_START_TIME_MS;
    const bool M_KEEP_RUNNING_AFTER_SUCCESS;

    /* Robot position in world coordinates */
    float m_robotEasting  = 0.0f;
    float m_robotNorthing = 0.0f;
    bool m_pipelineGoalCompleted = false;

    /* Pipeline */
    std::vector<Point2f> m_pipeline;  // in reverse order (last is first)
    cv::Mat m_groundTruthIntensity;   // Distance (in cm) to nearest pt on pipeline. At resolution of 0.2m per pixel.
                                      // Total area is 100x100m
    cv::Mat m_revealedMap;            // Colour representation. Same size as ground_truth_intensity.
                                      // But black where not revealed yet.

    /* UDP Socket */
    int32_t m_udpSocket = 0;
    sockaddr_in m_udpAddr{};
    uint16_t m_udpPort = 0;

    /* Serial */
    int32_t m_serialMasterFd    = 0;
    int32_t m_serialSlaveFd     = 0;
    std::string m_serialAddress = "";

    /* TCP Server */
    int32_t m_tcpServerSocketFd = -1;
    int32_t m_tcpClientSocketFd = -1;

    static bool keepRunningAfterSuccessEnabled() {
        const char *value = std::getenv("BEEX_KEEP_SIM_RUNNING_AFTER_SUCCESS");
        if (value == nullptr) {
            return false;
        }

        const std::string flag(value);
        return !(flag.empty() || flag == "0" || flag == "false" || flag == "False" || flag == "FALSE");
    }

public:
    uint64_t getCurrentTimeMs() const {
        return static_cast<uint64_t>(static_cast<double>(cv::getTickCount()) / cv::getTickFrequency() * 1'000);
    }

    int32_t random() const { return rand(); }

    explicit GameWorld(const uint16_t dstUdpPort = 8'085)
            : M_GAME_START_TIME_MS(getCurrentTimeMs()),
              M_KEEP_RUNNING_AFTER_SUCCESS(keepRunningAfterSuccessEnabled()),
              m_udpPort(dstUdpPort) {

        // Setup Serial
        char slaveName[STR_LEN];

        // Create a pseudo-terminal pair
        if (openpty(&m_serialMasterFd, &m_serialSlaveFd, slaveName, nullptr, nullptr) == -1) {
            perror("unable to open Serial openpty");
            exit(1);
        }

        // Set master file descriptor to non-blocking to prevent hanging
        const int32_t serialFlags = fcntl(m_serialMasterFd, F_GETFL, 0);
        if (serialFlags == -1) {
            perror("fcntl(F_GETFL)");
            exit(1);
        }
        if (fcntl(m_serialMasterFd, F_SETFL, serialFlags | O_NONBLOCK) == -1) {
            perror("fcntl(F_SETFL)");
            exit(1);
        }

        printf("Pseudo-terminal created. Slave device: %s\n", slaveName);
        m_serialAddress = slaveName;

        // Setup UDP Socket
        m_udpSocket = socket(AF_INET, SOCK_DGRAM, 0);
        if (m_udpSocket < 0) {
            perror("socket");
            exit(1);
        }

        m_udpAddr.sin_family      = AF_INET;
        m_udpAddr.sin_port        = htons(dstUdpPort);       // Port to send to
        m_udpAddr.sin_addr.s_addr = inet_addr("127.0.0.1");  // Localhost

        // Setup TCP Server
        setupTcpServer();

        // Random seed
        // srand(static_cast<unsigned int>(time(nullptr)));
        srand(100);  // Fixed seed for deterministic behavior

        setupGameBoard(false);
        m_incomingCmd = 0;
    }

    ~GameWorld() {
        // Close Serial
        close(m_serialMasterFd);
        close(m_serialSlaveFd);

        // Close UDP
        close(m_udpSocket);

        // Close TCP Sockets
        if (m_tcpClientSocketFd != -1) {
            close(m_tcpClientSocketFd);
        }
        if (m_tcpServerSocketFd != -1) {
            close(m_tcpServerSocketFd);
        }

        cv::destroyAllWindows();
    }

    bool spinOnce() {
        const uint64_t elapsedTimeMs = getCurrentTimeMs() - M_GAME_START_TIME_MS;

        // 1. Check if robot has reached current goal
        if (!m_pipelineGoalCompleted && hasReachedPipelineGoal()) {
            printf("Success! Pipeline goal reached. Elapsed time: %lu ms\n", elapsedTimeMs);
            m_pipelineGoalCompleted = true;
            if (!M_KEEP_RUNNING_AFTER_SUCCESS) {
                return false;
            }
            printf("Pipeline goal reached; simulator keep-alive mode is enabled.\n");
        }

        // 2. Check if robot is out of bounds
        if (!robotInBounds()) {
            printf("Failure! Robot out of bounds. Elapsed time: %lu ms\n", elapsedTimeMs);
            return false;
        }

        // 3. Handle TCP communication
        handleTcpClient();

        // Process command from TCP
        char command = m_incomingCmd.exchange(0);
        if (command != 0) {
            switch (command) {
            case 'w':
            case 'W':
                m_robotNorthing += 0.05;
                break;
            case 's':
            case 'S':
                m_robotNorthing -= 0.05;
                break;
            case 'a':
            case 'A':
                m_robotEasting -= 0.05;
                break;
            case 'd':
            case 'D':
                m_robotEasting += 0.05;
                break;
            }
        }

        // 4. Update revealed map
        const float voltageReading = revealArea() + ((random() % 1'000) / 10000.0f);

        // 5. Send UDP odometry
        writeOdomToUdp(m_robotNorthing, m_robotEasting);

        // 6. Send Serial sensor data
        writeSensorToSerial(voltageReading);

        // 7. Render game grid
        return renderGameWindow(voltageReading);
    }

private:
    bool hasReachedPipelineGoal() {
        if (m_pipeline.empty()) {
            return true;
        }

        const Point2f currentGoal = m_pipeline.back();
        const float distanceToGoal =
                sqrt(pow(m_robotEasting - currentGoal.x, 2) + pow(m_robotNorthing - currentGoal.y, 2));

        // Consider goal reached if within 0.3 meters
        if (distanceToGoal < 0.3f) {
            // Remove reached goal
            m_pipeline.pop_back();
        }

        return m_pipeline.empty();
    }

    void setupTcpServer() {
        m_tcpServerSocketFd = socket(AF_INET, SOCK_STREAM, 0);
        if (m_tcpServerSocketFd < 0) {
            perror("tcp socket");
            exit(1);
        }

        // Set to non-blocking
        const int32_t tcp_flags = fcntl(m_tcpServerSocketFd, F_GETFL, 0);
        fcntl(m_tcpServerSocketFd, F_SETFL, tcp_flags | O_NONBLOCK);

        // Allow address reuse
        int option = 1;
        setsockopt(m_tcpServerSocketFd, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));

        sockaddr_in serverAddr{};
        serverAddr.sin_family      = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port        = htons(TCP_PORT);

        if (bind(m_tcpServerSocketFd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
            perror("tcp bind");
            exit(1);
        }

        if (listen(m_tcpServerSocketFd, 1) < 0) {
            perror("tcp listen");
            exit(1);
        }
        printf("TCP Server listening on port %d\n", TCP_PORT);
    }

    void handleTcpClient() {
        // If no client is connected, try to accept a new one
        if (m_tcpClientSocketFd == -1) {
            m_tcpClientSocketFd = accept(m_tcpServerSocketFd, nullptr, nullptr);
            if (m_tcpClientSocketFd > 0) {
                printf("TCP client connected.\n");
                // Set client socket to non-blocking
                const int32_t clientFlags = fcntl(m_tcpClientSocketFd, F_GETFL, 0);
                fcntl(m_tcpClientSocketFd, F_SETFL, clientFlags | O_NONBLOCK);
            }
            return;  // Return after checking for new connection
        }

        // If a client is connected, read all available data from it
        char buffer[STR_LEN];
        ssize_t bytesRead;
        char lastChar = 0;

        while ((bytesRead = read(m_tcpClientSocketFd, buffer, sizeof(buffer))) > 0) {
            // Store the last character for movement commands
            lastChar = buffer[bytesRead - 1];
        }

        if (lastChar == 0) {
            // Do Nothing
        } else if (lastChar == 'Q') {
            printf("Received 'Q' from client.\n");

            char response[STR_LEN];
            int msgLen = snprintf(response, STR_LEN, "200OK!%s,127.0.0.1:%d", m_serialAddress.c_str(), m_udpPort);
            ssize_t bytesWritten = write(m_tcpClientSocketFd, response, msgLen);

            if (bytesWritten < 0) {
                perror("tcp write");
            } else {
                printf("Sent TCP response: %s\n", response);
            }
        } else {
            m_incomingCmd = lastChar;
        }

        if (bytesRead < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return;  // No more data to read
            }
            perror("tcp read");
            close(m_tcpClientSocketFd);
            m_tcpClientSocketFd = -1;
            return;
        }

        if (bytesRead == 0) {
            printf("TCP client disconnected.\n");
            close(m_tcpClientSocketFd);
            m_tcpClientSocketFd = -1;
            return;
        }
    }

    void setupGameBoard(const bool revealAll = false) {
        // Rand start position by +-2m
        m_robotEasting  = ((random() % 4'000) - 2'000) / 1000.0f;
        m_robotNorthing = ((random() % 4'000) - 2'000) / 1000.0f;

        printf("Starting Robot Position: Easting: %.3f m, Northing: %.3f m\n", m_robotEasting, m_robotNorthing);

        // Initialize pipeline with starting position
        // 1. Add starting position (+-3m, 3m)
        m_pipeline.push_back(Point2f(((random() % 6'000) - 3'000) / 1000.0F, 3.0F));

        // 2. Add intermediate positions (+-5m, 5m)
        m_pipeline.push_back(Point2f(((random() % 10'000) - 5'000) / 1000.0F, 5.0F));

        // 3. Add intermediate positions (+-7m, 7m)
        m_pipeline.push_back(Point2f(((random() % 14'000) - 7'000) / 1000.0F, 7.0F));

        // 4. Add final position (+-9m, 10m)
        m_pipeline.push_back(Point2f(((random() % 18'000) - 9'000) / 1000.0F, 10.0F));

        std::reverse(m_pipeline.begin(), m_pipeline.end());

        // Based on quadratic distance (L2 norm) of the cell center to the nearest pipeline polyline
        // If distance < 0.5m, color cell red, else blue with intensity based on distance.

        m_groundTruthIntensity = cv::Mat::zeros(
                static_cast<int32_t>(MAX_GAME_AREA / CELL_RESOLUTION),
                static_cast<int32_t>(MAX_GAME_AREA / CELL_RESOLUTION),
                CV_8UC1);

        m_revealedMap = cv::Mat::zeros(m_groundTruthIntensity.rows, m_groundTruthIntensity.cols, CV_8UC3);

        const float HALF_AREA = MAX_GAME_AREA / 2.0F;
        for (uint32_t rowIdx = 0; rowIdx < m_groundTruthIntensity.rows; rowIdx++) {
            const float cellCenterY = (rowIdx + 0.5f) * CELL_RESOLUTION - HALF_AREA;

            for (uint32_t colIdx = 0; colIdx < m_groundTruthIntensity.cols; colIdx++) {
                const float cellCenterX = (colIdx + 0.5f) * CELL_RESOLUTION - HALF_AREA;
                const Point2f cellCenter(cellCenterX, cellCenterY);

                // Find nearest pipeline point
                float minDistanceSq = std::numeric_limits<float>::max();
                for (int32_t pIdx = m_pipeline.size() - 1; pIdx > 0; --pIdx) {
                    const Point2f p1 = m_pipeline[pIdx];
                    const Point2f p2 = m_pipeline[pIdx - 1];

                    // Project cell center onto line segment p1-p2
                    const Point2f lineVec = p2 - p1;
                    const Point2f cellVec = cellCenter - p1;

                    const float lineLenSq = lineVec.dot(lineVec);
                    const float t =
                            std::max(0.0f, std::min(1.0f, cellVec.dot(lineVec) / lineLenSq));  // Clamp to segment

                    const Point2f projection = p1 + t * lineVec - cellCenter;
                    const float distSq       = projection.x * projection.x + projection.y * projection.y;

                    if (distSq < minDistanceSq) {
                        minDistanceSq = distSq;
                    }
                }

                // Color cell based on quadratic distance Red -> Blue. Max distance is 0.6m
                const float intensity                              = 1 - std::min(1.0, sqrt(minDistanceSq) / 0.6);
                m_groundTruthIntensity.at<uint8_t>(rowIdx, colIdx) = static_cast<uint8_t>(intensity * 100);
            }
        }

        namedWindow(M_WIN_NAME);

        if (revealAll) {
            for (int32_t r = 0; r < m_groundTruthIntensity.rows; r++) {
                for (int32_t c = 0; c < m_groundTruthIntensity.cols; c++) {
                    revealCell(r, c);
                }
            }

            cv::imshow("Ground Truth Intensity", m_groundTruthIntensity);
        }
    }

    bool robotInBounds(const float boundary = 4.0F) const {
        const float HALF_AREA = MAX_GAME_AREA / 2.0F;
        return ((boundary - HALF_AREA) < m_robotEasting && m_robotEasting < (HALF_AREA - boundary)  //
                && (boundary - HALF_AREA) < m_robotNorthing && m_robotNorthing <= (HALF_AREA - boundary));
    }


    float revealArea() {
        const float HALF_AREA = MAX_GAME_AREA / 2.0F;
        const float northCell = (HALF_AREA + m_robotNorthing) / CELL_RESOLUTION - 1;
        const float eastCell  = (HALF_AREA + m_robotEasting) / CELL_RESOLUTION - 1;

        revealCell(ceil(northCell), ceil(eastCell));
        revealCell(round(northCell), ceil(eastCell));
        revealCell(ceil(northCell), round(eastCell));

        return revealCell(round(northCell), round(eastCell));
    }

    float revealCell(const int32_t rowIdx, const int32_t colIdx) {
        // Reveal cell in revealed_map based on robot position
        const float val       = m_groundTruthIntensity.at<uint8_t>(rowIdx, colIdx);
        const float intensity = val / 100.0f;

        uint8_t *revPixel = m_revealedMap.ptr<uint8_t>(rowIdx, colIdx);
        revPixel[0]       = static_cast<uint8_t>((1.0f - intensity) * 255);  // Blue
        revPixel[1]       = 0;                                               // Green
        revPixel[2]       = static_cast<uint8_t>(intensity * 255);           // Red

        return intensity;
    }

    bool renderGameWindow(const float voltageReading) {

        // Overlay revealed map with offset based on robot position
        const float HALF_AREA   = MAX_GAME_AREA / 2.0F;
        const float cenOffsetFx = (HALF_AREA + m_robotEasting) / CELL_RESOLUTION;
        const float cenOffsetFy = (HALF_AREA + m_robotNorthing) / CELL_RESOLUTION;

        const int32_t cenOffsetX = static_cast<int32_t>(floor(cenOffsetFx));
        const int32_t cenOffsetY = static_cast<int32_t>(floor(cenOffsetFy));

        // Account for sub-cell offsets
        const int32_t topLeftX = cenOffsetX - 1 - M_GRID_SIZE / 2;
        const int32_t topLeftY = cenOffsetY - M_GRID_SIZE / 2;
        const Rect roiSrc(topLeftX, topLeftY, M_GRID_SIZE + 1, M_GRID_SIZE + 1);

        printf("Robot Pos: (%.2f, %.2f)\n", m_robotEasting, m_robotNorthing);

        // Resize revealed map to window size and clip to actual window
        Mat frame;
        const int32_t CELL_SIZE = M_WINDOW_SIZE / M_GRID_SIZE;
        cv::resize(m_revealedMap(roiSrc), frame, Size(M_WINDOW_SIZE + CELL_SIZE, M_WINDOW_SIZE + CELL_SIZE));
        cv::flip(frame, frame, 0);  // Flip vertically for correct orientation

        // Crop to window size while respecting sub-cell offsets
        const int32_t subcellOffsetX = (cenOffsetFx - floor(cenOffsetFx)) * CELL_SIZE;
        const int32_t subcellOffsetY = (ceil(cenOffsetFy) - cenOffsetFy) * CELL_SIZE;
        Rect roiDst(subcellOffsetX, subcellOffsetY, M_WINDOW_SIZE, M_WINDOW_SIZE);
        frame = frame(roiDst);

        // Draw robot as a small square at the center
        int robotSize = 10;
        Point winCenter(M_WINDOW_SIZE / 2, M_WINDOW_SIZE / 2);
        rectangle(
                frame,
                Point(winCenter.x, winCenter.y),
                Point(winCenter.x + robotSize, winCenter.y + robotSize),
                Scalar(0, 255, 0),
                FILLED);

        // Display world coordinates at top-left
        char buffer[80];  // Create a character array to store the formatted string
        snprintf(
                buffer,
                sizeof(buffer),
                "%+3.1fN %+3.1fE | +%.02fVolts",
                m_robotNorthing,
                m_robotEasting,
                voltageReading);
        putText(frame, buffer, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);

        // Show window
        cv::imshow(M_WIN_NAME, frame);

        // --- Handle input ---
        if (waitKey(100) == 27 || m_incomingCmd == 27) {  // ESC key
            printf("Window closed by user. Exiting.\n");
            return false;  // Exit if ESC is pressed
        }

        switch (m_incomingCmd.exchange(0)) {
        case 'w':
        case 'W':
            m_robotNorthing += 0.05;
            break;
        case 's':
        case 'S':
            m_robotNorthing -= 0.05;
            break;
        case 'a':
        case 'A':
            m_robotEasting -= 0.05;
            break;
        case 'd':
        case 'D':
            m_robotEasting += 0.05;
            break;
        default:
            break;
        }

        return true;
    }

    void writeOdomToUdp(const float northing, const float easting) {

        char odomBuffer[STR_LEN];
        // Print in format "<%.3fN,%.3fE>"
        const uint32_t msgSize = snprintf(odomBuffer, STR_LEN, "<%.3fN,%.3fE>\n", northing, easting);

        const uint32_t sent =
                sendto(m_udpSocket,
                       odomBuffer,
                       msgSize,
                       0,
                       reinterpret_cast<sockaddr *>(&m_udpAddr),
                       sizeof(m_udpAddr));

        if (sent < 0) {
            perror("sendto");
        } else {
            printf("Sent UDP Odometry: %s\n", odomBuffer);
        }
    }

    void writeSensorToSerial(const float voltage) {
        char sensorBuffer[STR_LEN];
        // Print in format "%fV==\n"
        const uint32_t msgSize = snprintf(sensorBuffer, STR_LEN, "%.4fV==\n", voltage);

        const ssize_t written = write(m_serialMasterFd, sensorBuffer, msgSize);
        if (written < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                perror("write to serial");
            }
        } else {
            printf("Wrote to Serial Sensor Data: %s\n", sensorBuffer);
        }
    }
};


int main() {

    GameWorld gameWorld(8'015);
    while (gameWorld.spinOnce()) {
        // Loop until game ends
    }

    return 0;
}
