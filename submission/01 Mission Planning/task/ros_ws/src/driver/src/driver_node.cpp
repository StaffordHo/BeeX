#include "driver/Command.h"
#include "driver/protocol.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <geometry_msgs/Point.h>
#include <netinet/in.h>
#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

namespace {

/* Small file-descriptor helpers mirror the simulator's direct POSIX style,
 * but keep cleanup paths short in the ROS node.
 */
bool setNonBlocking(const int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        return false;
    }
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

void closeIfOpen(int &fd) {
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

}  // namespace

class SimulatorDriver {
public:
    SimulatorDriver() : private_nh_("~") {
        // Keep the same defaults used by bx_grid_world.cpp: localhost TCP server on port 8091.
        private_nh_.param<std::string>("tcp_host", tcp_host_, "127.0.0.1");
        private_nh_.param<int>("tcp_port", tcp_port_, 8091);
        private_nh_.param<double>("io_period_s", io_period_s_, 0.02);
        private_nh_.param<double>("reconnect_period_s", reconnect_period_s_, 1.0);
        private_nh_.param<double>("command_interval_s", command_interval_s_, 0.105);
        private_nh_.param<double>("handshake_timeout_s", handshake_timeout_s_, 5.0);

        fg_pub_      = nh_.advertise<std_msgs::Float32>("/fg_readings", 10);
        odom_pub_    = nh_.advertise<geometry_msgs::Point>("/odom", 10);
        command_srv_ = nh_.advertiseService("/command", &SimulatorDriver::handleCommand, this);
        io_timer_    = nh_.createTimer(ros::Duration(io_period_s_), &SimulatorDriver::spinIo, this);
    }

    ~SimulatorDriver() { disconnect(); }

private:
    void spinIo(const ros::TimerEvent &) {
        // 1. Match GameWorld startup order: TCP handshake first, then serial and UDP endpoints.
        if (!connected_) {
            const ros::Time now = ros::Time::now();
            if (last_reconnect_attempt_.isZero() ||
                (now - last_reconnect_attempt_).toSec() >= reconnect_period_s_) {
                last_reconnect_attempt_ = now;
                connectToSimulator();
            }
            return;
        }

        // 2. Once connected, relay simulator sensor streams into ROS topics.
        readSerial();
        readUdp();
    }

    bool connectToSimulator() {
        disconnect();

        /* GameWorld::handleTcpClient() responds to 'Q' with the serial slave path
         * and UDP port. The driver must not assume /dev/pts/<X> or port 8015.
         */
        driver::HandshakeInfo handshake;
        tcp_fd_ = connectTcp();
        if (tcp_fd_ < 0) {
            ROS_WARN_THROTTLE(5.0, "Waiting for simulator TCP server at %s:%d", tcp_host_.c_str(), tcp_port_);
            return false;
        }

        if (!requestHandshake(handshake)) {
            ROS_WARN("Simulator handshake failed; will retry");
            disconnect();
            return false;
        }

        udp_fd_ = openUdpSocket(handshake.udp_port);
        if (udp_fd_ < 0) {
            ROS_WARN("Unable to bind UDP odometry socket on port %u", handshake.udp_port);
            disconnect();
            return false;
        }

        serial_fd_ = openSerial(handshake.serial_device);
        if (serial_fd_ < 0) {
            ROS_WARN("Unable to open serial FG device %s", handshake.serial_device.c_str());
            disconnect();
            return false;
        }

        setNonBlocking(tcp_fd_);
        connected_ = true;
        ROS_INFO("Connected to simulator: serial=%s udp=%s:%u",
                 handshake.serial_device.c_str(),
                 handshake.udp_host.c_str(),
                 handshake.udp_port);
        return true;
    }

    int connectTcp() const {
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            ROS_WARN("TCP socket failed: %s", std::strerror(errno));
            return -1;
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(static_cast<uint16_t>(tcp_port_));
        if (inet_pton(AF_INET, tcp_host_.c_str(), &addr.sin_addr) != 1) {
            ROS_ERROR("Invalid TCP host: %s", tcp_host_.c_str());
            close(fd);
            return -1;
        }

        if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
            close(fd);
            return -1;
        }

        return fd;
    }

    bool requestHandshake(driver::HandshakeInfo &handshake) {
        // The simulator uses a single-byte command protocol; 'Q' is the connection query.
        const char query = 'Q';
        if (write(tcp_fd_, &query, 1) != 1) {
            ROS_WARN("Failed to send simulator handshake query: %s", std::strerror(errno));
            return false;
        }

        std::string response;
        const ros::WallTime deadline = ros::WallTime::now() + ros::WallDuration(handshake_timeout_s_);
        while (ros::WallTime::now() < deadline) {
            fd_set read_set;
            FD_ZERO(&read_set);
            FD_SET(tcp_fd_, &read_set);

            timeval timeout{};
            timeout.tv_sec  = 0;
            timeout.tv_usec = 100000;

            const int ready = select(tcp_fd_ + 1, &read_set, nullptr, nullptr, &timeout);
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                ROS_WARN("Handshake select failed: %s", std::strerror(errno));
                return false;
            }
            if (ready == 0) {
                continue;
            }

            char buffer[256];
            const ssize_t bytes = read(tcp_fd_, buffer, sizeof(buffer));
            if (bytes <= 0) {
                ROS_WARN("Handshake read failed: %s", bytes == 0 ? "connection closed" : std::strerror(errno));
                return false;
            }

            response.append(buffer, static_cast<size_t>(bytes));
            if (driver::parseHandshakeResponse(response, handshake)) {
                return true;
            }
        }

        ROS_WARN("Handshake timed out; last response was '%s'", response.c_str());
        return false;
    }

    int openUdpSocket(const uint16_t port) const {
        const int fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (fd < 0) {
            ROS_WARN("UDP socket failed: %s", std::strerror(errno));
            return -1;
        }

        int reuse = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port        = htons(port);

        if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
            ROS_WARN("UDP bind failed on port %u: %s", port, std::strerror(errno));
            close(fd);
            return -1;
        }

        if (!setNonBlocking(fd)) {
            ROS_WARN("Unable to set UDP socket non-blocking: %s", std::strerror(errno));
            close(fd);
            return -1;
        }

        return fd;
    }

    int openSerial(const std::string &device) const {
        const int fd = open(device.c_str(), O_RDONLY | O_NOCTTY | O_NONBLOCK);
        if (fd < 0) {
            ROS_WARN("Serial open failed for %s: %s", device.c_str(), std::strerror(errno));
            return -1;
        }

        termios options{};
        if (tcgetattr(fd, &options) == 0) {
            cfmakeraw(&options);
            options.c_cc[VMIN]  = 0;
            options.c_cc[VTIME] = 0;
            tcsetattr(fd, TCSANOW, &options);
        }

        return fd;
    }

    void readSerial() {
        if (serial_fd_ < 0) {
            return;
        }

        char buffer[256];
        while (true) {
            const ssize_t bytes = read(serial_fd_, buffer, sizeof(buffer));
            if (bytes > 0) {
                serial_buffer_.append(buffer, static_cast<size_t>(bytes));
                publishSerialLines();
                continue;
            }

            if (bytes == 0) {
                // With VMIN=0/VTIME=0, a zero-byte read means no serial data is ready yet.
                return;
            }

            if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
                return;
            }

            ROS_WARN("Serial connection lost: %s", std::strerror(errno));
            disconnect();
            return;
        }
    }

    void publishSerialLines() {
        size_t newline = std::string::npos;
        while ((newline = serial_buffer_.find('\n')) != std::string::npos) {
            const std::string line = serial_buffer_.substr(0, newline + 1);
            serial_buffer_.erase(0, newline + 1);

            float voltage = 0.0F;
            if (driver::parseFgReading(line, voltage)) {
                // GameWorld::writeSensorToSerial() emits "%.4fV==\n"; ignore every other serial line.
                std_msgs::Float32 msg;
                msg.data = voltage;
                fg_pub_.publish(msg);
            }
        }

        constexpr size_t max_buffer = 4096;
        if (serial_buffer_.size() > max_buffer) {
            serial_buffer_.erase(0, serial_buffer_.size() - max_buffer);
        }
    }

    void readUdp() {
        if (udp_fd_ < 0) {
            return;
        }

        char buffer[256];
        while (true) {
            const ssize_t bytes = recvfrom(udp_fd_, buffer, sizeof(buffer) - 1, 0, nullptr, nullptr);
            if (bytes > 0) {
                buffer[bytes] = '\0';
                driver::OdomReading reading;
                if (driver::parseOdomDatagram(buffer, reading)) {
                    // Assignment mapping: Easting -> Point.x, Northing -> Point.y.
                    geometry_msgs::Point msg;
                    msg.x = reading.easting;
                    msg.y = reading.northing;
                    msg.z = 0.0;
                    odom_pub_.publish(msg);
                }
                continue;
            }

            if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
                return;
            }

            ROS_WARN("UDP odometry socket lost: %s", bytes == 0 ? "closed" : std::strerror(errno));
            disconnect();
            return;
        }
    }

    bool handleCommand(driver::Command::Request &request, driver::Command::Response &response) {
        if (!connected_) {
            response.success = false;
            response.message = "driver is not connected to simulator";
            return true;
        }

        char command = 0;
        if (!driver::parseCommandChar(request.command, command)) {
            response.success = false;
            response.message = "command must be one of W/A/S/D, north/south/east/west, or N/E/S/W";
            return true;
        }

        const ros::Time now = ros::Time::now();
        if (!last_command_time_.isZero()) {
            const double elapsed = (now - last_command_time_).toSec();
            if (elapsed < command_interval_s_) {
                // GameWorld only executes the last command in each 100 ms render/update window.
                ros::Duration(command_interval_s_ - elapsed).sleep();
            }
        }

        const ssize_t bytes = write(tcp_fd_, &command, 1);
        if (bytes != 1) {
            response.success = false;
            response.message = std::string("TCP command write failed: ") + std::strerror(errno);
            disconnect();
            return true;
        }

        last_command_time_ = ros::Time::now();
        response.success   = true;
        response.message   = std::string("sent ") + command;
        return true;
    }

    void disconnect() {
        closeIfOpen(serial_fd_);
        closeIfOpen(udp_fd_);
        closeIfOpen(tcp_fd_);
        serial_buffer_.clear();
        connected_ = false;
    }

    /* ROS Interface */
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    ros::Publisher fg_pub_;
    ros::Publisher odom_pub_;
    ros::ServiceServer command_srv_;
    ros::Timer io_timer_;

    /* Simulator Connection Parameters */
    std::string tcp_host_;
    int tcp_port_ = 8091;
    double io_period_s_ = 0.02;
    double reconnect_period_s_ = 1.0;
    double command_interval_s_ = 0.105;
    double handshake_timeout_s_ = 5.0;

    /* TCP / UDP / Serial File Descriptors */
    int tcp_fd_    = -1;
    int udp_fd_    = -1;
    int serial_fd_ = -1;

    /* Runtime State */
    bool connected_ = false;
    std::string serial_buffer_;
    ros::Time last_reconnect_attempt_;
    ros::Time last_command_time_;
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "driver_node");
    SimulatorDriver driver;
    ros::spin();
    return 0;
}
