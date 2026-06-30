#pragma once

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace driver {

/* Parsed form of GameWorld::handleTcpClient() response:
 * "200OK!/dev/pts/<X>,127.0.0.1:<Y>".
 */
struct HandshakeInfo {
    std::string serial_device;
    std::string udp_host;
    uint16_t udp_port = 0;
};

/* Parsed form of GameWorld::writeOdomToUdp():
 * "<%.3fN,%.3fE>\n". Northing remains mission Y; Easting remains mission X.
 */
struct OdomReading {
    double northing = 0.0;
    double easting  = 0.0;
};

inline std::string trim(const std::string &input) {
    const auto begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

inline std::string upperCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return value;
}

inline bool parsePort(const std::string &text, uint16_t &port) {
    if (text.empty()) {
        return false;
    }

    char *end = nullptr;
    errno     = 0;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<uint16_t>::max()) {
        return false;
    }

    port = static_cast<uint16_t>(parsed);
    return true;
}

inline bool parseHandshakeResponse(const std::string &response, HandshakeInfo &info) {
    // Simulator sends the serial path and UDP port over TCP only after receiving the byte 'Q'.
    const std::string text   = trim(response);
    const std::string prefix = "200OK!";
    if (text.rfind(prefix, 0) != 0) {
        return false;
    }

    const std::string payload = text.substr(prefix.size());
    const auto comma          = payload.find(',');
    if (comma == std::string::npos) {
        return false;
    }

    const std::string serial_device = payload.substr(0, comma);
    const std::string endpoint      = payload.substr(comma + 1);
    const auto colon                = endpoint.rfind(':');
    if (serial_device.empty() || colon == std::string::npos || colon == 0 || colon + 1 >= endpoint.size()) {
        return false;
    }

    uint16_t udp_port = 0;
    if (!parsePort(endpoint.substr(colon + 1), udp_port)) {
        return false;
    }

    info.serial_device = serial_device;
    info.udp_host      = endpoint.substr(0, colon);
    info.udp_port      = udp_port;
    return true;
}

inline bool parseFgReading(const std::string &line, float &voltage) {
    // Simulator serial messages are newline-delimited and valid readings end exactly with "V==".
    const std::string text = trim(line);
    const auto suffix      = text.find("V==");
    if (suffix == std::string::npos || suffix == 0 || suffix + 3 != text.size()) {
        return false;
    }

    const std::string number = text.substr(0, suffix);
    char *end                = nullptr;
    errno                    = 0;
    const float parsed       = std::strtof(number.c_str(), &end);
    if (errno != 0 || end == number.c_str() || *end != '\0') {
        return false;
    }

    voltage = parsed;
    return true;
}

inline bool parseOdomDatagram(const std::string &datagram, OdomReading &reading) {
    // Keep this strict so malformed UDP packets do not become plausible robot positions.
    const std::string text = trim(datagram);
    if (text.size() < 6 || text.front() != '<' || text.back() != '>') {
        return false;
    }

    const auto n_pos     = text.find('N');
    const auto comma_pos = text.find(',', n_pos);
    const auto e_pos     = text.find('E', comma_pos);
    if (n_pos == std::string::npos || comma_pos == std::string::npos || e_pos == std::string::npos ||
        e_pos + 1 != text.size() - 1) {
        return false;
    }

    const std::string north_text = text.substr(1, n_pos - 1);
    const std::string east_text  = text.substr(comma_pos + 1, e_pos - comma_pos - 1);

    char *north_end        = nullptr;
    char *east_end         = nullptr;
    errno                  = 0;
    const double northing  = std::strtod(north_text.c_str(), &north_end);
    const bool north_valid = errno == 0 && north_end != north_text.c_str() && *north_end == '\0';
    errno                  = 0;
    const double easting   = std::strtod(east_text.c_str(), &east_end);
    const bool east_valid  = errno == 0 && east_end != east_text.c_str() && *east_end == '\0';
    if (!north_valid || !east_valid) {
        return false;
    }

    reading.northing = northing;
    reading.easting  = easting;
    return true;
}

inline bool parseCommandChar(const std::string &input, char &command) {
    // Match GameWorld::spinOnce() movement bytes while accepting mission-readable aliases.
    const std::string text = upperCopy(trim(input));
    if (text.empty()) {
        return false;
    }

    if (text == "W" || text == "N" || text == "NORTH") {
        command = 'W';
        return true;
    }
    if (text == "S" || text == "SOUTH") {
        command = 'S';
        return true;
    }
    if (text == "D" || text == "E" || text == "EAST") {
        command = 'D';
        return true;
    }
    if (text == "A" || text == "WEST") {
        command = 'A';
        return true;
    }

    return false;
}

}  // namespace driver
