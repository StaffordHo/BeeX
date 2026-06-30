#include "driver/protocol.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void expectTrue(const bool condition, const std::string &message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expectNear(const double actual, const double expected, const double tolerance, const std::string &message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message << " actual=" << actual << " expected=" << expected << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    driver::HandshakeInfo info;
    expectTrue(
            driver::parseHandshakeResponse("200OK!/dev/pts/7,127.0.0.1:8015\n", info),
            "valid handshake parses");
    expectTrue(info.serial_device == "/dev/pts/7", "serial device parsed");
    expectTrue(info.udp_host == "127.0.0.1", "UDP host parsed");
    expectTrue(info.udp_port == 8015, "UDP port parsed");
    expectTrue(!driver::parseHandshakeResponse("500ERR", info), "invalid handshake rejected");

    float voltage = 0.0F;
    expectTrue(driver::parseFgReading("0.9876V==\n", voltage), "valid FG parses");
    expectNear(voltage, 0.9876, 1e-5, "FG voltage value");
    expectTrue(!driver::parseFgReading("noise\n", voltage), "invalid FG rejected");
    expectTrue(!driver::parseFgReading("0.1000V==junk\n", voltage), "FG with trailing junk rejected");

    driver::OdomReading odom;
    expectTrue(driver::parseOdomDatagram("<-1.250N,3.500E>\n", odom), "valid odom parses");
    expectNear(odom.northing, -1.25, 1e-9, "odom northing");
    expectNear(odom.easting, 3.5, 1e-9, "odom easting");
    expectTrue(!driver::parseOdomDatagram("1.0N,2.0E", odom), "invalid odom rejected");

    char command = 0;
    expectTrue(driver::parseCommandChar("north", command) && command == 'W', "north maps to W");
    expectTrue(driver::parseCommandChar("east", command) && command == 'D', "east maps to D");
    expectTrue(driver::parseCommandChar("west", command) && command == 'A', "west maps to A");
    expectTrue(!driver::parseCommandChar("jump", command), "bad command rejected");

    if (failures != 0) {
        std::cerr << failures << " parser test failure(s)\n";
        return 1;
    }

    std::cout << "All protocol parser tests passed\n";
    return 0;
}
