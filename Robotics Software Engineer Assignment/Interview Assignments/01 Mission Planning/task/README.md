# Assignment: Autonomous Subsea Pipeline Tracking
## 1. Introduction

This assignment simulates a simplified real-world challenge of autonomous underwater pipeline inspection. The objective is to develop a mission planner for a simulated Hovering Autonomous Underwater Vehicle (HAUV) tasked with locating and tracking a buried subsea pipeline.

The simulation environment models a low-visibility scenario where the primary sensor is a Field Gradient (FG) sensor. This sensor provides point-based measurements, yielding a voltage from 0V to +1V, which is proportional to the vehicle's proximity to the pipeline directly beneath it. A reading of +1V indicates the sensor is precisely over the pipeline's center-line.

## 2. System Architecture & Requirements

The solution must be developed using C++17 and ROS 1, and will be evaluated using ROS1 (Noetic) on an Ubuntu 20.04 platform. Your implementation should be structured into two distinct ROS packages: `driver` and `mission`.


### 2.1. `driver` Package: Simulator Interface

This package is responsible for all low-level communication with the simulation environment. It must abstract the hardware-level details and expose a standardized ROS interface.

**Responsibilities:**

1.  **Connection Initialization:**
    *   Establish a TCP connection to `127.0.0.1:8091`.
    *   Transmit the character `Q` (Case-sensitive) to the server to request communication port details.
    *   Parse the expected response format: `200OK!/dev/pts/<X>,127.0.0.1:<Y>`, where `<X>` is the serial port ID and `<Y>` is the UDP port number.

2.  **Serial Data Handling (FG Sensor):**
    *   Connect to the pseudo-terminal device specified by the TCP response (e.g., `/dev/pts/<X>`).
    *   Listen for incoming data streams, which are newline-terminated (`\n`).
    *   Parse messages matching the format `"%fV==` to extract the floating-point voltage reading. All other serial messages should be ignored.
    *   Publish valid FG sensor readings to the `/fg_readings` topic as `std_msgs/Float32` messages.

3.  **UDP Data Handling (Odometry):**
    *   Listen for UDP datagrams on the port specified by the TCP response (e.g., `127.0.0.1:<Y>`).
    *   Parse odometry data from the format `"<%.3fN,%.3fE>"`, representing the vehicle's position in meters.
    *   Publish the position to the `/odom` topic as a `geometry_msgs/Point` message, mapping Northing to `Point.y` and Easting to `Point.x`. Let `Point.z` be always zero.

4.  **Vehicle Motion Control:**
    *   Provide a ROS service on the `/command` topic. The service definition (`mycommand.srv`) is left to the your discretion.
    *   This service must translate high-level motion requests into single-character TCP commands sent to the simulator:
        *   `W`: Move North (+Y) by 0.05 meters.
        *   `S`: Move South (-Y) by 0.05 meters.
        *   `D`: Move East (+X) by 0.05 meters.
        *   `A`: Move West (-X) by 0.05 meters.
    *   Note: The simulator operates at 10Hz; only the last command received in a 100ms window is executed.

### 2.2. `mission` Package: Control Logic

This package contains the high-level business logic for accomplishing the mission objective. It must not interact directly with the simulator.

**Responsibilities:**

1.  Subscribe to the `/fg_readings` and `/odom` topics to acquire situational awareness.
2.  Implement an algorithm to process sensor and position data.
3.  Based on the algorithm's decisions, invoke the `/command` service to manoeuvre the vehicle to find and follow the pipeline.

## 3. Mission Objective & Constraints

The primary objective is to autonomously navigate the vehicle from its starting position to locate the pipeline, and then track it northwards from its southernmost point to its northern terminus.

**Success Criteria:**

*   **Completion Time:** The mission must be completed within 15 minutes of simulation time.
*   **Tracking Accuracy:** The vehicle must maintain a lateral distance of no more than 0.5 meters from the pipeline's centerline while tracking.
*   **Boundary Adherence:** The vehicle must remain within the 100m x 100m operational area centered at the origin. Exceeding these bounds will result in mission failure.

**Assumptions:**

*   The pipeline is oriented primarily along the North-South (Y) axis.
*   The vehicle's starting position is within a +/- 5-meter box around the world origin (0,0).
*   The existence of a single pipeline within the operational area is guaranteed.

## 4. Evaluation Criteria

Submissions will be evaluated based on the following:

1.  **Compliance:** Adherence to C++17 standard and ROS 1 conventions.
2.  **Functionality:** Correct implementation of TCP, UDP, and Serial communication protocols.
3.  **ROS:** Effective use of ROS topics and services for modular, inter-node communication.
4.  **Mission Planning:** Robustness and efficiency of strategy employed to locate and track the pipeline within the defined constraints.
5.  **Documentation:** Clarity of the README file, including a brief explanation of the implemented strategy and instructions for reproduction. If you used any LLM tools to assist in solving this assignment, please mention them here.

Do not use any external libraries beyond the standard C++17 library and ROS 1 packages available in Ubuntu 20.04 repositories.


## 5. Running the Simulator (GAME)

Compile the files using the `./compile.sh` script.
Run the simulation by executing `./bx_grid_world`.

**Note on Compatibility:** The official evaluation platform is Ubuntu 20.04 with ROS Noetic. While modifications to the simulator's dependencies are permissible for local development on other operating systems, please be advised that all submissions will be assessed exclusively against the original, unmodified simulation environment provided.


## 6. Assignment Submission
For submission, please only submit your code and README in Zip file. NOT THE BINARIES OR DOCKER IMAGE.
You are welcomed to use any LLM to aid you in solving this assignment. Kindly state in the README if such tools are used.
