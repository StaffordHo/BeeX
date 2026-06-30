#include "driver/Command.h"

#include <geometry_msgs/Point.h>
#include <ros/ros.h>
#include <std_msgs/Float32.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <vector>

/* Mission planner kept separate from simulator I/O by design:
 * bx_grid_world.cpp exposes only odometry, scalar FG voltage, and W/A/S/D commands.
 * This node therefore treats the pipe as an unknown signal field, not as map data.
 */
class MissionPlanner {
public:
    MissionPlanner() : private_nh_("~") {
        // FG off-pipe noise can reach roughly 0.10 V in GameWorld::spinOnce(); thresholds include margin.
        private_nh_.param<double>("seen_threshold", seen_threshold_, 0.20);
        private_nh_.param<double>("track_threshold", track_threshold_, 0.18);
        private_nh_.param<double>("strong_threshold", strong_threshold_, 0.45);
        private_nh_.param<double>("center_accept_threshold", center_accept_threshold_, 0.65);
        private_nh_.param<double>("row_step_m", row_step_m_, 0.50);
        private_nh_.param<double>("fine_row_step_m", fine_row_step_m_, 0.25);
        private_nh_.param<double>("search_half_width_m", search_half_width_m_, 5.0);
        private_nh_.param<double>("search_max_row_offset_m", search_max_row_offset_m_, 8.0);
        private_nh_.param<double>("search_row_step_m", search_row_step_m_, 0.50);
        private_nh_.param<double>("center_scan_radius_m", center_scan_radius_m_, 0.75);
        private_nh_.param<double>("south_scan_radius_m", south_scan_radius_m_, 0.85);
        private_nh_.param<double>("wide_scan_radius_m", wide_scan_radius_m_, 1.6);
        private_nh_.param<double>("reacquire_max_radius_m", reacquire_max_radius_m_, 4.0);
        private_nh_.param<double>("move_tolerance_m", move_tolerance_m_, 0.035);
        private_nh_.param<double>("step_wait_s", step_wait_s_, 0.11);
        private_nh_.param<double>("sample_window_s", sample_window_s_, 0.15);
        private_nh_.param<double>("mission_timeout_s", mission_timeout_s_, 840.0);
        private_nh_.param<double>("max_northing_m", max_northing_m_, 45.0);
        private_nh_.param<double>("operating_limit_m", operating_limit_m_, 45.0);
        private_nh_.param<double>("endpoint_step_m", endpoint_step_m_, 0.10);
        private_nh_.param<double>("endpoint_probe_m", endpoint_probe_m_, 0.80);

        odom_sub_ = nh_.subscribe("/odom", 20, &MissionPlanner::odomCallback, this);
        fg_sub_   = nh_.subscribe("/fg_readings", 20, &MissionPlanner::fgCallback, this);
        command_client_ = nh_.serviceClient<driver::Command>("/command");
    }

    bool waitForReady() {
        ROS_INFO("Waiting for /command service, /odom, and /fg_readings");
        if (!command_client_.waitForExistence(ros::Duration(30.0))) {
            ROS_ERROR("Timed out waiting for /command service");
            return false;
        }

        const ros::Time deadline = ros::Time::now() + ros::Duration(30.0);
        ros::Rate rate(20.0);
        while (ros::ok() && ros::Time::now() < deadline) {
            ros::spinOnce();
            if (has_odom_ && has_fg_) {
                ROS_INFO("Mission inputs ready at E=%.2f N=%.2f FG=%.3f", odom_.x, odom_.y, filtered_fg_);
                return true;
            }
            rate.sleep();
        }

        ROS_ERROR("Timed out waiting for odometry and FG readings");
        return false;
    }

    void run() {
        mission_start_ = ros::Time::now();

        // 1. Find any credible voltage rise near the start position.
        ROS_INFO("STATE SEARCH");
        CenterResult first_hit = searchForPipeline();
        if (!first_hit.found) {
            ROS_ERROR("Pipeline search failed before timeout");
            return;
        }

        // 2. Convert the first hit into a centerline sample with an east-west peak scan.
        ROS_INFO("STATE CENTER_ON_PIPE");
        CenterResult centered = centerAtY(first_hit.y, center_scan_radius_m_, seen_threshold_);
        if (!centered.found) {
            ROS_WARN("Initial local centering failed; trying wider scan");
            centered = centerAtY(first_hit.y, wide_scan_radius_m_, seen_threshold_);
        }
        if (!centered.found) {
            ROS_ERROR("Unable to center on the first pipeline detection");
            return;
        }
        centers_.push_back(centered);

        // 3. The assignment asks to start from the southernmost point before tracking north.
        ROS_INFO("STATE ACQUIRE_SOUTH_END");
        CenterResult south = acquireSouthEnd(centered);
        if (!south.found) {
            ROS_ERROR("Unable to acquire southern endpoint");
            return;
        }
        centers_.push_back(south);

        // 4. Follow the pipe north until sustained signal loss or timeout.
        ROS_INFO("STATE TRACK_NORTH");
        trackNorth(south);
    }

private:
    struct FgSample {
        ros::Time stamp;
        double value = 0.0;
    };

    struct CenterResult {
        bool found = false;
        double x   = 0.0;
        double y   = 0.0;
        double fg  = 0.0;
    };

    void odomCallback(const geometry_msgs::Point::ConstPtr &msg) {
        odom_     = *msg;
        has_odom_ = true;
    }

    void fgCallback(const std_msgs::Float32::ConstPtr &msg) {
        const double raw = msg->data;
        if (!has_fg_) {
            filtered_fg_ = raw;
        } else {
            // Light smoothing is enough for simulator noise without hiding the pipe edge.
            constexpr double alpha = 0.35;
            filtered_fg_ = alpha * raw + (1.0 - alpha) * filtered_fg_;
        }

        has_fg_ = true;
        fg_samples_.push_back({ros::Time::now(), raw});
        while (fg_samples_.size() > 100) {
            fg_samples_.pop_front();
        }
    }

    bool timedOut() const {
        return !mission_start_.isZero() && (ros::Time::now() - mission_start_).toSec() > mission_timeout_s_;
    }

    bool shouldStop() const { return mission_aborted_ || timedOut(); }

    void spinFor(const double seconds) {
        const ros::Time end = ros::Time::now() + ros::Duration(seconds);
        ros::Rate rate(50.0);
        while (ros::ok() && ros::Time::now() < end) {
            ros::spinOnce();
            rate.sleep();
        }
    }

    double recentFg(const double window_s) const {
        const ros::Time start = ros::Time::now() - ros::Duration(window_s);
        double sum = 0.0;
        int count  = 0;
        for (const FgSample &sample : fg_samples_) {
            if (sample.stamp >= start) {
                sum += sample.value;
                ++count;
            }
        }

        if (count > 0) {
            return sum / count;
        }
        return has_fg_ ? filtered_fg_ : 0.0;
    }

    double sampleFg(const double seconds) {
        spinFor(seconds);
        return recentFg(std::max(seconds, sample_window_s_));
    }

    double clampX(const double x) const { return std::clamp(x, -operating_limit_m_, operating_limit_m_); }

    double clampY(const double y) const {
        return std::clamp(y, -operating_limit_m_, std::min(operating_limit_m_, max_northing_m_));
    }

    bool sendCommand(const std::string &command) {
        driver::Command srv;
        srv.request.command = command;
        if (!command_client_.call(srv)) {
            ROS_WARN("Command service call failed for '%s'", command.c_str());
            mission_aborted_ = true;
            return false;
        }
        if (!srv.response.success) {
            ROS_WARN("Command '%s' rejected: %s", command.c_str(), srv.response.message.c_str());
            mission_aborted_ = true;
            return false;
        }
        // Driver handles 10 Hz pacing; this wait gives odom/FG time to reflect the accepted step.
        spinFor(step_wait_s_);
        return true;
    }

    bool moveOneStepTowardX(const double target_x) {
        const double error = clampX(target_x) - odom_.x;
        if (std::fabs(error) <= move_tolerance_m_) {
            return true;
        }
        return sendCommand(error > 0.0 ? "east" : "west");
    }

    bool moveOneStepTowardY(const double target_y) {
        const double error = clampY(target_y) - odom_.y;
        if (std::fabs(error) <= move_tolerance_m_) {
            return true;
        }
        return sendCommand(error > 0.0 ? "north" : "south");
    }

    bool moveToX(const double target_x, const double timeout_s = 60.0) {
        const double safe_target_x = clampX(target_x);
        const ros::Time deadline = ros::Time::now() + ros::Duration(timeout_s);
        while (ros::ok() && ros::Time::now() < deadline && !shouldStop()) {
            ros::spinOnce();
            if (std::fabs(safe_target_x - odom_.x) <= move_tolerance_m_) {
                return true;
            }
            if (!moveOneStepTowardX(safe_target_x)) {
                return false;
            }
        }
        return false;
    }

    bool moveToY(const double target_y, const double timeout_s = 60.0) {
        const double safe_target_y = clampY(target_y);
        const ros::Time deadline = ros::Time::now() + ros::Duration(timeout_s);
        while (ros::ok() && ros::Time::now() < deadline && !shouldStop()) {
            ros::spinOnce();
            if (std::fabs(safe_target_y - odom_.y) <= move_tolerance_m_) {
                return true;
            }
            if (!moveOneStepTowardY(safe_target_y)) {
                return false;
            }
        }
        return false;
    }

    bool moveTo(const double target_x, const double target_y, const double timeout_s = 120.0) {
        const double safe_target_x = clampX(target_x);
        const double safe_target_y = clampY(target_y);
        const ros::Time deadline = ros::Time::now() + ros::Duration(timeout_s);
        while (ros::ok() && ros::Time::now() < deadline && !shouldStop()) {
            ros::spinOnce();
            const double x_error = safe_target_x - odom_.x;
            const double y_error = safe_target_y - odom_.y;
            if (std::fabs(x_error) <= move_tolerance_m_ && std::fabs(y_error) <= move_tolerance_m_) {
                return true;
            }

            const bool moved =
                    std::fabs(x_error) > std::fabs(y_error) ? moveOneStepTowardX(safe_target_x) : moveOneStepTowardY(safe_target_y);
            if (!moved) {
                return false;
            }
        }
        return false;
    }

    std::vector<double> northFirstRowOffsets(const double max_offset, const double step) const {
        std::vector<double> offsets{0.0};
        for (double offset = step; offset <= max_offset + 1e-6; offset += step) {
            offsets.push_back(offset);
        }
        for (double offset = step; offset <= max_offset + 1e-6; offset += step) {
            offsets.push_back(-offset);
        }
        return offsets;
    }

    CenterResult scanToXForFirstHit(const double target_x, const double threshold) {
        CenterResult hit;
        const double safe_target_x = clampX(target_x);
        const ros::Time deadline = ros::Time::now() + ros::Duration(90.0);
        while (ros::ok() && ros::Time::now() < deadline && !shouldStop()) {
            ros::spinOnce();
            const double fg = recentFg(0.35);
            if (fg >= threshold) {
                hit.found = true;
                hit.x     = odom_.x;
                hit.y     = odom_.y;
                hit.fg    = fg;
                return hit;
            }

            if (std::fabs(safe_target_x - odom_.x) <= move_tolerance_m_) {
                return hit;
            }
            if (!moveOneStepTowardX(safe_target_x)) {
                return hit;
            }
        }
        return hit;
    }

    CenterResult searchForPipeline() {
        const double start_x = odom_.x;
        const double start_y = odom_.y;
        const auto offsets   = northFirstRowOffsets(search_max_row_offset_m_, search_row_step_m_);
        const double left_x  = start_x - search_half_width_m_;
        const double right_x = start_x + search_half_width_m_;
        bool scan_right      = true;

        // North-first zigzag: the pipe is North-South, so avoid wasting early rows south of the start.
        for (const double offset : offsets) {
            if (!ros::ok() || shouldStop()) {
                break;
            }

            const double row_y = clampY(start_y + offset);
            ROS_INFO("Search row N=%.2f", row_y);
            if (!moveToY(row_y)) {
                continue;
            }

            CenterResult hit = scanToXForFirstHit(scan_right ? right_x : left_x, seen_threshold_);
            if (hit.found) {
                ROS_INFO("Pipeline seen at E=%.2f N=%.2f FG=%.3f", hit.x, hit.y, hit.fg);
                return hit;
            }
            scan_right = !scan_right;
        }

        return {};
    }

    CenterResult centerAtY(const double target_y, const double radius, const double threshold) {
        CenterResult result;
        if (!moveToY(target_y)) {
            return result;
        }

        const double center_x = odom_.x;
        const double left_x   = clampX(center_x - radius);
        const double right_x  = clampX(center_x + radius);
        if (!moveToX(left_x)) {
            return result;
        }

        // The FG model peaks over the centerline, so local maximum is better than a fixed target voltage.
        double best_fg = -std::numeric_limits<double>::infinity();
        double best_x  = odom_.x;
        const ros::Time deadline = ros::Time::now() + ros::Duration(120.0);
        while (ros::ok() && ros::Time::now() < deadline && !shouldStop()) {
            ros::spinOnce();
            const double fg = recentFg(sample_window_s_);
            if (fg > best_fg) {
                best_fg = fg;
                best_x  = odom_.x;
            }

            if (std::fabs(right_x - odom_.x) <= move_tolerance_m_) {
                break;
            }
            if (!moveOneStepTowardX(right_x)) {
                break;
            }
        }

        if (best_fg < threshold) {
            ROS_WARN("Center scan at N=%.2f failed: best FG %.3f below %.3f", target_y, best_fg, threshold);
            return result;
        }

        if (!moveToX(best_x)) {
            return result;
        }
        const double centered_fg = sampleFg(sample_window_s_);
        result.found            = true;
        result.x                = odom_.x;
        result.y                = odom_.y;
        result.fg               = std::max(best_fg, centered_fg);
        ROS_INFO("Centered at E=%.2f N=%.2f FG=%.3f", result.x, result.y, result.fg);
        return result;
    }

    CenterResult fastCenterNearY(
            const double target_y,
            const double predicted_x,
            const double radius,
            const double threshold) {
        CenterResult result;
        if (!moveTo(predicted_x, target_y)) {
            return result;
        }

        const double predicted_fg = sampleFg(sample_window_s_);
        if (predicted_fg >= center_accept_threshold_) {
            result.found = true;
            result.x     = odom_.x;
            result.y     = odom_.y;
            result.fg    = predicted_fg;
            ROS_INFO("Fast-centered at E=%.2f N=%.2f FG=%.3f", result.x, result.y, result.fg);
            return result;
        }

        return centerAtY(target_y, radius, threshold);
    }

    CenterResult acquireSouthEnd(const CenterResult &start) {
        CenterResult last_good   = start;
        CenterResult last_strong = start.fg >= strong_threshold_ ? start : CenterResult{};
        CenterResult current     = start;
        int consecutive_losses   = 0;

        // Step south until the pipe signal is lost, then return to the last strong centered sample.
        for (int step = 0; ros::ok() && !shouldStop() && step < 80; ++step) {
            const double next_y = current.y - fine_row_step_m_;
            CenterResult centered = fastCenterNearY(next_y, current.x, south_scan_radius_m_, track_threshold_);
            if (!centered.found) {
                ++consecutive_losses;
            } else {
                current           = centered;
                last_good         = centered;
                consecutive_losses = centered.fg >= track_threshold_ ? 0 : consecutive_losses + 1;
                if (centered.fg >= strong_threshold_) {
                    last_strong = centered;
                }
                centers_.push_back(centered);
            }

            if (consecutive_losses >= 2) {
                break;
            }
        }

        CenterResult south = last_strong.found ? last_strong : last_good;
        if (!south.found) {
            return {};
        }

        moveTo(south.x, south.y);
        south.x  = odom_.x;
        south.y  = odom_.y;
        south.fg = sampleFg(sample_window_s_);
        ROS_INFO("Southern endpoint estimate E=%.2f N=%.2f FG=%.3f", south.x, south.y, south.fg);
        return south;
    }

    double predictedNextX(const CenterResult &last, const double next_y) const {
        if (centers_.size() < 2) {
            return last.x;
        }

        const CenterResult &a = centers_[centers_.size() - 2];
        const CenterResult &b = centers_[centers_.size() - 1];
        const double dy       = b.y - a.y;
        if (std::fabs(dy) < 1e-6) {
            return b.x;
        }

        // Clamp prediction slope so a noisy center sample cannot command a wild cross-track jump.
        const double slope = std::clamp((b.x - a.x) / dy, -3.0, 3.0);
        return b.x + slope * (next_y - b.y);
    }

    CenterResult reacquire(const double predicted_x, const double target_y) {
        const std::vector<double> y_offsets{0.0, row_step_m_, -row_step_m_, 2.0 * row_step_m_, -2.0 * row_step_m_};
        for (double radius = 1.0; radius <= reacquire_max_radius_m_ + 1e-6; radius += 1.0) {
            // Expanding scans handle the simulator's curved polyline without hard-coding hidden waypoints.
            for (const double y_offset : y_offsets) {
                if (!ros::ok() || shouldStop()) {
                    return {};
                }

                const double y = clampY(target_y + y_offset);
                moveTo(predicted_x, y);
                CenterResult centered = centerAtY(y, radius, track_threshold_);
                if (centered.found) {
                    ROS_INFO("Reacquired pipe at E=%.2f N=%.2f FG=%.3f", centered.x, centered.y, centered.fg);
                    return centered;
                }
            }
        }
        return {};
    }

    void finishNorthernEndpoint(const CenterResult &last) {
        CenterResult current = last;
        CenterResult best    = last;
        int consecutive_losses = 0;

        const double first_y = clampY(last.y - row_step_m_);
        const double last_y  = clampY(last.y + endpoint_probe_m_);
        ROS_INFO("STATE FINISH_NORTH_END from N=%.2f to N=%.2f", first_y, last_y);

        for (double y = first_y; ros::ok() && !shouldStop() && y <= last_y + 1e-6; y += endpoint_step_m_) {
            CenterResult centered = fastCenterNearY(y, current.x, center_scan_radius_m_, track_threshold_);
            if (!centered.found) {
                ++consecutive_losses;
                if (consecutive_losses >= 3 && y > last.y) {
                    break;
                }
                continue;
            }

            consecutive_losses = 0;
            current = centered;
            centers_.push_back(centered);
            if (centered.y >= best.y || centered.fg > best.fg) {
                best = centered;
            }
        }

        moveTo(best.x, best.y);
        ROS_INFO("Northern endpoint estimate E=%.2f N=%.2f FG=%.3f", best.x, best.y, best.fg);
    }

    void trackNorth(const CenterResult &south) {
        CenterResult last = south;
        int consecutive_losses = 0;

        while (ros::ok() && !shouldStop() && last.y < max_northing_m_) {
            const double next_y     = last.y + row_step_m_;
            const double predicted_x = predictedNextX(last, next_y);

            CenterResult centered = fastCenterNearY(next_y, predicted_x, center_scan_radius_m_, track_threshold_);
            if (!centered.found) {
                ROS_WARN("Track loss near E=%.2f N=%.2f; starting reacquire", predicted_x, next_y);
                centered = reacquire(predicted_x, next_y);
                if (centered.found && centered.y <= last.y + fine_row_step_m_) {
                    ROS_INFO("Reacquire returned to N=%.2f after loss ahead; treating this as the northern endpoint", centered.y);
                    centers_.push_back(centered);
                    finishNorthernEndpoint(centered);
                    return;
                }
            }

            if (!centered.found) {
                ++consecutive_losses;
                if (consecutive_losses >= 2 && last.y > south.y + 1.0) {
                    ROS_INFO("Sustained northward loss after last center E=%.2f N=%.2f", last.x, last.y);
                    finishNorthernEndpoint(last);
                    return;
                }
                continue;
            }

            consecutive_losses = 0;
            centers_.push_back(centered);
            last = centered;
        }

        if (mission_aborted_) {
            ROS_WARN("Mission stopped because the command service became unavailable");
        } else if (timedOut()) {
            ROS_WARN("Mission timeout reached");
        } else {
            ROS_INFO("Reached configured northing limit; mission stopping");
        }
    }

    /* ROS Interface */
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    ros::Subscriber odom_sub_;
    ros::Subscriber fg_sub_;
    ros::ServiceClient command_client_;

    /* Latest Sensor State */
    geometry_msgs::Point odom_;
    bool has_odom_ = false;
    bool has_fg_   = false;
    bool mission_aborted_ = false;
    double filtered_fg_ = 0.0;
    std::deque<FgSample> fg_samples_;
    std::vector<CenterResult> centers_;

    /* Mission History */
    ros::Time mission_start_;

    /* Tunable Strategy Parameters */
    double seen_threshold_ = 0.20;
    double track_threshold_ = 0.18;
    double strong_threshold_ = 0.45;
    double center_accept_threshold_ = 0.65;
    double row_step_m_ = 0.50;
    double fine_row_step_m_ = 0.25;
    double search_half_width_m_ = 5.0;
    double search_max_row_offset_m_ = 8.0;
    double search_row_step_m_ = 0.50;
    double center_scan_radius_m_ = 0.75;
    double south_scan_radius_m_ = 0.85;
    double wide_scan_radius_m_ = 1.6;
    double reacquire_max_radius_m_ = 4.0;
    double move_tolerance_m_ = 0.035;
    double step_wait_s_ = 0.11;
    double sample_window_s_ = 0.15;
    double mission_timeout_s_ = 840.0;
    double max_northing_m_ = 45.0;
    double operating_limit_m_ = 45.0;
    double endpoint_step_m_ = 0.10;
    double endpoint_probe_m_ = 0.80;
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "mission_node");
    MissionPlanner planner;
    if (!planner.waitForReady()) {
        return 1;
    }
    planner.run();
    return 0;
}
